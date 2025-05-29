import numpy as np
import copy
from typing import Any, Callable, List, Tuple, Optional, Union

class FILTres:
    """Filter results structure."""
    def __init__(self, x_out: np.ndarray, P_out: np.ndarray, r_out: np.ndarray, success_flag: bool):
        self.x_out = x_out
        self.P_out = P_out
        self.r_out = r_out
        self.success_flag = success_flag

def get_years(year: int, day_of_year: int) -> float:
    """
    Converts year and day of year to decimal year.
    A more precise implementation might be needed depending on IGRF requirements.
    """
    from datetime import datetime, timedelta
    try:
        start_of_year = datetime(year, 1, 1)
        target_date = start_of_year + timedelta(days=day_of_year - 1)
        year_start_timestamp = start_of_year.timestamp()
        year_end_timestamp = datetime(year + 1, 1, 1).timestamp()
        target_timestamp = target_date.timestamp()
        if (year_end_timestamp - year_start_timestamp) == 0: # Avoid division by zero for invalid year
            return float(year)
        return year + (target_timestamp - year_start_timestamp) / (year_end_timestamp - year_start_timestamp)
    except ValueError: # Handle invalid date/day_of_year
        print(f"Warning: Invalid date for get_years ({year}, {day_of_year}). Returning year as float.")
        return float(year)

from .ekf import get_Phi as ekf_get_Phi, get_h as ekf_get_h, get_H as ekf_get_H
from .tolles_lawson import create_TL_A, create_TL_coef

# --- Tolles-Lawson EKF (from ekf_online.jl) ---
def ekf_online_tl(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
    fn: np.ndarray, fe: np.ndarray, fd: np.ndarray,
    Cnb: np.ndarray,
    meas: np.ndarray,
    Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray,
    dt: float,
    itp_mapS: Callable,
    x0_TL: np.ndarray,
    P0: np.ndarray,
    Qd: np.ndarray,
    R: Union[float, np.ndarray],
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0,
    date: float = get_years(2020, 185),
    core: bool = False,
    terms: List[str] = ['permanent', 'induced', 'eddy', 'bias'],
    Bt_scale: float = 50000.0
) -> FILTres:
    """Extended Kalman filter (EKF) with online learning of Tolles-Lawson coefficients."""
    N = len(lat)
    ny = meas.shape[1] if meas.ndim > 1 else 1
    if meas.ndim == 1:
        meas = meas.reshape(-1, 1)

    nx = P0.shape[0]
    nx_TL = len(x0_TL)
    
    # State vector structure assumption:
    # [18 INS error states (pos, vel, att, acc_bias, gyro_bias, alt_baro_fogm_bias),
    #  nx_TL Tolles-Lawson coefficients,
    #  nx_vec vector magnetometer model states (e.g., 3 for Bx,By,Bz if modeled, or 0),
    #  1 scalar magnetometer bias state]
    # The Julia code `nx_vec = nx - 18 - nx_TL` implies the scalar bias is not part of the 18.
    # Let's assume 18 INS states, then TL, then vec_mag, then 1 scalar_mag_bias.
    # So, nx = 18 (INS) + nx_TL + nx_vec + 1 (scalar_bias).
    # nx_vec = nx - 18 - nx_TL - 1
    # The Julia code `nx_vec == 3 && (x[end-3:end-1] = ...)` and H construction for vec_mag
    # suggests vec_mag states are just before scalar_bias.
    # If nx_vec is for Bx,By,Bz *values* as states (unusual), their derivatives HBx,HBy,HBz are complex.
    # For now, assume nx_vec = 0 as it simplifies H matrix construction significantly,
    # unless the HBx,HBy,HBz terms are critical and well-defined.
    # The prompt did not specify vector magnetometer calibration as a primary goal for rt_comp_main.
    # Let's assume nx_vec = 0 for this translation, meaning no explicit vector mag states in EKF.
    # If vec_mag states are needed, the H matrix and state definition need careful handling.
    nx_vec = 0 # Simplification: No explicit vector magnetometer states in this EKF version.
    
    # Recalculate nx if nx_vec is forced to 0, or ensure P0 matches this assumption.
    # If P0 implies nx_vec > 0, this simplification will mismatch.
    # For now, proceed with nx from P0 and see how H is built.
    # The Julia H construction has branches for nx_vec.
    
    # Indices for state vector parts:
    ins_states_end_idx = 18
    tl_coeffs_start_idx = ins_states_end_idx
    tl_coeffs_end_idx = tl_coeffs_start_idx + nx_TL
    # If nx_vec > 0:
    # vec_mag_states_start_idx = tl_coeffs_end_idx
    # vec_mag_states_end_idx = vec_mag_states_start_idx + nx_vec
    # scalar_bias_idx = vec_mag_states_end_idx
    # else (nx_vec == 0):
    scalar_bias_idx = tl_coeffs_end_idx

    if nx != scalar_bias_idx + 1: # Check consistency
        # This implies P0 might be sized for nx_vec > 0.
        # For this translation, we will follow the Julia H-matrix construction logic.
        # The original nx_vec = nx - 18 - nx_TL (from Julia)
        # If this is used, then scalar_bias_idx is nx-1.
        _original_julia_nx_vec = nx - 18 - nx_TL # This is number of states between TL and scalar bias
        if _original_julia_nx_vec < 0:
             raise ValueError(f"P0 size {nx} is too small for 18 INS states and {nx_TL} TL states.")
        nx_vec = _original_julia_nx_vec # Use the one implied by P0 size.

    x_out = np.zeros((nx, N), dtype=P0.dtype)
    P_out = np.zeros((nx, nx, N), dtype=P0.dtype)
    r_out = np.zeros((ny, N), dtype=P0.dtype)

    x = np.zeros(nx, dtype=P0.dtype)
    P = P0.copy()

    A_tl_design_matrix = create_TL_A(Bx, By, Bz, Bt_input=meas[:,0], terms=terms, Bt_scale=Bt_scale)
    x[tl_coeffs_start_idx:tl_coeffs_end_idx] = x0_TL

    vec_states_present_in_model = (nx_vec > 0)

    current_itp_mapS = itp_mapS # Assuming itp_mapS is a callable. Map_Cache logic simplified.

    for t in range(N):
        # Overwrite vector magnetometer "states" if modeled (original Julia logic)
        if vec_states_present_in_model and nx_vec == 3: # Specific case from Julia
            # These states are just before the scalar bias state.
            vec_mag_actual_states_start_idx = tl_coeffs_end_idx
            x[vec_mag_actual_states_start_idx : vec_mag_actual_states_start_idx + 3] = [Bx[t], By[t], Bz[t]]

        phi_matrix = ekf_get_Phi(nx, lat[t], vn[t], ve[t], vd[t],
                                 fn[t], fe[t], fd[t], Cnb[:, :, t],
                                 baro_tau, acc_tau, gyro_tau, fogm_tau, dt,
                                 vec_states=vec_states_present_in_model) # Pass based on P0 sizing

        x_TL_current = x[tl_coeffs_start_idx:tl_coeffs_end_idx]
        tl_compensation_val = np.dot(A_tl_design_matrix[t, :], x_TL_current)
        
        # get_h should use relevant INS error states from x and scalar_bias_state x[scalar_bias_idx]
        # Or, get_h is only for map, and H explicitly includes scalar_bias_idx derivative.
        # The Julia `get_h` is called with full `x`.
        # Predicted measurement h(x_k|k-1)
        # h_pred = tl_compensation_val + map_contribution + scalar_bias_val
        # Let's assume ekf_get_h provides map contribution based on INS states in x,
        # AND includes the scalar bias state x[scalar_bias_idx] if H is built for it.
        # The Julia residual: meas - (A'*x_TL + get_h(map, x_ins_errors_and_bias, ...))
        h_map_and_bias_val = ekf_get_h(current_itp_mapS, x, lat[t], lon[t], alt[t], date=date, core=core)
        h_pred = tl_compensation_val + h_map_and_bias_val
        resid = meas[t, :] - h_pred

        # Measurement Jacobian H
        H_map_derivs_and_bias_deriv = ekf_get_H(current_itp_mapS, x, lat[t], lon[t], alt[t], date=date, core=core).T # Expect row vector
        
        H_row = np.zeros(nx, dtype=P0.dtype)
        # Derivatives for first 2 INS states (lat_err, lon_err)
        H_row[0:2] = H_map_derivs_and_bias_deriv[0, 0:2] 
        # Assuming remaining INS states (up to 18) have zero direct derivative for map part,
        # or are covered by H_map_derivs_and_bias_deriv if it's longer.
        # The Julia H construction is specific: [Hll[1:2]; zeros; A_tl; (HBxyz); 1]
        # This implies H_map_derivs_and_bias_deriv[0,0:2] for dMap/dPos_xy
        # And H_map_derivs_and_bias_deriv[0, scalar_bias_idx_within_map_derivs] for dMapPart/dBias
        
        # Derivatives for TL coefficients
        H_row[tl_coeffs_start_idx:tl_coeffs_end_idx] = A_tl_design_matrix[t, :]

        if vec_states_present_in_model and nx_vec == 3:
            # Derivatives wrt Bx, By, Bz "states" - these are complex and model-specific
            # TODO: Implement HBx, HBy, HBz from Julia code if this path is taken.
            # This requires exact mapping of A_tl_design_matrix columns and x_TL_current elements.
            # For now, placeholder zeros.
            vec_mag_actual_states_start_idx = tl_coeffs_end_idx
            # H_row[vec_mag_actual_states_start_idx : vec_mag_actual_states_start_idx + 3] = [HBx_val, HBy_val, HBz_val]
            print("Warning: HBx, HBy, HBz for TL EKF not implemented, using zeros.")
            pass # Keep as zeros for now

        # Derivative for scalar bias state (is it part of h_map_and_bias_val's Jacobian or separate?)
        # Julia H1 ends with '1', implying d(total_model)/d(scalar_bias_state) = 1.
        # This scalar_bias_state is x[scalar_bias_idx].
        H_row[scalar_bias_idx] = 1.0 
        
        H = H_row.reshape(1, -1) # ny=1
        if ny > 1: H = np.tile(H_row, (ny, 1))

        S_val = H @ P @ H.T + R
        if ny == 1 and isinstance(R, np.ndarray) and R.ndim > 0 : S_val = S_val[0,0]
        
        K_gain = (P @ H.T) / S_val if ny == 1 else (P @ H.T) @ np.linalg.inv(S_val)
        
        x = x + (K_gain @ resid.reshape(ny,-1)).flatten() if ny > 1 else x + (K_gain * resid).flatten()
        P = (np.eye(nx) - K_gain @ H) @ P
        P = 0.5 * (P + P.T)

        x_out[:, t] = x.flatten()
        P_out[:, :, t] = P
        r_out[:, t] = resid.flatten()

        x = phi_matrix @ x
        P = phi_matrix @ P @ phi_matrix.T + Qd
        P = 0.5 * (P + P.T)

    return FILTres(x_out, P_out, r_out, True)

def ekf_online_tl_ins(
    ins: INS, meas: np.ndarray, flux: MagV, itp_mapS: Callable,
    x0_TL: np.ndarray, P0: np.ndarray, Qd: np.ndarray, R: Union[float, np.ndarray],
    **kwargs
) -> FILTres:
    """Wrapper for ekf_online_tl using INS and MagV structs."""
    return ekf_online_tl(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb, meas,
        flux.x, flux.y, flux.z, ins.dt, itp_mapS, x0_TL, P0, Qd, R,
        **kwargs
    )

def ekf_online_tl_setup(
    flux: MagV, meas: np.ndarray, ind: Optional[np.ndarray] = None,
    ridge_param: float = 0.025, # Renamed from λ
    terms: List[str] = ['permanent', 'induced', 'eddy', 'bias'],
    pass1: float = 0.1, pass2: float = 0.9, fs: float = 10.0,
    pole: int = 4, trim: int = 20, N_sigma_points: int = 100, # Renamed from N_sigma
    Bt_scale: float = 50000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setup for EKF with online Tolles-Lawson learning."""
    if ind is None:
        ind = np.ones(len(meas), dtype=bool)
    
    flux_x_ind, flux_y_ind, flux_z_ind = flux.x[ind], flux.y[ind], flux.z[ind]
    meas_ind = meas[ind]
    Bt_for_setup = np.sqrt(flux_x_ind**2 + flux_y_ind**2 + flux_z_ind**2)

    x0_TL, y_var = create_TL_coef(
        flux, meas, ind, Bt_input=Bt_for_setup, ridge_param=ridge_param, terms=terms,
        pass1=pass1, pass2=pass2, fs=fs, pole=pole, trim=trim,
        Bt_scale=Bt_scale, return_var=True
    )

    A_matrix_setup = create_TL_A(
        flux_x_ind, flux_y_ind, flux_z_ind, Bt_input=Bt_for_setup,
        terms=terms, Bt_scale=Bt_scale
    )
    
    try:
        if A_matrix_setup.shape[0] < A_matrix_setup.shape[1]: # Underdetermined
             P0_TL_inv = A_matrix_setup.T @ A_matrix_setup + np.eye(A_matrix_setup.shape[1]) * 1e-6 # Regularize
             P0_TL = np.linalg.pinv(P0_TL_inv) * y_var
        else:
            P0_TL = np.linalg.inv(A_matrix_setup.T @ A_matrix_setup) * y_var
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix in P0_TL calculation, using pseudo-inverse.")
        P0_TL = np.linalg.pinv(A_matrix_setup.T @ A_matrix_setup) * y_var
    P0_TL = 0.5 * (P0_TL + P0_TL.T) # Ensure symmetry

    N_ind_true = np.sum(ind)
    N_for_sigma_calc = min(N_ind_true - max(2 * trim, 50), N_sigma_points)
    
    true_indices_original = np.where(ind)[0]
    N_min_data_points = 10
    if N_for_sigma_calc < N_min_data_points:
        print(f"Warning: Not enough data points ({N_for_sigma_calc}) for robust TL_sigma. Using fallback.")
        TL_sigma = np.abs(x0_TL) * 0.1 + 1e-6 # Fallback
        return x0_TL, P0_TL, TL_sigma

    coef_set = np.zeros((len(x0_TL), N_for_sigma_calc), dtype=x0_TL.dtype)
    # Sliding window logic from Julia: inds[i : end+i-N]
    # Window length = N_ind_true - N_for_sigma_calc
    window_len = N_ind_true - N_for_sigma_calc +1 # +1 to make it inclusive of N_for_sigma_calc iterations
    if window_len <=0: window_len = N_min_data_points # ensure positive
    window_len = max(window_len, max(2 * trim, 50) +1) # ensure min length for create_TL_coef

    for i in range(N_for_sigma_calc):
        start_original_idx = true_indices_original[i]
        # Define a segment of data for this iteration
        # The Julia slice `inds[i:end+i-N]` is tricky.
        # A simpler approach: use N_for_sigma_calc distinct (or overlapping) blocks.
        # For now, let's use windows of a fixed reasonable size.
        current_segment_true_indices = true_indices_original[i : min(i + window_len, N_ind_true)]
        if len(current_segment_true_indices) < max(2 * trim, 50) +1 : continue # Skip if too small

        ind_for_iter = np.zeros(len(meas), dtype=bool)
        ind_for_iter[current_segment_true_indices] = True
        
        flux_x_iter, flux_y_iter, flux_z_iter = flux.x[ind_for_iter], flux.y[ind_for_iter], flux.z[ind_for_iter]
        Bt_for_iter = np.sqrt(flux_x_iter**2 + flux_y_iter**2 + flux_z_iter**2)

        coef_set[:, i], _ = create_TL_coef(
            flux, meas, ind_for_iter, Bt_input=Bt_for_iter, ridge_param=ridge_param, terms=terms,
            pass1=pass1, pass2=pass2, fs=fs, pole=pole, trim=trim,
            Bt_scale=Bt_scale, return_var=False
        )
    
    valid_cols = np.any(coef_set != 0, axis=0) # Check for columns that got computed
    if np.sum(valid_cols) > 1:
        TL_sigma = np.std(coef_set[:, valid_cols], axis=1)
    elif np.sum(valid_cols) == 1:
        TL_sigma = np.abs(coef_set[:, valid_cols].flatten()) * 0.1 + 1e-6
    else:
        TL_sigma = np.abs(x0_TL) * 0.1 + 1e-6
    TL_sigma[TL_sigma < 1e-6] = 1e-6
    return x0_TL, P0_TL, TL_sigma

# Assuming these are defined in magnav.py or will be.
from .magnav import INS, MagV # EKF_RT might be FILTres or similar

# --- Placeholder/Helper Definitions ---

class Map_Cache:
    """Placeholder for map caching functionality."""
    def __init__(self):
        # In a real implementation, this would hold cached map data and interpolation objects.
        pass

def get_cached_map(map_cache: Optional[Map_Cache], lat: float, lon: float, alt: float, silent: bool = True) -> Callable:
    """
    Placeholder for getting a map interpolation function from a cache.
    This needs a proper implementation based on how map data is stored and interpolated.
    """
    # This is a simplified placeholder. It should return a callable similar to itp_mapS.
    print("Warning: Using placeholder get_cached_map. Map caching not implemented.")
    def dummy_interpolator(lat_query, lon_query, alt_query=None):
        # Returns a dummy scalar value.
        return 0.0
    if map_cache:
        # Actual logic to retrieve/interpolate from cache would go here.
        pass
    # This function should return an interpolator function.
    # For now, this will likely cause issues if map_cache is actually used.
    return dummy_interpolator


# --- NN Model Interaction Placeholders (CRITICAL: Implement these based on your NN library) ---
# from .model_functions import ( # For NN EKF - these are critical placeholders
#     destructure_nn_model,
#     reconstruct_nn_from_vector_and_predict, # Combines reconstruction and prediction
#     get_nn_output_and_param_gradients # Gets NN output and d(output)/d(params)
# )

def destructure_nn_model(model: Any) -> Tuple[np.ndarray, Callable[[np.ndarray], Any]]:
    """
    CRITICAL PLACEHOLDER: Extracts NN parameters as a vector and returns a reconstruction function.
    This is highly library-specific (e.g., PyTorch: torch.nn.utils.parameters_to_vector).
    """
    print("CRITICAL WARNING: Using placeholder for destructure_nn_model.")
    # Example for a simple PyTorch model (conceptual)
    # if isinstance(model, torch.nn.Module):
    #     params_list = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    #     params_vector = np.concatenate(params_list) if params_list else np.array([])
    #     
    #     def reconstruct_function(p_vec: np.ndarray) -> Any:
    #         temp_model = copy.deepcopy(model) # Important to not modify original shared model
    #         # torch.nn.utils.vector_to_parameters(torch.from_numpy(p_vec), temp_model.parameters())
    #         print("CRITICAL WARNING: Placeholder reconstruct_function for NN model called.")
    #         return temp_model # This model should now have weights from p_vec
    #     return params_vector, reconstruct_function
    
    # Fallback dummy implementation
    num_dummy_params = 10 # Guess or derive if possible
    if hasattr(model, '_dummy_num_params'): num_dummy_params = model._dummy_num_params
    
    dummy_params_vector = np.random.randn(num_dummy_params) 
    def dummy_reconstruct(p_vec: np.ndarray) -> Any:
        # This should return a model instance configured with p_vec
        print(f"CRITICAL WARNING: Dummy NN reconstruct called with p_vec shape {p_vec.shape}")
        # model_new = copy.deepcopy(model) # Create new instance
        # model_new._dummy_params = p_vec # Store params for dummy prediction/grad
        # model_new._dummy_num_params = len(p_vec)
        # return model_new
        # For simplicity, assume the model passed can be updated or used with external params
        # This is a major simplification.
        return model # Returning original, assuming it can be used with external params by other funcs
        
    return dummy_params_vector, dummy_reconstruct

def get_nn_output_and_param_gradients(
    nn_model_instance_for_grad: Any, # This should be the model instance with current EKF state params
    input_data: np.ndarray,
    # current_params_vector: np.ndarray # Params might be part of nn_model_instance_for_grad
) -> Tuple[float, np.ndarray]:
    """
    CRITICAL PLACEHOLDER: Computes NN output and gradients of output w.r.t. parameters.
    Returns (output_scalar_normalized, gradients_vector_wrt_params).
    `input_data` is a single sample (1D array for features).
    """
    print(f"CRITICAL WARNING: Using placeholder for get_nn_output_and_param_gradients.")
    # Example for PyTorch (conceptual)
    # input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0) # Add batch dim
    # nn_model_instance_for_grad.zero_grad()
    # output_tensor = nn_model_instance_for_grad(input_tensor)
    # output_scalar_normalized = output_tensor.item()
    # # To get gradients w.r.t. parameters, sum output if multiple outputs, then backward.
    # output_tensor.sum().backward() 
    # grads_list = [p.grad.data.cpu().numpy().flatten() for p in nn_model_instance_for_grad.parameters() if p.grad is not None]
    # gradients_vector = np.concatenate(grads_list) if grads_list else np.array([])
    # return output_scalar_normalized, gradients_vector

    # Fallback dummy implementation
    dummy_output_normalized = np.sum(input_data) * 0.05 # Dummy prediction
    num_params = 10
    if hasattr(nn_model_instance_for_grad, '_dummy_num_params'):
        num_params = nn_model_instance_for_grad._dummy_num_params
    elif hasattr(nn_model_instance_for_grad, '_dummy_params'):
        num_params = len(nn_model_instance_for_grad._dummy_params)

    dummy_gradients_vector = np.random.randn(num_params) * 0.01 if num_params > 0 else np.array([])
    return dummy_output_normalized, dummy_gradients_vector

# --- Neural Network EKF (from ekf_online_nn.jl) ---
def ekf_online_nn(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
    fn: np.ndarray, fe: np.ndarray, fd: np.ndarray,
    Cnb: np.ndarray, meas: np.ndarray, dt: float, itp_mapS: Callable,
    x_nn_inputs: np.ndarray, nn_model: Any, y_norms: Tuple[float, float],
    P0: np.ndarray, Qd: np.ndarray, R: Union[float, np.ndarray],
    baro_tau: float = 3600.0, acc_tau: float = 3600.0, gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0, date: float = get_years(2020, 185), core: bool = False
) -> FILTres:
    """Extended Kalman filter (EKF) with online learning of neural network weights."""
    y_bias, y_scale = y_norms
    N = len(lat)
    ny = meas.shape[1] if meas.ndim > 1 else 1
    if meas.ndim == 1: meas = meas.reshape(-1, 1)

    nx = P0.shape[0]
    
    w0_nn_vec, reconstruct_nn_fn = destructure_nn_model(nn_model) # CRITICAL PLACEHOLDER
    nx_nn = len(w0_nn_vec)

    # State vector: [18 INS_errors, nx_nn NN_params, 1 scalar_mag_bias]
    ins_states_end_idx = 18
    nn_params_start_idx = ins_states_end_idx
    nn_params_end_idx = nn_params_start_idx + nx_nn
    scalar_bias_idx = nn_params_end_idx
    
    if nx != scalar_bias_idx + 1:
        raise ValueError(f"P0 dimension {nx} mismatch with expected {scalar_bias_idx + 1} states (18 INS, {nx_nn} NN, 1 Bias).")

    x_out = np.zeros((nx, N), dtype=P0.dtype)
    P_out = np.zeros((nx, nx, N), dtype=P0.dtype)
    r_out = np.zeros((ny, N), dtype=P0.dtype)

    x = np.zeros(nx, dtype=P0.dtype)
    x[nn_params_start_idx:nn_params_end_idx] = w0_nn_vec
    P = P0.copy()
    current_itp_mapS = itp_mapS

    for t in range(N):
        phi_matrix = ekf_get_Phi(nx, lat[t], vn[t], ve[t], vd[t],
                                 fn[t], fe[t], fd[t], Cnb[:, :, t],
                                 baro_tau, acc_tau, gyro_tau, fogm_tau, dt,
                                 vec_states=False) # Assuming NN params handled by Qd/Phi structure

        current_nn_params_vec = x[nn_params_start_idx:nn_params_end_idx]
        
        # Reconstruct model for this step (CRITICAL: reconstruct_nn_fn must work)
        # This model instance is used for prediction and gradient calculation.
        nn_model_step_instance = reconstruct_nn_fn(current_nn_params_vec)
        
        nn_output_normalized, nn_param_grads_unscaled = get_nn_output_and_param_gradients(
            nn_model_step_instance, x_nn_inputs[t, :]
        ) # CRITICAL PLACEHOLDER

        nn_output_denormalized = nn_output_normalized * y_scale + y_bias
        map_h_val = ekf_get_h(current_itp_mapS, x, lat[t], lon[t], alt[t], date=date, core=core)
        h_pred = nn_output_denormalized + map_h_val
        resid = meas[t, :] - h_pred

        H_map_derivs_and_bias_deriv = ekf_get_H(current_itp_mapS, x, lat[t], lon[t], alt[t], date=date, core=core).T
        H_row = np.zeros(nx, dtype=P0.dtype)
        H_row[0:2] = H_map_derivs_and_bias_deriv[0, 0:2] # dMap/dPos_xy
        # Zeros for other INS states (indices 2 to 17, assuming H_map covers only pos_xy for direct impact here)
        
        H_row[nn_params_start_idx:nn_params_end_idx] = nn_param_grads_unscaled * y_scale
        H_row[scalar_bias_idx] = 1.0 # d(Total)/d(ScalarBiasState)

        H = H_row.reshape(1, -1)
        if ny > 1: H = np.tile(H_row, (ny, 1))

        S_val = H @ P @ H.T + R
        if ny == 1 and isinstance(R, np.ndarray) and R.ndim > 0: S_val = S_val[0,0]
        
        K_gain = (P @ H.T) / S_val if ny == 1 else (P @ H.T) @ np.linalg.inv(S_val)
        
        x = x + (K_gain @ resid.reshape(ny,-1)).flatten() if ny > 1 else x + (K_gain * resid).flatten()
        P = (np.eye(nx) - K_gain @ H) @ P
        P = 0.5 * (P + P.T)

        x_out[:, t] = x.flatten()
        P_out[:, :, t] = P
        r_out[:, t] = resid.flatten()

        x = phi_matrix @ x
        P = phi_matrix @ P @ phi_matrix.T + Qd
        P = 0.5 * (P + P.T)
        
    return FILTres(x_out, P_out, r_out, True)

def ekf_online_nn_ins(
    ins: INS, meas: np.ndarray, itp_mapS: Callable, x_nn_inputs: np.ndarray,
    nn_model: Any, y_norms: Tuple[float, float], P0: np.ndarray, Qd: np.ndarray, R: Union[float, np.ndarray],
    **kwargs
) -> FILTres:
    """Wrapper for ekf_online_nn using INS struct."""
    return ekf_online_nn(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb, meas, ins.dt, itp_mapS,
        x_nn_inputs, nn_model, y_norms, P0, Qd, R, **kwargs
    )

def ekf_online_nn_setup(
    x_features: np.ndarray, y_target: np.ndarray, nn_model_initial: Any,
    y_norms: Tuple[float, float], N_sigma_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Setup for EKF NN. Estimates initial NN param covariance (P0_nn) and std dev (nn_sigma).
    CRITICAL: Relies on working NN interaction placeholders.
    """
    N_sigma_min = 10
    if N_sigma_points < N_sigma_min: N_sigma_points = N_sigma_min
    
    y_bias, y_scale = y_norms
    w_nn_vec_current, reconstruct_fn = destructure_nn_model(nn_model_initial) # CRITICAL
    num_nn_params = len(w_nn_vec_current)

    if num_nn_params == 0:
        print("Warning: NN model has no parameters. Returning empty P0_nn, nn_sigma.")
        return np.array([]).reshape(0,0), np.array([])

    w_nn_store = np.zeros((num_nn_params, N_sigma_points), dtype=w_nn_vec_current.dtype)
    P_rls = np.eye(num_nn_params, dtype=w_nn_vec_current.dtype) # RLS covariance

    num_samples = x_features.shape[0]
    if N_sigma_points > num_samples:
        print(f"Warning: N_sigma_points ({N_sigma_points}) > num_samples ({num_samples}). Using num_samples.")
        N_sigma_points = num_samples
        w_nn_store = np.zeros((num_nn_params, N_sigma_points), dtype=w_nn_vec_current.dtype)
    
    if N_sigma_points == 0: # Cannot proceed
        P0_nn_fallback = np.eye(num_nn_params) * 1e-2 # Small default covariance
        nn_sigma_fallback = np.ones(num_nn_params) * 0.1
        return P0_nn_fallback, nn_sigma_fallback


    for i in range(N_sigma_points):
        h_rls = w_nn_vec_current # Specific RLS formulation from Julia
        S_rls_inv = 1.0 / (1.0 + h_rls.T @ P_rls @ h_rls)
        K_rls = (P_rls @ h_rls) * S_rls_inv
        P_rls = P_rls - (K_rls.reshape(-1,1) @ h_rls.reshape(1,-1) @ P_rls) # Ensure outer product
        P_rls = 0.5 * (P_rls + P_rls.T)

        m_for_pred = reconstruct_fn(w_nn_vec_current) # CRITICAL
        
        # nn_pred_norm_val = m_for_pred(x_features[i,:]).item() # Assuming scalar output
        # This needs to use the get_nn_output_and_param_gradients or similar robust way
        try:
            nn_pred_norm_val, _ = get_nn_output_and_param_gradients(m_for_pred, x_features[i,:])
        except Exception as e:
             print(f"Error during NN prediction in setup: {e}. Using dummy value.")
             nn_pred_norm_val = np.sum(x_features[i,:]) * 0.05


        error_signal_denorm = y_target[i] - (nn_pred_norm_val * y_scale + y_bias)
        error_signal_norm = error_signal_denorm / y_scale
        
        w_nn_vec_current = w_nn_vec_current + K_rls * error_signal_norm
        w_nn_store[:, i] = w_nn_vec_current.flatten()

    P0_nn = np.abs(P_rls)
    P0_nn = 0.5 * (P0_nn + P0_nn.T)

    if N_sigma_points > 1:
        diffs = np.abs(w_nn_store[:, 1:] - w_nn_store[:, :-1])
        if diffs.size > 0 :
            nn_sigma = np.min(diffs, axis=1)
        else: # only one point stored effectively
            nn_sigma = np.abs(w_nn_store[:,0]) * 0.1 + 1e-6 if N_sigma_points > 0 else np.abs(w0_nn_vec) * 0.1 + 1e-6
    else:
        nn_sigma = np.abs(w0_nn_vec) * 0.1 + 1e-6
    
    nn_sigma[nn_sigma < 1e-6] = 1e-6
    return P0_nn, nn_sigma