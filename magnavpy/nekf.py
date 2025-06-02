import numpy as np
from typing import Callable, Optional, Union, Tuple, List, Any
import math

from .magnav import FILTres, INS
from .common_types import get_cached_map, MapCache, XYZ, TRAJ # XYZ, TRAJ for training
from .model_functions import get_Phi, get_H, create_P0, create_Qd, get_h
from .core_utils import get_years
from .nav_utils import dlat2dn, dlon2de # For training loss
from .create_xyz import get_ins as get_ins_from_xyz, get_traj as get_traj_from_xyz # For nekf_train_xyz

# --- Placeholder ML Model Components (Conceptual) ---
# These are simplified placeholders to allow porting the structure of nekf_train.
# A real implementation would use PyTorch, TensorFlow, or JAX.

class SimpleLSTM:
    def __init__(self, input_features: int, hidden_size: int):
        self.input_features = input_features
        self.hidden_size = hidden_size
        # In a real model, these would be learnable parameters (weights, biases)
        self.Wh = np.random.randn(hidden_size, input_features) * 0.1
        self.Uh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros((hidden_size, 1))
        # Simplified: just one set of gates for brevity, real LSTM has 4
        self.Wi = np.random.randn(hidden_size, input_features) * 0.1
        self.Ui = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))

        self.params = [self.Wh, self.Uh, self.bh, self.Wi, self.Ui, self.bi]

    def __call__(self, x_seq: np.ndarray) -> np.ndarray:
        # x_seq: (features, seq_len)
        _features, seq_len = x_seq.shape
        h_t = np.zeros((self.hidden_size, 1))
        # c_t = np.zeros((self.hidden_size, 1)) # Cell state for full LSTM
        output_seq = np.zeros((self.hidden_size, seq_len))

        for t in range(seq_len):
            xt_col = x_seq[:, t].reshape(-1, 1)
            # Simplified LSTM-like update (not a full LSTM)
            i_t = sigmoid(self.Wi @ xt_col + self.Ui @ h_t + self.bi)
            h_tilde = np.tanh(self.Wh @ xt_col + self.Uh @ h_t + self.bh)
            h_t = i_t * h_tilde # Simplified update
            output_seq[:, t] = h_t.flatten()
        return output_seq

    def parameters(self):
        return self.params

class SimpleDense:
    def __init__(self, input_size: int, output_size: int, activation: Callable = lambda x: x):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        # Learnable parameters
        self.W = np.random.randn(output_size, input_size) * 0.1
        self.b = np.zeros((output_size, 1))
        self.params = [self.W, self.b]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (input_size, num_samples)
        return self.activation(self.W @ x + self.b)

    def parameters(self):
        return self.params

class SequentialModel:
    def __init__(self, layers: List[Any]):
        self.layers = layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                all_params.extend(layer.parameters())
        return all_params

# --- Activation Functions ---
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def swish(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

# --- Default NN model for inference if not provided by training ---
def default_nn_model(x_nn_sample_transposed: np.ndarray) -> np.ndarray:
    """
    Default placeholder for the neural network model if 'm' is not sophisticated.
    x_nn_sample_transposed: (num_features, N)
    Returns: (N,) array for R_nn component.
    """
    # This model simply returns a small constant value based on sum of features.
    # It needs to match the expected output shape for R_nn component.
    # R_nn = m(x_nn') where x_nn' is Nf x N. R_nn is then 1 x N or N.
    if x_nn_sample_transposed.ndim == 1: # Single feature over time
        return 0.01 * np.sum(x_nn_sample_transposed, axis=0) * np.ones(x_nn_sample_transposed.shape[0])
    return 0.01 * np.sum(x_nn_sample_transposed, axis=0)


# --- NEKF Inference Functions ---
def nekf(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
    fn: np.ndarray, fe: np.ndarray, fd: np.ndarray,
    Cnb: np.ndarray,
    meas: np.ndarray,
    dt: float,
    itp_mapS: Union[Callable, MapCache],
    x_nn: Optional[np.ndarray] = None, # N x Nf
    m: Optional[Callable[[np.ndarray], np.ndarray]] = None, # Model: (Nf, N_samples) -> (1, N_samples) or (N_samples,)
    P0: Optional[np.ndarray] = None,
    Qd: Optional[np.ndarray] = None,
    R: float = 1.0,
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0,
    date: Optional[float] = None,
    core: bool = False,
    der_mapS: Optional[Callable] = None,
    map_alt: float = 0
) -> FILTres:
    """
    Measurement noise covariance-adaptive neural extended Kalman filter (nEKF)
    for airborne magnetic anomaly navigation.
    """
    if P0 is None: P0 = create_P0()
    if Qd is None: Qd = create_Qd()
    if date is None: date = get_years(2020, 185)

    N = len(lat)
    nx = P0.shape[0]

    if meas.ndim == 1:
        ny = 1
        _meas_internal = meas.reshape(-1, 1) # (N,1)
    elif meas.ndim == 2 and meas.shape[1] == 1:
        ny = 1
        _meas_internal = meas
    else:
        # For future multi-dimensional measurements, ny would be meas.shape[1]
        raise ValueError("meas must be a 1D array or a 2D array with one column for scalar case.")

    x_out = np.zeros((nx, N), dtype=P0.dtype)
    P_out = np.zeros((nx, nx, N), dtype=P0.dtype)
    r_out = np.zeros((ny, N), dtype=P0.dtype) # Residuals

    # Handle x_nn and m defaults similar to Julia
    if x_nn is None:
        x_nn = _meas_internal.copy() # N x 1 (scalar measurement as feature)
    if m is None:
        # Use a default simple model if none provided.
        # This default model expects (Nf, N_samples) and returns (N_samples,)
        _m_internal = default_nn_model
    else:
        _m_internal = m
    
    # x_nn is (N, num_features), so x_nn.T is (num_features, N)
    R_nn_all = _m_internal(x_nn.T) # Expected (N,) or (1,N)
    if R_nn_all.ndim == 2 and R_nn_all.shape[0] == 1:
        R_nn_all = R_nn_all.flatten()
    if R_nn_all.shape[0] != N:
        raise ValueError(f"Output of neural model m should have length N={N}, but got {R_nn_all.shape}")

    x_state = np.zeros(nx, dtype=P0.dtype)
    P_cov = P0.copy()
    
    itp_mapS_arg = itp_mapS

    for t in range(N):
        current_itp_mapS_for_step = itp_mapS_arg
        current_der_mapS_for_step = der_mapS

        if isinstance(itp_mapS_arg, MapCache):
            current_itp_mapS_for_step = get_cached_map(itp_mapS_arg, lat[t], lon[t], alt[t], silent=True)

        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 and Cnb.shape[2] == N else Cnb # Assumes Cnb is (3,3) or (3,3,N)

        lat_t, lon_t, alt_t = lat[t], lon[t], alt[t]
        vn_t, ve_t, vd_t = vn[t], ve[t], vd[t]
        fn_t, fe_t, fd_t = fn[t], fe[t], fd[t]

        Phi_mat = get_Phi(nx, lat_t, vn_t, ve_t, vd_t, fn_t, fe_t, fd_t, _Cnb_t,
                          baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

        h_pred = get_h(current_itp_mapS_for_step, x_state, lat_t, lon_t, alt_t,
                       date=date, core=core, der_map=current_der_mapS_for_step, map_alt=map_alt)
        if isinstance(h_pred, (float, int, np.number)): h_pred = np.array([h_pred])
        if h_pred.ndim == 1: h_pred = h_pred.reshape(-1,1) # to (ny,1)

        resid = _meas_internal[t,:].reshape(-1,1) - h_pred # (ny,1)

        H_jac = get_H(current_itp_mapS_for_step, x_state, lat_t, lon_t, alt_t,
                      date=date, core=core, der_map=current_der_mapS_for_step, map_alt=map_alt) # (nx,) or (1,nx)
        if H_jac.ndim == 1: H_jac = H_jac.reshape(1, -1) # Ensure (1,nx) for scalar measurement
        # If ny > 1, H_jac would be (ny, nx) and repeat logic might be needed if get_H returns single row.
        # For scalar (ny=1), H_val is (1,nx)

        # Measurement residual covariance: S = H P H' + R_eff
        R_effective_t = R * (1 + R_nn_all[t]) # R_effective_t is scalar for ny=1
        S_val = H_jac @ P_cov @ H_jac.T + R_effective_t * np.eye(ny) # S is (ny,ny)
        if ny == 1 and S_val.ndim == 0 : S_val = S_val.reshape(1,1)


        # Kalman gain: K = P H' S^-1
        # K_val = (P_cov @ H_jac.T) @ np.linalg.inv(S_val) # (nx, ny)
        K_val = np.linalg.solve(S_val.T, H_jac @ P_cov.T).T


        # State and covariance update
        x_state = x_state + (K_val @ resid).flatten() # K_val is (nx,ny), resid is (ny,1)
        P_cov = (np.eye(nx, dtype=P_cov.dtype) - K_val @ H_jac) @ P_cov

        x_out[:, t] = x_state
        P_out[:, :, t] = P_cov
        r_out[:, t] = resid.flatten()

        # State and covariance propagate
        x_state = Phi_mat @ x_state
        P_cov = Phi_mat @ P_cov @ Phi_mat.T + Qd
    
    return FILTres(x_out, P_out, r_out, True)


def nekf_ins(
    ins: INS,
    meas: np.ndarray,
    itp_mapS: Union[Callable, MapCache],
    x_nn: Optional[np.ndarray] = None,
    m: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    P0: Optional[np.ndarray] = None,
    Qd: Optional[np.ndarray] = None,
    R: float = 1.0,
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0,
    date: Optional[float] = None,
    core: bool = False,
    der_mapS: Optional[Callable] = None,
    map_alt: float = 0
) -> FILTres:
    """ nEKF overload for INS data structure. """
    return nekf(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb,
        meas, ins.dt, itp_mapS,
        x_nn=x_nn, m=m,
        P0=P0, Qd=Qd, R=R,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        date=date, core=core, der_mapS=der_mapS, map_alt=map_alt
    )

def ekf_single(
    lat_t: float, lon_t: float, alt_t: float,
    Phi_t: np.ndarray, 
    meas_t: Union[float, np.ndarray], 
    itp_mapS_t: Callable, 
    P_prev: np.ndarray,
    Qd_step: np.ndarray, 
    R_base: float,
    R_nn_t: float, 
    x_prev: np.ndarray,
    date: Optional[float] = None,
    core: bool = False,
    der_mapS_t: Optional[Callable] = None,
    map_alt_t: float = 0,
    nx: Optional[int] = None 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper function to run an EKF for a single time step.
    Used in nEKF training. Assumes itp_mapS is not a Map_Cache.
    """
    if nx is None:
        nx = P_prev.shape[0]

    if isinstance(itp_mapS_t, MapCache):
        raise ValueError("Map_Cache not supported for ekf_single (nEKF training context)")

    if isinstance(meas_t, (float, int, np.number)):
        _meas_t_internal = np.array([[meas_t]]) # (1,1)
        ny = 1
    elif isinstance(meas_t, np.ndarray) and meas_t.size == 1:
        _meas_t_internal = meas_t.reshape(1,1)
        ny = 1
    elif isinstance(meas_t, np.ndarray): # For potential future multi-dim measurement
        _meas_t_internal = meas_t.reshape(-1,1)
        ny = _meas_t_internal.shape[0]
    else:
        raise ValueError("meas_t must be a scalar or NumPy array.")
    
    # Measurement residual
    h_pred = get_h(itp_mapS_t, x_prev, lat_t, lon_t, alt_t,
                   date=date, core=core, der_map=der_mapS_t, map_alt=map_alt_t)
    if isinstance(h_pred, (float, int, np.number)): h_pred = np.array([h_pred])
    if h_pred.ndim == 1: h_pred = h_pred.reshape(-1,1) # to (ny,1)

    resid = _meas_t_internal - h_pred # (ny,1)

    # Measurement Jacobian
    H_jac = get_H(itp_mapS_t, x_prev, lat_t, lon_t, alt_t,
                  date=date, core=core, der_map=der_mapS_t, map_alt=map_alt_t) # (nx,) or (ny,nx)
    if ny == 1 and H_jac.ndim == 1: H_jac = H_jac.reshape(1, -1) # (1,nx)
    # if ny > 1, get_H should return (ny,nx) or logic here needs to adapt

    # Measurement residual covariance
    R_effective_t = R_base * (1 + R_nn_t)
    S_val = H_jac @ P_prev @ H_jac.T + R_effective_t * np.eye(ny) # (ny,ny)
    if ny == 1 and S_val.ndim == 0 : S_val = S_val.reshape(1,1)


    # Kalman gain
    # K_val = (P_prev @ H_jac.T) @ np.linalg.inv(S_val) # (nx,ny)
    K_val = np.linalg.solve(S_val.T, H_jac @ P_prev.T).T


    # State & covariance update
    x_curr = x_prev + (K_val @ resid).flatten()
    P_curr = (np.eye(nx, dtype=P_prev.dtype) - K_val @ H_jac) @ P_prev

    # State & covariance propagate (predict for next step)
    x_next_pred = Phi_t @ x_curr
    P_next_pred = Phi_t @ P_curr @ Phi_t.T + Qd_step

    return P_next_pred, x_next_pred

# --- Helper functions for training ---
def _chunk_data_py(data: np.ndarray, target_data: Optional[np.ndarray], window_length: int) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], List[List[int]]]:
    """
    Chunks data into non-overlapping sequences.
    data: (N, Nf) features
    target_data: (N, Ny_target) targets, or None
    window_length: length of each sequence
    Returns:
        x_seqs: list of [ (Nf, window_length) ]
        y_seqs: list of [ (Ny_target, window_length) ] or None
        N_indices_seqs: list of [ [end_original_index_of_sequence] ]
    """
    N = data.shape[0]
    x_seqs = []
    y_seqs = [] if target_data is not None else None
    N_indices_seqs = []

    for i in range(0, N - (N % window_length), window_length):
        start_idx = i
        end_idx = i + window_length
        x_seqs.append(data[start_idx:end_idx, :].T.astype(np.float32)) # (Nf, window_length)
        if target_data is not None:
            y_seqs.append(target_data[start_idx:end_idx, :].T.astype(np.float32)) # (Ny_target, window_length)
        N_indices_seqs.append([end_idx]) # Store original end index of the chunk
        
    return x_seqs, y_seqs, N_indices_seqs

def _norm_sets_py(data: np.ndarray, norm_type: str = "standardize") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes the data.
    data: (N, Nf)
    norm_type: "standardize" supported
    Returns: (bias (mean), scale (std), normalized_data)
    """
    if norm_type == "standardize":
        bias = np.mean(data, axis=0)
        scale = np.std(data, axis=0)
        scale[scale == 0] = 1.0 # Avoid division by zero
        normalized_data = (data - bias) / scale
        return bias, scale, normalized_data
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")

# --- nEKF Training Functions ---
def nekf_train(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
    fn: np.ndarray, fe: np.ndarray, fd: np.ndarray, Cnb: np.ndarray,
    meas: np.ndarray, dt: float,
    itp_mapS: Callable, # Must be callable, not MapCache for training
    x_nn_features: np.ndarray, # (N, Nf) data matrix for NN features
    y_nn_targets: np.ndarray, # (N, 2) target matrix for NN ([latitude longitude])
    P0: Optional[np.ndarray] = None,
    Qd: Optional[np.ndarray] = None,
    R: float = 1.0,
    baro_tau: float = 3600.0, acc_tau: float = 3600.0, gyro_tau: float = 3600.0, fogm_tau: float = 600.0,
    eta_adam: float = 0.1,
    epoch_adam: int = 10,
    hidden_size: int = 1, # LSTM hidden layer size
    activation_func: Callable = swish, # Dense layer activation
    l_window: int = 50, # Temporal window length for LSTM sequences
    date: Optional[float] = None,
    core: bool = False
) -> Callable: # Returns the 'trained' model callable
    """
    Train a measurement noise covariance-adaptive neural EKF (nEKF) model.
    NOTE: This is a conceptual port. True training requires an ML framework
          for backpropagation and optimization. This implementation outlines the
          forward pass (loss calculation) and a conceptual training loop.
    """
    if P0 is None: P0 = create_P0()
    if Qd is None: Qd = create_Qd()
    if date is None: date = get_years(2020, 185)
    if isinstance(itp_mapS, MapCache):
        raise ValueError("itp_mapS must be a callable function for training, not MapCache.")

    N_samples, Nf = x_nn_features.shape
    Ny_model_output = 1 # Model m outputs a scalar correction for R_nn

    # Define the model structure (conceptually)
    # In a real scenario, this would use PyTorch, TensorFlow, etc.
    model_to_train = SequentialModel([
        SimpleLSTM(Nf, hidden_size),
        SimpleDense(hidden_size, Ny_model_output, activation=activation_func)
    ])

    # Prepare sequential data for LSTM
    # x_nn_features is (N, Nf), y_nn_targets is (N, 2)
    x_seqs, y_seqs, N_indices_seqs = _chunk_data_py(x_nn_features, y_nn_targets, l_window)
    
    if not x_seqs:
        print("Warning: No sequences created from x_nn_features, possibly due to N < l_window.")
        return model_to_train # Return untrained model

    # Pre-compute Phi matrices for all time steps
    N_total_ts = len(lat)
    nx = P0.shape[0]
    Phi_all_ts = np.zeros((nx, nx, N_total_ts), dtype=P0.dtype)
    for t in range(N_total_ts):
        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 and Cnb.shape[2] == N_total_ts else Cnb
        Phi_all_ts[:, :, t] = get_Phi(nx, lat[t], vn[t], ve[t], vd[t],
                                      fn[t], fe[t], fd[t], _Cnb_t,
                                      baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

    # Loss function per sequence
    def calculate_sequence_loss(
        model_params_ignored: Any, # In Flux, model is passed. Here, assume model_to_train is in scope.
        current_x_seq: np.ndarray, # (Nf, l_window)
        current_y_seq: np.ndarray, # (2, l_window) - true lat, lon
        current_N_indices: List[int] # [original_end_index_of_sequence]
    ) -> float:
        
        x_ekf_state = np.zeros(nx, dtype=P0.dtype) # Reset EKF state for each sequence
        P_ekf_cov = P0.copy() # Reset EKF covariance for each sequence

        # R_nn_values_for_seq will be (1, l_window) or (l_window,)
        R_nn_values_for_seq = model_to_train(current_x_seq)
        if R_nn_values_for_seq.ndim == 2: R_nn_values_for_seq = R_nn_values_for_seq.flatten()

        seq_len = current_x_seq.shape[1]
        total_squared_pos_error = 0.0
        
        original_end_idx = current_N_indices[0]

        for i in range(seq_len):
            # Determine original time index t for EKF inputs (lat, lon, meas, Phi_all_ts)
            # If current_N_indices[0] is the *end* index of the original data segment for this sequence
            t_original = original_end_idx - seq_len + i
            
            if not (0 <= t_original < N_total_ts):
                # Should not happen if chunking and N_indices_seqs are correct
                print(f"Warning: t_original {t_original} out of bounds {N_total_ts}")
                continue

            P_ekf_cov, x_ekf_state = ekf_single(
                lat[t_original], lon[t_original], alt[t_original],
                Phi_all_ts[:, :, t_original],
                meas[t_original], # Assuming meas is (N_total_ts,) or (N_total_ts,1)
                itp_mapS, # The callable map interpolator
                P_ekf_cov, Qd, R,
                R_nn_values_for_seq[i], # R_nn for this step in sequence
                x_ekf_state,
                date=date, core=core
                # der_mapS and map_alt are not passed to ekf_single from Julia's nekf_train loss
            )

            # Calculate position error
            # x_ekf_state[0] is delta_lat, x_ekf_state[1] is delta_lon
            est_lat = lat[t_original] + x_ekf_state[0]
            est_lon = lon[t_original] + x_ekf_state[1]
            
            true_lat_step = current_y_seq[0, i] # Target lat for this step in sequence
            true_lon_step = current_y_seq[1, i] # Target lon

            # dlat2dn(delta_lat, ref_lat), dlon2de(delta_lon, ref_lat)
            err_n = dlat2dn(true_lat_step - est_lat, est_lat)
            err_e = dlon2de(true_lon_step - est_lon, est_lat) # Use estimated lat for radius
            total_squared_pos_error += err_n**2 + err_e**2
            
        return math.sqrt(total_squared_pos_error / seq_len) if seq_len > 0 else 0.0 # DRMS for sequence

    # Conceptual Training Loop (Adam optimizer and backprop would be here)
    print(f"Starting conceptual training for {epoch_adam} epochs...")
    # In a real framework, optimizer would be initialized with model_to_train.parameters()
    # e.g., optimizer = torch.optim.Adam(model_to_train.parameters(), lr=eta_adam)

    for epoch in range(epoch_adam):
        total_epoch_loss = 0.0
        num_sequences_processed = 0

        # Iterate over data sequences (batches)
        for s_idx in range(len(x_seqs)):
            x_s = x_seqs[s_idx]
            y_s = y_seqs[s_idx]
            n_idx_s = N_indices_seqs[s_idx]

            # 1. Calculate loss (forward pass)
            #    In PyTorch/TF, this happens within a context that tracks gradients.
            loss_val = calculate_sequence_loss(None, x_s, y_s, n_idx_s)
            total_epoch_loss += loss_val
            num_sequences_processed +=1

            # 2. Compute gradients (backward pass) - ML FRAMEWORK NEEDED
            #    e.g., loss_val.backward() in PyTorch

            # 3. Update model parameters - ML FRAMEWORK NEEDED
            #    e.g., optimizer.step() in PyTorch
            #    Conceptually: params -= learning_rate * gradients
            #    For SimpleModel, one might manually adjust self.params based on a dummy gradient.
            #    This part is highly simplified here:
            for param_group in model_to_train.parameters():
                if isinstance(param_group, np.ndarray): # Assuming params are numpy arrays
                    grad_dummy = np.random.randn(*param_group.shape) * 0.01 # Dummy gradient
                    param_group -= eta_adam * grad_dummy 
        
        avg_epoch_loss = total_epoch_loss / num_sequences_processed if num_sequences_processed > 0 else 0
        print(f"Epoch {epoch+1}/{epoch_adam}, Avg DRMS Loss: {avg_epoch_loss:.4f}")

    print("Conceptual training finished.")
    return model_to_train


def nekf_train_ins(
    ins: INS,
    meas: np.ndarray,
    itp_mapS: Callable,
    x_nn_features: np.ndarray,
    y_nn_targets: np.ndarray,
    # ... (all other nekf_train params)
    P0: Optional[np.ndarray] = None, Qd: Optional[np.ndarray] = None, R: float = 1.0,
    baro_tau: float = 3600.0, acc_tau: float = 3600.0, gyro_tau: float = 3600.0, fogm_tau: float = 600.0,
    eta_adam: float = 0.1, epoch_adam: int = 10, hidden_size: int = 1,
    activation_func: Callable = swish, l_window: int = 50,
    date: Optional[float] = None, core: bool = False
) -> Callable:
    """ nEKF training overload for INS data structure. """
    return nekf_train(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb,
        meas, ins.dt, itp_mapS,
        x_nn_features, y_nn_targets,
        P0=P0, Qd=Qd, R=R,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        eta_adam=eta_adam, epoch_adam=epoch_adam, hidden_size=hidden_size,
        activation_func=activation_func, l_window=l_window,
        date=date, core=core
    )

def nekf_train_xyz(
    xyz: XYZ,
    ind: Union[np.ndarray, slice, List[int]], # Indices for selecting data from XYZ
    meas: np.ndarray, # Scalar magnetometer measurements corresponding to ind
    itp_mapS: Callable,
    x_features_raw: np.ndarray, # (N_selected, Nf_raw) raw feature matrix from XYZ data
    # ... (all other nekf_train params, except x_nn_features and y_nn_targets)
    P0: Optional[np.ndarray] = None, Qd: Optional[np.ndarray] = None, R: float = 1.0,
    baro_tau: float = 3600.0, acc_tau: float = 3600.0, gyro_tau: float = 3600.0, fogm_tau: float = 600.0,
    eta_adam: float = 0.1, epoch_adam: int = 10, hidden_size: int = 1,
    activation_func: Callable = swish, l_window: int = 50,
    date: Optional[float] = None, core: bool = False
) -> Tuple[Callable, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    nEKF training overload for XYZ data structure.
    Normalizes x_features_raw before passing to main nekf_train.
    Returns: (trained_model, data_normalizations)
    data_normalizations = (v_scale, x_bias, x_scale)
    """
    # Get trajectory and INS data for selected indices
    # Ensure meas and x_features_raw correspond to the selected 'ind'
    traj_selected: TRAJ = get_traj_from_xyz(xyz, ind)
    ins_selected: INS = get_ins_from_xyz(xyz, ind, N_zero_ll=1) # N_zero_ll as in Julia

    # Create y_nn_targets (true lat, lon) from selected trajectory
    y_nn_targets = np.column_stack((traj_selected.lat, traj_selected.lon)) # (N_selected, 2)

    # Normalize x_features_raw
    # (x_bias, x_scale, x_norm_std) = norm_sets(x;norm_type=:standardize)
    x_bias, x_scale, x_norm_std = _norm_sets_py(x_features_raw, norm_type="standardize")
    
    # (_,S,V) = svd(cov(x_norm))
    # v_scale = V[:,1:1]*inv(Diagonal(sqrt.(S[1:1])))
    # x_nn = x_norm * v_scale
    if x_norm_std.shape[0] < 2: # Need at least 2 samples for covariance
        print("Warning: Not enough samples to compute covariance for SVD scaling. Using standardized features directly.")
        x_nn_processed = x_norm_std
        # v_scale would be identity if Nf_raw features are kept, or problematic if trying to reduce to 1 feature
        # For simplicity, if SVD fails, use standardized features and v_scale implies no change or selection
        v_scale_factor = np.eye(x_norm_std.shape[1]) # Placeholder if SVD part is skipped
    else:
        cov_matrix = np.cov(x_norm_std, rowvar=False)
        if np.allclose(cov_matrix, 0): # Handle zero covariance case
             print("Warning: Covariance matrix is zero. Using standardized features directly.")
             x_nn_processed = x_norm_std
             v_scale_factor = np.eye(x_norm_std.shape[1])
        else:
            try:
                U, S_singular_values, Vt = np.linalg.svd(cov_matrix)
                # V in Julia's svd(A) is U*S*V', so Julia's V is Vt.T (U from np.linalg.svd)
                # Julia: v_scale = V[:,1:1]*inv(Diagonal(sqrt.(S[1:1])))
                # Python: V_julia_equivalent = U
                
                # We want to project onto the first principal component direction scaled by 1/sqrt(eigenvalue)
                # This reduces features to 1D.
                v_pc1 = U[:, 0].reshape(-1, 1) # First principal component (eigenvector)
                s_sqrt_inv_pc1 = 1.0 / math.sqrt(S_singular_values[0]) if S_singular_values[0] > 1e-9 else 1.0
                
                v_scale_factor = v_pc1 * s_sqrt_inv_pc1 # (Nf_raw, 1) scaling vector
                x_nn_processed = x_norm_std @ v_scale_factor # (N_selected, Nf_raw) @ (Nf_raw, 1) -> (N_selected, 1)
            except np.linalg.LinAlgError:
                print("Warning: SVD did not converge. Using standardized features directly.")
                x_nn_processed = x_norm_std
                v_scale_factor = np.eye(x_norm_std.shape[1]) # (Nf_raw, Nf_raw) identity


    data_norms_tuple = (v_scale_factor, x_bias, x_scale)

    # Call the main training function with processed x_nn and y_nn
    trained_model = nekf_train(
        ins_selected.lat, ins_selected.lon, ins_selected.alt,
        ins_selected.vn, ins_selected.ve, ins_selected.vd,
        ins_selected.fn, ins_selected.fe, ins_selected.fd,
        ins_selected.Cnb,
        meas, # This 'meas' should already be for the selected 'ind'
        ins_selected.dt,
        itp_mapS,
        x_nn_processed, # Processed features (potentially N_selected, 1)
        y_nn_targets,
        P0=P0, Qd=Qd, R=R,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        eta_adam=eta_adam, epoch_adam=epoch_adam, hidden_size=hidden_size,
        activation_func=activation_func, l_window=l_window,
        date=date, core=core
    )
    return trained_model, data_norms_tuple