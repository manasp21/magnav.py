import numpy as np
from scipy.linalg import cholesky, solve
import logging
from typing import Callable, Optional, Union, Any

# Attempt to import from existing modules within the magnavpy package
# These imports are based on common patterns in the MagNav project.
# If these exact functions/classes don't exist with these names,
# they would need to be adjusted to match the actual available utilities.

try:
    from .common_types import INS, FILTres, get_years
    # Assuming create_P0 and create_Qd might be general or EKF-specific
    # If specific versions for MPF are needed and named differently, adjust here.
    from .ekf import create_P0, create_Qd, get_Phi
    from .model_functions import get_h
    from .map_utils import Map_Cache # Assuming Map_Cache is a class
except ImportError:
    # This block is for placeholder purposes if the above imports fail.
    # In a real scenario, these dependencies must be correctly resolved.
    print("Warning: Could not import all MagNavPy dependencies for mpf.py. Using placeholders.")
    # Define dummy placeholders if imports fail, to allow syntax checking of mpf.py itself.
    class INS: pass
    class FILTres:
        def __init__(self, x_out, P_out, resid, converge):
            self.x_out, self.P_out, self.resid, self.converge = x_out, P_out, resid, converge
    class Map_Cache:
        def get_cached_map(self, lat, lon, alt, silent=True): return None # Placeholder
    def get_years(y,d): return float(y) # Placeholder
    def create_P0(): return np.eye(18) * 0.1 # Placeholder
    def create_Qd(): return np.eye(18) * 0.01 # Placeholder
    def get_Phi(nx,lat,vn,ve,vd,fn,fe,fd,Cnb,baro_tau,acc_tau,gyro_tau,fogm_tau,dt): return np.eye(nx) # Placeholder
    def get_h(itp_mapS, states, lat, lon, alt, date, core): return np.array([0.0]) # Placeholder

# Configure basic logging for divergence messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def sys_resample(weights: np.ndarray) -> np.ndarray:
    """
    Systematic resampling.
    Assumes weights sum to 1.
    Returns 0-indexed particle indices.
    """
    n_particles = len(weights)
    if n_particles == 0:
        return np.array([], dtype=int)
        
    cumulative_sum = np.cumsum(weights)
    # Ensure the last element is exactly 1.0 to avoid issues with searchsorted
    if n_particles > 0:
        cumulative_sum[-1] = 1.0

    # Generate stratified random numbers
    # Julia: u  = ((1:np) .- rand(eltype(q),1)) ./ np
    # This means a single random draw is subtracted from the sequence 1..np
    rand_offset = np.random.rand()
    u_values = (np.arange(1, n_particles + 1) - rand_offset) / n_particles

    # Find corresponding indices
    indices = np.searchsorted(cumulative_sum, u_values, side='right')
    
    return np.clip(indices, 0, n_particles - 1)

def part_cov(q: np.ndarray, x: np.ndarray, x_mean: np.ndarray, P_add: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate weighted particle covariance.
    P_out = sum_k q_k * (x_k - x_mean) @ (x_k - x_mean).T + P_add
    q: particle weights (np,)
    x: particle states (nx, np)
    x_mean: mean of particle states (nx,)
    P_add: additive covariance term (nx, nx), optional
    """
    nx, np = x.shape
    if np == 0: # Handle empty particles case
        P_out_particles = np.zeros((nx,nx), dtype=x.dtype)
    else:
        P_temp = x - x_mean.reshape(-1, 1)  # dx = x_k - x_mean, shape (nx, np)
        P_out_particles = (P_temp * q.reshape(1, -1)) @ P_temp.T
    
    if P_add is not None:
        P_out = P_out_particles + P_add
    else:
        P_out = P_out_particles
        
    return P_out

def filter_exit(Pl_out: np.ndarray, Pn_out: np.ndarray, t: int, N_total_steps: int, converge: bool = True) -> np.ndarray:
    """
    Combine linear and non-linear covariances and log divergence if any.
    Pl_out: linear part of covariance (nxl, nxl, N)
    Pn_out: non-linear part of covariance (nxn, nxn, N)
    t: current time step (relevant for divergence message, 0-indexed)
    N_total_steps: total number of steps the filter ran or was supposed to run.
    converge: boolean indicating filter convergence
    """
    nxl = Pl_out.shape[0]
    nxn = Pn_out.shape[0]
    nx = nxl + nxn
    
    P_out = np.zeros((nx, nx, N_total_steps), dtype=Pl_out.dtype)
    # Ensure slicing does not go out of bounds if N_total_steps is smaller than Pn_out/Pl_out last dim
    # This should not happen if Pn_out/Pl_out are correctly sized up to N_samples
    min_steps_dim = min(N_total_steps, Pn_out.shape[2], Pl_out.shape[2])

    P_out[0:nxn, 0:nxn, :min_steps_dim] = Pn_out[:,:,:min_steps_dim]
    P_out[nxn:nx, nxn:nx, :min_steps_dim] = Pl_out[:,:,:min_steps_dim]
    
    if not converge:
        logging.info(f"Filter diverged, particle weights ~0 at time step {t+1}") # t is 0-indexed
        
    return P_out

def mpf(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
        vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
        fn: np.ndarray, fe: np.ndarray, fd: np.ndarray,
        Cnb: np.ndarray, meas: np.ndarray, dt: float,
        itp_mapS: Union[Callable, Map_Cache], 
        P0_func: Callable = create_P0, 
        Qd_func: Callable = create_Qd, 
        R_val: float = 1.0,
        num_part: int = 1000,
        thresh_frac: float = 0.8, 
        baro_tau: float = 3600.0,
        acc_tau: float = 3600.0,
        gyro_tau: float = 3600.0,
        fogm_tau: float = 600.0,
        date_func: Callable = get_years, 
        default_year: int = 2020, 
        default_day: int = 185,   
        core: bool = False
       ) -> FILTres:
    """
    Rao-Blackwellized (marginalized) particle filter (MPF) for airborne magnetic
    anomaly navigation. Assumes linear dynamics and non-correlated scalar measurements.
    """
    
    P0 = P0_func()
    Qd = Qd_func()
    R_matrix = np.array([[R_val]]) 

    N_samples = len(lat)
    np_particles = num_part
    nx_total_states = P0.shape[0]
    nxn_nonlinear_states = 2  
    nxl_linear_states = nx_total_states - nxn_nonlinear_states
    
    if meas.ndim == 1:
        meas = meas.reshape(-1,1) 
    ny_meas_dim = meas.shape[1] 
    if ny_meas_dim != 1:
        raise ValueError("MPF currently supports only scalar measurements (ny_meas_dim=1)")

    dtype = P0.dtype

    H_jac = np.zeros((ny_meas_dim, nxl_linear_states), dtype=dtype)
    if nxl_linear_states > 0 : H_jac[0, -1] = 1.0 

    x_out = np.zeros((nx_total_states, N_samples), dtype=dtype)
    Pn_out = np.zeros((nxn_nonlinear_states, nxn_nonlinear_states, N_samples), dtype=dtype)
    Pl_out = np.zeros((nxl_linear_states, nxl_linear_states, N_samples), dtype=dtype)
    resid_out = np.zeros((ny_meas_dim, N_samples), dtype=dtype)

    P0_nl = P0[0:nxn_nonlinear_states, 0:nxn_nonlinear_states]
    try:
        L_P0_nl = cholesky(P0_nl, lower=True)
        xn_particles = L_P0_nl @ np.random.randn(nxn_nonlinear_states, np_particles).astype(dtype)
    except np.linalg.LinAlgError: 
        logging.warning("Initial non-linear covariance P0_nl is not positive definite. Initializing xn_particles with small random noise based on diagonal.")
        diag_P0_nl = np.diag(P0_nl).copy()
        diag_P0_nl[diag_P0_nl <=0] = np.finfo(dtype).eps # ensure positive for sqrt
        xn_particles = np.sqrt(diag_P0_nl).reshape(-1,1) * np.random.randn(nxn_nonlinear_states, np_particles).astype(dtype)

    xl_particles = np.zeros((nxl_linear_states, np_particles), dtype=dtype)
    Pl_covariance = P0[nxn_nonlinear_states:, nxn_nonlinear_states:].copy()
    q_weights = np.ones(np_particles, dtype=dtype) / np_particles
    current_date = date_func(default_year, default_day)

    map_cache_obj = itp_mapS if isinstance(itp_mapS, Map_Cache) else None
    itp_mapS_callable = itp_mapS if callable(itp_mapS) else None

    for t in range(N_samples):
        itp_mapS_current_step = itp_mapS_callable
        if map_cache_obj is not None:
            itp_mapS_current_step = map_cache_obj.get_cached_map(lat[t], lon[t], alt[t], silent=True)
        
        if itp_mapS_current_step is None:
            raise ValueError(f"itp_mapS is not a valid callable or Map_Cache object at step {t}.")

        Phi = get_Phi(nx_total_states, lat[t], vn[t], ve[t], vd[t],
                      fn[t], fe[t], fd[t], Cnb[:,:,t],
                      baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

        An_l = Phi[0:nxn_nonlinear_states, nxn_nonlinear_states:]      
        An_n = Phi[0:nxn_nonlinear_states, 0:nxn_nonlinear_states]     
        Al_l = Phi[nxn_nonlinear_states:, nxn_nonlinear_states:]      
        Al_n = Phi[nxn_nonlinear_states:, 0:nxn_nonlinear_states]     

        full_particle_states = np.vstack((xn_particles, xl_particles))
        
        y_hat_particles = get_h(itp_mapS_current_step, full_particle_states,
                                lat[t], lon[t], alt[t], date=current_date, core=core)
        
        if y_hat_particles.ndim == 1:
             y_hat_particles = y_hat_particles.reshape(ny_meas_dim, -1) # Ensure (ny, np)

        e_residuals = meas[t,:].reshape(-1,1) - y_hat_particles 
        resid_out[:,t] = np.mean(e_residuals, axis=1) 

        V_measurement_cov = H_jac @ Pl_covariance @ H_jac.T + R_matrix
        
        inv_V_scalar = 1.0 / V_measurement_cov[0,0]
        log_likelihood_per_particle = -0.5 * (e_residuals[0,:] ** 2) * inv_V_scalar
        max_log_likelihood = np.max(log_likelihood_per_particle) if np_particles > 0 else 0
        q_weights = q_weights * np.exp(log_likelihood_per_particle - max_log_likelihood)

        sum_q_weights = np.sum(q_weights)
        # Check for NaN or effectively zero sum of weights
        if np.isnan(sum_q_weights) or sum_q_weights <= np.finfo(dtype).eps:
            converge_status = False
            # Fill remaining x_out with NaN
            if t < N_samples:
                x_out[:, t:] = np.nan
            # P_out will be handled by filter_exit, ensure it's filled appropriately for diverged steps
            # For now, filter_exit will use what's in Pl_out, Pn_out up to min_steps_dim
            P_out_final = filter_exit(Pl_out, Pn_out, t, N_samples, converge_status)
            return FILTres(x_out, P_out_final, resid_out, converge_status)

        q_weights = q_weights / sum_q_weights

        x_out[0:nxn_nonlinear_states, t] = np.sum(q_weights * xn_particles, axis=1)
        Pn_out[:,:,t] = part_cov(q_weights, xn_particles, x_out[0:nxn_nonlinear_states, t])

        N_eff = 1.0 / np.sum(q_weights**2) if sum_q_weights > 0 else 0
        if N_eff < np_particles * thresh_frac and np_particles > 0:
            indices = sys_resample(q_weights)
            xn_particles = xn_particles[:, indices]
            xl_particles = xl_particles[:, indices]
            q_weights = np.ones(np_particles, dtype=dtype) / np_particles
        
        xl_particles_temp_for_prop = xl_particles.copy() 

        if nxl_linear_states > 0: # Proceed with KF update only if linear states exist
            K_gain = solve(V_measurement_cov.T, (Pl_covariance @ H_jac.T).T).T 
            Pl_covariance = Pl_covariance - K_gain @ V_measurement_cov @ K_gain.T
            xl_particles = xl_particles + K_gain @ e_residuals 
            x_out[nxn_nonlinear_states:, t] = np.sum(q_weights * xl_particles, axis=1)
            Pl_out[:,:,t] = part_cov(q_weights, xl_particles, x_out[nxn_nonlinear_states:, t], P_add=Pl_covariance)

            Qd_nl_part = Qd[0:nxn_nonlinear_states, 0:nxn_nonlinear_states]
            Qd_l_part  = Qd[nxn_nonlinear_states:, nxn_nonlinear_states:]

            M_prop_cov = An_l @ Pl_covariance @ An_l.T + Qd_nl_part 
            L_prop_gain = solve(M_prop_cov.T, (Al_l @ Pl_covariance @ An_l.T).T).T
            Pl_covariance_propagated = Al_l @ Pl_covariance @ Al_l.T + Qd_l_part - L_prop_gain @ M_prop_cov @ L_prop_gain.T
            Pl_covariance = (Pl_covariance_propagated + Pl_covariance_propagated.T) / 2.0
        else: # No linear states, Pl_covariance remains empty or zero, no KF update for it
            Pl_out[:,:,t] = np.zeros((nxl_linear_states,nxl_linear_states), dtype=dtype) # Should be empty if nxl=0
            M_prop_cov = Qd[0:nxn_nonlinear_states, 0:nxn_nonlinear_states] # Simplified if An_l is irrelevant
            # L_prop_gain would be zero or irrelevant

        try:
            L_M_prop = cholesky(M_prop_cov, lower=True)
            noise_nl_prop = L_M_prop @ np.random.randn(nxn_nonlinear_states, np_particles).astype(dtype)
        except np.linalg.LinAlgError:
            logging.warning(f"M_prop_cov not positive definite at step {t+1}. Using diagonal approx for noise.")
            diag_M = np.diag(M_prop_cov).copy()
            diag_M[diag_M <= 0] = np.finfo(dtype).eps
            noise_nl_prop = np.sqrt(diag_M).reshape(-1,1) * np.random.randn(nxn_nonlinear_states, np_particles).astype(dtype)
        
        xn_particles_propagated = An_n @ xn_particles + An_l @ xl_particles_temp_for_prop + noise_nl_prop
        
        if nxl_linear_states > 0:
            z_nl_diff = noise_nl_prop # As per Julia: z = xn_propagated - (An_n @ xn_particles_current + An_l @ xl_particles_temp_for_prop)
            xl_particles = Al_n @ xn_particles + Al_l @ xl_particles_temp_for_prop + \
                           L_prop_gain @ (z_nl_diff - (An_l @ xl_particles_temp_for_prop))
        else: # No linear states to propagate via KF mechanism
            pass # xl_particles remains empty or zero

        xn_particles = xn_particles_propagated

    converge_status = True
    P_out_final = filter_exit(Pl_out, Pn_out, N_samples -1, N_samples, converge_status)
    return FILTres(x_out, P_out_final, resid_out, converge_status)


def mpf_ins(ins_data: INS, meas: np.ndarray, itp_mapS: Union[Callable, Map_Cache], **kwargs) -> FILTres:
    """
    Wrapper for mpf function that takes an INS object.
    """
    # Extract P0_func and Qd_func from kwargs if provided, else they default in mpf
    P0_func = kwargs.pop('P0_func', create_P0)
    Qd_func = kwargs.pop('Qd_func', create_Qd)
    
    return mpf(lat=ins_data.lat, lon=ins_data.lon, alt=ins_data.alt,
               vn=ins_data.vn, ve=ins_data.ve, vd=ins_data.vd,
               fn=ins_data.fn, fe=ins_data.fe, fd=ins_data.fd,
               Cnb=ins_data.Cnb, meas=meas, dt=ins_data.dt,
               itp_mapS=itp_mapS,
               P0_func=P0_func, Qd_func=Qd_func,
               **kwargs)