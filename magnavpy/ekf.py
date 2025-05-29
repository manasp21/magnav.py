import numpy as np

# Assuming FILTres, EKF_RT, INS, Map_Cache, get_cached_map are defined/imported in .magnav
# If get_cached_map is elsewhere, adjust import.
from .magnav import FILTres, EKF_RT, INS
from .common_types import get_cached_map
from .common_types import MapCache
from .model_functions import get_Phi, get_H, create_P0, create_Qd
from .core_utils import get_years
from .model_functions import get_h # Explicitly import get_h from model_functions

def ekf(
    lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, meas, dt, itp_mapS,
    P0=None, Qd=None, R=1.0,
    baro_tau=3600.0, acc_tau=3600.0, gyro_tau=3600.0, fogm_tau=600.0,
    date=None,
    core=False,
    der_mapS=None,
    map_alt=0
):
    """
    Extended Kalman filter (EKF) for airborne magnetic anomaly navigation.

    Args:
        lat (np.ndarray): Latitude [rad], shape (N,)
        lon (np.ndarray): Longitude [rad], shape (N,)
        alt (np.ndarray): Altitude [m], shape (N,)
        vn (np.ndarray): North velocity [m/s], shape (N,)
        ve (np.ndarray): East velocity [m/s], shape (N,)
        vd (np.ndarray): Down velocity [m/s], shape (N,)
        fn (np.ndarray): North specific force [m/s^2], shape (N,)
        fe (np.ndarray): East specific force [m/s^2], shape (N,)
        fd (np.ndarray): Down specific force [m/s^2], shape (N,)
        Cnb (np.ndarray): Direction cosine matrix (body to navigation), shape (3,3,N) or (3,3)
        meas (np.ndarray): Scalar magnetometer measurement [nT], shape (N,) or (N,1)
        dt (float): Measurement time step [s]
        itp_mapS (callable or Map_Cache): Scalar map interpolation function (callable)
                                         or a Map_Cache object.
        P0 (np.ndarray, optional): Initial covariance matrix, shape (nx,nx).
                                   Defaults to create_P0().
        Qd (np.ndarray, optional): Discrete time process/system noise matrix, shape (nx,nx).
                                   Defaults to create_Qd().
        R (float or tuple, optional): Measurement noise variance or (R_min, R_max) for adaptation.
                                      Defaults to 1.0.
        baro_tau (float, optional): Barometer time constant [s]. Defaults to 3600.0.
        acc_tau (float, optional): Accelerometer time constant [s]. Defaults to 3600.0.
        gyro_tau (float, optional): Gyroscope time constant [s]. Defaults to 3600.0.
        fogm_tau (float, optional): FOGM catch-all time constant [s]. Defaults to 600.0.
        date (float, optional): Measurement date (decimal year) for IGRF [yr].
                                Defaults to get_years(2020,185).
        core (bool, optional): If true, include core magnetic field in measurement. Defaults to False.
        der_mapS (callable, optional): Scalar map vertical derivative map interpolation function.
                                       Defaults to None.
        map_alt (float, optional): Map altitude [m]. Defaults to 0.

    Returns:
        FILTres: Filter results struct.
    """
    if P0 is None:
        P0 = create_P0()
    if Qd is None:
        Qd = create_Qd()
    if date is None:
        date = get_years(2020, 185)

    N = len(lat)
    nx = P0.shape[0]

    if meas.ndim == 1:
        ny = 1
        _meas_internal = meas.reshape(-1, 1)  # Work with (N,1)
    elif meas.ndim == 2 and meas.shape[1] == 1:
        ny = 1
        _meas_internal = meas
    else:
        # Future: extend for ny > 1 if meas.shape[1] > 1
        raise ValueError("meas must be a 1D array or a 2D array with one column for scalar measurements.")

    x_out = np.zeros((nx, N), dtype=P0.dtype)
    P_out = np.zeros((nx, nx, N), dtype=P0.dtype)
    r_out = np.zeros((ny, N), dtype=P0.dtype)
    
    x = np.zeros(nx, dtype=P0.dtype)
    P = P0.copy()

    adapt = False
    R_val = R # Can be scalar or matrix if ny > 1
    R_min, R_max = None, None
    if isinstance(R, (list, tuple)) and len(R) == 2:
        adapt = True
        R_min, R_max = R
        R_val = np.mean(R) # Initial scalar R value for adaptation
    elif not isinstance(R, (float, int, np.number)):
        if isinstance(R, np.ndarray) and R.shape == (ny, ny):
            R_val = R.copy() # R is already a matrix
        else:
            raise ValueError("R must be a scalar, a (min,max) tuple for adaptation, or a (ny,ny) matrix.")

    itp_mapS_arg = itp_mapS # Keep original argument

    for t in range(N):
        current_itp_mapS_for_step = itp_mapS_arg
        current_der_mapS_for_step = der_mapS

        if isinstance(itp_mapS_arg, Map_Cache):
            # get_cached_map returns the interpolator for the current location
            current_itp_mapS_for_step = get_cached_map(itp_mapS_arg, lat[t], lon[t], alt[t], silent=True)
            # In Julia, if map_cache is used, der_mapS is effectively ignored for get_h in RT.
            # For batch, der_mapS is passed along. Here we assume der_mapS is independent or also cached.

        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 else Cnb

        Phi = get_Phi(nx, lat[t], vn[t], ve[t], vd[t], fn[t], fe[t], fd[t], _Cnb_t,
                      baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

        h_pred = get_h(current_itp_mapS_for_step, x, lat[t], lon[t], alt[t],
                       date=date, core=core, der_map=current_der_mapS_for_step, map_alt=map_alt)

        if isinstance(h_pred, (float, int, np.number)): h_pred = np.array([h_pred])
        if h_pred.ndim == 1: h_pred = h_pred.reshape(-1,1)

        resid = _meas_internal[t,:].reshape(-1,1) - h_pred

        r_out[:, t] = resid.flatten()

        H_m = get_H(current_itp_mapS_for_step, x, lat[t], lon[t], alt[t],
                    date=date, core=core) # Expected (nx,) or (1,nx)
        if H_m.ndim == 1: H_m = H_m.reshape(1, -1) # Ensure (1,nx)
        
        # If ny > 1, H_m would be tiled. For ny=1, H_m is (1,nx)
        # H_val = np.tile(H_m, (ny, 1)) # This is general
        H_val = H_m # Since ny=1 for scalar measurements

        if adapt and ny == 1: # Adaptive R for scalar case
            n_adapt_window = 10
            if t >= n_adapt_window:
                # Julia: r_out[:,(t-n):(t-1)], 1-based indices, window size n
                # Python: r_out[:, t-n_adapt_window : t]
                r_window = r_out[:, (t - n_adapt_window):t] # (1, n_adapt_window)
                
                term1 = (r_window @ r_window.T) / n_adapt_window 
                term2 = H_val @ P @ H_val.T
                R_candidate = (term1 - term2)[0,0]
                R_val = np.clip(R_candidate, R_min, R_max)
                if (t + 1) % n_adapt_window == 0:
                    print(f"EKF Info: Timestep {t+1}, Adaptive R = {R_val:.2f} (sqrt: {np.sqrt(R_val):.2f})")
        
        # S = H P H' + R
        S_matrix_term = H_val @ P @ H_val.T
        if ny == 1 and isinstance(R_val, (float, int, np.number)):
            S_val = S_matrix_term + R_val 
        elif isinstance(R_val, np.ndarray) and R_val.shape == (ny,ny):
             S_val = S_matrix_term + R_val
        else: # Fallback for ny > 1 and R_val scalar
            S_val = S_matrix_term + np.eye(ny) * R_val
        if S_val.ndim == 0: S_val = S_val.reshape(1,1) # Ensure S_val is 2D for solve

        # K = P H' S^-1
        # K_val = (P @ H_val.T) @ np.linalg.inv(S_val) # Direct inversion
        # Using solve: K^T = solve(S^T, (P H^T)^T) = solve(S^T, H P^T)
        K_val = (np.linalg.solve(S_val.T, H_val @ P.T)).T

        x = x + K_val @ resid.flatten()
        P = (np.eye(nx, dtype=P.dtype) - K_val @ H_val) @ P

        x_out[:, t] = x
        P_out[:, :, t] = P

        x = Phi @ x
        P = Phi @ P @ Phi.T + Qd
    
    return FILTres(x_out, P_out, r_out, True)


def ekf_ins(
    ins: INS,
    meas, 
    itp_mapS,
    P0=None,
    Qd=None,
    R=1.0,
    baro_tau=3600.0,
    acc_tau=3600.0,
    gyro_tau=3600.0,
    fogm_tau=600.0,
    date=None,
    core=False,
    der_mapS=None,
    map_alt=0
):
    """
    EKF overload for INS data structure.
    See `ekf` for argument details. `ins.dt` is used for the time step.
    """
    # The Julia version has a complex default for der_mapS involving map_itp.
    # Here, we rely on the main `ekf` function's default for der_mapS (which is None).
    # If a specific default like Julia's is needed, it should be constructed before calling.
    return ekf(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb,
        meas, ins.dt, itp_mapS,
        P0=P0, Qd=Qd, R=R,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        date=date, core=core, der_mapS=der_mapS, map_alt=map_alt
    )


def process_ekf_rt_step(
    ekf_rt: EKF_RT,
    lat_curr, lon_curr, alt_curr, vn_curr, ve_curr, vd_curr, 
    fn_curr, fe_curr, fd_curr, Cnb_curr, 
    meas_scalar_curr, t_curr, itp_mapS,
    der_mapS=None,
    map_alt=-1,
    dt_fallback=0.1
):
    """
    Processes a single step for the real-time EKF.
    Mutates ekf_rt object's t, P, x, r fields.
    Returns FILTres for the current step.

    Args:
        ekf_rt (EKF_RT): The EKF_RT data object (mutable).
        # ... (current step's navigation and measurement data)
        itp_mapS (callable or Map_Cache): Interpolator or cache.
        # ... (other optional EKF parameters)
    """
    if not (isinstance(meas_scalar_curr, (float, int, np.number)) or \
            (isinstance(meas_scalar_curr, np.ndarray) and meas_scalar_curr.size == 1)):
        raise ValueError("meas_scalar_curr must be a scalar or a single-element array.")
    
    _meas_input = np.array([[meas_scalar_curr]]).reshape(1,1) # Ensure (1,1)

    if ekf_rt.ny != 1:
         raise AssertionError("ekf_rt.ny must be 1 for scalar measurement processing.")

    dt_actual = dt_fallback if ekf_rt.t < 0 else (t_curr - ekf_rt.t)

    _Cnb_step = Cnb_curr
    if Cnb_curr.ndim == 3: # Should be 2D for a single step
        if Cnb_curr.shape[2] == 1: _Cnb_step = Cnb_curr[:,:,0]
        else: raise ValueError("Cnb_curr for EKF_RT step should be a 2D matrix.")

    Phi = get_Phi(ekf_rt.nx, lat_curr, vn_curr, ve_curr, vd_curr, 
                  fn_curr, fe_curr, fd_curr, _Cnb_step,
                  ekf_rt.baro_tau, ekf_rt.acc_tau, ekf_rt.gyro_tau, 
                  ekf_rt.fogm_tau, dt_actual)

    current_itp_mapS_for_step = itp_mapS
    current_der_mapS_for_step = der_mapS
    if isinstance(itp_mapS, Map_Cache):
        current_itp_mapS_for_step = get_cached_map(itp_mapS, lat_curr, lon_curr, alt_curr, silent=True)
        # Julia: der_mapS = nothing if itp_mapS is Map_Cache
        current_der_mapS_for_step = None 

    h_pred = get_h(current_itp_mapS_for_step, ekf_rt.x, lat_curr, lon_curr, alt_curr,
                   date=ekf_rt.date, core=ekf_rt.core, 
                   der_map=current_der_mapS_for_step, map_alt=map_alt if map_alt != -1 else 0) # map_alt default in get_h?
    
    if isinstance(h_pred, (float, int, np.number)): h_pred = np.array([h_pred])
    if h_pred.ndim == 1: h_pred = h_pred.reshape(-1,1) # to (1,1)

    resid = _meas_input - h_pred # (1,1)

    ekf_rt.t = t_curr
    ekf_rt.r = resid.flatten() # Store as (1,) array

    H_m = get_H(current_itp_mapS_for_step, ekf_rt.x, lat_curr, lon_curr, alt_curr,
                date=ekf_rt.date, core=ekf_rt.core) # (nx,) or (1,nx)
    if H_m.ndim == 1: H_m = H_m.reshape(1, -1)
    H_val = H_m # ny=1

    S_matrix_term = H_val @ ekf_rt.P @ H_val.T
    # ekf_rt.R is expected to be scalar for ny=1
    if not isinstance(ekf_rt.R, (float, int, np.number)):
        raise TypeError("ekf_rt.R must be a scalar for ny=1.")
    S_val = S_matrix_term + ekf_rt.R 
    if S_val.ndim == 0: S_val = S_val.reshape(1,1)

    K_val = (np.linalg.solve(S_val.T, H_val @ ekf_rt.P.T)).T

    ekf_rt.x = ekf_rt.x + K_val @ resid.flatten()
    ekf_rt.P = (np.eye(ekf_rt.nx, dtype=ekf_rt.P.dtype) - K_val @ H_val) @ ekf_rt.P

    x_out_step = ekf_rt.x[:, np.newaxis]
    P_out_step = ekf_rt.P[:, :, np.newaxis]
    r_out_step = ekf_rt.r.reshape(-1,1)

    ekf_rt.x = Phi @ ekf_rt.x
    ekf_rt.P = Phi @ ekf_rt.P @ Phi.T + ekf_rt.Qd

    return FILTres(x_out_step, P_out_step, r_out_step, True)


def process_ekf_rt_step_ins(
    ekf_rt: EKF_RT,
    ins: INS,
    meas_scalar_curr,
    itp_mapS,
    der_mapS=None,
    map_alt=-1 # Julia default for this overload
):
    """
    EKF_RT step processor using a single sample from an INS data object.
    Mutates ekf_rt.
    """
    if ins.N != 1:
        raise AssertionError("ins must contain a single sample for EKF_RT step processing.")
    
    idx = 0 # Python is 0-indexed

    # Ensure Cnb from INS is 2D for the single step
    _Cnb_ins_step = ins.Cnb
    if ins.Cnb.ndim == 3:
        if ins.Cnb.shape[2] == 1: _Cnb_ins_step = ins.Cnb[:,:,0]
        else: raise ValueError("ins.Cnb for EKF_RT step should correspond to a single 2D matrix.")


    return process_ekf_rt_step(
        ekf_rt,
        ins.lat[idx], ins.lon[idx], ins.alt[idx],
        ins.vn[idx], ins.ve[idx], ins.vd[idx],
        ins.fn[idx], ins.fe[idx], ins.fd[idx],
        _Cnb_ins_step,
        meas_scalar_curr,
        ins.tt[idx], # Time for the current sample
        itp_mapS,
        der_mapS=der_mapS,
        map_alt=map_alt,
        dt_fallback=ins.dt # dt from INS struct if ekf_rt.t is not initialized
    )
def crlb(*args, **kwargs):
    """Placeholder for Cramer-Rao Lower Bound calculation."""
    pass

def calc_crlb_pos(*args, **kwargs):
    """Placeholder for CRLB position calculation."""
    pass

def calc_crlb_vel(*args, **kwargs):
    """Placeholder for CRLB velocity calculation."""
    pass

def calc_crlb_att(*args, **kwargs):
    """Placeholder for CRLB attitude calculation."""
    pass

def calc_crlb_fogm(*args, **kwargs):
    """Placeholder for CRLB FOGM calculation."""
    pass

def calc_crlb_map(*args, **kwargs):
    """Placeholder for CRLB map calculation."""
    pass