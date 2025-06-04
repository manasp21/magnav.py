import numpy as np
from typing import List, Union # Add Union to this import
from .common_types import MapS, MapS3D, MapCache # Import MapS, MapS3D, and MapCache

# Assuming FILTres, EKF_RT, INS, get_cached_map are defined/imported in .magnav
# If get_cached_map is elsewhere, adjust import.
from .magnav import FILTres, EKF_RT, INS
from .common_types import get_cached_map
# MapCache is already imported from .common_types on line 3
from .model_functions import get_Phi, get_H, create_P0, create_Qd
from .map_utils import map_interpolate # Moved from local import
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

        if isinstance(itp_mapS_arg, MapCache):
            if not itp_mapS_arg.maps or not itp_mapS_arg.interpolators:
                # Fallback or error if MapCache is empty
                # For now, let's assume it might use its fallback_map's interpolator if available
                # This part needs robust handling based on MapCache's design for fallbacks
                if hasattr(itp_mapS_arg, 'fallback_map_interpolator') and itp_mapS_arg.fallback_map_interpolator is not None:
                    current_itp_mapS_for_step = itp_mapS_arg.fallback_map_interpolator
                elif hasattr(itp_mapS_arg, 'fallback_map') and itp_mapS_arg.fallback_map is not None:
                    # Attempt to create interpolator from fallback_map if not pre-cached
                    # This requires map_interpolate from map_utils
                    try:
                        # from .map_utils import map_interpolate # Moved to top
                        current_itp_mapS_for_step = map_interpolate(itp_mapS_arg.fallback_map)
                        if current_itp_mapS_for_step is None:
                            raise ValueError("Fallback interpolator creation failed.")
                    except Exception as e_interp:
                        print(f"Warning: MapCache empty and fallback interpolator failed/unavailable: {e_interp}")
                        # As a last resort, if no interpolator, get_h/get_H will fail.
                        # Or, we could make current_itp_mapS_for_step a dummy that returns NaNs.
                        # For now, let it proceed and fail in get_h/get_H if interpolator is None.
                        current_itp_mapS_for_step = None # This will likely cause issues downstream
                else:
                    print("Warning: MapCache is empty and no fallback available. Interpolation will likely fail.")
                    current_itp_mapS_for_step = None # This will likely cause issues downstream
            else:
                # Select interpolator from MapCache based on closest altitude
                map_altitudes = np.array([m.alt for m in itp_mapS_arg.maps])
                # Ensure alt[t] is a scalar for comparison
                current_alt_scalar = alt[t].item() if isinstance(alt[t], np.ndarray) else alt[t]
                closest_idx = np.argmin(np.abs(map_altitudes - current_alt_scalar))
                current_itp_mapS_for_step = itp_mapS_arg.interpolators[closest_idx]
                if current_itp_mapS_for_step is None:
                    # This specific interpolator might be None if its map was problematic
                    print(f"Warning: Selected interpolator from MapCache at index {closest_idx} is None.")
                    # Potentially use fallback logic here too
                    # For now, allow None to propagate; get_h/get_H should handle it or error out.

            # der_mapS handling: if itp_mapS is MapCache, Julia often ignores der_mapS for get_h.
            # If get_h/get_H are robust to current_itp_mapS_for_step being None, this is okay.
            # The original der_mapS (passed to ekf) is used for current_der_mapS_for_step.

        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 else Cnb

        # Ensure Cnb is 2D for get_Phi
        _Cnb_t_for_phi = _Cnb_t
        if _Cnb_t.ndim == 3:
            if _Cnb_t.shape[2] == 1: _Cnb_t_for_phi = _Cnb_t[:,:,0]
            else: raise ValueError("Cnb for get_Phi should be a 2D matrix for a single time step.")

        # Direct indexing, assuming lat, vn etc. are 1D arrays of scalars
        lat_t = lat[t]
        lon_t = lon[t]
        alt_t = alt[t]
        vn_t  = vn[t]
        ve_t  = ve[t]
        vd_t  = vd[t]
        fn_t  = fn[t]
        fe_t  = fe[t]
        fd_t  = fd[t]

        Phi = get_Phi(nx, lat_t, vn_t, ve_t, vd_t, fn_t, fe_t, fd_t, _Cnb_t_for_phi,
                      baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

        h_pred = get_h(current_itp_mapS_for_step, x, lat_t, lon_t, alt_t,
                        date=date, core=core, der_map=current_der_mapS_for_step, map_alt=map_alt)

        if isinstance(h_pred, (float, int, np.number)): h_pred = np.array([h_pred])
        if h_pred.ndim == 1: h_pred = h_pred.reshape(-1,1)

        resid = _meas_internal[t,:].reshape(-1,1) - h_pred

        r_out[:, t] = resid.flatten()

        H_m = get_H(current_itp_mapS_for_step, x, lat_t, lon_t, alt_t,
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
    if isinstance(itp_mapS, MapCache):
        current_itp_mapS_for_step = get_cached_map(itp_mapS, lat_curr, lon_curr, alt_curr, silent=True)
        # Julia: der_mapS = nothing if itp_mapS is Map_Cache
        current_der_mapS_for_step = None 

    h_pred = get_h(current_itp_mapS_for_step, ekf_rt.x, lat_curr, lon_curr, alt_curr,
                   date=ekf_rt.date, core=ekf_rt.core, 
                   der_map=current_der_mapS_for_step, map_alt=map_alt if map_alt != -1 else 0) 
    
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
def crlb(
    lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, dt, itp_mapS,
    P0=None, Qd=None, R=1.0,
    baro_tau=3600.0, acc_tau=3600.0, gyro_tau=3600.0, fogm_tau=600.0,
    date=None, core=False
):
    """
    Cramér-Rao lower bound (CRLB) computed with classic Kalman Filter.
    Equations evaluated about true trajectory.

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
        dt (float): Measurement time step [s]
        itp_mapS (callable or MapCache): Scalar map interpolation function or MapCache.
        P0 (np.ndarray, optional): Initial covariance matrix, shape (nx,nx).
                                   Defaults to create_P0().
        Qd (np.ndarray, optional): Discrete time process/system noise matrix, shape (nx,nx).
                                   Defaults to create_Qd().
        R (float or tuple, optional): Measurement noise variance. If tuple, mean is used.
                                      Defaults to 1.0.
        baro_tau (float, optional): Barometer time constant [s]. Defaults to 3600.0.
        acc_tau (float, optional): Accelerometer time constant [s]. Defaults to 3600.0.
        gyro_tau (float, optional): Gyroscope time constant [s]. Defaults to 3600.0.
        fogm_tau (float, optional): FOGM catch-all time constant [s]. Defaults to 600.0.
        date (float, optional): Measurement date (decimal year) for IGRF [yr].
                                Defaults to get_years(2020,185).
        core (bool, optional): If true, include core magnetic field in measurement. Defaults to False.

    Returns:
        np.ndarray: Non-linear covariance matrix P_out [nx,nx,N]
    """
    if P0 is None:
        P0 = create_P0()
    if Qd is None:
        Qd = create_Qd()
    if date is None:
        date = get_years(2020, 185)

    N = len(lat)
    nx = P0.shape[0]

    P_out = np.zeros((nx, nx, N), dtype=P0.dtype)
    x_true = np.zeros(nx, dtype=P0.dtype)  # CRLB evaluated about true trajectory (zero error state)
    P = P0.copy()

    R_val = R
    if isinstance(R, (list, tuple)) and len(R) == 2:
        R_val = np.mean(R)
    elif not isinstance(R, (float, int, np.number)):
        # Assuming R is already a scalar or a compatible numpy type
        if isinstance(R, np.ndarray) and R.size == 1:
            R_val = R.item()
        else:
            raise ValueError("R must be a scalar or a (min,max) tuple.")


    itp_mapS_arg = itp_mapS

    for t in range(N):
        current_itp_mapS_for_step = itp_mapS_arg
        if isinstance(itp_mapS_arg, MapCache):
            if not itp_mapS_arg.maps or not itp_mapS_arg.interpolators:
                # Fallback or error if MapCache is empty (similar to EKF logic)
                if hasattr(itp_mapS_arg, 'fallback_map_interpolator') and itp_mapS_arg.fallback_map_interpolator is not None:
                    current_itp_mapS_for_step = itp_mapS_arg.fallback_map_interpolator
                elif hasattr(itp_mapS_arg, 'fallback_map') and itp_mapS_arg.fallback_map is not None:
                    try:
                        # from .map_utils import map_interpolate # Moved to top
                        current_itp_mapS_for_step = map_interpolate(itp_mapS_arg.fallback_map)
                        if current_itp_mapS_for_step is None:
                            raise ValueError("Fallback interpolator creation failed for CRLB.")
                    except Exception as e_interp_crlb:
                        print(f"Warning: CRLB MapCache empty and fallback interpolator failed: {e_interp_crlb}")
                        current_itp_mapS_for_step = None
                else:
                    print("Warning: CRLB MapCache is empty and no fallback. Interpolation will likely fail.")
                    current_itp_mapS_for_step = None
            else:
                map_altitudes = np.array([m.alt for m in itp_mapS_arg.maps])
                current_alt_scalar = alt[t].item() if isinstance(alt[t], np.ndarray) else alt[t]
                closest_idx = np.argmin(np.abs(map_altitudes - current_alt_scalar))
                current_itp_mapS_for_step = itp_mapS_arg.interpolators[closest_idx]
                if current_itp_mapS_for_step is None:
                     print(f"Warning: Selected interpolator for CRLB from MapCache at index {closest_idx} is None.")
            # For CRLB, der_mapS is not used by get_H, so no special handling for it here.

        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 else Cnb
        
        # Ensure Cnb is 2D for get_Phi
        _Cnb_t_for_phi = _Cnb_t
        if _Cnb_t.ndim == 3:
            if _Cnb_t.shape[2] == 1: _Cnb_t_for_phi = _Cnb_t[:,:,0] # Should not happen if Cnb is (3,3,N)
            # else: raise ValueError("Cnb for get_Phi should be a 2D matrix for a single time step.")
            # This case is handled by Cnb[:,:,t] above for (3,3,N)

        lat_t, lon_t, alt_t = lat[t], lon[t], alt[t]
        vn_t, ve_t, vd_t = vn[t], ve[t], vd[t]
        fn_t, fe_t, fd_t = fn[t], fe[t], fd[t]

        Phi = get_Phi(nx, lat_t, vn_t, ve_t, vd_t, fn_t, fe_t, fd_t, _Cnb_t_for_phi,
                      baro_tau, acc_tau, gyro_tau, fogm_tau, dt)

        # H is (1, nx) for scalar measurement
        if not isinstance(current_itp_mapS_for_step, (MapS, MapS3D)):
            # This check assumes get_H expects MapS or MapS3D directly.
            raise TypeError(f'map_obj for get_H in crlb must be MapS or MapS3D, but got {type(current_itp_mapS_for_step)}')
        H_m = get_H(current_itp_mapS_for_step, x_true, lat_t, lon_t, alt_t, date=date, core=core)
        if H_m.ndim == 1: H_m = H_m.reshape(1, -1) # Ensure (1,nx)

        # S = H P H' + R
        S_val = H_m @ P @ H_m.T + R_val
        if S_val.ndim == 0: S_val = S_val.reshape(1,1) # Ensure S_val is 2D

        # K = P H' S^-1
        # K_val = (P @ H_m.T) @ np.linalg.inv(S_val) # Direct inversion
        K_val = (np.linalg.solve(S_val.T, H_m @ P.T)).T # Using solve

        # P_update = (I - K H) P
        P = (np.eye(nx, dtype=P.dtype) - K_val @ H_m) @ P
        
        P_out[:, :, t] = P

        # P_propagate = Phi P_update Phi' + Qd
        P = Phi @ P @ Phi.T + Qd

    return P_out

def crlb_ins(
    ins: INS, itp_mapS,
    P0=None, Qd=None, R=1.0,
    baro_tau=3600.0, acc_tau=3600.0, gyro_tau=3600.0, fogm_tau=600.0,
    date=None, core=False
):
    """
    CRLB overload for INS data structure.
    See `crlb` for argument details. `ins.dt` is used for the time step.
    """
    return crlb(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb,
        ins.dt, itp_mapS,
        P0=P0, Qd=Qd, R=R,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        date=date, core=core
    )

def calc_crlb_pos(P_crlb):
    """
    Calculate position CRLB (standard deviation) from CRLB covariance matrices.
    Assumes P_crlb is [nx,nx,N] or [nx,nx].
    States 0,1,2 are position errors (N, E, D).
    Returns sqrt(diag(P_pos)) for each time step, shape (3,N) or (3,).
    """
    if P_crlb.ndim == 3:
        return np.sqrt(np.diagonal(P_crlb[0:3, 0:3, :], axis1=0, axis2=1))
    elif P_crlb.ndim == 2:
        return np.sqrt(np.diag(P_crlb[0:3, 0:3]))
    else:
        raise ValueError("P_crlb must be 2D or 3D array")

def calc_crlb_vel(P_crlb):
    """
    Calculate velocity CRLB (standard deviation) from CRLB covariance matrices.
    Assumes P_crlb is [nx,nx,N] or [nx,nx].
    States 3,4,5 are velocity errors (N, E, D).
    Returns sqrt(diag(P_vel)) for each time step, shape (3,N) or (3,).
    """
    if P_crlb.ndim == 3:
        return np.sqrt(np.diagonal(P_crlb[3:6, 3:6, :], axis1=0, axis2=1))
    elif P_crlb.ndim == 2:
        return np.sqrt(np.diag(P_crlb[3:6, 3:6]))
    else:
        raise ValueError("P_crlb must be 2D or 3D array")

def calc_crlb_att(P_crlb):
    """
    Calculate attitude CRLB (standard deviation) from CRLB covariance matrices.
    Assumes P_crlb is [nx,nx,N] or [nx,nx].
    States 6,7,8 are attitude errors.
    Returns sqrt(diag(P_att)) for each time step, shape (3,N) or (3,).
    """
    if P_crlb.ndim == 3:
        return np.sqrt(np.diagonal(P_crlb[6:9, 6:9, :], axis1=0, axis2=1))
    elif P_crlb.ndim == 2:
        return np.sqrt(np.diag(P_crlb[6:9, 6:9]))
    else:
        raise ValueError("P_crlb must be 2D or 3D array")

def calc_crlb_fogm(P_crlb):
    """
    Calculate FOGM bias CRLB (standard deviation) from CRLB covariance matrices.
    Assumes P_crlb is [nx,nx,N] or [nx,nx].
    State 16 is FOGM bias (assuming nx >= 17).
    Returns sqrt(P_fogm_bias) for each time step, shape (N,) or scalar.
    """
    if P_crlb.shape[0] < 17: # nx must be at least 17
        raise ValueError("P_crlb does not have enough states for FOGM bias (expected nx >= 17).")
    if P_crlb.ndim == 3:
        return np.sqrt(P_crlb[16, 16, :])
    elif P_crlb.ndim == 2:
        return np.sqrt(P_crlb[16, 16])
    else:
        raise ValueError("P_crlb must be 2D or 3D array")

def calc_crlb_map(P_crlb):
    """
    Calculate map bias CRLB (standard deviation) from CRLB covariance matrices.
    Assumes P_crlb is [nx,nx,N] or [nx,nx].
    State 17 is map bias (assuming nx >= 18).
    Returns sqrt(P_map_bias) for each time step, shape (N,) or scalar.
    """
    if P_crlb.shape[0] < 18: # nx must be at least 18
        raise ValueError("P_crlb does not have enough states for map bias (expected nx >= 18).")
    if P_crlb.ndim == 3:
        return np.sqrt(P_crlb[17, 17, :])
    elif P_crlb.ndim == 2:
        return np.sqrt(P_crlb[17, 17])
    else:
        raise ValueError("P_crlb must be 2D or 3D array")
from .common_types import MagV # MagV needed for ekf_online
from .tolles_lawson import create_TL_A, create_TL_coef, get_TL_term_ind # For online TL EKF
from scipy.linalg import block_diag # For constructing block diagonal matrices

# Default terms for Tolles-Lawson in online EKF, mapping from common names to short codes
# Ensure these short codes are compatible with create_TL_A and create_TL_coef
DEFAULT_TL_TERMS_ONLINE = ['p', 'i', 'e', 'b'] # permanent, induced, eddy, bias

def ekf_online_setup(
    flux: MagV,
    meas: np.ndarray,
    ind: np.ndarray = None,
    Bt: np.ndarray = None,
    lam: float = 0.025, # lam is lambda in Julia
    terms: List[str] = None, # Uses short codes e.g. ['p', 'i', 'e', 'b']
    pass1: float = 0.1,
    pass2: float = 0.9,
    fs: float = 10.0,
    pole: int = 4,
    trim: int = 20,
    N_sigma: int = 100,
    Bt_scale: float = 50000.0
):
    """
    Setup for Extended Kalman Filter (EKF) with online learning of Tolles-Lawson coefficients.
    Ports MagNav.jl/src/ekf_online.jl -> ekf_online_setup.

    Args:
        flux (MagV): Vector magnetometer measurement struct.
        meas (np.ndarray): Scalar magnetometer measurement [nT].
        ind (np.ndarray, optional): Selected data indices. Defaults to all true.
        Bt (np.ndarray, optional): Magnitude of vector magnetometer measurements or
                                   scalar magnetometer measurements for modified Tolles-Lawson [nT].
                                   Defaults to norm of flux components.
        lam (float, optional): Ridge parameter for create_TL_coef. Defaults to 0.025.
        terms (List[str], optional): Tolles-Lawson terms to use (e.g., ['p','i','e','b']).
                                     Defaults to ['p', 'i', 'e', 'b'].
        pass1 (float, optional): First passband frequency [Hz] for create_TL_coef. Defaults to 0.1.
        pass2 (float, optional): Second passband frequency [Hz] for create_TL_coef. Defaults to 0.9.
        fs (float, optional): Sampling frequency [Hz] for create_TL_coef. Defaults to 10.0.
        pole (int, optional): Number of poles for Butterworth filter in create_TL_coef. Defaults to 4.
        trim (int, optional): Number of elements to trim after filtering in create_TL_coef. Defaults to 20.
        N_sigma (int, optional): Number of Tolles-Lawson coefficient sets to use to create TL_sigma.
                                 Defaults to 100.
        Bt_scale (float, optional): Scaling factor for induced & eddy current terms [nT]. Defaults to 50000.0.

    Returns:
        tuple: (x0_TL, P0_TL, TL_sigma)
            - x0_TL (np.ndarray): Initial Tolles-Lawson coefficient states.
            - P0_TL (np.ndarray): Initial Tolles-Lawson covariance matrix.
            - TL_sigma (np.ndarray): Tolles-Lawson coefficients process noise std dev.
    """
    if ind is None:
        ind = np.ones(len(meas), dtype=bool)
    if terms is None:
        terms = DEFAULT_TL_TERMS_ONLINE

    # Calculate Bt if not provided, using only indexed data for consistency
    if Bt is None:
        if not (flux.x.shape == flux.y.shape == flux.z.shape):
            raise ValueError("flux.x, flux.y, flux.z must have the same shape.")
        Bt_full = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
        Bt_calc = Bt_full[ind]
    else:
        Bt_calc = Bt[ind] if Bt.shape == ind.shape else Bt # Assume Bt is already indexed if not matching ind shape

    x0_TL, y_var = create_TL_coef(
        flux_or_Bx=flux, meas_or_By=meas, # Pass full flux and meas
        ind=ind, # Pass indexer
        Bt=Bt_calc, # Pass indexed Bt
        lam=lam, terms=terms, pass1=pass1, pass2=pass2, fs=fs,
        pole=pole, trim=trim, Bt_scale=Bt_scale, return_y_var=True
    )

    A = create_TL_A(
        flux_or_Bx=flux, ind=ind, # Pass indexer
        Bt=Bt_calc, # Pass indexed Bt
        terms=terms, Bt_scale=Bt_scale
    )

    if A.shape[0] == 0: # No data selected by ind
        raise ValueError("No data selected by 'ind', A matrix is empty.")
    if A.shape[0] < A.shape[1]:
        # Not enough data points for a full rank A'A, use pseudo-inverse
        P0_TL = np.linalg.pinv(A.T @ A) * y_var if y_var is not None else np.linalg.pinv(A.T @ A)
    else:
        try:
            P0_TL = np.linalg.inv(A.T @ A) * y_var if y_var is not None else np.linalg.inv(A.T @ A)
        except np.linalg.LinAlgError: # If singular
             P0_TL = np.linalg.pinv(A.T @ A) * y_var if y_var is not None else np.linalg.pinv(A.T @ A)


    true_indices = np.where(ind)[0]
    N_ind = len(true_indices)
    
    # Ensure N_sigma calculation is robust
    min_data_for_tl_coef = max(2 * trim, 50) # Minimum data points for one create_TL_coef call
    if N_ind < min_data_for_tl_coef:
        raise ValueError(f"Not enough data points ({N_ind}) after indexing. Need at least {min_data_for_tl_coef}.")

    # N is the number of windows to process
    # window_len is the length of data in each window
    # N_sigma is the desired number of coefficient sets
    
    # Max possible windows of at least min_data_for_tl_coef length
    max_possible_N = N_ind - min_data_for_tl_coef + 1
    N = min(max_possible_N, N_sigma)

    N_min_loops = 10 # Julia has N_min = 10
    if N < N_min_loops and N_ind >= min_data_for_tl_coef : # if we can make at least one window
        print(f"Warning: ekf_online_setup: N_sigma reduced from {N_sigma} to {N} due to limited data ({N_ind} points). Minimum {N_min_loops} preferred.")
    elif N_ind < min_data_for_tl_coef:
         raise ValueError(f"ekf_online_setup: Not enough data ({N_ind} points) to create even one TL coefficient set. Need at least {min_data_for_tl_coef}.")
    if N <= 0 : N = 1 # Ensure at least one loop if possible

    window_len = N_ind - N + 1 # Length of data for each call to create_TL_coef

    coef_set = np.zeros((len(x0_TL), N))

    for i in range(N):
        current_window_indices = true_indices[i : i + window_len]
        
        # Bt for the current window
        if Bt is None:
             Bt_window = np.sqrt(flux.x[current_window_indices]**2 + \
                                 flux.y[current_window_indices]**2 + \
                                 flux.z[current_window_indices]**2)
        else: # If Bt was provided, assume it's full and slice, or it's pre-indexed and we need to be careful
             # This part is tricky if Bt is pre-indexed. Assuming full Bt for simplicity here.
             Bt_window = Bt[current_window_indices]


        coef_set[:, i] = create_TL_coef(
            flux_or_Bx=flux, meas_or_By=meas, # Pass full flux and meas
            ind=current_window_indices, # Pass current window's indices
            Bt=Bt_window,
            lam=lam, terms=terms, pass1=pass1, pass2=pass2, fs=fs,
            pole=pole, trim=trim, Bt_scale=Bt_scale, return_y_var=False
        )

    TL_sigma = np.std(coef_set, axis=1)

    return x0_TL, P0_TL, TL_sigma


def ekf_online(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray,
    fn: np.ndarray, fe: np.ndarray, fd: np.ndarray, Cnb: np.ndarray,
    meas: np.ndarray, flux: MagV, dt: float, itp_mapS,
    x0_TL: np.ndarray, P0_tl: np.ndarray, tl_proc_noise_std: np.ndarray,
    R: Union[float, np.ndarray] = 1.0,
    P0_nav: np.ndarray = None, Qd_nav: np.ndarray = None,
    baro_tau: float = 3600.0, acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0, fogm_tau: float = 600.0,
    date: float = None, core: bool = False,
    terms: List[str] = None, Bt_scale: float = 50000.0,
    map_alt: float = 0.0,
    der_mapS=None # For consistency with batch EKF, though get_H might not use it for scalar
):
    """
    Extended Kalman Filter (EKF) with online learning of Tolles-Lawson coefficients.
    Ports MagNav.jl/src/ekf_online.jl -> ekf_online.
    Assumes nx_vec = 0 (no estimation of vector magnetometer errors in state).

    Args:
        lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb: Standard navigation inputs.
        meas (np.ndarray): Scalar magnetometer measurement [nT].
        flux (MagV): Vector magnetometer measurements (for TL A matrix).
        dt (float): Measurement time step [s].
        itp_mapS: Scalar map interpolation function or MapCache.
        x0_TL (np.ndarray): Initial Tolles-Lawson coefficient states.
        P0_tl (np.ndarray): Initial covariance for TL states.
        tl_proc_noise_std (np.ndarray): Process noise standard deviation for TL states.
        R (float or np.ndarray): Measurement noise variance. Defaults to 1.0.
        P0_nav (np.ndarray, optional): Initial covariance for navigation states (18 states).
                                       Defaults to create_P0().
        Qd_nav (np.ndarray, optional): Process noise for navigation states (18 states).
                                       Defaults to create_Qd().
        baro_tau, acc_tau, gyro_tau, fogm_tau: Time constants.
        date (float, optional): Measurement date for IGRF. Defaults to get_years(2020,185).
        core (bool, optional): If true, include core field. Defaults to False.
        terms (List[str], optional): Tolles-Lawson terms for A matrix. Defaults to ['p','i','e','b'].
        Bt_scale (float, optional): Scaling for TL A matrix. Defaults to 50000.0.
        map_alt (float, optional): Map altitude. Defaults to 0.
        der_mapS (callable, optional): Scalar map vertical derivative interpolator.

    Returns:
        FILTres: Filter results struct.
    """
    if P0_nav is None: P0_nav = create_P0()
    if Qd_nav is None: Qd_nav = create_Qd()
    if date is None: date = get_years(2020, 185)
    if terms is None: terms = DEFAULT_TL_TERMS_ONLINE

    N_data = len(lat)
    nx_nav = P0_nav.shape[0] # Should be 18
    nx_TL = len(x0_TL)
    nx_full = nx_nav + nx_TL

    if P0_nav.shape != (nx_nav, nx_nav) or Qd_nav.shape != (nx_nav, nx_nav):
        raise ValueError(f"P0_nav and Qd_nav must be square matrices of size {nx_nav}x{nx_nav}")
    if P0_tl.shape != (nx_TL, nx_TL):
        raise ValueError(f"P0_tl must be a square matrix of size {nx_TL}x{nx_TL}")
    if tl_proc_noise_std.shape != (nx_TL,):
        raise ValueError(f"tl_proc_noise_std must be of length {nx_TL}")

    # Construct full P0 and Qd
    P0_full = block_diag(P0_nav, P0_tl)
    Qd_tl = np.diag(tl_proc_noise_std**2)
    Qd_full = block_diag(Qd_nav, Qd_tl)

    if meas.ndim == 1:
        ny = 1
        _meas_internal = meas.reshape(-1, 1)
    elif meas.ndim == 2 and meas.shape[1] == 1:
        ny = 1
        _meas_internal = meas
    else:
        raise ValueError("meas must be a 1D array or a 2D array with one column.")

    x_out = np.zeros((nx_full, N_data))
    P_out = np.zeros((nx_full, nx_full, N_data))
    r_out = np.zeros((ny, N_data))

    x = np.zeros(nx_full)
    x[nx_nav : nx_full] = x0_TL # Nav states start at zero error
    P = P0_full.copy()

    # Create the Tolles-Lawson A matrix (design matrix)
    # Bt for A matrix construction - use full flux data for consistency with Julia
    # The create_TL_A function handles indexing if `ind` is passed, but here we need A for all time steps.
    # So, we pass full flux and it should return A for all time steps.
    Bt_for_A = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
    A_tl_matrix = create_TL_A(flux_or_Bx=flux, Bt=Bt_for_A, terms=terms, Bt_scale=Bt_scale)

    if A_tl_matrix.shape[0] != N_data or A_tl_matrix.shape[1] != nx_TL:
        raise ValueError(f"A_tl_matrix shape mismatch. Expected ({N_data}, {nx_TL}), got {A_tl_matrix.shape}")

    itp_mapS_arg = itp_mapS # To handle MapCache if passed

    for t in range(N_data):
        current_itp_mapS_for_step = itp_mapS_arg
        current_der_mapS_for_step = der_mapS
        if isinstance(itp_mapS_arg, MapCache):
            # This assumes a function _get_interpolator_from_cache exists or
            # get_h/get_H can handle MapCache directly.
            # For now, pass the cache and let get_h/get_H resolve.
            # If get_h/get_H expect only interpolators, this needs adjustment:
            # current_itp_mapS_for_step = _get_interpolator_from_cache(itp_mapS_arg, lat[t], lon[t], alt[t])
            # current_der_mapS_for_step = _get_interpolator_from_cache(der_mapS, lat[t], lon[t], alt[t]) if isinstance(der_mapS, MapCache) else der_mapS
            pass # Assuming get_h/get_H handle MapCache

        _Cnb_t = Cnb[:, :, t] if Cnb.ndim == 3 else Cnb

        # 1. Construct full Phi
        Phi_nav_t = get_Phi(nx_nav, lat[t], vn[t], ve[t], vd[t], fn[t], fe[t], fd[t], _Cnb_t,
                            baro_tau, acc_tau, gyro_tau, fogm_tau, dt)
        Phi_tl_t = np.eye(nx_TL) # TL states are random walk or constant
        Phi_full_t = block_diag(Phi_nav_t, Phi_tl_t)

        # 2. Predict measurement
        x_nav_t = x[:nx_nav]
        x_tl_t = x[nx_nav:]
        
        A_row_t = A_tl_matrix[t, :]
        h_tl_comp = A_row_t @ x_tl_t
        
        # get_h expects nav states (including map bias if it's part of its state definition)
        h_map_pred = get_h(current_itp_mapS_for_step, x_nav_t, lat[t], lon[t], alt[t],
                           date=date, core=core, der_map=current_der_mapS_for_step, map_alt=map_alt)
        
        if isinstance(h_map_pred, (float, int, np.number)): h_map_pred = np.array([h_map_pred])
        if h_map_pred.ndim == 1: h_map_pred = h_map_pred.reshape(-1,1) # to (1,1)

        h_full_pred = h_tl_comp + h_map_pred
        resid = _meas_internal[t, :].reshape(-1,1) - h_full_pred
        r_out[:, t] = resid.flatten()

        # 3. Construct full H
        # get_H returns Jacobian for nav states (1 x nx_nav)
        H_nav_t = get_H(current_itp_mapS_for_step, x_nav_t, lat[t], lon[t], alt[t],
                        date=date, core=core)
        if H_nav_t.ndim == 1: H_nav_t = H_nav_t.reshape(1, -1) # Ensure (1, nx_nav)

        H_tl_t = A_row_t.reshape(1, -1) # (1 x nx_TL)
        H_full_t = np.concatenate((H_nav_t, H_tl_t), axis=1) # (1 x nx_full)

        # 4. Kalman Update
        S_matrix_term = H_full_t @ P @ H_full_t.T
        _R_val = R
        if isinstance(R, np.ndarray) and R.ndim == 0: _R_val = R.item() # Handle 0-dim array R

        if ny == 1 and isinstance(_R_val, (float, int, np.number)):
            S_val = S_matrix_term + _R_val
        elif isinstance(_R_val, np.ndarray) and _R_val.shape == (ny, ny):
            S_val = S_matrix_term + _R_val
        else: # Fallback for ny > 1 and R_val scalar (though ny=1 here)
            S_val = S_matrix_term + np.eye(ny) * _R_val
        if S_val.ndim == 0: S_val = S_val.reshape(1,1)

        K_val = (np.linalg.solve(S_val.T, H_full_t @ P.T)).T

        x = x + K_val @ resid.flatten()
        P = (np.eye(nx_full) - K_val @ H_full_t) @ P
        
        x_out[:, t] = x
        P_out[:, :, t] = P

        # 5. Propagate
        x = Phi_full_t @ x
        P = Phi_full_t @ P @ Phi_full_t.T + Qd_full
        
    return FILTres(x_out, P_out, r_out, True)


def ekf_online_ins(
    ins: INS,
    meas: np.ndarray,
    flux: MagV,
    itp_mapS,
    x0_TL: np.ndarray, P0_tl: np.ndarray, tl_proc_noise_std: np.ndarray,
    R: Union[float, np.ndarray] = 1.0,
    P0_nav: np.ndarray = None, Qd_nav: np.ndarray = None,
    baro_tau: float = 3600.0, acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0, fogm_tau: float = 600.0,
    date: float = None, core: bool = False,
    terms: List[str] = None, Bt_scale: float = 50000.0,
    map_alt: float = 0.0,
    der_mapS=None
):
    """
    EKF with online Tolles-Lawson learning, overload for INS data structure.
    See `ekf_online` for detailed argument descriptions.
    """
    return ekf_online(
        ins.lat, ins.lon, ins.alt, ins.vn, ins.ve, ins.vd,
        ins.fn, ins.fe, ins.fd, ins.Cnb,
        meas, flux, ins.dt, itp_mapS,
        x0_TL, P0_tl, tl_proc_noise_std,
        R=R, P0_nav=P0_nav, Qd_nav=Qd_nav,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau,
        date=date, core=core, terms=terms, Bt_scale=Bt_scale,
        map_alt=map_alt, der_mapS=der_mapS
    )