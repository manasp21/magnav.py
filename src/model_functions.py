import math
import numpy as np
from scipy.linalg import expm
from scipy.linalg import cholesky as scipy_cholesky

# Attempt to import constants and utilities from other project modules
# These imports assume specific functions and constants exist in these files.
# If they don't, these would need to be defined or adjusted.
try:
    from .magnav import r_earth, omega_earth # ω_earth as omega_earth
    # Placeholder for IGRF calculation functions, assuming they might be in magnav or a dedicated util
    # These would need to be properly implemented or imported from a library like pyIGRF,
    # ensuring correct unit handling (e.g., IGRF typically uses degrees for lat/lon, km for alt).
    def _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad, geodetic_val=None):
        # Placeholder: Replace with actual IGRF calculation
        # Example: Convert units, call pyIGRF, return norm
        # print(f"Warning: _calculate_igrf_intensity_at_point is a placeholder for {date}, {alt_m}, {lat_rad}, {lon_rad}")
        return 1.0 # Placeholder value
    
    def _calculate_igrf_vector_at_point(date, alt_m, lat_rad, lon_rad, geodetic_val=None):
        # Placeholder: Replace with actual IGRF calculation returning (X,Y,Z) components
        # print(f"Warning: _calculate_igrf_vector_at_point is a placeholder for {date}, {alt_m}, {lat_rad}, {lon_rad}")
        return np.array([0.5, 0.5, np.sqrt(0.5)]) # Placeholder vector (norm = 1)

except ImportError:
    print("Warning: Could not import r_earth, omega_earth from .magnav. Using default values.")
    r_earth = 6371000.0  # Mean Earth radius in meters
    omega_earth = 7.292115e-5  # Earth rotation rate in rad/s
    def _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad, geodetic_val=None):
        return 1.0 
    def _calculate_igrf_vector_at_point(date, alt_m, lat_rad, lon_rad, geodetic_val=None):
        return np.array([0.5, 0.5, np.sqrt(0.5)])


try:
    from .analysis_util import get_years, dlat2dn, dn2dlat, de2dlon
except ImportError:
    print("Warning: Could not import from .analysis_util. Defining placeholders.")
    def get_years(year, day_of_year): return year + (day_of_year -1) / 365.25 # Simplified
    def dlat2dn(dlat_rad, lat_rad): return dlat_rad * r_earth # Simplified
    def dn2dlat(dn_m, lat_rad): return dn_m / r_earth # Simplified
    def de2dlon(de_m, lat_rad): return de_m / (r_earth * math.cos(lat_rad)) # Simplified


def deg2rad(degrees):
    return math.radians(degrees)

def create_P0(lat1_rad=deg2rad(45),
              init_pos_sigma=3.0,
              init_alt_sigma=0.001,
              init_vel_sigma=0.01,
              init_att_sigma_rad=deg2rad(0.01), # Renamed from init_att_sigma for clarity
              ha_sigma=0.001,
              a_hat_sigma=0.01,
              acc_sigma=0.000245,
              gyro_sigma=0.00000000727,
              fogm_sigma=3.0,
              vec_sigma=1000.0,
              vec_states: bool = False,
              fogm_state: bool = True,
              P0_TL=None):
    """
    Create initial covariance matrix P0.

    Args:
        lat1_rad: initial approximate latitude [rad]
        init_pos_sigma: initial position uncertainty [m]
        init_alt_sigma: initial altitude uncertainty [m]
        init_vel_sigma: initial velocity uncertainty [m/s]
        init_att_sigma_rad: initial attitude uncertainty [rad]
        ha_sigma: barometer aiding altitude bias [m]
        a_hat_sigma: barometer aiding vertical accel bias [m/s^2]
        acc_sigma: accelerometer bias [m/s^2]
        gyro_sigma: gyroscope bias [rad/s]
        fogm_sigma: FOGM catch-all bias [nT]
        vec_sigma: vector magnetometer noise std dev
        vec_states: if true, include vector magnetometer states
        fogm_state: if true, include FOGM catch-all bias state
        P0_TL: initial Tolles-Lawson covariance matrix (NumPy array)

    Returns:
        P0: initial covariance matrix (NumPy array)
    """
    if P0_TL is None:
        P0_TL = np.array([])

    nx_TL = P0_TL.shape[0] if P0_TL.size > 0 else 0
    nx_vec = 3 if vec_states else 0
    nx_fogm = 1 if fogm_state else 0

    nx = 17 + nx_TL + nx_vec + nx_fogm
    P0 = np.zeros((nx, nx), dtype=np.float64)

    P0[0, 0] = dn2dlat(init_pos_sigma, lat1_rad)**2
    P0[1, 1] = de2dlon(init_pos_sigma, lat1_rad)**2
    P0[2, 2] = init_alt_sigma**2
    P0[3, 3] = init_vel_sigma**2
    P0[4, 4] = init_vel_sigma**2
    P0[5, 5] = init_vel_sigma**2
    P0[6, 6] = init_att_sigma_rad**2
    P0[7, 7] = init_att_sigma_rad**2
    P0[8, 8] = init_att_sigma_rad**2
    P0[9, 9] = ha_sigma**2
    P0[10, 10] = a_hat_sigma**2
    P0[11, 11] = acc_sigma**2
    P0[12, 12] = acc_sigma**2
    P0[13, 13] = acc_sigma**2
    P0[14, 14] = gyro_sigma**2
    P0[15, 15] = gyro_sigma**2
    P0[16, 16] = gyro_sigma**2

    idx_start_TL = 17 # 0-indexed start for TL block (original 18)
    
    if nx_TL > 0:
        P0[idx_start_TL : idx_start_TL + nx_TL, idx_start_TL : idx_start_TL + nx_TL] = P0_TL
    
    idx_start_vec = idx_start_TL + nx_TL
    if nx_vec > 0:
        P0[idx_start_vec : idx_start_vec + nx_vec, idx_start_vec : idx_start_vec + nx_vec] = np.diag(np.full(nx_vec, vec_sigma**2))
        
    idx_start_fogm = idx_start_vec + nx_vec
    if nx_fogm > 0:
        P0[idx_start_fogm, idx_start_fogm] = fogm_sigma**2
        
    return P0

def create_Qd(dt=0.1,
              VRW_sigma=0.000238,
              ARW_sigma=0.000000581,
              baro_sigma=1.0,
              acc_sigma=0.000245,
              gyro_sigma=0.00000000727,
              fogm_sigma=3.0,
              vec_sigma=1000.0,
              TL_sigma=None, # Expect NumPy array
              baro_tau=3600.0,
              acc_tau=3600.0,
              gyro_tau=3600.0,
              fogm_tau=600.0,
              vec_states: bool = False,
              fogm_state: bool = True):
    """
    Create the discrete time process/system noise matrix Qd.
    """
    if TL_sigma is None:
        TL_sigma = np.array([])

    VRW_var = VRW_sigma**2
    ARW_var = ARW_sigma**2
    baro_drive = 2 * baro_sigma**2 / baro_tau
    acc_drive = 2 * acc_sigma**2 / acc_tau
    gyro_drive = 2 * gyro_sigma**2 / gyro_tau
    fogm_drive = 2 * fogm_sigma**2 / fogm_tau
    TL_var = TL_sigma**2 if TL_sigma.size > 0 else np.array([]) # Element-wise square
    vec_var = vec_sigma**2

    nx_TL = TL_sigma.shape[0] if TL_sigma.size > 0 else 0
    nx_vec = 3 if vec_states else 0
    nx_fogm = 1 if fogm_state else 0

    Q_diag_elements = np.concatenate([
        np.full(3, 1e-30),
        np.full(3, VRW_var),
        np.full(3, ARW_var),
        np.array([baro_drive]),
        np.array([1e-30]),
        np.full(3, acc_drive),
        np.full(3, gyro_drive),
        np.zeros(nx_TL + nx_vec + nx_fogm, dtype=np.float64) # Placeholder for TL, vec, fogm parts
    ])
    
    # Indices for TL, vec, fogm parts in Q_diag_elements (0-based)
    # Base states are 17 (0-16)
    idx_start_TL_in_Q = 17

    if nx_TL > 0:
        Q_diag_elements[idx_start_TL_in_Q : idx_start_TL_in_Q + nx_TL] = TL_var.flatten()
        
    idx_start_vec_in_Q = idx_start_TL_in_Q + nx_TL
    if nx_vec > 0:
        Q_diag_elements[idx_start_vec_in_Q : idx_start_vec_in_Q + nx_vec] = np.full(nx_vec, vec_var)
        
    idx_start_fogm_in_Q = idx_start_vec_in_Q + nx_vec
    if nx_fogm > 0:
        Q_diag_elements[idx_start_fogm_in_Q] = fogm_drive
        
    Qd = np.diag(Q_diag_elements) * dt
    return Qd

def create_model(dt=0.1, lat1_rad=deg2rad(45),
                 init_pos_sigma=3.0,
                 init_alt_sigma=0.001,
                 init_vel_sigma=0.01,
                 init_att_sigma_rad=deg2rad(0.01),
                 meas_var=3.0**2,
                 VRW_sigma=0.000238,
                 ARW_sigma=0.000000581,
                 baro_sigma=1.0,
                 ha_sigma=0.001,
                 a_hat_sigma=0.01,
                 acc_sigma=0.000245,
                 gyro_sigma=0.00000000727,
                 fogm_sigma=3.0,
                 vec_sigma=1000.0,
                 TL_sigma=None,
                 baro_tau=3600.0,
                 acc_tau=3600.0,
                 gyro_tau=3600.0,
                 fogm_tau=600.0,
                 vec_states: bool = False,
                 fogm_state: bool = True,
                 P0_TL=None):
    """
    Create a magnetic navigation filter model for use in an EKF or a MPF.
    """
    P0 = create_P0(lat1_rad=lat1_rad,
                   init_pos_sigma=init_pos_sigma,
                   init_alt_sigma=init_alt_sigma,
                   init_vel_sigma=init_vel_sigma,
                   init_att_sigma_rad=init_att_sigma_rad,
                   ha_sigma=ha_sigma,
                   a_hat_sigma=a_hat_sigma,
                   acc_sigma=acc_sigma,
                   gyro_sigma=gyro_sigma,
                   fogm_sigma=fogm_sigma,
                   vec_sigma=vec_sigma,
                   vec_states=vec_states,
                   fogm_state=fogm_state,
                   P0_TL=P0_TL)

    Qd = create_Qd(dt=dt,
                   VRW_sigma=VRW_sigma,
                   ARW_sigma=ARW_sigma,
                   baro_sigma=baro_sigma,
                   acc_sigma=acc_sigma,
                   gyro_sigma=gyro_sigma,
                   fogm_sigma=fogm_sigma,
                   vec_sigma=vec_sigma,
                   TL_sigma=TL_sigma,
                   baro_tau=baro_tau,
                   acc_tau=acc_tau,
                   gyro_tau=gyro_tau,
                   fogm_tau=fogm_tau,
                   vec_states=vec_states,
                   fogm_state=fogm_state)

    R_val = meas_var  # R is typically a scalar or matrix. Here it's scalar.
    return P0, Qd, R_val
def get_pinson(nx: int, lat_rad, vn, ve, vd, fn, fe, fd, Cnb,
               baro_tau=3600.0,
               acc_tau=3600.0,
               gyro_tau=3600.0,
               fogm_tau=600.0,
               vec_states: bool = False,
               fogm_state: bool = True,
               k1=3e-2, k2=3e-4, k3=1e-6):
    """
    Get the nx x nx Pinson dynamics matrix F.
    States (errors) are 0-indexed:
    0: lat, 1: lon, 2: alt
    3: vn, 4: ve, 5: vd
    6: tn, 7: te, 8: td (tilts)
    9: ha (baro alt bias)
    10: a_hat (baro accel bias)
    11: ax_bias, 12: ay_bias, 13: az_bias (accel biases)
    14: gx_bias, 15: gy_bias, 16: gz_bias (gyro biases)
    (Optional) Tolles-Lawson coeffs
    (Optional) Vector mag biases
    (Optional) FOGM catch-all S
    """
    tan_l = math.tan(lat_rad)
    cos_l = math.cos(lat_rad)
    # sin_l = math.sin(lat_rad) # Not used directly in original F matrix construction with these vars

    F = np.zeros((nx, nx), dtype=np.float64)

    # Pinson matrix population (0-indexed)
    # Row 0 (d(lat_err)/dt)
    F[0, 2] = -vn / r_earth**2
    F[0, 3] = 1 / r_earth

    # Row 1 (d(lon_err)/dt)
    F[1, 0] = ve * tan_l / (r_earth * cos_l)
    F[1, 2] = -ve / (cos_l * r_earth**2)
    F[1, 4] = 1 / (r_earth * cos_l)

    # Row 2 (d(alt_err)/dt)
    F[2, 2] = -k1
    F[2, 5] = -1
    F[2, 9] = k1 # ha_bias_err (state 10 in Julia, 9 here)

    # Row 3 (d(vn_err)/dt)
    F[3, 0] = -ve * (2 * omega_earth * cos_l + ve / (r_earth * cos_l**2))
    F[3, 2] = (ve**2 * tan_l - vn * vd) / r_earth**2
    F[3, 3] = vd / r_earth
    F[3, 4] = -2 * (omega_earth * math.sin(lat_rad) + ve * tan_l / r_earth)
    F[3, 5] = vn / r_earth
    F[3, 7] = -fd # te_err (state 8 in Julia, 7 here)
    F[3, 8] = fe  # td_err (state 9 in Julia, 8 here)

    # Row 4 (d(ve_err)/dt)
    F[4, 0] = 2 * omega_earth * (vn * cos_l - vd * math.sin(lat_rad)) + vn * ve / (r_earth * cos_l**2)
    F[4, 2] = -ve * ((vn * tan_l + vd) / r_earth**2)
    F[4, 3] = 2 * omega_earth * math.sin(lat_rad) + ve * tan_l / r_earth
    F[4, 4] = (vn * tan_l + vd) / r_earth
    F[4, 5] = 2 * omega_earth * cos_l + ve / r_earth
    F[4, 6] = fd  # tn_err (state 7 in Julia, 6 here)
    F[4, 8] = -fn # td_err

    # Row 5 (d(vd_err)/dt)
    F[5, 0] = 2 * omega_earth * ve * math.sin(lat_rad)
    F[5, 2] = (vn**2 + ve**2) / r_earth**2 + k2
    F[5, 3] = -2 * vn / r_earth
    F[5, 4] = -2 * (omega_earth * cos_l + ve / r_earth)
    F[5, 6] = -fe # tn_err
    F[5, 7] = fn  # te_err
    F[5, 9] = -k2 # ha_bias_err
    F[5, 10] = 1   # a_hat_bias_err (state 11 in Julia, 10 here)

    # Row 6 (d(tn_err)/dt)
    F[6, 0] = -omega_earth * math.sin(lat_rad)
    F[6, 2] = -ve / r_earth**2 # Original had ve^2, seems like a typo based on typical Pinson forms, using ve
    F[6, 4] = 1 / r_earth
    F[6, 7] = -omega_earth * math.sin(lat_rad) - ve * tan_l / r_earth # te_err
    F[6, 8] = vn / r_earth # td_err

    # Row 7 (d(te_err)/dt)
    F[7, 2] = vn / r_earth**2
    F[7, 3] = -1 / r_earth
    F[7, 6] = omega_earth * math.sin(lat_rad) + ve * tan_l / r_earth # tn_err
    F[7, 8] = omega_earth * cos_l + ve / r_earth # td_err

    # Row 8 (d(td_err)/dt)
    F[8, 0] = -omega_earth * cos_l - ve / (r_earth * cos_l**2)
    F[8, 2] = ve * tan_l / r_earth**2
    F[8, 4] = -tan_l / r_earth
    F[8, 6] = -vn / r_earth # tn_err
    F[8, 7] = -omega_earth * cos_l - ve / r_earth # te_err

    # Sensor biases
    F[9, 9]   = -1 / baro_tau # ha_bias_err
    F[10, 2]  = k3            # a_hat_bias_err depends on alt_err
    F[10, 9]  = -k3           # a_hat_bias_err depends on ha_bias_err
    # F[10,10] = 0 by default for a_hat_bias if not driven by other terms

    F[11, 11] = -1 / acc_tau  # ax_bias
    F[12, 12] = -1 / acc_tau  # ay_bias
    F[13, 13] = -1 / acc_tau  # az_bias
    F[14, 14] = -1 / gyro_tau # gx_bias
    F[15, 15] = -1 / gyro_tau # gy_bias
    F[16, 16] = -1 / gyro_tau # gz_bias

    # Coupling Cnb to accel/gyro biases
    # d(vel_err)/dt depends on accel_bias_err (states 12,13,14 in Julia -> 11,12,13 Python)
    F[3:6, 11:14] = Cnb
    # d(tilt_err)/dt depends on gyro_bias_err (states 15,16,17 in Julia -> 14,15,16 Python)
    F[6:9, 14:17] = -Cnb
    
    # Optional states at the end
    current_max_std_idx = 16 # Max index for standard 17 states (0-16)
    
    # Tolles-Lawson states are assumed to be simple random walks (zero entries in F)
    # unless specified otherwise, so their part of F remains zero.
    # nx_TL = ... (not explicitly calculated here, but assumed part of nx)

    idx_vec_start = nx - (3 if vec_states else 0) - (1 if fogm_state else 0)

    if vec_states:
        # Assuming vec_mag biases are modeled as very fast decaying states or placeholders
        # The -1e9 implies a very fast decay to zero if these are error states.
        # Or, it might be a placeholder for a different model.
        # This part needs clarification on the physical meaning in the original model.
        # For now, direct translation of the large negative diagonal.
        F[idx_vec_start, idx_vec_start]     = -1e9 # Bx_bias
        F[idx_vec_start+1, idx_vec_start+1] = -1e9 # By_bias
        F[idx_vec_start+2, idx_vec_start+2] = -1e9 # Bz_bias

    if fogm_state:
        F[nx-1, nx-1] = -1 / fogm_tau # S_err (FOGM catch-all)

    return F

def get_Phi(nx: int, lat_rad, vn, ve, vd, fn, fe, fd, Cnb,
            baro_tau, acc_tau, gyro_tau, fogm_tau, dt,
            vec_states: bool = False,
            fogm_state: bool = True,
            # k1,k2,k3 for get_pinson are using defaults if not passed
            **pinson_kwargs): 
    """
    Get Pinson matrix exponential (state transition matrix Phi).
    **pinson_kwargs are passed to get_pinson (e.g. k1, k2, k3).
    """
    F_matrix = get_pinson(nx, lat_rad, vn, ve, vd, fn, fe, fd, Cnb,
                   baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                   fogm_tau=fogm_tau, vec_states=vec_states, fogm_state=fogm_state,
                   **pinson_kwargs)
    return expm(F_matrix * dt) # Corrected variable name

def map_grad(itp_mapS, lat_rad, lon_rad, alt_m, delta_rad=1.0e-8):
    """
    Internal helper function to get local map gradient.
    itp_mapS: scalar map interpolation function f(lat_rad, lon_rad, alt_m)
    Returns:
        mapS_grad: [dmap/dlat (nT/rad), dmap/dlon (nT/rad), dmap/dalt (nT/m)]
    """
    dlat_rad = dlon_rad = delta_rad
    # Convert delta_rad used for lat/lon to an equivalent meter step for altitude
    dalt_m = dlat2dn(delta_rad, lat_rad) # Using dlat2dn for consistency with Julia

    grad_lat = (itp_mapS(lat_rad + dlat_rad, lon_rad, alt_m) -
                itp_mapS(lat_rad - dlat_rad, lon_rad, alt_m)) / (2 * dlat_rad)
    grad_lon = (itp_mapS(lat_rad, lon_rad + dlon_rad, alt_m) -
                itp_mapS(lat_rad, lon_rad - dlon_rad, alt_m)) / (2 * dlon_rad)
    grad_alt = (itp_mapS(lat_rad, lon_rad, alt_m + dalt_m) -
                itp_mapS(lat_rad, lon_rad, alt_m - dalt_m)) / (2 * dalt_m)
    
    return np.array([grad_lat, grad_lon, grad_alt])

def igrf_grad(lat_rad, lon_rad, alt_m, date=get_years(2020, 185), delta_rad=1.0e-8):
    """
    Internal helper function to get core magnetic field gradient using IGRF model.
    Assumes _calculate_igrf_intensity_at_point is available and handles units.
    Returns:
        core_grad: [dcore/dlat (nT/rad), dcore/dlon (nT/rad), dcore/dalt (nT/m)]
    """
    dlat_rad = dlon_rad = delta_rad
    dalt_m = dlat2dn(delta_rad, lat_rad)

    # Using the placeholder IGRF calculation function
    # The actual IGRF function would take date, alt, lat, lon
    # Ensure units are consistent (e.g. pyIGRF uses lat/lon in deg, alt in km)
    
    # Gradient w.r.t. latitude
    val_plus_dlat = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad + dlat_rad, lon_rad)
    val_minus_dlat = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad - dlat_rad, lon_rad)
    grad_lat = (val_plus_dlat - val_minus_dlat) / (2 * dlat_rad)

    # Gradient w.r.t. longitude
    val_plus_dlon = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad + dlon_rad)
    val_minus_dlon = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad - dlon_rad)
    grad_lon = (val_plus_dlon - val_minus_dlon) / (2 * dlon_rad)

    # Gradient w.r.t. altitude
    val_plus_dalt = _calculate_igrf_intensity_at_point(date, alt_m + dalt_m, lat_rad, lon_rad)
    val_minus_dalt = _calculate_igrf_intensity_at_point(date, alt_m - dalt_m, lat_rad, lon_rad)
    grad_alt = (val_plus_dalt - val_minus_dalt) / (2 * dalt_m)
    
    return np.array([grad_lat, grad_lon, grad_alt])

def get_H(itp_mapS, x_state_error: np.ndarray, lat_true_rad, lon_true_rad, alt_true_m,
          date=get_years(2020, 185),
          core: bool = False):
    """
    Internal helper function to get expected magnetic measurement Jacobian H.
    x_state_error: current state error estimate [lat_err, lon_err, alt_err, ..., S_err]
    lat_true_rad, lon_true_rad, alt_true_m: current true/reference position
    """
    # Position errors are x_state_error[0], x_state_error[1], x_state_error[2]
    # FOGM state error S_err is x_state_error[-1]
    
    # Calculate gradient at the estimated true position (true + error)
    est_lat_rad = lat_true_rad + x_state_error[0]
    est_lon_rad = lon_true_rad + x_state_error[1]
    est_alt_m   = alt_true_m   + x_state_error[2]

    map_gradient = map_grad(itp_mapS, est_lat_rad, est_lon_rad, est_alt_m)
    
    if core:
        core_field_gradient = igrf_grad(est_lat_rad, est_lon_rad, est_alt_m, date=date)
        total_gradient_pos = map_gradient + core_field_gradient
    else:
        total_gradient_pos = map_gradient
        
    # H matrix row vector: [d(mag)/d(pos_err), zeros for vel,att,biases etc., d(mag)/d(S_err)]
    # Assuming S_err (FOGM bias) is the last state if fogm_state=True in P0/Qd setup.
    # The size of x_state_error determines the number of zeros.
    
    # num_intermediate_states = len(x_state_error) - 3 - 1 # Total - pos_states - S_state
    
    H_row = np.zeros(len(x_state_error), dtype=np.float64)
    H_row[0:3] = total_gradient_pos # d(mag)/d(lat_err), d(mag)/d(lon_err), d(mag)/d(alt_err)
    # H_row[3 : 3 + num_intermediate_states] remains zero
    if len(x_state_error) > 3 : # Check if S_err state exists
        H_row[-1] = 1.0 # d(mag)/d(S_err) assuming S is additive bias
    
    return H_row # This is a 1D array, effectively a row vector

# Overloaded get_h function. Python handles this with default args or different names.
# For clarity, I'll name them get_h_basic and get_h_with_derivative.

def get_h_basic(itp_mapS, x_state_error: np.ndarray, lat_true_rad, lon_true_rad, alt_true_m,
                date=get_years(2020, 185),
                core: bool = False):
    """
    Expected magnetic measurement h(x).
    x_state_error is a 1D array for a single point, or 2D (num_states, num_points) for multiple.
    This implementation assumes x_state_error is for a single point or needs vectorization for itp_mapS.
    The Julia version itp_mapS.(...) implies itp_mapS can take array args or is broadcasted.
    We assume itp_mapS and _calculate_igrf_intensity_at_point handle scalar inputs here,
    and vectorization would be done by caller if needed for multiple particles.
    """
    
    # If x_state_error is for multiple particles (e.g., shape [num_states, num_particles])
    if x_state_error.ndim == 2:
        est_lat_rad = lat_true_rad + x_state_error[0, :] # Array of lats
        est_lon_rad = lon_true_rad + x_state_error[1, :] # Array of lons
        est_alt_m   = alt_true_m   + x_state_error[2, :] # Array of alts
        s_bias      = x_state_error[-1, :] if x_state_error.shape[0] > 3 else 0.0 # Array of S_biases
        
        # itp_mapS needs to be able to handle array inputs for lat, lon, alt
        # or we need to loop. Assuming it can handle arrays:
        map_val = itp_mapS(est_lat_rad, est_lon_rad, est_alt_m)
    else: # Single point
        est_lat_rad = lat_true_rad + x_state_error[0]
        est_lon_rad = lon_true_rad + x_state_error[1]
        est_alt_m   = alt_true_m   + x_state_error[2]
        s_bias      = x_state_error[-1] if len(x_state_error) > 3 else 0.0
        map_val = itp_mapS(est_lat_rad, est_lon_rad, est_alt_m)

    if core:
        # _calculate_igrf_intensity_at_point needs to handle array inputs or be looped
        if x_state_error.ndim == 2:
            core_val = np.array([_calculate_igrf_intensity_at_point(date, alt, lat, lon)
                                 for alt, lat, lon in zip(est_alt_m, est_lat_rad, est_lon_rad)])
        else:
            core_val = _calculate_igrf_intensity_at_point(date, est_alt_m, est_lat_rad, est_lon_rad)
        return map_val + s_bias + core_val
    else:
        return map_val + s_bias

def get_h_with_derivative(itp_mapS, der_mapS, x_state_error: np.ndarray,
                          lat_true_rad, lon_true_rad, alt_true_m, map_comp_alt_m, # map_alt in Julia
                          date=get_years(2020, 185),
                          core: bool = False):
    """
    Expected magnetic measurement h(x) including vertical derivative.
    map_comp_alt_m: map compilation altitude [m]
    Similar vectorization considerations as get_h_basic.
    """
    if x_state_error.ndim == 2:
        est_lat_rad = lat_true_rad + x_state_error[0, :]
        est_lon_rad = lon_true_rad + x_state_error[1, :]
        est_alt_m   = alt_true_m   + x_state_error[2, :]
        s_bias      = x_state_error[-1, :] if x_state_error.shape[0] > 3 else 0.0
        
        map_val = itp_mapS(est_lat_rad, est_lon_rad, est_alt_m)
        der_val = der_mapS(est_lat_rad, est_lon_rad, est_alt_m) # Vertical derivative from map
    else: # Single point
        est_lat_rad = lat_true_rad + x_state_error[0]
        est_lon_rad = lon_true_rad + x_state_error[1]
        est_alt_m   = alt_true_m   + x_state_error[2]
        s_bias      = x_state_error[-1] if len(x_state_error) > 3 else 0.0
        map_val = itp_mapS(est_lat_rad, est_lon_rad, est_alt_m)
        der_val = der_mapS(est_lat_rad, est_lon_rad, est_alt_m)

    # Term for vertical correction: der_val * (current_alt - map_compilation_alt)
    vertical_correction = der_val * (est_alt_m - map_comp_alt_m)
    
    h_val = map_val + s_bias + vertical_correction

    if core:
        if x_state_error.ndim == 2:
            core_val = np.array([_calculate_igrf_intensity_at_point(date, alt, lat, lon)
                                 for alt, lat, lon in zip(est_alt_m, est_lat_rad, est_lon_rad)])
        else:
            core_val = _calculate_igrf_intensity_at_point(date, est_alt_m, est_lat_rad, est_lon_rad)
        return h_val + core_val
    else:
        return h_val

def fogm(sigma, tau, dt, N):
    """
    First-order Gauss-Markov stochastic process.
    """
    x = np.zeros(N, dtype=np.float64)
    if N == 0:
        return x
        
    x[0] = sigma * np.random.randn() # randn() gives a single float
    phi_gm = math.exp(-dt / tau) # Renamed from Phi to avoid conflict with state transition matrix
    
    q_drive_var_ct = 2 * sigma**2 / tau
    q_drive_var_dt = q_drive_var_ct * dt 
    
    std_dev_noise_term = math.sqrt(q_drive_var_dt)

    for i in range(1, N):
        x[i] = phi_gm * x[i-1] + std_dev_noise_term * np.random.randn()
        
    return x

def chol(M: np.ndarray):
    """
    Internal helper function to get the Cholesky factorization of matrix M,
    returning the upper triangular factor U such that M = U.T @ U.
    The matrix is made symmetric before factorization to handle potential
    minor asymmetries from floating point errors.
    """
    M_np = np.asarray(M)
    M_sym = (M_np + M_np.T) / 2.0
    try:
        U = scipy_cholesky(M_sym, lower=False)
        return U
    except np.linalg.LinAlgError as e:
        print(f"Cholesky decomposition failed: {e}. Matrix might not be positive definite.")
        raise 

if __name__ == '__main__':
    print("MagNavPy model_functions.py loaded and appended content.")
    
    P0_test = create_P0(lat1_rad=deg2rad(34.0))
    print(f"P0 shape: {P0_test.shape}")

    Qd_test = create_Qd(dt=0.05)
    print(f"Qd shape: {Qd_test.shape}")

    P0m, Qdm, Rm = create_model(dt=0.05, lat1_rad=deg2rad(34.0))
    print(f"P0m shape: {P0m.shape}, Qdm shape: {Qdm.shape}, Rm: {Rm}")

    Cnb_test = np.eye(3)
    F_test = get_pinson(nx=17, lat_rad=deg2rad(34), vn=100, ve=10, vd=1,
                        fn=0, fe=0, fd=9.8, Cnb=Cnb_test)
    print(f"F_pinson shape: {F_test.shape}")

    fogm_data = fogm(sigma=3.0, tau=600.0, dt=0.1, N=100)
    print(f"FOGM data example (first 5): {fogm_data[:5]}")

    test_matrix = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=float)
    U_chol = chol(test_matrix)
    print(f"Cholesky U for test_matrix:\n{U_chol}")
    print(f"U.T @ U:\n{U_chol.T @ U_chol}")

    def dummy_itp_mapS(lat, lon, alt):
        return 10 * lat + 5 * lon + 0.1 * alt 
    
    map_g = map_grad(dummy_itp_mapS, deg2rad(34), deg2rad(-118), 1000)
    print(f"map_grad example: {map_g}")

    igrf_g = igrf_grad(deg2rad(34), deg2rad(-118), 1000, date=2021.5)
    print(f"igrf_grad example: {igrf_g}") 

    x_err_test = np.zeros(17) 
    H_test = get_H(dummy_itp_mapS, x_err_test, deg2rad(34), deg2rad(-118), 1000)
    print(f"get_H example: {H_test}")

    h_basic_test = get_h_basic(dummy_itp_mapS, x_err_test, deg2rad(34), deg2rad(-118), 1000)
    print(f"get_h_basic example: {h_basic_test}")