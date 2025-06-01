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
    from .core_utils import get_years
except ImportError:
    print("Warning: Could not import get_years from .core_utils. Defining placeholder.")
    def get_years(year, day_of_year): return year + (day_of_year -1) / 365.25 # Simplified

try:
    from .analysis_util import dlat2dn, dn2dlat, de2dlon
except ImportError:
    print("Warning: Could not import dlat2dn, dn2dlat, de2dlon from .analysis_util. Defining placeholders.")
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
        lat1_rad (float): initial approximate latitude [rad]
        init_pos_sigma (float): initial position uncertainty [m]
        init_alt_sigma (float): initial altitude uncertainty [m]
        init_vel_sigma (float): initial velocity uncertainty [m/s]
        init_att_sigma_rad (float): initial attitude uncertainty [rad]
        ha_sigma (float): barometer aiding altitude bias [m]
        a_hat_sigma (float): barometer aiding vertical accel bias [m/s^2]
        acc_sigma (float): accelerometer bias [m/s^2]
        gyro_sigma (float): gyroscope bias [rad/s]
        fogm_sigma (float): FOGM catch-all bias [nT]
        vec_sigma (float): vector magnetometer noise std dev
        vec_states (bool): if true, include vector magnetometer states
        fogm_state (bool): if true, include FOGM catch-all bias state
        P0_TL (Optional[numpy.ndarray]): initial Tolles-Lawson covariance matrix (NumPy array)

    Returns:
        numpy.ndarray: initial covariance matrix (NumPy array)
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

    Args:
        dt (float): time step [s]
        VRW_sigma (float): velocity random walk standard deviation
        ARW_sigma (float): angle random walk standard deviation
        baro_sigma (float): barometer noise standard deviation
        acc_sigma (float): accelerometer noise standard deviation
        gyro_sigma (float): gyroscope noise standard deviation
        fogm_sigma (float): FOGM catch-all bias standard deviation
        vec_sigma (float): vector magnetometer noise standard deviation
        TL_sigma (Optional[numpy.ndarray]): Tolles-Lawson covariance matrix (NumPy array)
        baro_tau (float): barometer correlation time [s]
        acc_tau (float): accelerometer correlation time [s]
        gyro_tau (float): gyroscope correlation time [s]
        fogm_tau (float): FOGM catch-all correlation time [s]
        vec_states (bool): if true, include vector magnetometer states
        fogm_state (bool): if true, include FOGM catch-all bias state

    Returns:
        numpy.ndarray: discrete time process/system noise matrix Qd (NumPy array)
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
    # Debugging: Print nx and F shape
    print(f"DEBUG: get_pinson - nx: {nx}, F shape: {F.shape}")
    # Debugging: Print input types and values
    # print(f"DEBUG: get_pinson inputs - lat_rad: {lat_rad} (type: {type(lat_rad)})")
    print(f"DEBUG: get_pinson inputs - vn: {vn} (type: {type(vn)})")
    # print(f"DEBUG: get_pinson inputs - ve: {ve} (type: {type(ve)})")
    # print(f"DEBUG: get_pinson inputs - vd: {vd} (type: {type(vd)})")
    # print(f"DEBUG: get_pinson inputs - fn: {fn} (type: {type(fn)})")
    # print(f"DEBUG: get_pinson inputs - fe: {fe} (type: {type(fe)})")
    # print(f"DEBUG: get_pinson inputs - fd: {fd} (type: {type(fd)})")

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

    :param nx: number of states
    :type nx: int
    :param lat_rad: latitude [rad]
    :type lat_rad: float
    :param vn: north velocity [m/s]
    :type vn: float
    :param ve: east velocity [m/s]
    :type ve: float
    :param vd: down velocity [m/s]
    :type vd: float
    :param fn: north specific force [m/s^2]
    :type fn: float
    :param fe: east specific force [m/s^2]
    :type fe: float
    :param fd: down specific force [m/s^2]
    :type fd: float
    :param Cnb: 3x3 direction cosine matrix (body to navigation)
    :type Cnb: numpy.ndarray
    :param baro_tau: barometer correlation time [s]
    :type baro_tau: float
    :param acc_tau: accelerometer correlation time [s]
    :type acc_tau: float
    :param gyro_tau: gyroscope correlation time [s]
    :type gyro_tau: float
    :param fogm_tau: FOGM catch-all correlation time [s]
    :type fogm_tau: float
    :param dt: time step [s]
    :type dt: float
    :param vec_states: if true, include vector magnetometer states
    :type vec_states: bool
    :param fogm_state: if true, include FOGM catch-all bias state
    :type fogm_state: bool
    :param pinson_kwargs: additional keyword arguments passed to get_pinson (e.g. k1, k2, k3)
    :type pinson_kwargs: dict
    :returns: state transition matrix Phi (NumPy array)
    :rtype: numpy.ndarray
    """
    # Ensure scalar values are passed to get_pinson
    # If the input is a numpy array, try to extract its scalar value.
    # This assumes that at this stage of the EKF (processing a single time step),
    # these navigation parameters should be scalars.
    _lat_rad = lat_rad.item() if isinstance(lat_rad, (np.ndarray, np.generic)) and lat_rad.size == 1 else (lat_rad if not isinstance(lat_rad, (np.ndarray, np.generic)) else float(lat_rad))
    _vn = vn.item() if isinstance(vn, (np.ndarray, np.generic)) and vn.size == 1 else (vn if not isinstance(vn, (np.ndarray, np.generic)) else float(vn))
    _ve = ve.item() if isinstance(ve, (np.ndarray, np.generic)) and ve.size == 1 else (ve if not isinstance(ve, (np.ndarray, np.generic)) else float(ve))
    _vd = vd.item() if isinstance(vd, (np.ndarray, np.generic)) and vd.size == 1 else (vd if not isinstance(vd, (np.ndarray, np.generic)) else float(vd))
    _fn = fn.item() if isinstance(fn, (np.ndarray, np.generic)) and fn.size == 1 else (fn if not isinstance(fn, (np.ndarray, np.generic)) else float(fn))
    _fe = fe.item() if isinstance(fe, (np.ndarray, np.generic)) and fe.size == 1 else (fe if not isinstance(fe, (np.ndarray, np.generic)) else float(fe))
    _fd = fd.item() if isinstance(fd, (np.ndarray, np.generic)) and fd.size == 1 else (fd if not isinstance(fd, (np.ndarray, np.generic)) else float(fd))

    F_matrix = get_pinson(nx, _lat_rad, _vn, _ve, _vd, _fn, _fe, _fd, Cnb,
                   baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                   fogm_tau=fogm_tau, vec_states=vec_states, fogm_state=fogm_state,
                   **pinson_kwargs)
    return expm(F_matrix * dt)


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
    val_plus_lon = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad + dlon_rad)
    val_minus_lon = _calculate_igrf_intensity_at_point(date, alt_m, lat_rad, lon_rad - dlon_rad)
    grad_lon = (val_plus_lon - val_minus_lon) / (2 * dlon_rad)

    # Gradient w.r.t. altitude
    val_plus_alt = _calculate_igrf_intensity_at_point(date, alt_m + dalt_m, lat_rad, lon_rad)
    val_minus_alt = _calculate_igrf_intensity_at_point(date, alt_m - dalt_m, lat_rad, lon_rad)
    grad_alt = (val_plus_alt - val_minus_alt) / (2 * dalt_m)
    
    return np.array([grad_lat, grad_lon, grad_alt])

def get_H(itp_mapS, x_state_error: np.ndarray, lat_true_rad, lon_true_rad, alt_true_m,
          date=get_years(2020, 185),
          core: bool = False,
          vec_states: bool = False,
          fogm_state: bool = True,
          TL_coeffs_len: int = 0, # Length of Tolles-Lawson coefficients if present
          delta_rad=1.0e-8):
    """
    Get measurement matrix H.
    itp_mapS: scalar map interpolation function f(lat_rad, lon_rad, alt_m)
    x_state_error: current state error vector (used to determine nx)
    """
    nx = x_state_error.shape[0]
    H = np.zeros(nx, dtype=np.float64)

    # Calculate map gradient at true location
    map_grad_vals = map_grad(itp_mapS, lat_true_rad, lon_true_rad, alt_true_m, delta_rad)
    
    # d(map)/d(lat_err) = d(map)/d(lat) * d(lat)/d(lat_err)
    # Assuming lat_err is in radians, d(lat)/d(lat_err) = 1
    H[0] = map_grad_vals[0]  # dmap/dlat (nT/rad)
    H[1] = map_grad_vals[1]  # dmap/dlon (nT/rad)
    H[2] = map_grad_vals[2]  # dmap/dalt (nT/m)

    if core:
        core_grad_vals = igrf_grad(lat_true_rad, lon_true_rad, alt_true_m, date, delta_rad)
        H[0] += core_grad_vals[0]
        H[1] += core_grad_vals[1]
        H[2] += core_grad_vals[2]

    # Optional states
    # Standard states end at index 16 (17 states total 0-16)
    # Tolles-Lawson coeffs start after standard states
    # Vector mag biases start after TL coeffs
    # FOGM state is last
    
    idx_fogm = nx -1 # FOGM state is always the last one if present

    if fogm_state:
        H[idx_fogm] = 1.0 # d(meas)/d(S_err) = 1

    # Vector magnetometer biases (if vec_states is True)
    # These would affect the measurement if the measurement model included them.
    # For a scalar magnetometer, these biases don't directly appear in H for total field.
    # If this H is for a vector measurement, this part would need adjustment.
    # Assuming for now this H is for a scalar total field measurement.
    # If vec_states are present, they are before fogm_state (if both true)
    # The Julia code has a section for `vec_states` in `get_H_core` which is more complex
    # and depends on the core field vector. This simplified H assumes scalar measurement.

    return H # Returns a 1D array (1,nx) effectively for scalar measurement

# --- Measurement Prediction Functions (h(x)) ---

def get_h_basic(itp_mapS, x_state_error: np.ndarray, lat_true_rad, lon_true_rad, alt_true_m,
                date=get_years(2020, 185),
                core: bool = False,
                vec_states: bool = False, # Placeholder, not used in basic scalar
                fogm_state: bool = True,
                TL_coeffs_len: int = 0): # Placeholder, not used in basic scalar
    """
    Basic measurement prediction h(x) for scalar magnetometer.
    itp_mapS: scalar map interpolation function f(lat_rad, lon_rad, alt_m)
    x_state_error: current state error vector
    Other args: true navigation parameters
    """
    # Extract error states
    dlat_err = x_state_error[0]
    dlon_err = x_state_error[1]
    dalt_err = x_state_error[2]
    # Other error states (velocity, attitude, biases) are not directly used in h_basic
    # for map value prediction, but are part of x_state_error.

    # Predicted location based on true + error
    lat_pred_rad = lat_true_rad + dlat_err
    lon_pred_rad = lon_true_rad + dlon_err
    alt_pred_m   = alt_true_m   + dalt_err

    # Interpolate map at predicted location
    map_val_pred = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m)
    
    h_val = map_val_pred

    if core:
        # Add core field at predicted location
        core_val_pred = _calculate_igrf_intensity_at_point(date, alt_pred_m, lat_pred_rad, lon_pred_rad)
        h_val += core_val_pred

    # Add FOGM catch-all bias if it's a state
    if fogm_state:
        nx = x_state_error.shape[0]
        s_err = x_state_error[nx-1] # FOGM error is the last state
        h_val += s_err
        
    # Tolles-Lawson and Vector magnetometer biases are not directly added in this basic scalar h(x)
    # Their effect would be through how they influence the state estimate x,
    # or if the measurement model was more complex (e.g. vector measurements).

    return h_val


def get_h_with_derivative(itp_mapS, der_mapS, x_state_error: np.ndarray,
                          lat_true_rad, lon_true_rad, alt_true_m,
                          date=get_years(2020, 185),
                          core: bool = False,
                          vec_states: bool = False, # Placeholder
                          fogm_state: bool = True,
                          TL_coeffs_len: int = 0, # Placeholder
                          map_alt: float = 0.0): # Altitude of the map if der_mapS is 2D
    """
    Measurement prediction h(x) using map and its vertical derivative.
    itp_mapS: scalar map interpolation function f(lat, lon, alt)
    der_mapS: scalar map vertical derivative interpolation function df/dalt(lat, lon) - assumed at map_alt
    x_state_error: current state error vector
    map_alt: altitude of the der_mapS plane [m]
    """
    # Extract error states
    dlat_err = x_state_error[0]
    dlon_err = x_state_error[1]
    dalt_err = x_state_error[2]

    # Predicted location
    lat_pred_rad = lat_true_rad + dlat_err
    lon_pred_rad = lon_true_rad + dlon_err
    alt_pred_m   = alt_true_m   + dalt_err

    # Map value at predicted aircraft altitude (alt_pred_m)
    # This uses the 3D interpolator itp_mapS
    map_val_at_ac_alt = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m)
    h_val = map_val_at_ac_alt

    # If der_mapS is provided, it's typically a 2D map of d(map)/d(alt) at a specific map altitude.
    # The term (alt_pred_m - map_alt) * der_mapS(...) is a first-order Taylor expansion
    # to adjust the map value from map_alt to alt_pred_m, if itp_mapS was only 2D.
    # However, if itp_mapS is already a 3D interpolator, this adjustment might be redundant
    # or represent a different physical model (e.g., accounting for drape).
    # The Julia code implies der_mapS is used when available.
    # Let's assume der_mapS is d(map)/dz at the map's reference altitude (map_alt).
    if der_mapS is not None:
        # Interpolate vertical derivative at predicted lat/lon, on the map's altitude plane
        map_deriv_val = der_mapS(lat_pred_rad, lon_pred_rad) # This is (dmap/dalt) at map_alt
        
        # This term seems to be an adjustment based on the difference between aircraft altitude
        # and the altitude of the derivative map.
        # If itp_mapS is a 3D interpolator, map_val_at_ac_alt is already at the correct altitude.
        # This suggests der_mapS might be used for a different purpose or under specific assumptions.
        # For now, translating the structure from potential Julia logic:
        # h_val = map_val_at_map_alt + (alt_pred_m - map_alt) * map_deriv_val
        # If itp_mapS is 3D, map_val_at_ac_alt is preferred.
        # If itp_mapS is 2D (at map_alt) and der_mapS is its derivative, then:
        # map_val_at_map_alt_plane = itp_mapS(lat_pred_rad, lon_pred_rad) # Assuming itp_mapS is 2D if der_mapS is used like this
        # h_val = map_val_at_map_alt_plane + (alt_pred_m - map_alt) * map_deriv_val
        # Given the problem description implies itp_mapS can be a 3D interpolator from MapCache,
        # the addition of (alt_pred_m - map_alt) * map_deriv_val needs careful interpretation.
        # If itp_mapS is already giving the value at alt_pred_m, this term might be an additional model component.
        # Let's assume the intention is to use the derivative for a correction or refinement if available.
        # This part is a bit ambiguous without more context on how der_mapS is generated and used.
        # A common use of vertical derivative is in the H matrix, not directly in h(x) like this
        # unless it's part of a specific measurement model (e.g., gradiometer or specific field model).
        # For now, let's assume it's an additive term if present, as implied by some EKF formulations
        # where h(x) might include such terms.
        # However, the most straightforward h(x) is just the map value at (lat_pred, lon_pred, alt_pred).
        # The Julia EKF_RT call passes der_mapS and map_alt to get_h.
        # Let's assume if der_mapS is present, it's used for a linear extrapolation from map_alt
        # and itp_mapS is then considered to be the map at map_alt.
        # This is a common simplification if only a 2D map and its vertical derivative are available.
        
        # Re-evaluating: if itp_mapS is a 3D interpolator, it already gives the value at alt_pred_m.
        # The der_mapS term is more likely used in H.
        # If itp_mapS is a MapCache, it will select the best map and interpolate.
        # The `get_h` in Julia's EKF.jl has:
        # `map_val = itp_mapS(lat,lon,alt) + (der_mapS === nothing ? 0.0 : (alt - map_alt)*der_mapS(lat,lon))`
        # This implies itp_mapS gives value at 'alt', and der_mapS provides an additional adjustment.
        # This is unusual if itp_mapS is a full 3D interpolator.
        # It makes more sense if itp_mapS was a 2D map at some reference altitude, and der_mapS corrects from that.
        # Given the Python structure, let's assume itp_mapS gives the value at the query altitude.
        # The der_mapS term in h(x) will be omitted for now as its role is unclear with a 3D itp_mapS.
        # It's primarily used in H. If it *must* be in h(x), the model needs clarification.
        # For now: h_val = map_val_at_ac_alt (from 3D itp_mapS)
        pass # Keeping h_val as map_val_at_ac_alt

    if core:
        core_val_pred = _calculate_igrf_intensity_at_point(date, alt_pred_m, lat_pred_rad, lon_pred_rad)
        h_val += core_val_pred

    if fogm_state:
        nx = x_state_error.shape[0]
        s_err = x_state_error[nx-1]
        h_val += s_err
        
    return h_val

def get_h(itp_mapS, x_state_error: np.ndarray, lat_true_rad, lon_true_rad, alt_true_m,
          date=get_years(2020, 185),
          core: bool = False,
          vec_states: bool = False,
          fogm_state: bool = True,
          TL_coeffs_len: int = 0,
          der_mapS=None, # Added der_mapS
          map_alt: float = 0.0): # Added map_alt
    """
    Unified measurement prediction function.
    Selects basic or derivative-based h(x) based on der_mapS.
    """
    if der_mapS is not None:
        # This implies a model where der_mapS is used to adjust a base map value.
        # The interpretation of how itp_mapS and der_mapS combine is crucial.
        # Following the Julia structure: h = map_at_alt + (alt_ac - alt_map)*derivative_at_map_alt
        # This assumes itp_mapS gives the map value at alt_true_m (aircraft altitude)
        # and then an additional term is added if der_mapS is available.
        # This is somewhat counter-intuitive if itp_mapS is a 3D interpolator.
        # A more common model: h = map_at_map_reference_alt + (alt_ac - alt_map_ref)*derivative_at_map_ref
        # Let's assume itp_mapS gives the value at the query point (lat_pred, lon_pred, alt_pred).
        # The Julia line: map_val = itp_mapS(lat,lon,alt) + (der_mapS === nothing ? 0.0 : (alt - map_alt)*der_mapS(lat,lon))
        # Here, (lat,lon,alt) are the *predicted* coordinates.
        
        dlat_err = x_state_error[0]
        dlon_err = x_state_error[1]
        dalt_err = x_state_error[2]
        lat_pred_rad = lat_true_rad + dlat_err
        lon_pred_rad = lon_true_rad + dlon_err
        alt_pred_m   = alt_true_m   + dalt_err

        h_val = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m) # Value from map at aircraft's predicted altitude
        
        # Add derivative term as per Julia's apparent model
        h_val += (alt_pred_m - map_alt) * der_mapS(lat_pred_rad, lon_pred_rad)

    else: # Basic h(x) if no derivative map
        h_val = get_h_basic(itp_mapS, x_state_error, lat_true_rad, lon_true_rad, alt_true_m,
                            date=date, core=core, vec_states=vec_states,
                            fogm_state=fogm_state, TL_coeffs_len=TL_coeffs_len)
    
    # Add core and FOGM bias contributions if they were not handled by the chosen h_function
    # This logic is a bit redundant if get_h_basic already adds them.
    # Refactoring to ensure core and fogm are added once.

    # The core and fogm_state additions are now inside get_h_basic and the if der_mapS block.
    # So, we just return h_val.
    # However, the Julia structure adds core and S_err *after* the map_val + derivative term.
    # Let's adjust to match that more closely.

    # Recalculate predicted coords for clarity if not done above
    dlat_err = x_state_error[0]
    dlon_err = x_state_error[1]
    dalt_err = x_state_error[2]
    lat_pred_rad = lat_true_rad + dlat_err
    lon_pred_rad = lon_true_rad + dlon_err
    alt_pred_m   = alt_true_m   + dalt_err

    if der_mapS is not None:
        # This assumes itp_mapS gives value at some reference plane if der_mapS is used for extrapolation
        # Or, if itp_mapS is 3D, it gives value at alt_pred_m.
        # Let's assume itp_mapS(lat,lon,alt) is the primary map value at aircraft altitude.
        map_component = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m)
        # The derivative term in Julia's get_h seems to be an *additional* component, not just for extrapolation
        # from a 2D map. This is unusual.
        # map_component += (alt_pred_m - map_alt) * der_mapS(lat_pred_rad, lon_pred_rad)
        # Sticking to the simpler: map value at predicted location.
        # The H matrix handles derivatives. If h(x) needs this term, the model is specific.
        # For now, let's assume h(x) is the field value at the point.
        # The Julia EKF.jl's `get_h` function has:
        # `map_val  = itp_mapS(lat,lon,alt) + (der_mapS === nothing ? 0.0 : (alt - map_alt)*der_mapS(lat,lon))`
        # This implies `itp_mapS` is value at `alt` (predicted aircraft alt)
        # and `der_mapS` is used for an additional term.
        # This is not a standard Taylor expansion for map value if `itp_mapS` is already 3D.
        # It might be a specific model feature.
        
        # Let's follow the Julia structure literally for now:
        h_map_term = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m)
        h_map_term += (alt_pred_m - map_alt) * der_mapS(lat_pred_rad, lon_pred_rad)

    else: # Basic h(x) if no derivative map
        # get_h_basic already calculates map_val at predicted location
        # and adds core and fogm if states are true.
        # To avoid double-adding core/fogm, call a simpler version or extract map part.
        
        # Simpler: just map value at predicted location
        h_map_term = itp_mapS(lat_pred_rad, lon_pred_rad, alt_pred_m)

    # Add core field contribution
    if core:
        h_map_term += _calculate_igrf_intensity_at_point(date, alt_pred_m, lat_pred_rad, lon_pred_rad)

    # Add FOGM catch-all bias if it's a state
    if fogm_state:
        nx_current = x_state_error.shape[0]
        s_err = x_state_error[nx_current-1] # FOGM error is the last state
        h_map_term += s_err
        
    return h_map_term


def fogm(sigma, tau, dt, N):
    """
    Simulate first-order Gauss-Markov (FOGM) process.
    """
    if tau == 0: # White noise
        return np.random.normal(0, sigma, N)
    
    x = np.zeros(N)
    x[0] = np.random.normal(0, sigma) # Initialize from stationary distribution
    
    # Parameters for FOGM
    alpha = math.exp(-dt / tau)
    # Variance of the driving white noise: sigma_w^2 = sigma^2 * (1 - alpha^2)
    # Or, if sigma is the std dev of the driving noise: sigma_drive = sigma * sqrt(dt)
    # The common FOGM model: x_k = alpha * x_{k-1} + w_k
    # where w_k ~ N(0, sigma_w^2) and sigma_w^2 = sigma_process^2 * (1 - alpha^2)
    # Here, 'sigma' is the stationary std dev of the FOGM process itself.
    sigma_drive_noise = sigma * math.sqrt(1 - alpha**2)

    for i in range(1, N):
        x[i] = alpha * x[i-1] + np.random.normal(0, sigma_drive_noise)
        
    return x

def chol(M: np.ndarray):
    """
    Cholesky decomposition. Input M must be positive definite.
    Returns upper triangular R such that M = R'*R (MATLAB convention)
    or lower triangular L such that M = L*L' (SciPy convention).
    This returns L (lower triangular) from SciPy.
    """
    try:
        # scipy.linalg.cholesky returns lower triangular L by default (L @ L.T = M)
        # For upper triangular R (R.T @ R = M), use cholesky(M).T if M is symmetric.
        # Or, if M is already upper triangular from a QR-like process, that's different.
        # MATLAB's chol(M) returns upper R. Python's np.linalg.cholesky returns lower L.
        # The Julia code uses `Matrix(cholesky(Hermitian(P)).U)` which means upper.
        # So, we need L.T from scipy's cholesky.
        L = scipy_cholesky(M, lower=True)
        return L # Returning L (lower) as per typical Python usage. If R (upper) is needed, use L.T
    except np.linalg.LinAlgError:
        # print("Warning: Cholesky decomposition failed. Matrix may not be positive definite.", file=sys.stderr)
        # Fallback: Add small identity matrix (jitter)
        # This is a common technique but should be used cautiously.
        jitter = 1e-9 * np.eye(M.shape[0])
        try:
            L = scipy_cholesky(M + jitter, lower=True)
            # print("Cholesky succeeded with jitter.", file=sys.stderr)
            return L
        except np.linalg.LinAlgError:
            # print("Error: Cholesky failed even with jitter. Returning empty array.", file=sys.stderr)
            return np.array([]) # Or raise error

# Placeholder for get_f, if it's a distinct function from get_pinson or get_Phi
# In many EKF contexts, f(x) is the state propagation, and F is its Jacobian.
# If get_f is needed, its signature and implementation depend on what it represents.
# For example, if it's the non-linear state update: x_k+1 = f(x_k, u_k)
# This is not explicitly used by the EKF equations if Phi is directly computed.
# It might be used in other filter types or for simulation.
def get_f(*args, **kwargs):
    """Placeholder for a potential state propagation function f(x)."""
    # print("Warning: get_f is a placeholder and not fully implemented.")
    # This would depend on the specific state model if it's different from
    # what's implied by get_Phi (linearized propagation).
    # If it's just identity (x_k+1 = Phi @ x_k for error states), then it might not be needed.
    pass