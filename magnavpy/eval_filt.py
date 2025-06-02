"""
Module for evaluating navigation filter performance.
Ported from MagNav.jl/src/eval_filt.jl
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict, Callable # Added Dict, Callable

# Assuming these types are defined elsewhere, e.g., in common_types.py or magnav.py
# from .common_types import Traj, INS, MagV, FILTres # Example
# from .nav_utils import dlat2dn, dlon2de # Example
# from .ekf import ekf_py, ekf_online_py, ekf_online_nn_py # Example placeholders for filters
# from .crlb import crlb_py # Example placeholder for crlb

# Placeholder for actual type definitions until they are confirmed
@dataclass
class Traj:
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    Cnb: np.ndarray # Shape (N, 3, 3) or (3, 3, N) - assuming (N,3,3) for consistency
    tt: np.ndarray
    dt: float
    N: int

@dataclass
class INS:
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    Cnb: np.ndarray # Shape (N, 3, 3) or (3, 3, N)
    P: Optional[np.ndarray] = None # Covariance matrix (num_states, num_states, N)
    # Add other fields as necessary, e.g., time 'tt'

@dataclass
class MagV: # Placeholder
    fx: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    fy: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    fz: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    t: np.ndarray = field(default_factory=lambda: np.array([0.0]))

@dataclass
class FILTres: # Filter results (input to eval_filt_py)
    x: np.ndarray # State corrections (num_states, N)
    P: np.ndarray # Covariance matrix (num_states, num_states, N)
    # Add other fields if the filter returns more, e.g., filter_type

# Placeholder for utility functions (normally imported)
def dlat2dn(dlat: np.ndarray, lat: np.ndarray) -> np.ndarray:
    R0 = 6378137.0  # WGS84 Equatorial radius
    return dlat * (R0 * (1 - np.sin(lat)**2 * 0.00669437999014)) # Simplified, use proper nav_utils

def dlon2de(dlon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    R0 = 6378137.0  # WGS84 Equatorial radius
    return dlon * (R0 * (1 - np.sin(lat)**2 * 0.00669437999014) * np.cos(lat)) # Simplified

def get_years(year: int, day_of_year: int) -> float: # Placeholder
    return year + (day_of_year -1) / 365.25

# --- Output Dataclasses ---
@dataclass
class CRLBout:
    lat_std: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_std: np.ndarray = field(default_factory=lambda: np.array([]))
    alt_std: np.ndarray = field(default_factory=lambda: np.array([]))
    vn_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ve_std: np.ndarray = field(default_factory=lambda: np.array([]))
    vd_std: np.ndarray = field(default_factory=lambda: np.array([]))
    tn_std: np.ndarray = field(default_factory=lambda: np.array([]))
    te_std: np.ndarray = field(default_factory=lambda: np.array([]))
    td_std: np.ndarray = field(default_factory=lambda: np.array([]))
    fogm_std: np.ndarray = field(default_factory=lambda: np.array([]))
    n_std: np.ndarray = field(default_factory=lambda: np.array([]))
    e_std: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class INSout:
    lat_std: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_std: np.ndarray = field(default_factory=lambda: np.array([]))
    alt_std: np.ndarray = field(default_factory=lambda: np.array([]))
    n_std: np.ndarray = field(default_factory=lambda: np.array([]))
    e_std: np.ndarray = field(default_factory=lambda: np.array([]))
    lat_err: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_err: np.ndarray = field(default_factory=lambda: np.array([]))
    alt_err: np.ndarray = field(default_factory=lambda: np.array([]))
    n_err: np.ndarray = field(default_factory=lambda: np.array([]))
    e_err: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class FILTout:
    N: int
    dt: float
    tt: np.ndarray = field(default_factory=lambda: np.array([]))
    lat: np.ndarray = field(default_factory=lambda: np.array([]))
    lon: np.ndarray = field(default_factory=lambda: np.array([]))
    alt: np.ndarray = field(default_factory=lambda: np.array([]))
    vn: np.ndarray = field(default_factory=lambda: np.array([]))
    ve: np.ndarray = field(default_factory=lambda: np.array([]))
    vd: np.ndarray = field(default_factory=lambda: np.array([]))
    tn: np.ndarray = field(default_factory=lambda: np.array([]))
    te: np.ndarray = field(default_factory=lambda: np.array([]))
    td: np.ndarray = field(default_factory=lambda: np.array([]))
    ha: np.ndarray = field(default_factory=lambda: np.array([])) # baro_bias
    ah: np.ndarray = field(default_factory=lambda: np.array([])) # alt_bias (was δhb)
    ax: np.ndarray = field(default_factory=lambda: np.array([])) # acc_bias_x
    ay: np.ndarray = field(default_factory=lambda: np.array([])) # acc_bias_y
    az: np.ndarray = field(default_factory=lambda: np.array([])) # acc_bias_z
    gx: np.ndarray = field(default_factory=lambda: np.array([])) # gyro_bias_x
    gy: np.ndarray = field(default_factory=lambda: np.array([])) # gyro_bias_y
    gz: np.ndarray = field(default_factory=lambda: np.array([])) # gyro_bias_z
    fogm: np.ndarray = field(default_factory=lambda: np.array([])) # fogm_bias

    lat_std: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_std: np.ndarray = field(default_factory=lambda: np.array([]))
    alt_std: np.ndarray = field(default_factory=lambda: np.array([]))
    vn_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ve_std: np.ndarray = field(default_factory=lambda: np.array([]))
    vd_std: np.ndarray = field(default_factory=lambda: np.array([]))
    tn_std: np.ndarray = field(default_factory=lambda: np.array([]))
    te_std: np.ndarray = field(default_factory=lambda: np.array([]))
    td_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ha_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ah_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ax_std: np.ndarray = field(default_factory=lambda: np.array([]))
    ay_std: np.ndarray = field(default_factory=lambda: np.array([]))
    az_std: np.ndarray = field(default_factory=lambda: np.array([]))
    gx_std: np.ndarray = field(default_factory=lambda: np.array([]))
    gy_std: np.ndarray = field(default_factory=lambda: np.array([]))
    gz_std: np.ndarray = field(default_factory=lambda: np.array([]))
    fogm_std: np.ndarray = field(default_factory=lambda: np.array([]))
    n_std: np.ndarray = field(default_factory=lambda: np.array([]))
    e_std: np.ndarray = field(default_factory=lambda: np.array([]))

    lat_err: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_err: np.ndarray = field(default_factory=lambda: np.array([]))
    alt_err: np.ndarray = field(default_factory=lambda: np.array([]))
    vn_err: np.ndarray = field(default_factory=lambda: np.array([]))
    ve_err: np.ndarray = field(default_factory=lambda: np.array([]))
    vd_err: np.ndarray = field(default_factory=lambda: np.array([]))
    tn_err: np.ndarray = field(default_factory=lambda: np.array([]))
    te_err: np.ndarray = field(default_factory=lambda: np.array([]))
    td_err: np.ndarray = field(default_factory=lambda: np.array([]))
    n_err: np.ndarray = field(default_factory=lambda: np.array([]))
    e_err: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        # Initialize arrays if N is known, otherwise they remain empty
        if self.N > 0:
            array_fields = [
                f.name for f in self.__dataclass_fields__.values()
                if f.default_factory is not list and isinstance(getattr(self, f.name), np.ndarray)
            ]
            for fname in array_fields:
                if fname not in ['N', 'dt']: # N, dt are not arrays to be sized
                    # Check if field already has data (e.g. tt from traj)
                    if getattr(self, fname).size == 0 :
                        setattr(self, fname, np.zeros(self.N))


def eval_crlb_py(traj: Traj, crlb_P: np.ndarray) -> CRLBout:
    """
    Extract Cramér–Rao lower bound (CRLB) results.
    """
    N = traj.N
    crlb_out = CRLBout()

    # Assuming crlb_P is (num_states, num_states, N)
    # State indices: 0:lat, 1:lon, 2:alt, 3:vn, 4:ve, 5:vd, 6:tn, 7:te, 8:td ... 17:fogm
    crlb_out.lat_std  = np.sqrt(crlb_P[0,0,:])
    crlb_out.lon_std  = np.sqrt(crlb_P[1,1,:])
    crlb_out.alt_std  = np.sqrt(crlb_P[2,2,:])
    crlb_out.vn_std   = np.sqrt(crlb_P[3,3,:])
    crlb_out.ve_std   = np.sqrt(crlb_P[4,4,:])
    crlb_out.vd_std   = np.sqrt(crlb_P[5,5,:])
    crlb_out.tn_std   = np.sqrt(crlb_P[6,6,:])
    crlb_out.te_std   = np.sqrt(crlb_P[7,7,:])
    crlb_out.td_std   = np.sqrt(crlb_P[8,8,:])
    # Assuming 18 states total, fogm is state 17 (0-indexed)
    if crlb_P.shape[0] > 17:
         crlb_out.fogm_std = np.sqrt(crlb_P[17,17,:])
    else: # Handle cases with fewer states if necessary
        crlb_out.fogm_std = np.full(N, np.nan)


    crlb_out.n_std = dlat2dn(crlb_out.lat_std, traj.lat)
    crlb_out.e_std = dlon2de(crlb_out.lon_std, traj.lat)

    crlb_DRMS = int(round(np.sqrt(np.mean(crlb_out.n_std**2 + crlb_out.e_std**2))))
    print(f"CRLB DRMS error = {crlb_DRMS} m")

    return crlb_out

def eval_ins_py(traj: Traj, ins: INS) -> INSout:
    """
    Extract INS results and compare with trajectory.
    """
    N = traj.N
    ins_out = INSout()

    # Initialize arrays
    ins_out.lat_std = np.full(N, np.nan)
    ins_out.lon_std = np.full(N, np.nan)
    ins_out.alt_std = np.full(N, np.nan)
    ins_out.n_std = np.full(N, np.nan)
    ins_out.e_std = np.full(N, np.nan)

    if ins.P is not None and ins.P.size > 0:
        ins_out.lat_std = np.sqrt(ins.P[0,0,:])
        ins_out.lon_std = np.sqrt(ins.P[1,1,:])
        ins_out.alt_std = np.sqrt(ins.P[2,2,:])
        ins_out.n_std   = dlat2dn(ins_out.lat_std, ins.lat)
        ins_out.e_std   = dlon2de(ins_out.lon_std, ins.lat)

    ins_out.lat_err = ins.lat - traj.lat
    ins_out.lon_err = ins.lon - traj.lon
    ins_out.alt_err = ins.alt - traj.alt
    ins_out.n_err   = dlat2dn(ins_out.lat_err, ins.lat)
    ins_out.e_err   = dlon2de(ins_out.lon_err, ins.lat)

    ins_DRMS = int(round(np.sqrt(np.mean(ins_out.n_err**2 + ins_out.e_err**2))))
    print(f"INS  DRMS error = {ins_DRMS} m")

    return ins_out

def eval_filt_py(traj: Traj, ins: INS, filt_res: FILTres) -> FILTout:
    """
    Extract filter results, combine with INS, and compare with trajectory.
    """
    N  = traj.N
    dt = traj.dt
    filt_out = FILTout(N=N, dt=dt) # Initializes arrays via __post_init__

    filt_out.tt = traj.tt

    # Corrected states (INS + filter estimate)
    # filt_res.x is (num_states, N)
    filt_out.lat = ins.lat + filt_res.x[0,:]
    filt_out.lon = ins.lon + filt_res.x[1,:]
    filt_out.alt = ins.alt + filt_res.x[2,:]
    filt_out.vn  = ins.vn  + filt_res.x[3,:]
    filt_out.ve  = ins.ve  + filt_res.x[4,:]
    filt_out.vd  = ins.vd  + filt_res.x[5,:]
    filt_out.tn  =           filt_res.x[6,:]
    filt_out.te  =           filt_res.x[7,:]
    filt_out.td  =           filt_res.x[8,:]
    filt_out.ha  =           filt_res.x[9,:]  # baro_bias
    filt_out.ah  =           filt_res.x[10,:] # alt_bias (δhb)
    filt_out.ax  =           filt_res.x[11,:] # acc_bias_x
    filt_out.ay  =           filt_res.x[12,:] # acc_bias_y
    filt_out.az  =           filt_res.x[13,:] # acc_bias_z
    filt_out.gx  =           filt_res.x[14,:] # gyro_bias_x
    filt_out.gy  =           filt_res.x[15,:] # gyro_bias_y
    filt_out.gz  =           filt_res.x[16,:] # gyro_bias_z
    if filt_res.x.shape[0] > 17: # Check if FOGM state exists
        filt_out.fogm = filt_res.x[17,:] # fogm_bias
    else:
        filt_out.fogm = np.zeros(N)


    # Standard deviations from filter covariance P
    # filt_res.P is (num_states, num_states, N)
    filt_out.lat_std  = np.sqrt(filt_res.P[0,0,:])
    filt_out.lon_std  = np.sqrt(filt_res.P[1,1,:])
    filt_out.alt_std  = np.sqrt(filt_res.P[2,2,:])
    filt_out.vn_std   = np.sqrt(filt_res.P[3,3,:])
    filt_out.ve_std   = np.sqrt(filt_res.P[4,4,:])
    filt_out.vd_std   = np.sqrt(filt_res.P[5,5,:])
    filt_out.tn_std   = np.sqrt(filt_res.P[6,6,:])
    filt_out.te_std   = np.sqrt(filt_res.P[7,7,:])
    filt_out.td_std   = np.sqrt(filt_res.P[8,8,:])
    filt_out.ha_std   = np.sqrt(filt_res.P[9,9,:])
    filt_out.ah_std   = np.sqrt(filt_res.P[10,10,:])
    filt_out.ax_std   = np.sqrt(filt_res.P[11,11,:])
    filt_out.ay_std   = np.sqrt(filt_res.P[12,12,:])
    filt_out.az_std   = np.sqrt(filt_res.P[13,13,:])
    filt_out.gx_std   = np.sqrt(filt_res.P[14,14,:])
    filt_out.gy_std   = np.sqrt(filt_res.P[15,15,:])
    filt_out.gz_std   = np.sqrt(filt_res.P[16,16,:])
    if filt_res.P.shape[0] > 17:
        filt_out.fogm_std = np.sqrt(filt_res.P[17,17,:])
    else:
        filt_out.fogm_std = np.zeros(N)


    filt_out.n_std = dlat2dn(filt_out.lat_std, filt_out.lat)
    filt_out.e_std = dlon2de(filt_out.lon_std, filt_out.lat)

    # Errors (Filter - Trajectory)
    filt_out.lat_err = filt_out.lat - traj.lat
    filt_out.lon_err = filt_out.lon - traj.lon
    filt_out.alt_err = filt_out.alt - traj.alt
    filt_out.vn_err  = filt_out.vn  - traj.vn
    filt_out.ve_err  = filt_out.ve  - traj.ve
    filt_out.vd_err  = filt_out.vd  - traj.vd

    # Attitude errors
    # ins.Cnb and traj.Cnb are assumed (N, 3, 3)
    n_tilt_truth = np.zeros(N)
    e_tilt_truth = np.zeros(N)
    d_tilt_truth = np.zeros(N)

    for k in range(N):
        # Error DCM: C_filter_to_truth = C_body_to_filter @ C_truth_to_body
        # C_body_to_filter is approximated by ins.Cnb (or could be derived from filter attitude)
        # Here, we use ins.Cnb as the reference for filter's body frame,
        # and traj.Cnb as the truth for body frame.
        # The filter's attitude tn, te, td are corrections to the INS attitude.
        # So, the "true" tilt errors are against the INS's error w.r.t trajectory.
        # tilt_temp = ins.Cnb[k] @ traj.Cnb[k].T # This is C_nav_ins w.r.t nav_truth
        
        # The Julia code calculates tilt error of INS relative to Trajectory
        # Cnb_ins = ins.Cnb[k,:,:]
        # Cnb_traj = traj.Cnb[k,:,:]
        # tilt_temp_dcm = Cnb_ins @ Cnb_traj.T
        
        # Let's assume ins.Cnb is the Cnb used by the filter as its reference.
        # The filter estimates tn, te, td as the misalignment of this reference.
        # The "truth" for these tn, te, td values is the actual misalignment of ins.Cnb wrt traj.Cnb
        
        # If ins.Cnb is (N,3,3)
        Cnb_ins_k = ins.Cnb[k]
        Cnb_traj_k = traj.Cnb[k]
        
        # If ins.Cnb is (3,3,N)
        # Cnb_ins_k = ins.Cnb[:,:,k]
        # Cnb_traj_k = traj.Cnb[:,:,k]

        err_dcm = Cnb_ins_k @ Cnb_traj_k.T # Error DCM of INS relative to Trajectory
        n_tilt_truth[k] = err_dcm[2,1] # tilt_temp[2,1] in 0-indexed for Julia's [3,2]
        e_tilt_truth[k] = err_dcm[0,2] # tilt_temp[0,2] in 0-indexed for Julia's [1,3]
        d_tilt_truth[k] = err_dcm[1,0] # tilt_temp[1,0] in 0-indexed for Julia's [2,1]

    filt_out.tn_err = filt_out.tn - n_tilt_truth
    filt_out.te_err = filt_out.te - e_tilt_truth
    filt_out.td_err = filt_out.td - d_tilt_truth

    filt_out.n_err = dlat2dn(filt_out.lat_err, filt_out.lat)
    filt_out.e_err = dlon2de(filt_out.lon_err, filt_out.lat)

    filt_DRMS = int(round(np.sqrt(np.mean(filt_out.n_err**2 + filt_out.e_err**2))))
    print(f"FILT DRMS error = {filt_DRMS} m")

    return filt_out

def eval_results_py(traj: Traj, ins: INS, filt_res: FILTres, crlb_P: np.ndarray) -> Tuple[CRLBout, INSout, FILTout]:
    """
    Extract CRLB, INS, & filter results.
    """
    crlb_out = eval_crlb_py(traj, crlb_P)
    ins_out  = eval_ins_py(traj, ins)
    filt_out = eval_filt_py(traj, ins, filt_res)
    return crlb_out, ins_out, filt_out


# --- Placeholder filter/CRLB functions (to be imported from actual modules) ---
def placeholder_ekf_py(*args, **kwargs) -> FILTres:
    print("Warning: Using placeholder EKF function.")
    # Determine N from ins or traj if possible
    ins = kwargs.get('ins', args[0] if args else None)
    N_val = ins.N if hasattr(ins, 'N') else 100 # Default N
    num_states = 18 # Default number of states
    return FILTres(x=np.random.randn(num_states, N_val) * 0.01,
                   P=np.array([np.eye(num_states) * 0.1 for _ in range(N_val)]).transpose(1,2,0))

def placeholder_crlb_py(*args, **kwargs) -> np.ndarray:
    print("Warning: Using placeholder CRLB function.")
    ins = kwargs.get('ins', args[0] if args else None)
    N_val = ins.N if hasattr(ins, 'N') else 100
    num_states = 18
    return np.array([np.eye(num_states) * 0.05 for _ in range(N_val)]).transpose(1,2,0)

# --- Main `run_filt` ---
# Map filter type symbols/strings to actual filter functions
FILTER_DISPATCHER: Dict[str, Callable[..., FILTres]] = {
    "ekf": placeholder_ekf_py,
    # "ekf_online": ekf_online_py, # Add actual functions when available
    # "ekf_online_nn": ekf_online_nn_py,
    # "mpf": mpf_py,
    # "nekf": nekf_py,
}

CRLB_FUNCTION: Callable[..., np.ndarray] = placeholder_crlb_py


def run_filt_py(
    traj: Traj,
    ins: INS,
    meas: np.ndarray, # Scalar magnetometer measurements
    itp_mapS: Callable, # Map interpolation function f(lat,lon) or f(lat,lon,alt)
    filt_type: str = "ekf",
    P0: Optional[np.ndarray] = None, # create_P0_py()
    Qd: Optional[np.ndarray] = None, # create_Qd_py()
    R: float = 1.0,
    num_part: int = 1000, # For MPF
    thresh: float = 0.8,  # For MPF
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0,
    date: float = get_years(2020,185), # Decimal year
    core: bool = False, # Include core field in measurement
    map_alt: float = 0.0,
    x_nn: Optional[np.ndarray] = None, # For NN models
    m_nn: Optional[Any] = None,        # NN model itself
    y_norms_nn: Optional[Tuple[float,float]] = None, # NN y_bias, y_scale
    terms_tl: List[str] = ['permanent','induced','eddy','bias'], # Tolles-Lawson
    flux_tl: Optional[MagV] = None, # Vector mag data for online TL
    x0_tl: Optional[np.ndarray] = None, # Initial TL states
    extract: bool = True,
    run_crlb: bool = True
) -> Any: # Returns can be (CRLBout, INSout, FILTout), or FILTout, or (FILTres, crlb_P), or FILTres
    """
    Run navigation filter and optionally compute Cramér–Rao lower bound (CRLB).
    """
    if flux_tl is None:
        flux_tl = MagV()

    # Initialize P0, Qd, x0_tl if not provided (using placeholders for create_P0/Qd)
    # num_states = 18 # Common number of states
    # if P0 is None: P0 = np.eye(num_states) * 0.1 # Placeholder create_P0()
    # if Qd is None: Qd = np.eye(num_states) * 1e-4 # Placeholder create_Qd()
    # if x0_tl is None: x0_tl = np.ones(19) # Default for Tolles-Lawson

    filter_func = FILTER_DISPATCHER.get(filt_type)
    if not filter_func:
        raise ValueError(f"Filter type '{filt_type}' not defined in FILTER_DISPATCHER.")

    # Prepare arguments for the specific filter function
    # This is a simplified example; actual filter functions will have specific signatures
    filter_args = {
        "ins": ins, "meas": meas, "itp_mapS": itp_mapS,
        "P0": P0, "Qd": Qd, "R": R,
        "baro_tau": baro_tau, "acc_tau": acc_tau, "gyro_tau": gyro_tau, "fogm_tau": fogm_tau,
        "date": date, "core": core, "map_alt": map_alt
    }
    if filt_type == "mpf":
        filter_args.update({"num_part": num_part, "thresh": thresh})
    # Add args for ekf_online, ekf_online_nn, nekf as needed, e.g.:
    # if filt_type == "ekf_online":
    #     filter_args.update({"flux": flux_tl, "x0_TL": x0_tl, "terms": terms_tl})
    # if "nn" in filt_type or filt_type == "nekf":
    #     filter_args.update({"x_nn": x_nn, "m": m_nn, "y_norms": y_norms_nn})

    filt_res = filter_func(**filter_args)

    if run_crlb:
        # Prepare args for CRLB function
        crlb_args = {
            "traj": traj, "itp_mapS": itp_mapS, # Assuming crlb_py needs traj
            "P0": P0, "Qd": Qd, "R": R,
            "baro_tau": baro_tau, "acc_tau": acc_tau, "gyro_tau": gyro_tau, "fogm_tau": fogm_tau,
            "date": date, "core": core
        }
        crlb_P_matrix = CRLB_FUNCTION(**crlb_args)
        if extract:
            return eval_results_py(traj, ins, filt_res, crlb_P_matrix)
        else:
            return filt_res, crlb_P_matrix
    else:
        if extract:
            # eval_filt_py is the one that returns FILTout
            # The Julia version calls eval_filt(traj,ins,filt_res) which returns FILTout
            return eval_filt_py(traj, ins, filt_res)
        else:
            return filt_res


def run_filt_py_multiple(
    traj: Traj,
    ins: INS,
    meas: np.ndarray,
    itp_mapS: Callable,
    filt_types: List[str], # e.g., ["ekf", "ekf_online_nn"]
    **kwargs # Pass through all other keyword args from the single run_filt_py
) -> None:
    """
    Run multiple filter models and print their DRMS errors (nothing returned).
    """
    # Default extract to True and run_crlb to False for this multi-run scenario
    # as per Julia version's behavior (prints DRMS, implies extraction, no CRLB typically)
    kwargs.setdefault('extract', True)
    kwargs.setdefault('run_crlb', False)

    for filt_type_str in filt_types:
        print(f"--- Running {filt_type_str} filter ---")
        # The result of run_filt_py will be FILTout if extract=True, run_crlb=False
        # The DRMS is printed within eval_filt_py (called by run_filt_py)
        _ = run_filt_py(
            traj=traj, ins=ins, meas=meas, itp_mapS=itp_mapS,
            filt_type=filt_type_str,
            **kwargs
        )
        print(f"--- Completed {filt_type_str} filter ---")

if __name__ == '__main__':
    # Example Usage (requires actual Traj, INS, meas, itp_mapS, and filter/CRLB implementations)
    print("eval_filt.py example usage (requires full setup)")

    # Dummy data for demonstration if run directly
    _N = 100
    _dt = 0.1
    _tt = np.arange(_N) * _dt
    
    _traj = Traj(lat=np.random.randn(_N), lon=np.random.randn(_N), alt=np.random.randn(_N),
                 vn=np.random.randn(_N), ve=np.random.randn(_N), vd=np.random.randn(_N),
                 Cnb=np.array([np.eye(3) for _ in range(_N)]), tt=_tt, dt=_dt, N=_N)
    _ins = INS(lat=_traj.lat + np.random.randn(_N)*0.1, lon=_traj.lon + np.random.randn(_N)*0.1,
               alt=_traj.alt + np.random.randn(_N)*0.1, vn=_traj.vn, ve=_traj.ve, vd=_traj.vd,
               Cnb=np.array([np.eye(3) for _ in range(_N)]), P=np.array([np.eye(3)*0.01 for _ in range(_N)]).transpose(1,2,0) ) # Simplified P for pos only
    _meas = np.random.randn(_N) * 50000
    def _itp_mapS_dummy(lat, lon, alt=None): return 50000 + np.random.randn()

    # Test eval_filt_py directly
    _num_states = 18
    _filt_res_dummy = FILTres(
        x = np.random.randn(_num_states, _N) * 0.01,
        P = np.array([np.eye(_num_states) * 0.01 for _ in range(_N)]).transpose(1,2,0) # (18,18,N)
    )
    print("\nTesting eval_filt_py:")
    filt_out_obj = eval_filt_py(_traj, _ins, _filt_res_dummy)
    # print(f"FILTout lat error mean: {np.mean(filt_out_obj.lat_err)}")

    # Test run_filt_py (using placeholder filter)
    print("\nTesting run_filt_py (with placeholder EKF and CRLB):")
    # Need to provide P0, Qd for placeholder_crlb_py and placeholder_ekf_py if they use it
    # For simplicity, they currently don't strictly require them in the placeholder logic
    # but a real implementation would.
    _P0_dummy = np.eye(_num_states) * 0.1
    _Qd_dummy = np.eye(_num_states) * 1e-4

    results = run_filt_py(
        traj=_traj, ins=_ins, meas=_meas, itp_mapS=_itp_mapS_dummy,
        filt_type="ekf", P0=_P0_dummy, Qd=_Qd_dummy,
        extract=True, run_crlb=True
    )
    if results and len(results) == 3:
        crlb_o, ins_o, filt_o = results
        print(f"run_filt_py returned CRLB DRMS (from print): ...") # DRMS printed inside
