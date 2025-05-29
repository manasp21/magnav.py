# -*- coding: utf-8 -*-
"""
Main module for MagNavPy, translated from MagNav.jl.

This module provides core data structures, constants, utility functions,
and access to datasets for magnetic navigation.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Any, Union # Any for some complex types initially
from .common_types import (
    Map, MapS, MapSd, MapS3D, MapV, MapCache, MAP_S_NULL, MagV
)
from abc import ABC # This is only needed if Path or XYZ are defined here and extend ABC

# Attempt to import packages, with placeholders for those not in requirements.txt
import numpy as np
import pandas as pd
import h5py
import toml # Standard in Python 3.11+, or pip install toml
import zipfile

# SciPy imports
from scipy import signal
from scipy.linalg import expm
from scipy.stats import multivariate_normal, norm, uniform
from scipy.interpolate import UnivariateSpline, interp1d, RegularGridInterpolator, interpn
from scipy.io import loadmat, savemat
from scipy.optimize import minimize as scipy_minimize # Renamed to avoid conflict if we define minimize
from scipy.special import gamma, gammainc, gammaincinv
from .tolles_lawson import create_TL_A
from .dcm_util import dcm2euler, euler2dcm # New import for dcm functions
from .fdm_util import fdm # New import for fdm function

# Matplotlib
import matplotlib.pyplot as plt

# PyTorch imports (Flux equivalent)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Scikit-learn imports (for GLMNet, MLJLinearModels, KernelFunctions, NearestNeighbors)
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.neighbors import KDTree as SklearnKDTree, NearestNeighbors as SklearnNearestNeighbors
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

# Statsmodels imports (for StatsBase)
from statsmodels.tsa.stattools import acf as sm_acf # Renamed

# GDAL import
try:
    from osgeo import gdal
except ImportError:
    gdal = None # Placeholder
    print("Warning: GDAL not found. Some functionalities might be unavailable.", file=sys.stderr)

# JAX imports (for ForwardDiff, Zygote)
try:
    import jax
    import jax.numpy as jnp
    # For JAX, enable float64 if needed, as it defaults to float32
    # from jax.config import config
    # config.update("jax_enable_x64", True)
except ImportError:
    jax = None
    jnp = None # Placeholder
    print("Warning: JAX not found. Autodiff functionalities might be unavailable.", file=sys.stderr)


# --- Helper Functions for path manipulation (from Julia's internal logic) ---
def add_extension(name: str, ext: str) -> str:
    """Adds extension if not already present."""
    if not name.endswith(ext):
        return name + ext
    return name

def remove_extension(name: str, suffix_to_remove: str) -> str:
    """Removes specified suffix if present."""
    if name.endswith(suffix_to_remove):
        return name[:-len(suffix_to_remove)]
    return name

# --- Project Version ---
# Path to the original Julia Project.toml to read version
# This assumes MagNavPy/src/magnav.py and MagNav.jl/Project.toml relative paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_TOML_PATH = os.path.join(_SCRIPT_DIR, "..", "..", "MagNav.jl", "Project.toml")

MAGNAV_VERSION_STR = "0.0.0" # Default
try:
    with open(_PROJECT_TOML_PATH, 'r') as f:
        _project_data = toml.load(f)
    MAGNAV_VERSION_STR = _project_data.get("version", "0.0.0")
except FileNotFoundError:
    print(f"Warning: Project.toml not found at {_PROJECT_TOML_PATH}. Using default version.", file=sys.stderr)
except Exception as e:
    print(f"Warning: Could not read version from Project.toml: {e}", file=sys.stderr)

# --- Constants ---
NUM_MAG_MAX = 6  # Maximum number of scalar & vector magnetometers (each)
E_EARTH = 0.0818191908426  # First eccentricity of Earth [-]
G_EARTH = 9.80665  # Gravity of Earth [m/s^2]
R_EARTH = 6378137  # WGS-84 radius of Earth [m]
OMEGA_EARTH = 7.2921151467e-5  # Rotation rate of Earth [rad/s]

# Artifacts: Paths to data files.
# Assumption: An ARTIFACT_DIR environment variable or a known relative path.
# For now, using a placeholder. Users should configure this.
# Example: ARTIFACT_DIR = os.getenv("MAGNAV_ARTIFACT_DIR", "artifacts")
# If ARTIFACT_DIR is relative, it's relative to where the script is run or a known base.
# For simplicity in this conversion, we'll use relative paths from a hypothetical "artifacts" dir
# that would be a sibling to MagNav.jl and MagNavPy or configured.
_ARTIFACT_BASE_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "MagNav.jl", "test", "test_data") # Fallback to test_data for some, needs proper setup
_DEFAULT_ARTIFACT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "artifacts_data") # A more general placeholder

# It's better to define a function or a configuration system for artifacts.
# For this translation, paths will be constructed assuming _DEFAULT_ARTIFACT_DIR.
# Users will need to ensure these files exist at the expected locations.
print(f"Note: Artifact paths are placeholders. Ensure data exists at expected locations relative to '{_DEFAULT_ARTIFACT_DIR}' or configure properly.", file=sys.stderr)

USGS_COLOR_SCALE = os.path.join(_DEFAULT_ARTIFACT_DIR, "util_files", "util_files", "color_scale_usgs.csv")
ICON_CIRCLE = os.path.join(_DEFAULT_ARTIFACT_DIR, "util_files", "util_files", "icon_circle.dae")

SILENT_DEBUG = True # Internal flag. If true, no verbose print outs.

def sgl_fields_path(f: str = "") -> str:
    """
    Data fields in SGL flight data collections.
    Args:
        f: (optional) name of data file (.csv extension optional)
    Returns:
        path of folder or f data file
    """
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_fields", "sgl_fields")
    if f:
        return os.path.join(p_base, add_extension(str(f), ".csv"))
    return p_base

def sgl_2020_train_path(f: str = "") -> str:
    """Flight data from the 2020 SGL flight data collection - training portion."""
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_2020_train", "sgl_2020_train")
    if f:
        d = remove_extension(str(f), "_train")
        return os.path.join(p_base, add_extension(d, "_train.h5"))
    return p_base

def sgl_2021_train_path(f: str = "") -> str:
    """Flight data from the 2021 SGL flight data collection - training portion."""
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_2021_train", "sgl_2021_train")
    if f:
        d = remove_extension(str(f), "_train")
        return os.path.join(p_base, add_extension(d, "_train.h5"))
    return p_base

EMAG2_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "EMAG2", "EMAG2.h5")
EMM720_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "EMM720_World", "EMM720_World.h5")
NAMAD_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "NAMAD_305", "NAMAD_305.h5") # Used as default fallback map

def ottawa_area_maps_path(f: str = "") -> str:
    """Magnetic anomaly maps near Ottawa, Ontario, Canada."""
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "ottawa_area_maps", "ottawa_area_maps")
    if f:
        return os.path.join(p_base, add_extension(str(f), ".h5"))
    return p_base

def ottawa_area_maps_gxf_path(f: str = "") -> str:
    """GXF versions of small magnetic anomaly maps near Ottawa, Ontario, Canada."""
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "ottawa_area_maps_gxf", "ottawa_area_maps_gxf")
    if f:
        d = remove_extension(str(f), "_Mag")
        return os.path.join(p_base, add_extension(d, "_Mag.gxf"))
    return p_base

# --- Data Structures (Structs to Dataclasses) --- (Map related classes moved to common_types.py)

class Path(ABC):
    """Abstract type for a flight path."""
    N: int          # number of samples
    dt: float       # measurement time step [s]
    tt: np.ndarray  # time [s]
    lat: np.ndarray # latitude [rad]
    lon: np.ndarray # longitude [rad]
    alt: np.ndarray # altitude [m]
    vn: np.ndarray  # north velocity [m/s]
    ve: np.ndarray  # east velocity [m/s]
    vd: np.ndarray  # down velocity [m/s]
    
@dataclass
class Traj(Path):
    """Trajectory struct (e.g., GPS or truth flight data)."""
    N: int
    dt: float
    tt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    fn: np.ndarray  # north specific force [m/s] - Note: Julia uses [m/s], often specific force is [m/s^2]
    fe: np.ndarray  # east specific force [m/s]
    fd: np.ndarray  # down specific force [m/s]
    Cnb: np.ndarray # 3x3xN direction cosine matrix (body to navigation)

@dataclass
class INS(Path):
    """Inertial navigation system (INS) struct."""
    N: int
    dt: float
    tt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    fn: np.ndarray
    fe: np.ndarray
    fd: np.ndarray
    Cnb: np.ndarray
    P: np.ndarray   # 17x17xN covariance matrix (or appropriate size)

class XYZ(ABC):
    """Abstract type for flight data."""
    info: str
    traj: Traj
    ins: INS
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray

@dataclass
class XYZ0(XYZ):
    """Minimum dataset required for MagNav."""
    info: str
    traj: Traj
    ins: INS
    flux_a: MagV
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    mag_1_c: np.ndarray  # Mag 1 compensated scalar measurements [nT]
    mag_1_uc: np.ndarray # Mag 1 uncompensated scalar measurements [nT]

@dataclass
class XYZ1(XYZ):
    """Flexible dataset for future use."""
    info: str
    traj: Traj
    ins: INS
    flux_a: MagV
    flux_b: MagV
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    mag_1_c: np.ndarray
    mag_2_c: np.ndarray
    mag_3_c: np.ndarray
    mag_1_uc: np.ndarray
    mag_2_uc: np.ndarray
    mag_3_uc: np.ndarray
    aux_1: np.ndarray
    aux_2: np.ndarray
    aux_3: np.ndarray

@dataclass
class XYZ20(XYZ):
    """Dataset for 2020 SGL datasets."""
    info: str
    traj: Traj
    ins: INS
    flux_a: MagV
    flux_b: MagV
    flux_c: MagV
    flux_d: MagV
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    utm_x: np.ndarray
    utm_y: np.ndarray
    utm_z: np.ndarray
    msl: np.ndarray
    baro: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    mag_1_c: np.ndarray
    mag_1_lag: np.ndarray
    mag_1_dc: np.ndarray
    mag_1_igrf: np.ndarray
    mag_1_uc: np.ndarray
    mag_2_uc: np.ndarray
    mag_3_uc: np.ndarray
    mag_4_uc: np.ndarray
    mag_5_uc: np.ndarray
    mag_6_uc: np.ndarray
    ogs_mag: np.ndarray
    ogs_alt: np.ndarray
    ins_wander: np.ndarray
    ins_roll: np.ndarray
    ins_pitch: np.ndarray
    ins_yaw: np.ndarray
    roll_rate: np.ndarray
    pitch_rate: np.ndarray
    yaw_rate: np.ndarray
    ins_acc_x: np.ndarray
    ins_acc_y: np.ndarray
    ins_acc_z: np.ndarray
    lgtl_acc: np.ndarray
    ltrl_acc: np.ndarray
    nrml_acc: np.ndarray
    pitot_p: np.ndarray
    static_p: np.ndarray
    total_p: np.ndarray
    cur_com_1: np.ndarray
    cur_ac_hi: np.ndarray
    cur_ac_lo: np.ndarray
    cur_tank: np.ndarray
    cur_flap: np.ndarray
    cur_strb: np.ndarray
    cur_srvo_o: np.ndarray
    cur_srvo_m: np.ndarray
    cur_srvo_i: np.ndarray
    cur_heat: np.ndarray
    cur_acpwr: np.ndarray
    cur_outpwr: np.ndarray
    cur_bat_1: np.ndarray
    cur_bat_2: np.ndarray
    vol_acpwr: np.ndarray
    vol_outpwr: np.ndarray
    vol_bat_1: np.ndarray
    vol_bat_2: np.ndarray
    vol_res_p: np.ndarray
    vol_res_n: np.ndarray
    vol_back_p: np.ndarray
    vol_back_n: np.ndarray
    vol_gyro_1: np.ndarray
    vol_gyro_2: np.ndarray
    vol_acc_p: np.ndarray
    vol_acc_n: np.ndarray
    vol_block: np.ndarray
    vol_back: np.ndarray
    vol_srvo: np.ndarray
    vol_cabt: np.ndarray
    vol_fan: np.ndarray
    aux_1: np.ndarray
    aux_2: np.ndarray
    aux_3: np.ndarray

@dataclass
class XYZ21(XYZ):
    """Dataset for 2021 SGL datasets."""
    info: str
    traj: Traj
    ins: INS
    flux_a: MagV
    flux_b: MagV
    flux_c: MagV
    flux_d: MagV
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    utm_x: np.ndarray
    utm_y: np.ndarray
    utm_z: np.ndarray
    msl: np.ndarray
    baro: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    mag_1_c: np.ndarray
    mag_1_uc: np.ndarray
    mag_2_uc: np.ndarray
    mag_3_uc: np.ndarray
    mag_4_uc: np.ndarray
    mag_5_uc: np.ndarray
    cur_com_1: np.ndarray
    cur_ac_hi: np.ndarray
    cur_ac_lo: np.ndarray
    cur_tank: np.ndarray
    cur_flap: np.ndarray
    cur_strb: np.ndarray
    vol_block: np.ndarray
    vol_back: np.ndarray
    vol_cabt: np.ndarray
    vol_fan: np.ndarray
    aux_1: np.ndarray
    aux_2: np.ndarray
    aux_3: np.ndarray

@dataclass
class FILTres:
    """Filter results struct."""
    x: np.ndarray    # Filtered states
    P: np.ndarray    # Non-linear covariance matrix
    r: np.ndarray    # Measurement residuals [nT]
    c: bool          # True if filter converged

@dataclass
class CRLBout:
    """Cramér–Rao lower bound extracted output struct."""
    lat_std: np.ndarray
    lon_std: np.ndarray
    alt_std: np.ndarray
    vn_std: np.ndarray
    ve_std: np.ndarray
    vd_std: np.ndarray
    tn_std: np.ndarray
    te_std: np.ndarray
    td_std: np.ndarray
    fogm_std: np.ndarray
    n_std: np.ndarray
    e_std: np.ndarray

@dataclass
class INSout:
    """Inertial navigation system extracted output struct."""
    lat_std: np.ndarray
    lon_std: np.ndarray
    alt_std: np.ndarray
    n_std: np.ndarray
    e_std: np.ndarray
    lat_err: np.ndarray
    lon_err: np.ndarray
    alt_err: np.ndarray
    n_err: np.ndarray

@dataclass
class FILTout(Path):
    """Filter output struct."""
    x: np.ndarray
    P: np.ndarray
    r: np.ndarray
    c: bool
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    f: np.ndarray
    h: np.ndarray
    mag: np.ndarray
    bias: np.ndarray
    scale: np.ndarray
    Mis_e: np.ndarray
    Mis_a: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    t_m: np.ndarray
    t_f: np.ndarray
    t_h: np.ndarray
    t_bias: np.ndarray
    t_scale: np.ndarray
    t_mis_e: np.ndarray
    t_mis_a: np.ndarray
    t_diurnal: np.ndarray
    t_igrf: np.ndarray
    t_xyz: np.ndarray

class CompParams(ABC):
    """Abstract base type for compensation parameters."""
    fields: List[str] # Fields used in the regression or training
    
@dataclass
class LinCompParams(CompParams):
    """Linear aeromagnetic compensation parameters struct."""
    fields: List[str]
    a: np.ndarray
    a_m: np.ndarray
    a_f: np.ndarray
    a_h: np.ndarray
    a_bias: np.ndarray
    a_scale: np.ndarray
    a_mis_e: np.ndarray
    a_mis_a: np.ndarray
    
@dataclass
class NNCompParams(CompParams):
    """Neural network-based aeromagnetic compensation parameters struct."""
    fields: List[str]
    nn: Any
    data_norms: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray] = field(
        default_factory=lambda: tuple(np.array([1.0]) for _ in range(7))
    )

@dataclass
class TempParams:
    """Temp related parameters, primarily for mag correction."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray

def _ekf_rt_create_P0_default() -> np.ndarray:
    return np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

def _ekf_rt_create_Qd_default() -> np.ndarray:
    return np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

@dataclass
class EKF_RT:
    """Real-time (RT) extended Kalman filter (EKF) struct, mutable."""
    P: np.ndarray = field(default_factory=_ekf_rt_create_P0_default)
    Qd: np.ndarray = field(default_factory=_ekf_rt_create_Qd_default)
    R: float = 1.0

# --- Helper functions for access / general utilities ---

try:
    from .analysis_util import get_bpf, bpf_data, linreg, calc_crlb_pos, calc_crlb_vel, calc_crlb_att, calc_crlb_fogm, calc_crlb_map
    from .create_xyz import create_xyz0
    from .tolles_lawson import create_tl_a, create_tl_coef
    from .model_functions import create_model, create_P0 as create_p0, create_Qd as create_qd, get_Phi, get_H, get_h, get_f
    from .ekf import crlb, ekf
    from .get_map import get_map, ottawa_area_maps, namad, emag2, emm720, upward_fft, map_interpolate
    from .plot_functions import plot_mag
    # NOTE: These imports are not directly used in the current version of magnav.py as per
    # the existing logic and might be part of an unused "data_loader" internal Julia module.
    # We define placeholders so Sphinx doesn't complain about missing references later.
    from .get_xyz import get_xyz0, get_xyz20, get_XYZ, sgl_2020_train, sgl_2021_train
    from .google_earth import create_kml
except ImportError as e:
    print(f"Warning: Could not import MagNavPy sub-modules. Some functions might be unavailable: {e}", file=sys.stderr)
    # Define placeholder functions/classes so code that calls them doesn't error immediately
    # during Sphinx autodoc processing. These are minimal and will likely cause runtime errors.

    def get_bpf(*args, **kwargs): raise NotImplementedError("get_bpf requires analysis_util module.")
    def bpf_data(*args, **kwargs): raise NotImplementedError("bpf_data requires analysis_util module.")
    def linreg(*args, **kwargs): raise NotImplementedError("linreg requires analysis_util module.")
    def calc_crlb_pos(*args, **kwargs): raise NotImplementedError("calc_crlb_pos requires analysis_util module.")
    def calc_crlb_vel(*args, **kwargs): raise NotImplementedError("calc_crlb_vel requires analysis_util module.")
    def calc_crlb_att(*args, **kwargs): raise NotImplementedError("calc_crlb_att requires analysis_util module.")
    def calc_crlb_fogm(*args, **kwargs): raise NotImplementedError("calc_crlb_fogm requires analysis_util module.")
    def calc_crlb_map(*args, **kwargs): raise NotImplementedError("calc_crlb_map requires analysis_util module.")
    def create_xyz0(*args, **kwargs): raise NotImplementedError("create_xyz0 requires create_xyz module.")
    def create_tl_a(*args, **kwargs): raise NotImplementedError("create_tl_a requires tolles_lawson module.")
    def create_tl_coef(*args, **kwargs): raise NotImplementedError("create_tl_coef requires tolles_lawson module.")
    def create_model(*args, **kwargs): raise NotImplementedError("create_model requires model_functions module.")
    def create_p0(*args, **kwargs): raise NotImplementedError("create_p0 requires model_functions module.") # Case sensitive in real code
    def create_qd(*args, **kwargs): raise NotImplementedError("create_qd requires model_functions module.") # Case sensitive in real code
    def get_Phi(*args, **kwargs): raise NotImplementedError("get_Phi requires model_functions module.")
    def get_H(*args, **kwargs): raise NotImplementedError("get_H requires model_functions module.")
    def get_h(*args, **kwargs): raise NotImplementedError("get_h requires model_functions module.")
    def get_f(*args, **kwargs): raise NotImplementedError("get_f requires model_functions module.")
    def crlb(*args, **kwargs): raise NotImplementedError("crlb requires ekf module.")
    def ekf(*args, **kwargs): raise NotImplementedError("ekf requires ekf module.")
    def get_map(*args, **kwargs): raise NotImplementedError("get_map requires get_map module.")
    def ottawa_area_maps(*args, **kwargs): raise NotImplementedError("ottawa_area_maps requires get_map module.")
    def namad(*args, **kwargs): raise NotImplementedError("namad requires get_map module.")
    def emag2(*args, **kwargs): raise NotImplementedError("emag2 requires get_map module.")
    def emm720(*args, **kwargs): raise NotImplementedError("emm720 requires get_map module.")
    def upward_fft(*args, **kwargs): raise NotImplementedError("upward_fft requires get_map module.")
    def map_interpolate(*args, **kwargs): raise NotImplementedError("map_interpolate requires get_map module.")
    def plot_mag(*args, **kwargs): raise NotImplementedError("plot_mag requires plot_functions module.")
    def get_xyz0(*args, **kwargs): raise NotImplementedError("get_xyz0 requires get_xyz module.")
    def get_xyz20(*args, **kwargs): raise NotImplementedError("get_xyz20 requires get_xyz module.")
    def get_XYZ(*args, **kwargs): raise NotImplementedError("get_XYZ requires get_xyz module.")
    def sgl_2020_train(*args, **kwargs): raise NotImplementedError("sgl_2020_train requires get_xyz module.")
    def sgl_2021_train(*args, **kwargs): raise NotImplementedError("sgl_2021_train requires get_xyz module.")
    def create_kml(*args, **kwargs): raise NotImplementedError("create_kml requires google_earth module.")

# Constants as defined in magnav.jl for gravity and Earth's rotation based on coordinate system setup details
# This should also account for the difference if from Julia's GeoData or other packages
# It's better to ensure these are consistent with Python's usual geo libraries (e.g., pyproj, geographiclib)
# For now, adopting values directly from MagNav.jl as presented.
r_earth = R_EARTH
omega_earth = OMEGA_EARTH

def get_cached_map(maps: Union[List[MapS], MapS], fallback: MapS = MAP_S_NULL, dz: Union[int, float] = 100) -> MapCache:
    """
    Retrieves or creates a cached map object.

    Args:
        maps: A list of `MapS` objects or a single `MapS` object to use as the base map(s).
        fallback: An optional `MapS` object to use as a fallback if no suitable map is found in `maps`.
                 Defaults to `MAP_S_NULL`.
        dz: The altitude discretization for the cache [m].

    Returns:
        A `MapCache` object containing the interpolated maps.
    """
    maps_list = [maps] if isinstance(maps, MapS) else maps
    return MapCache(maps_list, fallback, dz)

def run_filt(ins: INS, xyz: Union[XYZ0, XYZ1, XYZ20, XYZ21], filter_params: Any,
             mag_map: MapS = MAP_S_NULL, temp_params: TempParams = None) -> FILTres:
    """
    Runs the specified filter (Extended Kalman Filter or Neural EKF for now)
    and returns its results.

    Args:
        ins: INS data.
        xyz: Flight data (XYZ0, XYZ1, XYZ20, or XYZ21).
        filter_params: Filter parameters (e.g., EKF_RT).
        mag_map: A `MapS` object representing the magnetic anomaly map. Defaults to `MAP_S_NULL`.
        temp_params: (Optional) Temperature compensation parameters.

    Returns:
        A `FILTres` object containing the filter results.
    """

    if isinstance(filter_params, EKF_RT):
        # Placeholder for calling the EKF function
        # from .ekf import ekf # This import will be needed
        x_est, P_est, r, converged = ekf(ins, xyz, filter_params, mag_map)
        return FILTres(x=x_est, P=P_est, r=r, c=converged)
    else:
        raise NotImplementedError("Only EKF_RT is currently implemented for filtering.")

# Define placeholder for _undefined_in_all for modules that might use it
_undefined_in_all = [
    # Placeholder for types/functions that might be imported but were not available in original Julia setup
    # and might cause issues during direct translation.
    # This list would contain string names like "some_type", "some_function" if they were found and not translated.
]