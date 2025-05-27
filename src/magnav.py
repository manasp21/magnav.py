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
from .common_types import MapCache

# Attempt to import packages, with placeholders for those not in requirements.txt
import numpy as np
import pandas as pd
import h5py
import toml # Standard in Python 3.11+, or pip install toml
import zipfile
from abc import ABC # For abstract base classes

# SciPy imports
from scipy import signal
from scipy.linalg import expm
from scipy.stats import multivariate_normal, norm, uniform
from scipy.interpolate import UnivariateSpline, interp1d, RegularGridInterpolator, interpn
from scipy.io import loadmat, savemat
from scipy.optimize import minimize as scipy_minimize # Renamed to avoid conflict if we define minimize
from scipy.special import gamma, gammainc, gammaincinv

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

# --- Data Structures (Structs to Dataclasses) ---

class Map(ABC):
    """Abstract base type for a magnetic anomaly map."""
    info: str
    xx: np.ndarray  # 1D array for longitude coordinates [rad]
    yy: np.ndarray  # 1D array for latitude coordinates [rad]
    mask: np.ndarray # Boolean mask for valid data

@dataclass
class MapS(Map):
    """Scalar magnetic anomaly map struct."""
    info: str
    map: np.ndarray  # 2D array (ny x nx) of scalar magnetic anomaly [nT]
    xx: np.ndarray  # 1D array (nx) of map x-direction (longitude) coordinates [rad]
    yy: np.ndarray  # 1D array (ny) of map y-direction (latitude) coordinates [rad]
    alt: float       # Map altitude [m]
    mask: np.ndarray  # 2D boolean array (ny x nx) for valid map data

@dataclass
class MapSd(Map):
    """Scalar magnetic anomaly map struct for drape maps (variable altitude)."""
    info: str
    map: np.ndarray  # 2D array (ny x nx) of scalar magnetic anomaly [nT]
    xx: np.ndarray
    yy: np.ndarray
    alt: np.ndarray  # 2D array (ny x nx) of altitude map [m]
    mask: np.ndarray

@dataclass
class MapS3D(Map):
    """3D (multi-level) scalar magnetic anomaly map struct."""
    info: str
    map: np.ndarray  # 3D array (ny x nx x nz) of scalar magnetic anomaly [nT]
    xx: np.ndarray
    yy: np.ndarray
    alt: np.ndarray  # 1D array (nz) of map altitude levels [m]
    mask: np.ndarray  # 3D boolean array for valid map data

@dataclass
class MapV(Map):
    """Vector magnetic anomaly map struct."""
    info: str
    map_x: np.ndarray # 2D array (ny x nx) x-direction magnetic anomaly [nT]
    map_y: np.ndarray # 2D array (ny x nx) y-direction magnetic anomaly [nT]
    map_z: np.ndarray # 2D array (ny x nx) z-direction magnetic anomaly [nT]
    xx: np.ndarray
    yy: np.ndarray
    alt: float       # Map altitude [m]
    mask: np.ndarray

@dataclass
class MagV:
    """Vector magnetometer measurement struct."""
    x: np.ndarray  # 1D array x-direction magnetic field [nT]
    y: np.ndarray  # 1D array y-direction magnetic field [nT]
    z: np.ndarray  # 1D array z-direction magnetic field [nT]
    t: np.ndarray  # 1D array total magnetic field [nT]

# Null map for default arguments
MAP_S_NULL = MapS(info="Null map",
                  map=np.zeros((1, 1)),
                  xx=np.array([0.0]),
                  yy=np.array([0.0]),
                  alt=0.0,
                  mask=np.array([[True]]))

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
    e_err: np.ndarray

@dataclass
class FILTout(Path):
    """Filter extracted output struct."""
    N: int
    dt: float
    tt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    tn: np.ndarray  # north tilt (attitude) [rad]
    te: np.ndarray  # east tilt (attitude) [rad]
    td: np.ndarray  # down tilt (attitude) [rad]
    ha: np.ndarray  # barometer aiding altitude [m]
    ah: np.ndarray  # barometer aiding vertical acceleration [m/s^2]
    ax: np.ndarray  # x accelerometer [m/s^2]
    ay: np.ndarray
    az: np.ndarray
    gx: np.ndarray  # x gyroscope [rad/s]
    gy: np.ndarray
    gz: np.ndarray
    fogm: np.ndarray # FOGM catch-all [nT]
    lat_std: np.ndarray
    lon_std: np.ndarray
    alt_std: np.ndarray
    vn_std: np.ndarray
    ve_std: np.ndarray
    vd_std: np.ndarray
    tn_std: np.ndarray
    te_std: np.ndarray
    td_std: np.ndarray
    ha_std: np.ndarray
    ah_std: np.ndarray
    ax_std: np.ndarray
    ay_std: np.ndarray
    az_std: np.ndarray
    gx_std: np.ndarray
    gy_std: np.ndarray
    gz_std: np.ndarray
    fogm_std: np.ndarray
    n_std: np.ndarray
    e_std: np.ndarray
    lat_err: np.ndarray
    lon_err: np.ndarray
    alt_err: np.ndarray
    vn_err: np.ndarray
    ve_err: np.ndarray
    vd_err: np.ndarray
    tn_err: np.ndarray
    te_err: np.ndarray
    td_err: np.ndarray
    n_err: np.ndarray
    e_err: np.ndarray

class CompParams(ABC):
    # Import EKF functions after all relevant data structures are defined
    """Abstract type for aeromagnetic compensation parameters."""
    pass

@dataclass
class LinCompParams(CompParams):
    """Linear aeromagnetic compensation parameters struct."""
    version: str = MAGNAV_VERSION_STR
    features_setup: List[str] = field(default_factory=lambda: [":mag_1_uc",":TL_A_flux_a"]) # Julia symbols as strings
    features_no_norm: List[str] = field(default_factory=list)
    model_type: str = ":plsr"
    y_type: str = ":d"
    use_mag: str = ":mag_1_uc"
    use_vec: str = ":flux_a"
    data_norms: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.zeros((1,1)), np.zeros((1,1)), np.array([0.0]), np.array([0.0]))
    )
    model: Tuple[np.ndarray, float] = field(default_factory=lambda: (np.array([0.0]), 0.0))
    terms: List[str] = field(default_factory=lambda: [":permanent",":induced",":eddy"])
    terms_A: List[str] = field(default_factory=lambda: [":permanent",":induced",":eddy",":bias"])
    sub_diurnal: bool = False
    sub_igrf: bool = False
    bpf_mag: bool = False
    reorient_vec: bool = False
    norm_type_A: str = ":none"
    norm_type_x: str = ":none"
    norm_type_y: str = ":none"
    k_plsr: int = 18
    lambda_TL: float = 0.025 # λ_TL in Julia

@dataclass
class NNCompParams(CompParams):
    """Neural network-based aeromagnetic compensation parameters struct."""
    version: str = MAGNAV_VERSION_STR
    features_setup: List[str] = field(default_factory=lambda: [":mag_1_uc",":TL_A_flux_a"])
    features_no_norm: List[str] = field(default_factory=list)
    model_type: str = ":m1"
    y_type: str = ":d"
    use_mag: str = ":mag_1_uc"
    use_vec: str = ":flux_a"
    data_norms: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray] = field(
        default_factory=lambda: (
            np.zeros((1,1), dtype=np.float32), np.zeros((1,1), dtype=np.float32),
            np.zeros((1,1), dtype=np.float32), np.zeros((1,1), dtype=np.float32),
            np.zeros((1,1), dtype=np.float32), np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32)
        )
    )
    model: nn.Sequential = field(default_factory=nn.Sequential) # Flux.Chain -> torch.nn.Sequential
    terms: List[str] = field(default_factory=lambda: [":permanent",":induced",":eddy"])
    terms_A: List[str] = field(default_factory=lambda: [":permanent",":induced",":eddy",":bias"])
    sub_diurnal: bool = False
    sub_igrf: bool = False
    bpf_mag: bool = False
    reorient_vec: bool = False
    norm_type_A: str = ":none"
    norm_type_x: str = ":standardize"
    norm_type_y: str = ":standardize"
    TL_coef: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=float)) # Tolles-Lawson coefficients
    eta_adam: float = 0.001 # η_adam
    epoch_adam: int = 5
    epoch_lbfgs: int = 0
    hidden: List[int] = field(default_factory=lambda: [8])
    activation: Callable = field(default_factory=lambda: F.silu) # swish -> torch.nn.functional.silu or nn.SiLU()
    loss: Callable = field(default_factory=lambda: F.mse_loss) # mse -> torch.nn.functional.mse_loss
    batchsize: int = 2048
    frac_train: float = 14/17
    alpha_sgl: float = 1.0 # α_sgl
    lambda_sgl: float = 0.0 # λ_sgl
    k_pca: int = -1
    drop_fi: bool = False
    drop_fi_bson: str = "drop_fi"
    drop_fi_csv: str = "drop_fi"
    perm_fi: bool = False
    perm_fi_csv: str = "perm_fi"

@dataclass
class TempParams:
    """Temporary parameters struct for temporal models."""
    sigma_curriculum: float = 1.0 # σ_curriculum
    l_window: int = 5
    window_type: str = ":sliding"
    tf_layer_type: str = ":postlayer"
    tf_norm_type: str = ":batch"
    dropout_prob: float = 0.2
    N_tf_head: int = 8
    tf_gain: float = 1.0

# Placeholder functions for EKF_RT and Map_Cache defaults
# These would need proper implementation or import from other modules
def _ekf_rt_create_P0_default() -> np.ndarray:
    # Default P matrix (e.g., 17x17 identity or from create_P0.jl)
    # This is a placeholder, actual dimensions and values are important.
    print("Warning: Using placeholder for EKF_RT.P default.", file=sys.stderr)
    return np.eye(17) # Based on create_P0.jl, it's 17x17

def _ekf_rt_create_Qd_default() -> np.ndarray:
    # Default Qd matrix (e.g., from create_Qd.jl)
    print("Warning: Using placeholder for EKF_RT.Qd default.", file=sys.stderr)
    # Based on create_Qd.jl, it's complex. Placeholder:
    return np.diag([0.01**2]*3 + [0.01**2]*3 + [0.1**2]*3 + [0.0]*2 + [0.01**2]*3 + [0.0]*3 + [0.0])[:17,:17] # Simplified

def _ekf_rt_get_years_default(year: int, doy: int) -> float:
    # From get_XYZ.jl (simplified)
    return float(year) + (float(doy) - 1) / (366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365)

# Forward declaration for MapS needed by Map_Cache's default fallback
# Will be defined above, this is just for conceptual ordering if get_map was here
# def get_map(map_file_path: str, map_type: str = "mat") -> MapS:
# print("Warning: get_map function is a placeholder.", file=sys.stderr)
# return MAP_S_NULL # Placeholder

class EKF_RT:
    """Real-time (RT) extended Kalman filter (EKF) struct, mutable."""
    def __init__(self, P: np.ndarray = None, Qd: np.ndarray = None, R: float = 1.0,
                 baro_tau: float = 3600.0, acc_tau: float = 3600.0, gyro_tau: float = 3600.0,
                 fogm_tau: float = 600.0, date: float = None, core: bool = False,
                 nx: int = None, ny: int = 1, t: float = -1.0, x: np.ndarray = None,
                 r: np.ndarray = None):
        self.P = P if P is not None else _ekf_rt_create_P0_default()
        self.Qd = Qd if Qd is not None else _ekf_rt_create_Qd_default()
        self.R = R
        self.baro_tau = baro_tau
        self.acc_tau = acc_tau
        self.gyro_tau = gyro_tau
        self.fogm_tau = fogm_tau
        self.date = date if date is not None else _ekf_rt_get_years_default(2020, 185)
        self.core = core
        self.nx = nx if nx is not None else self.P.shape[0]
        self.ny = ny
        self.t = t
        self.x = x if x is not None else np.zeros(self.nx)
        self.r = r if r is not None else np.zeros(self.ny)

# MapCache class has been moved to common_types.py


# --- Moved relative imports to resolve circular dependencies ---
# Original "Included Files" comments removed and replaced by this section.
# All relative imports from MagNavPy.src submodules are consolidated here.
from .rt_comp_main import (
    ekf_online_tl,
    ekf_online_tl_ins,
    ekf_online_tl_setup,
    ekf_online_nn,
    ekf_online_nn_ins,
    ekf_online_nn_setup,
    # FILTres as RTCompFILTres # Alias if FILTres from rt_comp_main is different and needed
)
from .compensation import *
from .ekf import ekf, ekf_ins, process_ekf_rt_step, process_ekf_rt_step_ins
from .analysis_util import *
from .plot_functions import *
from .create_xyz import *
# --- Exports (Mimicking Julia's export and @compat(public, ...)) ---
# In Python, this is typically managed by __all__ or by not using underscore prefixes.
# For explicit control similar to Julia's export, we define __all__.

# Note: The functions and types listed in Julia's export need to be defined in this
# file or imported from the "included" files above. Since those are commented out,
# this __all__ list will cause errors if items are not defined/imported.
# This list is based on the `export` and `@compat(public, ...)` block in MagNav.jl.

__all__ = [
    # Constants from artifacts/paths
    "ottawa_area_maps_gxf_path", "EMAG2_PATH", "EMM720_PATH", "NAMAD_PATH",
    "sgl_2020_train_path", "sgl_2021_train_path", "ottawa_area_maps_path",
    # Structs / Dataclasses
    "MapS", "MapSd", "MapS3D", "MapV", "MagV", "Traj", "INS",
    "XYZ0", "XYZ1", "XYZ20", "XYZ21",
    "FILTres", "CRLBout", "INSout", "FILTout", "TempParams",
    "LinCompParams", "NNCompParams", "EKF_RT", "MapCache", # Renamed Map_Cache
    # Functions (many of these would come from the included files)
    # From @compat(public, ...)
    "linreg", "get_x", "get_y", "get_Axy", "get_nn_m", "sparse_group_lasso",
    "chunk_data", "predict_rnn_full", "predict_rnn_windowed", "krr_fit", "krr_test",
    "project_body_field_to_2d_igrf", "get_optimal_rotation_matrix",
    "filter_events_inplace", "filter_events", # filter_events! -> filter_events_inplace
    "TL_vec2mat", "TL_mat2vec", "plsr_fit", "elasticnet_fit", "linear_fit", "linear_test",
    "create_mag_c", "corrupt_mag",
    "eval_results", "eval_crlb", "eval_ins",
    "downward_L", "psd",
    "map_get_gxf", "map_correct_igrf_inplace", "map_correct_igrf", "map_chessboard_inplace",
    "map_chessboard", "map_utm2lla_inplace", "map_utm2lla", "map_resample", "get_step",
    "create_P0", "create_Qd", "get_pinson", "fogm",
    "fdm",
    "compare_fields",
    # From export ...
    "dn2dlat", "de2dlon", "dlat2dn", "dlon2de", "detrend", "get_bpf", "bpf_data", "bpf_data_inplace", # bpf_data!
    "err_segs", "norm_sets", "denorm_sets", "get_ind", "eval_shapley", "plot_shapley", "eval_gsa",
    "get_IGRF", "get_igrf", "get_years", "gif_animation_m3", "plot_basic", "plot_activation",
    "plot_mag", "plot_mag_c", "plot_frequency", "plot_correlation", "plot_correlation_matrix",
    "comp_train", "comp_test", "comp_m2bc_test", "comp_m3_test", "comp_train_test",
    "create_XYZ0", "create_traj", "create_ins", "create_flux", "create_informed_xyz",
    "euler2dcm", "dcm2euler",
    "ekf", "crlb",
    "ekf_online_nn", "ekf_online_nn_setup",
    "ekf_online", "ekf_online_setup",
    "eval_filt", "run_filt",
    "plot_filt_inplace", "plot_filt", "plot_filt_err", "plot_mag_map", "plot_mag_map_err", # plot_filt!
    "get_autocor", "plot_autocor", "gif_ellipse",
    "get_map", "save_map", "get_comp_params", "save_comp_params",
    "get_XYZ20", "get_XYZ21", "get_XYZ", "get_xyz", "get_XYZ0", "get_XYZ1",
    "get_flux", "get_magv", "get_MagV", "get_traj", "get_Traj", "get_ins", "get_INS",
    "map2kmz", "path2kml",
    "upward_fft", "vector_fft", "map_expand",
    "map_interpolate", "map_itp", "map_trim", "map_fill_inplace", "map_fill", "map_gxf2h5", # map_fill!
    "plot_map_inplace", "plot_map", "plot_path_inplace", "plot_path", "plot_events_inplace", "plot_events", # plot_map!, plot_path!, plot_events!
    "map_check", "get_map_val", "get_cached_map", "map_border", "map_combine",
    "create_model",
    "mpf",
    "nekf", "nekf_train",
    "create_TL_A", "create_TL_coef",
    "xyz2h5",
    # Core constants
    "NUM_MAG_MAX", "E_EARTH", "G_EARTH", "R_EARTH", "OMEGA_EARTH",
    "USGS_COLOR_SCALE", "ICON_CIRCLE", "SILENT_DEBUG", "MAP_S_NULL",
    "MAGNAV_VERSION_STR"
]

# Placeholder for functions that were in __all__ but are not yet defined/imported
# This is to avoid NameError if this file is imported as is.
# In a complete conversion, these would be imported from their respective modules.
_undefined_in_all = [
    name for name in __all__
    if name not in globals() and
       name not in locals() and # Check locals for class definitions too
       not (name.endswith("_path") and hasattr(sys.modules[__name__], name)) # Path functions
]

if _undefined_in_all:
    print(f"Warning: The following symbols in __all__ are not defined in magnav.py: {_undefined_in_all}", file=sys.stderr)
    print("This is expected if their corresponding modules (e.g., analysis_util.py) have not been converted and imported yet.", file=sys.stderr)
    # Optionally, create placeholders for them:
    # for _name in _undefined_in_all:
    #     if not hasattr(sys.modules[__name__], _name): # Check again to be safe
    #         exec(f"{_name} = lambda *args, **kwargs: print(f'Placeholder for {_name} called')")

# --- End of MagNav.py ---