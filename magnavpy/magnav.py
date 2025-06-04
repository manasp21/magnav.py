# -*- coding: utf-8 -*-
"""
Main module for MagNavPy, translated from MagNav.jl.

This module provides core data structures, constants, utility functions,
and access to datasets for magnetic navigation.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Any, Union, Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import h5py
import toml
import zipfile

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

# Scikit-learn imports
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.neighbors import KDTree as SklearnKDTree, NearestNeighbors as SklearnNearestNeighbors
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

# Statsmodels imports
from statsmodels.tsa.stattools import acf as sm_acf # Renamed

# GDAL import
try:
    from osgeo import gdal
except ImportError:
    gdal = None # Placeholder
    print("Warning: GDAL not found. Some functionalities might be unavailable.", file=sys.stderr)

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    # from jax.config import config
    # config.update("jax_enable_x64", True)
except ImportError:
    jax = None
    jnp = None # Placeholder
    print("Warning: JAX not found. Autodiff functionalities might be unavailable.", file=sys.stderr)

# MagNavPy sub-module imports
from .common_types import (
    Map, MapS, MapSd, MapS3D, MapV, MagV, MapCache, MAP_S_NULL
)
from .tolles_lawson import create_TL_A, create_TL_coef # create_TL_coef might be needed for defaults
from .dcm_util import dcm2euler, euler2dcm
from .fdm_util import fdm
from .signal_util import linreg_matrix # Assuming this is equivalent to Julia's linreg(y,x)
from .core_utils import dn2dlat, de2dlon, dlat2dn, dlon2de, get_years # Import core utilities

try:
    from .model_functions import get_f, create_P0 as model_create_P0, create_Qd as model_create_Qd
except ImportError:
    print("Warning: Could not import some functions from .model_functions. Defining placeholders.", file=sys.stderr)
    def get_f(*args, **kwargs):
        raise NotImplementedError("get_f could not be imported from model_functions.")
    def model_create_P0(*args, **kwargs):
        # Default P0 for EKF_RT if model_functions.create_P0 is not available
        # Based on Julia's EKF_RT default P = create_P0() which likely initializes a 17x17 matrix
        return np.diag(np.array([
            1e0**2, 1e0**2, 1e1**2,      # Lat, Lon, Alt
            1e-1**2, 1e-1**2, 1e-1**2,  # Vn, Ve, Vd
            (1*np.pi/180)**2, (1*np.pi/180)**2, (1*np.pi/180)**2, # Tilt n, e, d
            1e-1**2,                    # Baro Bias
            (0.1*9.81)**2, (0.1*9.81)**2, (0.1*9.81)**2, # Accel Bias X, Y, Z
            (1*np.pi/180/3600)**2, (1*np.pi/180/3600)**2, (1*np.pi/180/3600)**2, # Gyro Bias X, Y, Z
            1e1**2                      # FOGM Bias
        ], dtype=float))
    def model_create_Qd(*args, **kwargs):
        # Default Qd for EKF_RT if model_functions.create_Qd is not available
        # Based on Julia's EKF_RT default Qd = create_Qd()
        dt = 0.1 # Assuming a default dt for placeholder
        return np.diag(np.array([
            (1e-8*dt)**2, (1e-8*dt)**2, (1e-1*dt)**2, # Lat, Lon, Alt
            (1e-3*dt)**2, (1e-3*dt)**2, (1e-3*dt)**2, # Vn, Ve, Vd
            (1e-7*dt)**2, (1e-7*dt)**2, (1e-7*dt)**2, # Tilt n, e, d
            (1e-3*dt)**2,                             # Baro Bias
            (1e-4*dt)**2, (1e-4*dt)**2, (1e-4*dt)**2, # Accel Bias X, Y, Z
            (1e-8*dt)**2, (1e-8*dt)**2, (1e-8*dt)**2, # Gyro Bias X, Y, Z
            (1e-1*dt)**2                              # FOGM Bias
        ], dtype=float))


# --- Project Version ---
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
NUM_MAG_MAX = 6
E_EARTH = 0.0818191908426
G_EARTH = 9.80665
R_EARTH = 6378137
OMEGA_EARTH = 7.2921151467e-5
SILENT_DEBUG = True

_DEFAULT_ARTIFACT_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "artifacts_data")
print(f"Note: Artifact paths are placeholders. Ensure data exists at expected locations relative to '{_DEFAULT_ARTIFACT_DIR}' or configure properly.", file=sys.stderr)

USGS_COLOR_SCALE = os.path.join(_DEFAULT_ARTIFACT_DIR, "util_files", "util_files", "color_scale_usgs.csv")
ICON_CIRCLE = os.path.join(_DEFAULT_ARTIFACT_DIR, "util_files", "util_files", "icon_circle.dae")
EMAG2_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "EMAG2", "EMAG2.h5")
EMM720_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "EMM720_World", "EMM720_World.h5")
NAMAD_PATH = os.path.join(_DEFAULT_ARTIFACT_DIR, "NAMAD_305", "NAMAD_305.h5")

# --- Helper Functions for path manipulation ---
def add_extension(name: str, ext: str) -> str:
    if not name.endswith(ext):
        return name + ext
    return name

def remove_extension(name: str, suffix_to_remove: str) -> str:
    if name.endswith(suffix_to_remove):
        return name[:-len(suffix_to_remove)]
    return name

# --- Artifact Path Functions ---
def sgl_fields_path(f: str = "") -> str:
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_fields", "sgl_fields")
    if f:
        return os.path.join(p_base, add_extension(str(f), ".csv"))
    return p_base

def sgl_2020_train_path(f: str = "") -> str:
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_2020_train", "sgl_2020_train")
    if f:
        d = remove_extension(str(f), "_train")
        return os.path.join(p_base, add_extension(d, "_train.h5"))
    return p_base

def sgl_2021_train_path(f: str = "") -> str:
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "sgl_2021_train", "sgl_2021_train")
    if f:
        d = remove_extension(str(f), "_train")
        return os.path.join(p_base, add_extension(d, "_train.h5"))
    return p_base

def ottawa_area_maps_path(f: str = "") -> str:
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "ottawa_area_maps", "ottawa_area_maps")
    if f:
        return os.path.join(p_base, add_extension(str(f), ".h5"))
    return p_base

def ottawa_area_maps_gxf_path(f: str = "") -> str:
    p_base = os.path.join(_DEFAULT_ARTIFACT_DIR, "ottawa_area_maps_gxf", "ottawa_area_maps_gxf")
    if f:
        d = remove_extension(str(f), "_Mag")
        return os.path.join(p_base, add_extension(d, "_Mag.gxf"))
    return p_base

# --- Data Structures ---
class Path(ABC):
    N: int
    dt: float
    tt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray

@dataclass
class Traj(Path):
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

@dataclass
class INS(Path):
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
    P: np.ndarray

class XYZ(ABC):
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
    mag_1_c: np.ndarray
    mag_1_uc: np.ndarray

@dataclass
class XYZ1(XYZ):
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
    info: str; traj: Traj; ins: INS; flux_a: MagV; flux_b: MagV; flux_c: MagV; flux_d: MagV;
    flight: np.ndarray; line: np.ndarray; year: np.ndarray; doy: np.ndarray; utm_x: np.ndarray;
    utm_y: np.ndarray; utm_z: np.ndarray; msl: np.ndarray; baro: np.ndarray; diurnal: np.ndarray;
    igrf: np.ndarray; mag_1_c: np.ndarray; mag_1_lag: np.ndarray; mag_1_dc: np.ndarray;
    mag_1_igrf: np.ndarray; mag_1_uc: np.ndarray; mag_2_uc: np.ndarray; mag_3_uc: np.ndarray;
    mag_4_uc: np.ndarray; mag_5_uc: np.ndarray; mag_6_uc: np.ndarray; ogs_mag: np.ndarray;
    ogs_alt: np.ndarray; ins_wander: np.ndarray; ins_roll: np.ndarray; ins_pitch: np.ndarray;
    ins_yaw: np.ndarray; roll_rate: np.ndarray; pitch_rate: np.ndarray; yaw_rate: np.ndarray;
    ins_acc_x: np.ndarray; ins_acc_y: np.ndarray; ins_acc_z: np.ndarray; lgtl_acc: np.ndarray;
    ltrl_acc: np.ndarray; nrml_acc: np.ndarray; pitot_p: np.ndarray; static_p: np.ndarray;
    total_p: np.ndarray; cur_com_1: np.ndarray; cur_ac_hi: np.ndarray; cur_ac_lo: np.ndarray;
    cur_tank: np.ndarray; cur_flap: np.ndarray; cur_strb: np.ndarray; cur_srvo_o: np.ndarray;
    cur_srvo_m: np.ndarray; cur_srvo_i: np.ndarray; cur_heat: np.ndarray; cur_acpwr: np.ndarray;
    cur_outpwr: np.ndarray; cur_bat_1: np.ndarray; cur_bat_2: np.ndarray; vol_acpwr: np.ndarray;
    vol_outpwr: np.ndarray; vol_bat_1: np.ndarray; vol_bat_2: np.ndarray; vol_res_p: np.ndarray;
    vol_res_n: np.ndarray; vol_back_p: np.ndarray; vol_back_n: np.ndarray; vol_gyro_1: np.ndarray;
    vol_gyro_2: np.ndarray; vol_acc_p: np.ndarray; vol_acc_n: np.ndarray; vol_block: np.ndarray;
    vol_back: np.ndarray; vol_srvo: np.ndarray; vol_cabt: np.ndarray; vol_fan: np.ndarray;
    aux_1: np.ndarray; aux_2: np.ndarray; aux_3: np.ndarray;

@dataclass
class XYZ21(XYZ):
    info: str; traj: Traj; ins: INS; flux_a: MagV; flux_b: MagV; flux_c: MagV; flux_d: MagV;
    flight: np.ndarray; line: np.ndarray; year: np.ndarray; doy: np.ndarray; utm_x: np.ndarray;
    utm_y: np.ndarray; utm_z: np.ndarray; msl: np.ndarray; baro: np.ndarray; diurnal: np.ndarray;
    igrf: np.ndarray; mag_1_c: np.ndarray; mag_1_uc: np.ndarray; mag_2_uc: np.ndarray;
    mag_3_uc: np.ndarray; mag_4_uc: np.ndarray; mag_5_uc: np.ndarray; cur_com_1: np.ndarray;
    cur_ac_hi: np.ndarray; cur_ac_lo: np.ndarray; cur_tank: np.ndarray; cur_flap: np.ndarray;
    cur_strb: np.ndarray; vol_block: np.ndarray; vol_back: np.ndarray; vol_cabt: np.ndarray;
    vol_fan: np.ndarray; aux_1: np.ndarray; aux_2: np.ndarray; aux_3: np.ndarray;

@dataclass
class FILTres:
    x: np.ndarray
    P: np.ndarray
    r: np.ndarray
    c: bool

@dataclass
class CRLBout:
    lat_std: np.ndarray; lon_std: np.ndarray; alt_std: np.ndarray;
    vn_std: np.ndarray; ve_std: np.ndarray; vd_std: np.ndarray;
    tn_std: np.ndarray; te_std: np.ndarray; td_std: np.ndarray;
    fogm_std: np.ndarray; n_std: np.ndarray; e_std: np.ndarray;

@dataclass
class INSout:
    lat_std: np.ndarray; lon_std: np.ndarray; alt_std: np.ndarray;
    n_std: np.ndarray; e_std: np.ndarray; lat_err: np.ndarray;
    lon_err: np.ndarray; alt_err: np.ndarray; n_err: np.ndarray; e_err: np.ndarray;

@dataclass
class FILTout(Path): # Path provides N, dt, tt, lat, lon, alt, vn, ve, vd
    N: int; dt: float; tt: np.ndarray; lat: np.ndarray; lon: np.ndarray; alt: np.ndarray;
    vn: np.ndarray; ve: np.ndarray; vd: np.ndarray;
    tn: np.ndarray; te: np.ndarray; td: np.ndarray; ha: np.ndarray; ah: np.ndarray;
    ax: np.ndarray; ay: np.ndarray; az: np.ndarray; gx: np.ndarray; gy: np.ndarray; gz: np.ndarray;
    fogm: np.ndarray; lat_std: np.ndarray; lon_std: np.ndarray; alt_std: np.ndarray;
    vn_std: np.ndarray; ve_std: np.ndarray; vd_std: np.ndarray; tn_std: np.ndarray;
    te_std: np.ndarray; td_std: np.ndarray; ha_std: np.ndarray; ah_std: np.ndarray;
    ax_std: np.ndarray; ay_std: np.ndarray; az_std: np.ndarray; gx_std: np.ndarray;
    gy_std: np.ndarray; gz_std: np.ndarray; fogm_std: np.ndarray; n_std: np.ndarray;
    e_std: np.ndarray; lat_err: np.ndarray; lon_err: np.ndarray; alt_err: np.ndarray;
    vn_err: np.ndarray; ve_err: np.ndarray; vd_err: np.ndarray; tn_err: np.ndarray;
    te_err: np.ndarray; td_err: np.ndarray; n_err: np.ndarray; e_err: np.ndarray;

class CompParams(ABC):
    """Abstract base type for compensation parameters."""
    pass

def _default_lin_comp_data_norms():
    return (np.zeros((1,1), dtype=float), np.zeros((1,1), dtype=float),
            np.array([0.0], dtype=float), np.array([0.0], dtype=float))

def _default_lin_comp_model():
    return (np.array([0.0], dtype=float), 0.0)

@dataclass
class LinCompParams(CompParams):
    version: str = MAGNAV_VERSION_STR
    features_setup: List[str] = field(default_factory=lambda: ['mag_1_uc', 'TL_A_flux_a'])
    features_no_norm: List[str] = field(default_factory=list)
    model_type: str = 'plsr'  # :TL, :mod_TL, :map_TL, :elasticnet, :plsr
    y_type: str = 'd'  # :a, :b, :c, :d, :e
    use_mag: str = 'mag_1_uc'
    use_vec: str = 'flux_a'
    data_norms: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = field(default_factory=_default_lin_comp_data_norms)
    model: Tuple[np.ndarray, float] = field(default_factory=_default_lin_comp_model)
    terms: List[str] = field(default_factory=lambda: ['permanent', 'induced', 'eddy'])
    terms_A: List[str] = field(default_factory=lambda: ['permanent', 'induced', 'eddy', 'bias'])
    sub_diurnal: bool = False
    sub_igrf: bool = False
    bpf_mag: bool = False
    reorient_vec: bool = False
    norm_type_A: str = 'none'  # :standardize, :normalize, :scale, :none
    norm_type_x: str = 'none'
    norm_type_y: str = 'none'
    k_plsr: int = 18
    lambda_TL: float = 0.025  # λ_TL in Julia

def _default_nn_comp_data_norms():
    return tuple(np.zeros((1,1), dtype=np.float32) if i < 5 else np.array([0.0], dtype=np.float32) for i in range(7))

@dataclass
class NNCompParams(CompParams):
    version: str = MAGNAV_VERSION_STR
    features_setup: List[str] = field(default_factory=lambda: ['mag_1_uc', 'TL_A_flux_a'])
    features_no_norm: List[str] = field(default_factory=list)
    model_type: str = 'm1' # :m1, :m2a, :m2b, :m2c, :m2d, :m3tl, :m3s, :m3v, :m3sc, :m3vc, :m3w, :m3tf
    y_type: str = 'd' # :a, :b, :c, :d, :e
    use_mag: str = 'mag_1_uc'
    use_vec: str = 'flux_a'
    data_norms: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] = field(default_factory=_default_nn_comp_data_norms)
    model: nn.Sequential = field(default_factory=nn.Sequential)
    terms: List[str] = field(default_factory=lambda: ['permanent', 'induced', 'eddy'])
    terms_A: List[str] = field(default_factory=lambda: ['permanent', 'induced', 'eddy', 'bias'])
    sub_diurnal: bool = False
    sub_igrf: bool = False
    bpf_mag: bool = False
    reorient_vec: bool = False
    norm_type_A: str = 'none' # :standardize, :normalize, :scale, :none
    norm_type_x: str = 'standardize'
    norm_type_y: str = 'standardize'
    TL_coef: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=float))
    eta_adam: float = 0.001
    epoch_adam: int = 5
    epoch_lbfgs: int = 0
    hidden: List[int] = field(default_factory=lambda: [8])
    activation: Callable = F.silu # Corresponds to swish
    loss: Callable = F.mse_loss
    batchsize: int = 2048
    frac_train: float = 14/17
    alpha_sgl: float = 1.0 # α_sgl in Julia
    lambda_sgl: float = 0.0 # λ_sgl in Julia
    k_pca: int = -1
    drop_fi: bool = False
    drop_fi_bson: str = "drop_fi"
    drop_fi_csv: str = "drop_fi"
    perm_fi: bool = False
    perm_fi_csv: str = "perm_fi"

@dataclass
class TempParams:
    sigma_curriculum: float = 1.0 # σ_curriculum in Julia
    l_window: int = 5
    window_type: str = 'sliding' # :sliding, :contiguous
    tf_layer_type: str = 'postlayer' # :prelayer, :postlayer
    tf_norm_type: str = 'batch' # :batch, :layer, :none
    dropout_prob: float = 0.2
    N_tf_head: int = 8
    tf_gain: float = 1.0

def _get_default_ekf_rt_date() -> float:
    # Corresponds to Julia's get_years(2020,185)
    # Simplified: (year - 1) + (day_of_year / days_in_year)
    # For simplicity, using a fixed value, proper get_years would be better.
    return 2020 + (185 -1) / 366.0 # 2020 is a leap year

@dataclass
class EKF_RT:
    P: np.ndarray = field(default_factory=model_create_P0)
    Qd: np.ndarray = field(default_factory=model_create_Qd)
    R: float = 1.0
    baro_tau: float = 3600.0
    acc_tau: float = 3600.0
    gyro_tau: float = 3600.0
    fogm_tau: float = 600.0
    date: float = field(default_factory=_get_default_ekf_rt_date)
    core: bool = False
    nx: int = field(init=False)
    ny: int = 1
    t: float = -1.0
    x: np.ndarray = field(init=False)
    r: np.ndarray = field(init=False)

    def __post_init__(self):
        self.nx = self.P.shape[0]
        self.x = np.zeros(self.nx, dtype=float)
        self.r = np.zeros(self.ny, dtype=float)

# --- Utility Functions (ported from analysis_util.jl) ---
# dn2dlat, de2dlon, dlat2dn, dlon2de moved to core_utils.py

def linreg_vector(y: np.ndarray, lambda_val: float = 0) -> np.ndarray:
    """Linear regression to determine best fit line for x = eachindex(y)."""
    x_mat = np.vstack([np.ones_like(y), np.arange(len(y))]).T
    return linreg_matrix(y, x_mat, lambda_val=lambda_val)

def detrend(y: np.ndarray, x: Optional[np.ndarray] = None, lambda_val: float = 0, mean_only: bool = False) -> np.ndarray:
    """Detrend signal (remove mean and optionally slope)."""
    if mean_only:
        return y - np.mean(y)
    else:
        if x is None:
            x_mat = np.arange(len(y))
        else:
            x_mat = x
        
        if x_mat.ndim == 1:
             x_mat_for_reg = np.vstack([np.ones_like(y), x_mat]).T
        else: # Assuming x is already a matrix where columns are features, add intercept
            x_mat_for_reg = np.hstack([np.ones((len(y),1)), x_mat])

        coef = linreg_matrix(y, x_mat_for_reg, lambda_val=lambda_val)
        return y - x_mat_for_reg @ coef

# --- Main Functions ---
def run_filt(ins: INS, xyz: Union[XYZ0, XYZ1, XYZ20, XYZ21], filter_params: Any,
             mag_map: MapS = MAP_S_NULL, temp_params: Optional[TempParams] = None) -> FILTres:
    """
    Runs the specified filter (Extended Kalman Filter or Neural EKF for now)
    and returns its results.
    """
    # This function's full implementation depends on ekf.py
    # For now, it's a placeholder calling a potential ekf function.
    # from .ekf import ekf # Ensure ekf is importable
    if isinstance(filter_params, EKF_RT):
        # The actual ekf function needs to be implemented/imported from .ekf
        # x_est, P_est, r_val, converged = ekf(ins, xyz, filter_params, mag_map) # Hypothetical call
        # return FILTres(x=x_est, P=P_est, r=r_val, c=converged)
        raise NotImplementedError("Full EKF execution from .ekf module is not yet integrated here.")
    else:
        raise NotImplementedError("Only EKF_RT is currently supported as filter_params type.")


# --- Public API ---
__all__ = [
    # Constants
    "NUM_MAG_MAX", "E_EARTH", "G_EARTH", "R_EARTH", "OMEGA_EARTH", "MAGNAV_VERSION_STR",
    "USGS_COLOR_SCALE", "ICON_CIRCLE", "SILENT_DEBUG",
    "EMAG2_PATH", "EMM720_PATH", "NAMAD_PATH",
    # Path Functions
    "sgl_fields_path", "sgl_2020_train_path", "sgl_2021_train_path",
    "ottawa_area_maps_path", "ottawa_area_maps_gxf_path",
    # Core Data Structures (Local)
    "Path", "Traj", "INS", "XYZ", "XYZ0", "XYZ1", "XYZ20", "XYZ21",
    "FILTres", "CRLBout", "INSout", "FILTout",
    # Parameter Data Structures (Local)
    "CompParams", "LinCompParams", "NNCompParams", "TempParams", "EKF_RT",
    # Data Structures (Imported from common_types)
    "Map", "MapS", "MapSd", "MapS3D", "MapV", "MagV", "MapCache", "MAP_S_NULL",
    # Utility Functions (Local)
    "add_extension", "remove_extension",
    "dn2dlat", "de2dlon", "dlat2dn", "dlon2de", "linreg_vector", "detrend",
    # Key Functions (Imported from submodules - for convenience, matching Julia's exports)
    "dcm2euler", "euler2dcm",  # from .dcm_util
    "fdm",  # from .fdm_util
    "create_TL_A", "create_TL_coef", # from .tolles_lawson
    "linreg_matrix", # from .signal_util
    # Main functions (stubs or to be fully implemented)
    "run_filt",
    # Potentially more from other modules if they become central & frequently used
    # "create_model", "ekf", "crlb", "get_map", etc.
]