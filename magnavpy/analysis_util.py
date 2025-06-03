# -*- coding: utf-8 -*-
"""
Utility functions for data processing and analysis, translated from MagNav.jl/src/analysis_util.jl.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import polynomial_kernel # For KRR
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Dict, Any, Union, Optional

from .signal_util import linreg_matrix, get_bpf_sos, bpf_data, bpf_data_inplace
from .tolles_lawson import create_TL_A
from .map_utils import get_map_val, get_map # Import get_map_val and get_map from map_utils
from .nav_utils import get_step # Import get_step from nav_utils
from .magnav import (
    # create_TL_A, # Removed as it's now imported from .tolles_lawson
    R_EARTH, E_EARTH, NUM_MAG_MAX, SILENT_DEBUG, XYZ, MapS,
    Traj, INS, XYZ0, XYZ1, XYZ20, XYZ21, # Specific XYZ types
    LinCompParams, NNCompParams, TempParams, # Parameter structs
    # Functions that were in MagNav.jl and are assumed to be in magnav.py or other modules
    # get_map, get_step, # Moved to map_utils and nav_utils respectively
    # For type hinting if needed, though specific XYZ types are better
)
from .common_types import MagV # Import MagV from common_types
from .dcm_util import dcm2euler, euler2dcm # Import dcm functions from dcm_util
from .fdm_util import fdm # Import fdm function from fdm_util
from .common_types import MagV # Import MagV from common_types
from .dcm_util import dcm2euler, euler2dcm # Import dcm functions from dcm_util

# Placeholder for IGRF functionality, replace with a proper IGRF library
def pyigrf_calc(date_decimal_year: float, alt_km: float, lat_deg: float, lon_deg: float) -> Tuple[float, float, float, float, float, float, float]:
    """Placeholder for IGRF calculation. Returns (D, I, H, X, Y, Z, F)."""
    # print("Warning: Using placeholder IGRF calculation.")
    # Example: return (0,0,30000,30000,0,-45000,50000) # D, I, H, X, Y, Z, F
    # For a more realistic placeholder, one might use a fixed value or simple model.
    # This needs to be replaced with a call to a proper IGRF library like pyIGRF or geomagpy.
    # The Julia code `igrf()` returns [Bx, By, Bz] (North, East, Down) in nT.
    # This placeholder should be adapted to return that.
    # For now, returning dummy values that might somewhat match the expected structure [X,Y,Z]
    return 30000.0, 0.0, -45000.0 # Placeholder for X, Y, Z (N, E, D)

# Placeholder for field_check, assuming it returns list of attribute names of a certain type
def field_check(obj: Any, type_to_check: type) -> List[str]:
    """
    Checks attributes of an object that are instances of type_to_check.
    Returns a list of names of such attributes.
    """
    # A more robust implementation might be needed depending on actual Julia behavior
    # This is a simplified version based on common usage patterns.
    # print(f"Warning: Using placeholder field_check for type {type_to_check}.")
    checked_fields = []
    if hasattr(obj, '__dict__'): # For standard class instances
        for attr_name, attr_value in vars(obj).items():
            if isinstance(attr_value, type_to_check):
                checked_fields.append(attr_name)
    elif hasattr(obj, '__slots__'): # For classes with __slots__
         for slot_name in obj.__slots__:
            if hasattr(obj, slot_name):
                attr_value = getattr(obj, slot_name)
                if isinstance(attr_value, type_to_check):
                    checked_fields.append(slot_name)
    # This might need to be adapted if XYZ types are dataclasses and MagV fields are known
    if isinstance(obj, (XYZ0, XYZ1, XYZ20, XYZ21)) and type_to_check == MagV:
        # Manually list known MagV fields for XYZ types if introspection is tricky
        known_magv_fields = ["flux_a", "flux_b", "flux_c", "flux_d"]
        for f_name in known_magv_fields:
            if hasattr(obj, f_name) and isinstance(getattr(obj, f_name), MagV):
                if f_name not in checked_fields: # Avoid duplicates if already found
                    checked_fields.append(f_name)
    return checked_fields


# Constants from MagNav.jl (if not already in magnav.py)
# r_earth = R_EARTH (already imported)
# e_earth = E_EARTH (already imported)

def dn2dlat(dn: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert north-south position (northing) difference to latitude difference.

    Args:
        dn: north-south position (northing) difference [m]
        lat: nominal latitude [rad]

    Returns:
        dlat: latitude difference [rad]
    """
    dlat = dn * np.sqrt(1 - (E_EARTH * np.sin(lat))**2) / R_EARTH
    return dlat

def de2dlon(de: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert east-west position (easting) difference to longitude difference.

    Args:
        de: east-west position (easting) difference [m]
        lat: nominal latitude [rad]

    Returns:
        dlon: longitude difference [rad]
    """
    dlon = de * np.sqrt(1 - (E_EARTH * np.sin(lat))**2) / R_EARTH / np.cos(lat)
    return dlon

def dlat2dn(dlat: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert latitude difference to north-south position (northing) difference.

    Args:
        dlat: latitude difference [rad]
        lat: nominal latitude [rad]

    Returns:
        dn: north-south position (northing) difference [m]
    """
    dn = dlat / np.sqrt(1 - (E_EARTH * np.sin(lat))**2) * R_EARTH
    return dn

def dlon2de(dlon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert longitude difference to east-west position (easting) difference.

    Args:
        dlon: longitude difference [rad]
        lat: nominal latitude [rad]

    Returns:
        de: east-west position (easting) difference [m]
    """
    de = dlon / np.sqrt(1 - (E_EARTH * np.sin(lat))**2) * R_EARTH * np.cos(lat)
    return de


def get_ind_segs(line_all: np.ndarray, lines: Union[int, List[int], str] = "all") -> List[np.ndarray]:
    """
    Get all data segment indices for specified flight line(s).

    Args:
        line_all: Vector of line numbers for all data points.
        lines: Flight line number(s) to get data segment indices for.
               Can be an integer, list of integers, or "all".

    Returns:
        ind_segs: List of boolean arrays, each indicating a segment for the specified lines.
    """
    if isinstance(lines, str) and lines.lower() == "all":
        unique_lines = np.unique(line_all)
    elif isinstance(lines, int):
        unique_lines = np.array([lines])
    elif isinstance(lines, list):
        unique_lines = np.array(lines)
    else:
        raise ValueError("lines must be an int, list of ints, or 'all'")

    ind_segs = []
    for l_val in unique_lines:
        if np.any(line_all == l_val): # Check if line exists
            ind_segs.append(line_all == l_val)
    return ind_segs


def get_segments(line_all: np.ndarray, lines: Union[int, List[int], str] = "all",
                 N_min: int = 3, N_buffer: int = 0) -> Tuple[List[np.ndarray], List[int]]:
    """
    Get all data segments for specified flight line(s).

    Args:
        line_all: Vector of line numbers for all data points.
        lines: Flight line number(s) to get data segments for.
        N_min: Minimum number of data points for a segment.
        N_buffer: Number of data points to remove from beginning and end of each segment.

    Returns:
        ind_segs_out: List of boolean arrays, each indicating a valid segment.
        lines_out: List of line numbers corresponding to each segment in ind_segs_out.
    """
    if isinstance(lines, str) and lines.lower() == "all":
        unique_lines_to_process = np.unique(line_all)
    elif isinstance(lines, int):
        unique_lines_to_process = np.array([lines])
    elif isinstance(lines, list):
        unique_lines_to_process = np.array(lines)
    else:
        raise ValueError("lines must be an int, list of ints, or 'all'")

    ind_segs_out = []
    lines_out = []

    for l_val in unique_lines_to_process:
        line_indices = np.where(line_all == l_val)[0]
        if len(line_indices) == 0:
            continue

        # Find contiguous segments within this line
        # (Assumes line_indices are sorted, which np.where provides)
        diffs = np.diff(line_indices)
        segment_breaks = np.where(diffs > 1)[0]
        
        start_idx = 0
        current_segments = []
        if len(segment_breaks) == 0: # Single contiguous segment for this line
            current_segments.append(line_indices)
        else:
            current_segments.append(line_indices[start_idx : segment_breaks[0]+1])
            for i in range(len(segment_breaks) - 1):
                current_segments.append(line_indices[segment_breaks[i]+1 : segment_breaks[i+1]+1])
            current_segments.append(line_indices[segment_breaks[-1]+1 :])

        for seg_indices in current_segments:
            N_seg = len(seg_indices)
            if N_seg >= N_min + 2 * N_buffer:
                valid_indices_in_segment = seg_indices[N_buffer : N_seg - N_buffer]
                if len(valid_indices_in_segment) >= N_min:
                    seg_bool = np.zeros(len(line_all), dtype=bool)
                    seg_bool[valid_indices_in_segment] = True
                    ind_segs_out.append(seg_bool)
                    lines_out.append(l_val)
                    
    return ind_segs_out, lines_out


def get_ind(
    line_all: np.ndarray,
    lines: Union[int, List[int], str] = "all",
    N_total: Optional[int] = None, # Total number of points, if line_all is not full length
    l_window: int = -1,
    N_max: int = -1,
    N_min: int = 3,
    N_seg_min: int = 3, # Min points per contiguous segment within a line before N_buffer
    N_buffer: int = 0,
    mod_val: int = 1,
    mod_rem: Union[int, List[int]] = 0,
    # KRR related parameters (placeholders for now)
    # valid_frac_thresh: float = 0.9, line_val_perc: float = 0.05,
    # krr_cov: bool = False, krr_pca: bool = False, krr_gamma: float = 1.0,
    # krr_lambda: float = 1e-6, krr_poly_deg: int = 3, krr_poly_coef0: float = 1.0,
    # krr_mean_max: float = np.inf, krr_std_max: float = np.inf,
    # krr_norm: bool = True, krr_type: str = "scalar", # or "vector"
    silent: bool = False
) -> np.ndarray:
    """
    Get selected data indices based on various criteria.
    This is a partial port of MagNav.jl's get_ind. KRR outlier rejection is not yet implemented.

    Args:
        line_all: Vector of line numbers for all data points.
        lines: Flight line number(s) to get data for. Can be int, list of ints, or "all".
        N_total: Total number of points if line_all is a subset. Defaults to len(line_all).
        l_window: Window length. If > 0, select points within windows around line transitions.
        N_max: Maximum number of data points to select.
        N_min: Minimum number of data points to select.
        N_seg_min: Minimum number of data points for a contiguous segment within a line.
        N_buffer: Number of data points to remove from beginning and end of each segment.
        mod_val: Select every `mod_val`-th data point.
        mod_rem: Remainder(s) for `mod_val` selection. Can be int or list of ints.
        silent: Suppress print statements.

    Returns:
        ind_final: Boolean array of selected data indices.
    """
    if N_total is None:
        N_total = len(line_all)
    
    ind_lines = np.zeros(N_total, dtype=bool)

    if isinstance(lines, str) and lines.lower() == "all":
        selected_line_numbers = np.unique(line_all)
    elif isinstance(lines, int):
        selected_line_numbers = [lines]
    elif isinstance(lines, list):
        selected_line_numbers = lines
    else:
        raise ValueError("lines must be an int, list of ints, or 'all'")

    for l_val in selected_line_numbers:
        ind_this_line = (line_all == l_val)
        
        # Handle N_seg_min and N_buffer for contiguous segments within this line
        line_abs_indices = np.where(ind_this_line)[0]
        if len(line_abs_indices) == 0:
            continue

        diffs = np.diff(line_abs_indices)
        segment_breaks = np.where(diffs > 1)[0]
        
        current_pos = 0
        processed_indices_for_line = np.zeros(N_total, dtype=bool)

        for i in range(len(segment_breaks) + 1):
            start_idx_in_line_abs = current_pos
            if i < len(segment_breaks):
                end_idx_in_line_abs = segment_breaks[i]
            else:
                end_idx_in_line_abs = len(line_abs_indices) -1
            
            current_segment_abs_indices = line_abs_indices[start_idx_in_line_abs : end_idx_in_line_abs + 1]
            current_pos = end_idx_in_line_abs + 1

            if len(current_segment_abs_indices) >= N_seg_min:
                if N_buffer > 0:
                    if len(current_segment_abs_indices) > 2 * N_buffer:
                        buffered_segment_abs_indices = current_segment_abs_indices[N_buffer : -N_buffer]
                        processed_indices_for_line[buffered_segment_abs_indices] = True
                    # else segment too short after buffering, effectively skipped
                else:
                    processed_indices_for_line[current_segment_abs_indices] = True
        ind_lines = ind_lines | processed_indices_for_line

    ind_final = ind_lines.copy()

    # l_window logic (select points around line transitions or within lines)
    if l_window > 0 and np.any(ind_final):
        ind_window = np.zeros(N_total, dtype=bool)
        line_changes = np.where(np.diff(line_all) != 0)[0] + 1
        
        # Points considered "on-line" are those selected by ind_final
        on_line_indices = np.where(ind_final)[0]

        if len(on_line_indices) > 0:
            for idx in on_line_indices:
                start = max(0, idx - l_window // 2)
                end = min(N_total, idx + l_window // 2 + (l_window % 2)) # +1 if odd
                ind_window[start:end] = True
            ind_final = ind_final & ind_window # Intersect with original line selections
        else: # If no lines selected initially, l_window might not apply or select all if not careful
            ind_final = np.zeros(N_total, dtype=bool)


    # mod_val / mod_rem
    if mod_val > 1 and np.any(ind_final):
        ind_mod = np.zeros(N_total, dtype=bool)
        selected_indices_abs = np.where(ind_final)[0]
        
        if isinstance(mod_rem, int):
            mod_rem = [mod_rem]
        
        for rem_val in mod_rem:
            # Apply modulo on the indices *within the currently selected set*
            # This matches Julia: `(eachindex(ind_all)[ind_all] .% mod_val .== mod_rem)`
            # However, Julia's `eachindex` is 1-based. Python is 0-based.
            # A direct translation of the Julia logic:
            #   `idx_where_ind_final_is_true = np.where(ind_final)[0]`
            #   `condition = np.isin((idx_where_ind_final_is_true % mod_val), mod_rem)`
            #   `ind_mod[idx_where_ind_final_is_true[condition]] = True`
            # Simpler: apply to all indices and then AND with ind_final
            temp_mod_selector = np.zeros(N_total, dtype=bool)
            for i in range(N_total):
                if i % mod_val == rem_val: # 0-based index modulo
                    temp_mod_selector[i] = True
            ind_mod = ind_mod | temp_mod_selector
            
        ind_final = ind_final & ind_mod

    # TODO: KRR outlier rejection from Julia:
    # This involves setting up features (potentially using get_x),
    # training KRR, predicting, and removing points with high error.
    # if !silent && (krr_cov || krr_pca || krr_type != "none")
    #    println("KRR outlier rejection not yet implemented in Python get_ind.")
    # end

    # N_max: downsample if too many points
    if N_max > 0 and np.sum(ind_final) > N_max:
        selected_indices_abs = np.where(ind_final)[0]
        step = int(np.ceil(len(selected_indices_abs) / N_max))
        downsampled_indices_abs = selected_indices_abs[::step]
        
        ind_final.fill(False)
        ind_final[downsampled_indices_abs] = True

    # N_min: ensure minimum number of points
    if np.sum(ind_final) < N_min:
        if not silent:
            print(f"Warning: Fewer than N_min ({N_min}) points selected ({np.sum(ind_final)}). Returning empty selection.")
        ind_final.fill(False)
        
    return ind_final


def linreg_vector(y: np.ndarray, lambda_ridge: float = 0) -> np.ndarray:
    """
    Linear regression to determine best fit line for x = 0, 1, ..., N-1.

    Args:
        y: length-N observed data vector.
        lambda_ridge: (optional) ridge parameter.

    Returns:
        coef: length-2 vector of linear regression coefficients [intercept, slope].
    """
    N = len(y)
    x_values = np.arange(N)
    x_matrix = np.vstack([np.ones(N), x_values]).T  # Intercept and slope terms
    coef = linreg_matrix(y, x_matrix, lambda_ridge=lambda_ridge)
    return coef

def detrend(y: np.ndarray, x_input: Optional[np.ndarray] = None, lambda_ridge: float = 0, mean_only: bool = False) -> np.ndarray:
    """
    Detrend signal (remove mean and optionally slope).

    Args:
        y: length-N observed data vector.
        x_input: (optional) N x Nf input data matrix for regression-based detrending.
                 If None, simple linear detrend against indices is performed (if not mean_only).
        lambda_ridge: (optional) ridge parameter for regression.
        mean_only: (optional) if true, only remove mean (not slope).

    Returns:
        y_detrended: length-N observed data vector, detrended.
    """
    y_out = y.copy()
    if mean_only:
        y_out = y_out - np.mean(y_out)
    else:
        if x_input is None:
            # Detrend against indices [0, 1, ..., N-1]
            N = len(y_out)
            x_reg = np.vstack([np.ones(N), np.arange(N)]).T
        else:
            # Detrend against provided x_input, adding an intercept column
            if x_input.ndim == 1:
                x_input = x_input.reshape(-1,1)
            x_reg = np.hstack([np.ones((len(y_out), 1)), x_input])
        
        coef = linreg_matrix(y_out, x_reg, lambda_ridge=lambda_ridge)
        y_out = y_out - (x_reg @ coef)
    return y_out


def rmse(error: np.ndarray) -> float:
    """
    Calculate root mean squared error (RMSE).

    Args:
        error: Error vector.

    Returns:
        Root mean squared error.
    """
    return np.sqrt(np.mean(error**2))


def mae(error: np.ndarray) -> float:
    """
    Calculate mean absolute error (MAE).

    Args:
        error: Error vector.

    Returns:
        Mean absolute error.
    """
    return np.mean(np.abs(error))


def std_err(error: np.ndarray, remove_mean: bool = False) -> float:
    """
    Calculate standard deviation of error.

    Args:
        error: Error vector.
        remove_mean: If true, remove mean from error before calculating std.

    Returns:
        Standard deviation of error.
    """
    if remove_mean:
        error = error - np.mean(error)
    return np.std(error)


def downsample(data: Union[np.ndarray, pd.DataFrame], n_max: int = 1000) -> Union[np.ndarray, pd.DataFrame]:
    """
    Downsample data to n_max (or fewer) data points.

    Args:
        data: Data vector or matrix (rows are samples).
        n_max: (optional) maximum number of data points.

    Returns:
        data_downsampled: Downsampled data.
    """
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()
        is_df = True
    elif isinstance(data, np.ndarray):
        data_np = data
        is_df = False
    else:
        raise TypeError("Input data must be a NumPy array or Pandas DataFrame.")

    n_samples = data_np.shape[0]

    if n_samples <= n_max:
        return data.copy() # Return a copy to match deepcopy behavior in Julia
    else:
        step = int(np.ceil(n_samples / n_max))
        idx = np.arange(0, n_samples, step)
        
        if data_np.ndim == 1:
            res_np = data_np[idx]
        elif data_np.ndim == 2:
            res_np = data_np[idx, :]
        else:
            raise ValueError("Data must be 1D or 2D")

        if is_df:
            return pd.DataFrame(res_np, columns=data.columns)
        else:
            return res_np

# Note: The `get_x`, `get_y`, `get_Axy` functions are very complex and rely on many
# other functions (create_TL_A, dcm2euler, fdm, etc.) and specific data structures (XYZ).
# A full translation requires these dependencies to be available.
# The following are structural translations with placeholders/assumptions for dependencies.

def get_x(xyz: XYZ,
          ind: Optional[np.ndarray] = None,
          features_setup: List[str] = None, # Default: ['mag_1_uc', 'TL_A_flux_a']
          features_no_norm: List[str] = None, # Default: []
          terms: List[str] = None, # Default: ['permanent', 'induced', 'eddy']
          sub_diurnal: bool = False,
          sub_igrf: bool = False,
          bpf_mag_data: bool = False # Renamed from bpf_mag to avoid conflict
         ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Get x data matrix.

    Args:
        xyz: XYZ flight data object.
        ind: Selected data indices (boolean array). If None, all data used.
        features_setup: List of features to include. Symbols like :mag_1_uc become "mag_1_uc".
        features_no_norm: List of features to not normalize.
        terms: Tolles-Lawson terms to use.
        sub_diurnal: If true, subtract diurnal from scalar magnetometer measurements.
        sub_igrf: If true, subtract IGRF from scalar magnetometer measurements.
        bpf_mag_data: If true, bpf scalar magnetometer measurements.

    Returns:
        x_matrix: N x Nf data matrix.
        no_norm_mask: Boolean array indicating features not to normalize.
        feature_names_out: List of final feature names.
        l_segs: Lengths of unique line segments.
    """
    if features_setup is None:
        features_setup = ["mag_1_uc", "TL_A_flux_a"]
    if features_no_norm is None:
        features_no_norm = []
    if terms is None:
        terms = ["permanent", "induced", "eddy"]

    if ind is None:
        ind = np.ones(xyz.traj.N, dtype=bool)

    line_data = xyz.line[ind]
    N = len(line_data)
    assert N > 2, "ind must contain at least 3 data points"

    # Using a dictionary to store intermediate feature arrays
    d: Dict[str, np.ndarray] = {}
    
    # Tolles-Lawson A matrix components
    # Assuming field_check and create_TL_A are available
    # MagV fields in XYZ objects (e.g., flux_a, flux_b)
    magv_field_names = field_check(xyz, MagV) # Placeholder behavior
    for use_vec_str in magv_field_names:
        mag_v_data = getattr(xyz, use_vec_str) # This is a MagV object
        # Assuming create_TL_A handles indexing internally using the `ind` boolean array
        A_matrix = create_TL_A(mag_v_data, ind, terms=terms)
        if A_matrix.shape[0] == N: # Ensure A_matrix has the correct number of rows
            d[f"TL_A_{use_vec_str}"] = A_matrix
        else:
            # This case should ideally not happen if create_TL_A and ind are correct
            # Fallback or error, for now creating an empty array of correct row size
            # to prevent downstream errors if A_matrix is unexpectedly shaped.
            # A more robust solution would be to ensure create_TL_A always returns N rows
            # or handle the feature absence gracefully in the assembly loop.
            if A_matrix.ndim == 2 and A_matrix.shape[1] > 0:
                 print(f"Warning: TL_A_{use_vec_str} matrix from create_TL_A has {A_matrix.shape[0]} rows, expected {N}. Using zeros.")
                 d[f"TL_A_{use_vec_str}"] = np.zeros((N, A_matrix.shape[1]))
            else: # If A_matrix is empty or not 2D
                 print(f"Warning: TL_A_{use_vec_str} matrix from create_TL_A is empty or not 2D. Skipping feature.")
                 # To skip, ensure it's not added or handle in feature assembly loop.
                 # For now, let's assign an empty array that will have 0 columns if accessed.
                 d[f"TL_A_{use_vec_str}"] = np.empty((N,0))


    # Subtraction term for diurnal/IGRF
    sub = np.zeros(N, dtype=float)
    if sub_diurnal and hasattr(xyz, 'diurnal'):
        sub += xyz.diurnal[ind]
    if sub_igrf and hasattr(xyz, 'igrf'):
        sub += xyz.igrf[ind]

    # Scalar magnetometer features
    # Get all fields of xyz
    xyz_fields = [attr for attr in dir(xyz) if not callable(getattr(xyz, attr)) and not attr.startswith("__")]
    
    mags_c_names = [f"mag_{i}_c" for i in range(1, NUM_MAG_MAX + 1) if f"mag_{i}_c" in xyz_fields]
    mags_uc_names = [f"mag_{i}_uc" for i in range(1, NUM_MAG_MAX + 1) if f"mag_{i}_uc" in xyz_fields]
    mags_all_names = mags_c_names + mags_uc_names

    for mag_name in mags_all_names:
        val = getattr(xyz, mag_name)[ind] - sub
        if bpf_mag_data:
            # Assuming bpf_data takes 1D array and returns 1D array
            val = bpf_data(val) # Default BPF params might be needed
        d[mag_name] = val.reshape(-1,1) # Ensure 2D for hstack later

        # Derivatives
        if hasattr(xyz.traj, 'dt'):
            dt = xyz.traj.dt
            flat_val = val.flatten() # fdm expects 1D array
            d[f"{mag_name}_dot"] = fdm(flat_val, dt=dt, scheme="central").reshape(-1,1)
            d[f"{mag_name}_dot4"] = fdm(flat_val, dt=dt, scheme="fourth").reshape(-1,1)
        else: # Fallback if dt is not available, though fdm might default dt=1
            print(f"Warning: xyz.traj.dt not found for derivative calculation of {mag_name}. Derivatives may be incorrect.")
            d[f"{mag_name}_dot"] = np.zeros_like(val).reshape(-1,1)
            d[f"{mag_name}_dot4"] = np.zeros_like(val).reshape(-1,1)

        # Lags (Julia: val[[1:i;1:end-i]])
        # This corresponds to vcat(val[1:i], val[1:end-i]) in Julia 1-based indexing
        current_mag_scalar_data_for_lag = d[mag_name].flatten() # Use the (potentially BPF'd) data
        for i_j_lag in range(1, 4): # Julia i = 1, 2, 3
            if len(current_mag_scalar_data_for_lag) >= i_j_lag and len(current_mag_scalar_data_for_lag) - i_j_lag >= 0 :
                # Python 0-indexed slices:
                # Julia val[1:i] -> Python val[0:i_j_lag]
                # Julia val[1:end-i] -> Python val[0:len(val)-i_j_lag]
                part1 = current_mag_scalar_data_for_lag[0:i_j_lag]
                part2 = current_mag_scalar_data_for_lag[0:len(current_mag_scalar_data_for_lag)-i_j_lag]
                if len(part1) + len(part2) == len(current_mag_scalar_data_for_lag): # Ensure concat results in same length
                    lagged_feature = np.concatenate((part1, part2))
                    d[f"{mag_name}_lag_{i_j_lag}"] = lagged_feature.reshape(-1,1)
                else: # Should not happen if logic is correct and data is long enough
                    d[f"{mag_name}_lag_{i_j_lag}"] = np.zeros_like(current_mag_scalar_data_for_lag).reshape(-1,1)
            else:
                # Data too short for this lag combination
                d[f"{mag_name}_lag_{i_j_lag}"] = np.zeros_like(current_mag_scalar_data_for_lag).reshape(-1,1)

    # Differences between magnetometers
    for i, mag1_name in enumerate(mags_c_names):
        for j, mag2_name in enumerate(mags_c_names):
            # if i == j: continue # Typically diff with self is not a feature
            val_diff = (getattr(xyz, mag1_name) - getattr(xyz, mag2_name))[ind]
            d[f"mag_{i+1}_{j+1}_c"] = val_diff.reshape(-1,1)

    for i, mag1_name in enumerate(mags_uc_names):
        for j, mag2_name in enumerate(mags_uc_names):
            # if i == j: continue
            val_diff = (getattr(xyz, mag1_name) - getattr(xyz, mag2_name))[ind]
            d[f"mag_{i+1}_{j+1}_uc"] = val_diff.reshape(-1,1)
            
    # Attitude features (dcm2euler, euler2dcm are external)
    if hasattr(xyz.ins, 'Cnb'):
        Cnb_ind = xyz.ins.Cnb[ind, :, :] # Apply boolean index to the first (N) axis
        # Assuming dcm2euler takes (3,3,N) and returns (N,), (N,), (N,) for roll, pitch, yaw
        roll, pitch, yaw = dcm2euler(Cnb_ind, order='body2nav') # Use appropriate order string

        # Assuming euler2dcm takes (N,) arrays and returns (N,3,3)
        dcm_nav2body_N33 = euler2dcm(roll, pitch, yaw, order='zyx') # Assuming 'zyx' is the correct sequence for nav to body

        # d["dcm"] feature: Nx9 matrix, each row is a flattened (Fortran order) 3x3 DCM
        d["dcm"] = np.array([m.flatten('F') for m in dcm_nav2body_N33])

        # Individual DCM components (dcm_1 to dcm_9)
        # These correspond to column-major traversal of each 3x3 matrix
        d["dcm_1"] = dcm_nav2body_N33[:, 0, 0].reshape(-1,1) # C11
        d["dcm_2"] = dcm_nav2body_N33[:, 1, 0].reshape(-1,1) # C21
        d["dcm_3"] = dcm_nav2body_N33[:, 2, 0].reshape(-1,1) # C31
        d["dcm_4"] = dcm_nav2body_N33[:, 0, 1].reshape(-1,1) # C12
        d["dcm_5"] = dcm_nav2body_N33[:, 1, 1].reshape(-1,1) # C22
        d["dcm_6"] = dcm_nav2body_N33[:, 2, 1].reshape(-1,1) # C32
        d["dcm_7"] = dcm_nav2body_N33[:, 0, 2].reshape(-1,1) # C13
        d["dcm_8"] = dcm_nav2body_N33[:, 1, 2].reshape(-1,1) # C23
        d["dcm_9"] = dcm_nav2body_N33[:, 2, 2].reshape(-1,1) # C33

        # Trigonometric combinations
        cos_roll, sin_roll = np.cos(roll), np.sin(roll)
        cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        d["crcy"]   = (cos_roll * cos_yaw).reshape(-1,1)
        d["cpcy"]   = (cos_pitch * cos_yaw).reshape(-1,1)
        d["crsy"]   = (cos_roll * sin_yaw).reshape(-1,1)
        d["cpsy"]   = (cos_pitch * sin_yaw).reshape(-1,1)
        d["srcy"]   = (sin_roll * cos_yaw).reshape(-1,1)
        d["spcy"]   = (sin_pitch * cos_yaw).reshape(-1,1)
        d["srsy"]   = (sin_roll * sin_yaw).reshape(-1,1)
        d["spsy"]   = (sin_pitch * sin_yaw).reshape(-1,1)
        d["crcpcy"] = (cos_roll * cos_pitch * cos_yaw).reshape(-1,1)
        d["srcpcy"] = (sin_roll * cos_pitch * cos_yaw).reshape(-1,1)
        d["crspcy"] = (cos_roll * sin_pitch * cos_yaw).reshape(-1,1)
        d["srspcy"] = (sin_roll * sin_pitch * cos_yaw).reshape(-1,1)
        d["crcpsy"] = (cos_roll * cos_pitch * sin_yaw).reshape(-1,1)
        d["srcpsy"] = (sin_roll * cos_pitch * sin_yaw).reshape(-1,1)
        d["crspsy"] = (cos_roll * sin_pitch * sin_yaw).reshape(-1,1)
        d["srspsy"] = (sin_roll * sin_pitch * sin_yaw).reshape(-1,1)
        d["crcp"]   = (cos_roll * cos_pitch).reshape(-1,1)
        d["srcp"]   = (sin_roll * cos_pitch).reshape(-1,1)
        d["crsp"]   = (cos_roll * sin_pitch).reshape(-1,1)
        d["srsp"]   = (sin_roll * sin_pitch).reshape(-1,1)

        # Derivatives of roll, pitch, yaw and their sin/cos
        if hasattr(xyz.traj, 'dt'):
            dt = xyz.traj.dt
            for rpy_val, rpy_name_str in zip([roll, pitch, yaw], ["roll", "pitch", "yaw"]):
                rpy_val_flat = rpy_val.flatten()
                d[f"{rpy_name_str}_fdm"] = fdm(rpy_val_flat, dt=dt, scheme="central").reshape(-1,1)
                sin_rpy_val = np.sin(rpy_val_flat)
                cos_rpy_val = np.cos(rpy_val_flat)
                d[f"{rpy_name_str}_sin"] = sin_rpy_val.reshape(-1,1)
                d[f"{rpy_name_str}_cos"] = cos_rpy_val.reshape(-1,1)
                d[f"{rpy_name_str}_sin_fdm"] = fdm(sin_rpy_val, dt=dt, scheme="central").reshape(-1,1)
                d[f"{rpy_name_str}_cos_fdm"] = fdm(cos_rpy_val, dt=dt, scheme="central").reshape(-1,1)
        else:
            print(f"Warning: xyz.traj.dt not found for roll/pitch/yaw derivative calculations. Skipping these features.")
            for rpy_name_str in ["roll", "pitch", "yaw"]:
                d[f"{rpy_name_str}_fdm"] = np.zeros((N,1))
                d[f"{rpy_name_str}_sin_fdm"] = np.zeros((N,1))
                d[f"{rpy_name_str}_cos_fdm"] = np.zeros((N,1))
                # _sin and _cos features are still added if roll/pitch/yaw are valid
                if rpy_name_str == "roll": rpy_val_flat = roll.flatten()
                elif rpy_name_str == "pitch": rpy_val_flat = pitch.flatten()
                else: rpy_val_flat = yaw.flatten()
                d[f"{rpy_name_str}_sin"] = np.sin(rpy_val_flat).reshape(-1,1)
                d[f"{rpy_name_str}_cos"] = np.cos(rpy_val_flat).reshape(-1,1)

    # Low-pass filter current sensors
    if N > 12 and hasattr(xyz.traj, 'dt'):
        fs_lpf = 1.0 / xyz.traj.dt
        sos_lpf = get_bpf_sos(pass1=0.0, pass2=0.2, fs=fs_lpf) # Lowpass
        current_sensor_fields = ["cur_strb", "cur_outpwr", "cur_ac_hi", "cur_ac_lo", "cur_com_1"]
        for cs_field in current_sensor_fields:
            if hasattr(xyz, cs_field):
                cs_val = getattr(xyz, cs_field)[ind]
                d[f"lpf_{cs_field}"] = bpf_data(cs_val, sos=sos_lpf).reshape(-1,1)
                
    # INS data
    if hasattr(xyz.ins, 'lat'):
        d["ins_lat"] = xyz.ins.lat[ind].reshape(-1,1)
        d["ins_lon"] = xyz.ins.lon[ind].reshape(-1,1)
        d["ins_alt"] = xyz.ins.alt[ind].reshape(-1,1)

    # Assemble final x_matrix and feature names
    x_cols_list = []
    no_norm_list = []
    feature_names_out = []

    for f_name_setup in features_setup:
        f_name_base = f_name_setup.lstrip(':') # Remove Julia symbol colon if present
        
        u: Optional[np.ndarray] = None
        if f_name_base in d:
            u = d[f_name_base]
        elif hasattr(xyz, f_name_base):
            attr_val = getattr(xyz, f_name_base)
            if isinstance(attr_val, np.ndarray):
                u = attr_val[ind]
                if u.ndim == 1: u = u.reshape(-1,1)
            # Could also handle MagV here if features_setup can refer to them directly
            # else: print(f"Warning: XYZ attribute {f_name_base} is not an ndarray.")
        
        if u is None:
            raise ValueError(f"Feature '{f_name_base}' is invalid or not found.")

        if np.isnan(np.sum(u)):
            raise ValueError(f"Feature '{f_name_base}' contains NaNs.")

        num_sub_features = u.shape[1]
        is_no_norm_feature = (f_name_base in features_no_norm) or (f_name_setup in features_no_norm)
        
        x_cols_list.append(u)
        no_norm_list.extend([is_no_norm_feature] * num_sub_features)
        
        if num_sub_features > 1:
            feature_names_out.extend([f"{f_name_base}_{i}" for i in range(num_sub_features)])
        else:
            feature_names_out.append(f_name_base)
            
    if not x_cols_list: # If no features were added
        x_matrix = np.empty((N, 0))
    else:
        x_matrix = np.hstack(x_cols_list)
        
    no_norm_mask = np.array(no_norm_list, dtype=bool)

    # Segment lengths
    unique_lines, counts = np.unique(line_data, return_counts=True)
    l_segs = counts # Order might differ from Julia if unique doesn't sort same way.
                    # Julia's `[sum(line .== l) for l in unique(line)]` preserves order of `unique(line)`.
                    # `np.unique` sorts `unique_lines`. If original order of appearance matters, more work needed.
                    # For now, assuming sorted unique lines is acceptable.

    return x_matrix, no_norm_mask, feature_names_out, l_segs


def get_x_multiple_xyz(xyz_vec: List[XYZ],
                       ind_vec: List[np.ndarray],
                       **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Get x data matrix from multiple XYZ flight data objects.
    """
    if not xyz_vec or not ind_vec or len(xyz_vec) != len(ind_vec):
        raise ValueError("xyz_vec and ind_vec must be non-empty and of the same length.")

    x_list, l_segs_list = [], []
    no_norm_mask_out, features_out = None, None

    for i, (xyz_item, ind_item) in enumerate(zip(xyz_vec, ind_vec)):
        x_i, no_norm_i, features_i, l_segs_i = get_x(xyz_item, ind_item, **kwargs)
        x_list.append(x_i)
        l_segs_list.append(l_segs_i)
        if i == 0:
            no_norm_mask_out = no_norm_i
            features_out = features_i
        else: # Sanity check
            if not np.array_equal(no_norm_mask_out, no_norm_i) or features_out != features_i:
                raise ValueError("Feature set or no_norm mask mismatch between XYZ items.")
    
    x_matrix_combined = np.vstack(x_list)
    l_segs_combined = np.concatenate(l_segs_list)
    
    return x_matrix_combined, no_norm_mask_out, features_out, l_segs_combined


def get_x_from_dataframes(lines: Union[int, List[int]],
                          df_line: pd.DataFrame,
                          df_flight: pd.DataFrame,
                          features_setup: List[str] = None,
                          features_no_norm: List[str] = None,
                          terms: List[str] = None,
                          sub_diurnal: bool = False,
                          sub_igrf: bool = False,
                          bpf_mag_data: bool = False,
                          reorient_vec: bool = False, # Passed to get_XYZ
                          l_window: int = -1, # Passed to get_ind
                          silent: bool = True
                         ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Get x data matrix from multiple flight lines, possibly multiple flights, using DataFrames.
    """
    if features_setup is None: features_setup = ["mag_1_uc", "TL_A_flux_a"]
    if features_no_norm is None: features_no_norm = []
    if terms is None: terms = ["permanent", "induced", "eddy"]

    if isinstance(lines, int):
        lines = [lines]

    # Filter lines present in df_line
    valid_lines = [l for l in lines if l in df_line['line'].values]
    if not silent and len(valid_lines) < len(lines):
        skipped_lines = set(lines) - set(valid_lines)
        print(f"Info: Lines {skipped_lines} not in df_line, skipping.")
    
    if not valid_lines:
        # Return empty structures if no valid lines
        return np.empty((0,0)), np.array([], dtype=bool), [], np.array([], dtype=int)

    if len(valid_lines) != len(set(valid_lines)): # Check duplicates
        raise ValueError(f"Duplicate lines found in {valid_lines}")

    # Check flight data compatibility (xyz_set)
    # This logic needs careful translation from Julia's DataFrame indexing
    # flights_for_lines = [df_line.loc[df_line['line'] == l, 'flight'].iloc[0] for l in valid_lines]
    # xyz_sets = [df_flight.loc[df_flight['flight'] == f, 'xyz_set'].iloc[0] for f in flights_for_lines]
    # if len(set(xyz_sets)) > 1:
    #     raise ValueError("Incompatible xyz_sets in df_flight for the selected lines.")
    # Simplified check:
    line_details = df_line[df_line['line'].isin(valid_lines)]
    flight_names_for_lines = line_details['flight'].unique()
    flight_details = df_flight[df_flight['flight'].isin(flight_names_for_lines)]
    if flight_details['xyz_set'].nunique() > 1:
         raise ValueError("Incompatible xyz_sets in df_flight for the selected lines.")


    x_list, l_segs_list_for_output = [], []
    current_xyz: Optional[XYZ] = None
    current_flight_name: Optional[str] = None
    no_norm_mask_out, features_out = None, None
    
    # Assuming get_XYZ is a function that loads XYZ data for a flight name
    # from .get_XYZ import get_XYZ # This would be the import

    for line_num in valid_lines:
        line_info = df_line[df_line['line'] == line_num].iloc[0]
        flight_name = line_info['flight']

        if flight_name != current_flight_name:
            # current_xyz = get_XYZ(flight_name, df_flight, reorient_vec=reorient_vec, silent=silent) # Actual call
            print(f"Warning: Placeholder for get_XYZ('{flight_name}'). Returning dummy XYZ0.")
            dummy_traj = Traj(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10),
                              vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10),
                              fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0))
            dummy_ins = INS(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10),
                            vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10),
                            fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0), P=np.zeros((17,17,10)))
            dummy_magv = MagV(x=np.zeros(10),y=np.zeros(10),z=np.zeros(10),t=np.zeros(10))
            current_xyz = XYZ0(info="dummy", traj=dummy_traj, ins=dummy_ins, flux_a=dummy_magv,
                               flight=np.array([flight_name]*10), line=np.full(10,line_num), year=np.full(10,2020),
                               doy=np.ones(10), diurnal=np.zeros(10), igrf=np.zeros(10),
                               mag_1_c=np.zeros(10), mag_1_uc=np.zeros(10))
            current_flight_name = flight_name
        
        if current_xyz is None: continue # Should not happen if get_XYZ works

        # ind_for_line = get_ind_from_df(current_xyz, line_num, df_line, l_window=l_window) # Needs get_ind_from_df
        # Placeholder for get_ind logic:
        t_start = line_info.get('t_start', current_xyz.traj.tt.min())
        t_end = line_info.get('t_end', current_xyz.traj.tt.max())
        line_indices_in_xyz = (current_xyz.line == line_num) & \
                              (current_xyz.traj.tt >= t_start) & \
                              (current_xyz.traj.tt <= t_end)
        
        if l_window > 0:
            num_valid_pts = np.sum(line_indices_in_xyz)
            if num_valid_pts > 0:
                n_trim = num_valid_pts % l_window
                if n_trim > 0:
                    true_indices = np.where(line_indices_in_xyz)[0]
                    line_indices_in_xyz[true_indices[-n_trim:]] = False # Trim from end
        
        ind_for_line = line_indices_in_xyz

        if not np.any(ind_for_line):
            l_segs_list_for_output.append(0) # Add 0 length segment
            continue

        x_i, no_norm_i, features_i, _ = get_x(current_xyz, ind_for_line,
                                              features_setup=features_setup,
                                              features_no_norm=features_no_norm,
                                              terms=terms,
                                              sub_diurnal=sub_diurnal,
                                              sub_igrf=sub_igrf,
                                              bpf_mag_data=bpf_mag_data)
        x_list.append(x_i)
        l_segs_list_for_output.append(x_i.shape[0]) # Length of segment for this line

        if no_norm_mask_out is None: # First valid line processed
            no_norm_mask_out = no_norm_i
            features_out = features_i
        elif not np.array_equal(no_norm_mask_out, no_norm_i) or features_out != features_i:
             raise ValueError("Feature set or no_norm mask mismatch between lines/flights.")

    if not x_list: # If all lines were skipped or empty
        # Infer number of features from a dummy call if possible, or default
        num_features = 0
        if features_out: num_features = len(features_out)
        elif features_setup and "TL_A_flux_a" in features_setup: # Rough guess
             num_features = len(features_setup) + 10 # Placeholder
        elif features_setup:
             num_features = len(features_setup)

        return np.empty((0, num_features)), np.array([], dtype=bool), features_out or [], np.array(l_segs_list_for_output, dtype=int)

    x_matrix_combined = np.vstack(x_list)
    l_segs_combined = np.array(l_segs_list_for_output, dtype=int)
    
    return x_matrix_combined, no_norm_mask_out, features_out, l_segs_combined

# ... (Other functions like get_y, get_Axy, LPE, batchnorm, get_nn_m, etc. would follow)
# For brevity, I will stop here. A full translation would include all functions.
# The provided snippet covers the initial structure and some key function translations.
# The remaining functions involve more complex logic with ML models, signal processing,
# and external library interactions (Shapley, GSA, IGRF, plotting) which would
# require careful, step-by-step translation and testing.

# Example of how a more complex function like get_y would start:
def get_y(xyz: XYZ,
          ind: Optional[np.ndarray] = None,
          map_val: Union[float, np.ndarray] = -1,
          y_type: str = "d",
          use_mag_str: str = "mag_1_uc",
          use_mag_c_str: str = "mag_1_c",
          sub_diurnal: bool = False,
          sub_igrf: bool = False,
          bpf_scalar_mag: bool = False,
          fs_override: Optional[float] = None
         ) -> np.ndarray:
    """
    Get y target vector.

    Args:
        xyz: XYZ flight data object.
        ind: Selected data indices (boolean array). If None, all data used.
        map_val: Scalar magnetic anomaly map values. Used for y_type 'b' or 'c'.
                 Should be an array of same length as selected data if used.
        y_type: Target type:
            'a': Anomaly field #1 (compensated tail stinger total field)
            'b': Anomaly field #2 (interpolated magnetic anomaly map values)
            'c': Aircraft field #1 (uncomp_mag - map_val)
            'd': Aircraft field #2 (uncomp_mag - comp_mag)
            'e': BPF'd total field (bandpass filtered uncompensated cabin total field)
        use_mag_str: Uncompensated scalar magnetometer field name (e.g., "mag_1_uc").
        use_mag_c_str: Compensated scalar magnetometer field name (e.g., "mag_1_c").
        sub_diurnal: If true, subtract diurnal from scalar magnetometer measurements.
        sub_igrf: If true, subtract IGRF from scalar magnetometer measurements.
        bpf_scalar_mag: If true, bandpass filter the relevant scalar mag data for y_type 'e'.
        fs_override: Sampling frequency for BPF if xyz.traj.dt is not available.

    Returns:
        y: Length-N target vector.
    """
    if ind is None:
        ind = np.ones(xyz.traj.N, dtype=bool)
    
    N_selected = np.sum(ind)
    if N_selected == 0:
        return np.array([])

    mag_uc_selected: Optional[np.ndarray] = None
    mag_c_selected: Optional[np.ndarray] = None

    if y_type in ['c', 'd', 'e']:
        if not hasattr(xyz, use_mag_str):
            raise ValueError(f"Field {use_mag_str} not found in xyz for y_type {y_type}")
        mag_data_uncomp = getattr(xyz, use_mag_str)
        if isinstance(mag_data_uncomp, MagV):
             mag_uc_selected = mag_data_uncomp.t[ind]
        elif isinstance(mag_data_uncomp, np.ndarray):
             mag_uc_selected = mag_data_uncomp[ind]
        else:
            raise TypeError(f"Unsupported type for {use_mag_str}: {type(mag_data_uncomp)}")

    if y_type in ['a', 'd']:
        if not hasattr(xyz, use_mag_c_str):
            raise ValueError(f"Field {use_mag_c_str} not found in xyz for y_type {y_type}")
        mag_data_comp = getattr(xyz, use_mag_c_str)
        if isinstance(mag_data_comp, MagV): # This should ideally not be a MagV for compensated data
             mag_c_selected = mag_data_comp.t[ind]
        elif isinstance(mag_data_comp, np.ndarray):
             mag_c_selected = mag_data_comp[ind]
        else:
            raise TypeError(f"Unsupported type for {use_mag_c_str}: {type(mag_data_comp)}")

    sub = np.zeros(N_selected, dtype=float)
    if sub_diurnal and hasattr(xyz, 'diurnal') and xyz.diurnal is not None:
        # Ensure xyz.diurnal has data for the selected indices
        if len(xyz.diurnal) >= np.max(np.where(ind)[0]) + 1 if N_selected > 0 else True:
             sub += xyz.diurnal[ind]
        elif N_selected > 0 :
             print(f"Warning: xyz.diurnal might be too short for selected indices. Length: {len(xyz.diurnal)}, Max_ind: {np.max(np.where(ind)[0])}")


    if sub_igrf and hasattr(xyz, 'igrf') and xyz.igrf is not None:
        if len(xyz.igrf) >= np.max(np.where(ind)[0]) + 1 if N_selected > 0 else True:
            sub += xyz.igrf[ind]
        elif N_selected > 0:
            print(f"Warning: xyz.igrf might be too short for selected indices. Length: {len(xyz.igrf)}, Max_ind: {np.max(np.where(ind)[0])}")


    y_out: np.ndarray
    if y_type == 'a':
        if mag_c_selected is None: raise ValueError("mag_c_selected is None for y_type 'a'")
        y_out = mag_c_selected - sub
    elif y_type == 'b':
        if not isinstance(map_val, np.ndarray) or len(map_val) != N_selected:
            # Allow map_val to be a scalar if N_selected is 1
            if N_selected == 1 and isinstance(map_val, (float, int)):
                map_val = np.array([map_val]) # Convert to array
            elif map_val == -1 and N_selected > 0 : # Default value check
                raise ValueError(f"map_val must be provided (not -1) and be a NumPy array of length {N_selected} or scalar if N_selected=1 for y_type 'b'")
            elif map_val != -1 : # map_val is something else but not correct type/shape
                 raise ValueError(f"map_val must be a NumPy array of length {N_selected} or scalar if N_selected=1 for y_type 'b'. Got {type(map_val)} of len {len(map_val) if isinstance(map_val, np.ndarray) else 'N/A'}")
            else: # map_val is -1 and N_selected is 0 (empty ind)
                y_out = np.array([]) # Should be handled by N_selected == 0 check earlier
        if N_selected > 0 : y_out = map_val
        else: y_out = np.array([])

    elif y_type == 'c':
        if mag_uc_selected is None: raise ValueError("mag_uc_selected is None for y_type 'c'")
        if not isinstance(map_val, np.ndarray) or len(map_val) != N_selected:
            if N_selected == 1 and isinstance(map_val, (float, int)):
                map_val = np.array([map_val])
            elif map_val == -1 and N_selected > 0:
                raise ValueError(f"map_val must be provided (not -1) and be a NumPy array of length {N_selected} or scalar if N_selected=1 for y_type 'c'")
            elif map_val != -1:
                 raise ValueError(f"map_val must be a NumPy array of length {N_selected} or scalar if N_selected=1 for y_type 'c'. Got {type(map_val)} of len {len(map_val) if isinstance(map_val, np.ndarray) else 'N/A'}")
            else: # map_val is -1 and N_selected is 0
                pass # y_out will be assigned based on mag_uc_selected - sub if N_selected > 0
        
        if N_selected > 0:
            y_out = mag_uc_selected - sub - map_val
        else:
            y_out = np.array([])


    elif y_type == 'd':
        if mag_uc_selected is None: raise ValueError("mag_uc_selected is None for y_type 'd'")
        if mag_c_selected is None: raise ValueError("mag_c_selected is None for y_type 'd'")
        y_out = mag_uc_selected - mag_c_selected
    elif y_type == 'e':
        if mag_uc_selected is None: raise ValueError("mag_uc_selected is None for y_type 'e'")
        data_to_filter = mag_uc_selected - sub
        if bpf_scalar_mag:
            fs = fs_override
            if fs is None:
                if hasattr(xyz.traj, 'dt') and xyz.traj.dt > 0:
                    fs = 1.0 / xyz.traj.dt
                else:
                    raise ValueError("Cannot determine sampling frequency for BPF. Provide fs_override or ensure xyz.traj.dt is valid.")
            sos = get_bpf_sos(pass1=0.1, pass2=0.9, fs=fs)
            y_out = bpf_data(data_to_filter, sos=sos)
        else:
            y_out = data_to_filter
    else:
        raise ValueError(f"Invalid y_type: {y_type}. Choose from 'a', 'b', 'c', 'd', 'e'.")

    return y_out.flatten()

def get_y_from_dataframes(lines: Union[int, List[int]],
                            df_line: pd.DataFrame,
                            df_flight: pd.DataFrame,
                            df_map: pd.DataFrame,
                            y_type: str = "d",
                            use_mag_str: str = "mag_1_uc",
                            use_mag_c_str: str = "mag_1_c",
                            sub_diurnal: bool = False,
                            sub_igrf: bool = False,
                            bpf_scalar_mag: bool = False,
                            l_window: int = -1,
                            reorient_vec: bool = False,
                            silent: bool = True
                           ) -> np.ndarray:
    """
    Get y target vector from multiple flight lines, possibly multiple flights, using DataFrames.
    """
    if isinstance(lines, int):
        lines = [lines]

    # Ensure lines are unique before processing
    unique_input_lines = sorted(list(set(lines)))

    valid_lines_from_df = df_line['line'].unique()
    
    processed_lines = []
    for l in unique_input_lines:
        if l in valid_lines_from_df:
            processed_lines.append(l)
        elif not silent:
            print(f"Info: Line {l} not in df_line, skipping.")
    
    if not processed_lines:
        return np.array([])

    y_list = []
    current_xyz: Optional[XYZ] = None
    current_flight_name: Optional[str] = None
    
    get_XYZ_func = None
    get_map_func = None
    get_map_val_func = None

    try:
        from .get_XYZ import get_XYZ as get_XYZ_imported
        get_XYZ_func = get_XYZ_imported
    except ImportError:
        if not silent: print("Warning: get_XYZ could not be imported. Using placeholder for XYZ data in get_y_from_dataframes.")

    try:
        from .map_utils import get_map as get_map_imported
        get_map_func = get_map_imported
        from .map_utils import get_map_val as get_map_val_imported
        get_map_val_func = get_map_val_imported
    except ImportError:
        if not silent: print("Warning: get_map or get_map_val could not be imported. Using placeholders in get_y_from_dataframes.")

    for line_num in processed_lines:
        line_info_df = df_line[df_line['line'] == line_num]
        if line_info_df.empty: # Should not happen due to pre-filtering, but as a safeguard
            if not silent: print(f"Warning: Line {line_num} details not found in df_line post filtering. Skipping.")
            continue
        line_info = line_info_df.iloc[0]
        flight_name = line_info['flight']

        if flight_name != current_flight_name:
            if get_XYZ_func:
                current_xyz = get_XYZ_func(flight_name, df_flight, reorient_vec=reorient_vec, silent=silent)
            else:
                dummy_traj = Traj(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10),
                                  vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10),
                                  fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0))
                dummy_ins = INS(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10),
                                vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10),
                                fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0), P=np.zeros((17,17,10)))
                dummy_magv = MagV(x=np.zeros(10),y=np.zeros(10),z=np.zeros(10),t=np.zeros(10))
                current_xyz = XYZ0(info="dummy_y_df", traj=dummy_traj, ins=dummy_ins, flux_a=dummy_magv,
                                   flight=np.array([flight_name]*10, dtype=object), line=np.full(10,line_num), year=np.full(10,2020),
                                   doy=np.ones(10), diurnal=np.zeros(10), igrf=np.zeros(10),
                                   mag_1_c=np.zeros(10), mag_1_uc=np.zeros(10))
            current_flight_name = flight_name
        
        if current_xyz is None:
            if not silent: print(f"Warning: Could not load XYZ data for flight {flight_name}. Skipping line {line_num}.")
            continue

        # Use the main get_ind function for consistent index selection logic
        # This requires current_xyz.line to be correctly populated by get_XYZ
        ind_for_line = get_ind(current_xyz.line, lines=line_num, N_total=current_xyz.traj.N,
                               l_window=l_window, N_seg_min=1, N_buffer=0, silent=silent)
        
        # Further filter by t_start and t_end from df_line for this specific line segment
        t_start = line_info.get('t_start', -np.inf)
        t_end = line_info.get('t_end', np.inf)
        
        # Ensure current_xyz.traj.tt is available and has same length as current_xyz.line
        if hasattr(current_xyz.traj, 'tt') and len(current_xyz.traj.tt) == current_xyz.traj.N:
            time_mask = (current_xyz.traj.tt >= t_start) & (current_xyz.traj.tt <= t_end)
            ind_for_line = ind_for_line & time_mask
        elif not silent:
            print(f"Warning: Trajectory time 'tt' not available or mismatched for line {line_num}, flight {flight_name}. Skipping time-based filtering from df_line.")

        if not np.any(ind_for_line):
            if not silent: print(f"Warning: No data selected for line {line_num} after applying indices. Skipping.")
            continue

        map_val_for_line: Union[float, np.ndarray] = -1
        if y_type in ['b', 'c']:
            map_name_series = line_info.get('map_name')
            if map_name_series is None or pd.isna(map_name_series):
                raise ValueError(f"map_name not found in df_line for line {line_num} when y_type is {y_type}")
            map_name = str(map_name_series)
            
            if get_map_func and get_map_val_func:
                current_map_data = get_map_func(map_name, df_map)
                map_val_for_line = get_map_val_func(current_map_data, current_xyz.traj, ind_for_line, alpha=200)
            else:
                 if not silent: print(f"Warning: get_map or get_map_val not available. Using zeros for map_val_for_line for line {line_num}.")
                 map_val_for_line = np.zeros(np.sum(ind_for_line))
        
        fs_for_y = 1.0/current_xyz.traj.dt if hasattr(current_xyz.traj, 'dt') and current_xyz.traj.dt > 0 else None

        y_i = get_y(current_xyz, ind_for_line, map_val=map_val_for_line,
                    y_type=y_type, use_mag_str=use_mag_str, use_mag_c_str=use_mag_c_str,
                    sub_diurnal=sub_diurnal, sub_igrf=sub_igrf,
                    bpf_scalar_mag=bpf_scalar_mag, fs_override=fs_for_y)
        y_list.append(y_i)

    if not y_list:
        return np.array([])
    return np.concatenate(y_list)


def get_data_stats(data: np.ndarray, data_name: str = "data", digits: int = 2,
                   silent: bool = False) -> pd.DataFrame:
    """
    Get statistics (mean, std dev, min, max, RMSE, MAE) of data.

    Args:
        data: Input data array (1D or 2D). If 2D, stats are computed column-wise.
        data_name: Name for the data, used in DataFrame column names.
        digits: Number of digits to round statistics to.
        silent: If true, suppress print statements.

    Returns:
        df_stats: DataFrame containing the statistics.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.size == 0: # Handle empty array before checking ndim
        if not silent:
            print(f"Warning: Empty data provided for {data_name}. Returning empty stats DataFrame.")
        # Define columns even for empty DataFrame for consistency
        stat_cols = ["mean", "std", "min", "max", "rmse", "mae"]
        return pd.DataFrame(columns=stat_cols)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
        col_names = [data_name]
    elif data.ndim == 2:
        col_names = [f"{data_name}_{i}" for i in range(data.shape[1])]
    else:
        raise ValueError("Input data must be 1D or 2D.")

    stats_list = []
    for i in range(data.shape[1]):
        col_data = data[:, i]
        # Filter out NaNs for calculations that are sensitive to them or don't have nan-aware versions
        col_data_no_nan = col_data[~np.isnan(col_data)]

        if col_data_no_nan.size == 0: # All NaNs or originally empty after filtering
            s_mean, s_std, s_min, s_max, s_rmse, s_mae = (np.nan,) * 6
        else:
            s_mean = np.round(np.mean(col_data_no_nan), digits) # np.nanmean not needed due to pre-filtering
            s_std  = np.round(np.std(col_data_no_nan), digits)  # np.nanstd not needed
            s_min  = np.round(np.min(col_data_no_nan), digits)  # np.nanmin not needed
            s_max  = np.round(np.max(col_data_no_nan), digits)  # np.nanmax not needed
            s_rmse = np.round(rmse(col_data_no_nan), digits) # rmse/mae handle their own NaNs if any slip through
            s_mae  = np.round(mae(col_data_no_nan), digits)
        
        stats_list.append({
            "mean": s_mean, "std": s_std, "min": s_min, "max": s_max,
            "rmse": s_rmse, "mae": s_mae
        })
        if not silent:
            print(f"{col_names[i]} stats | mean: {s_mean}, std: {s_std}, min: {s_min}, max: {s_max}, rmse: {s_rmse}, mae: {s_mae}")

    df_stats = pd.DataFrame(stats_list, index=col_names)
    return df_stats

# ... (Continue with other function translations)
# norm_sets, denorm_sets, err_segs, get_ind, etc.
# Then the more complex ML/signal processing related functions.

# For functions like eval_shapley, eval_gsa, gif_animation_m3,
# it would be:
# import shap
# from SALib.sample import morris as morris_sample
# from SALib.analyze import morris as morris_analyze
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import pyIGRF # or other IGRF library

# And then translate the logic using these libraries.
# This is a substantial task beyond a single step for the entire file.
# The provided code forms a starting point for analysis_util.py.
def err_segs(y_hat: np.ndarray, y: np.ndarray, l_segs: np.ndarray, silent: bool = True) -> np.ndarray:
    """
    Remove mean error from multiple individual flight lines within a larger dataset.

    Args:
        y_hat: Prediction vector.
        y: Target vector.
        l_segs: Vector of lengths of line segments. sum(l_segs) should be len(y).
        silent: If true, no print outs.

    Returns:
        err: Mean-corrected (per line) error.
    """
    if len(y_hat) != len(y):
        raise ValueError("y_hat and y must have the same length.")
    if np.sum(l_segs) != len(y):
        raise ValueError("Sum of l_segs must equal the length of y and y_hat.")

    err = y_hat - y
    current_pos = 0
    for i, seg_len in enumerate(l_segs):
        if seg_len == 0:
            continue
        i1 = current_pos
        i2 = current_pos + seg_len
        segment_error = err[i1:i2]
        if len(segment_error) > 0: # Ensure segment is not empty
            err[i1:i2] -= np.mean(segment_error)
            err_std = np.round(np.std(segment_error), digits=2) # Assuming std_err is available or use np.std
            if not silent:
                print(f"Line #{i+1} error: {err_std} nT (length: {seg_len})")
        current_pos = i2
    return err

def norm_sets(train_data: np.ndarray, 
              test_data: Optional[np.ndarray] = None, 
              val_data: Optional[np.ndarray] = None,
              norm_type: str = "standardize", 
              no_norm_mask: Optional[np.ndarray] = None
             ) -> tuple:
    """
    Normalize (or standardize) features (columns) of training data and optionally
    apply the same transformation to test and validation data.

    Args:
        train_data: N_train x Nf training data.
        test_data: (Optional) N_test x Nf testing data.
        val_data: (Optional) N_val x Nf validation data.
        norm_type: Normalization type:
            "standardize": Z-score normalization (mean=0, std=1).
            "normalize": Min-max normalization (range [0,1] or specified).
            "scale": Scale by maximum absolute value (bias=0).
            "none": No normalization (scale by 1, bias=0).
        no_norm_mask: Boolean array of shape (Nf,) indicating features to not normalize.

    Returns:
        A tuple containing:
            train_bias (1xNf array of biases)
            train_scale (1xNf array of scales)
            train_data_norm (normalized training data)
            (if test_data provided) test_data_norm (normalized test data)
            (if val_data provided) val_data_norm (normalized validation data)
    """
    Nf = train_data.shape[1]
    if no_norm_mask is None:
        no_norm_mask = np.zeros(Nf, dtype=bool)
    elif not isinstance(no_norm_mask, np.ndarray) or no_norm_mask.dtype != bool or no_norm_mask.shape != (Nf,):
        # Attempt to convert if it's a list of indices or similar
        if isinstance(no_norm_mask, (list, np.ndarray)) and np.array(no_norm_mask).ndim == 1:
            temp_mask = np.zeros(Nf, dtype=bool)
            valid_indices = [idx for idx in no_norm_mask if isinstance(idx, int) and 0 <= idx < Nf]
            temp_mask[valid_indices] = True
            no_norm_mask = temp_mask
        else:
            raise ValueError("no_norm_mask must be a boolean array of shape (Nf,)")

    train_bias = np.zeros((1, Nf), dtype=train_data.dtype)
    train_scale = np.ones((1, Nf), dtype=train_data.dtype)

    for i in range(Nf):
        if no_norm_mask[i]:
            continue # Bias remains 0, scale remains 1

        col_data = train_data[:, i]
        if norm_type == "standardize":
            mu = np.mean(col_data)
            sigma = np.std(col_data)
            train_bias[0, i] = mu
            train_scale[0, i] = sigma if sigma > 1e-9 else 1.0 # Avoid division by zero
        elif norm_type == "normalize":
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            train_bias[0, i] = min_val
            train_scale[0, i] = (max_val - min_val) if (max_val - min_val) > 1e-9 else 1.0
        elif norm_type == "scale":
            # Bias is already 0
            max_abs_val = np.max(np.abs(col_data))
            train_scale[0, i] = max_abs_val if max_abs_val > 1e-9 else 1.0
        elif norm_type == "none":
            # Bias is 0, scale is 1, already set
            pass
        else:
            raise ValueError(f"{norm_type} normalization type not defined")

    train_data_norm = (train_data - train_bias) / train_scale
    
    results = [train_bias, train_scale, train_data_norm]

    if val_data is not None:
        val_data_norm = (val_data - train_bias) / train_scale
        results.append(val_data_norm)
        
    if test_data is not None:
        test_data_norm = (test_data - train_bias) / train_scale
        results.append(test_data_norm)
        
    return tuple(results)


def denorm_sets(*args: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Denormalize (or destandardize) features (columns) of data.
    Can take (bias, scale, data1, data2, ...) or (bias, scale, data_tuple).

    Args:
        train_bias: 1xNf training data biases.
        train_scale: 1xNf training data scaling factors.
        *data_sets: One or more N x Nf data arrays to denormalize.

    Returns:
        A single denormalized data array if one was passed, otherwise a tuple of them.
    """
    if len(args) < 3:
        raise ValueError("denorm_sets requires at least train_bias, train_scale, and one data array.")
    
    train_bias = args[0]
    train_scale = args[1]
    data_to_denorm = args[2:]

    denormalized_sets = []
    for data_set in data_to_denorm:
        if not isinstance(data_set, np.ndarray):
            raise TypeError("All data sets to denormalize must be NumPy arrays.")
        if data_set.shape[1] != train_bias.shape[1] or data_set.shape[1] != train_scale.shape[1]:
            raise ValueError("Data sets must have the same number of features as bias and scale.")
        denormalized_sets.append(data_set * train_scale + train_bias)

    return denormalized_sets[0] if len(denormalized_sets) == 1 else tuple(denormalized_sets)
def unpack_data_norms(data_norms: tuple) -> tuple:
    """
    Internal helper function to unpack data normalizations.
    Python equivalent of MagNav.jl's unpack_data_norms.

    Args:
        data_norms: Tuple of data normalizations of length 4 to 7.
            4: (x_bias, x_scale, y_bias, y_scale)
            5: (v_scale, x_bias, x_scale, y_bias, y_scale)
            6: (A_bias, A_scale, x_bias, x_scale, y_bias, y_scale)
            7: (A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale)

    Returns:
        A 7-tuple: (A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale)
        Defaulting missing A_bias, A_scale to 0, 1 and v_scale to identity-like.
    """
    len_dn = len(data_norms)
    A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale = (None,) * 7

    if len_dn == 7:
        A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale = data_norms
    elif len_dn == 6:
        A_bias, A_scale, x_bias, x_scale, y_bias, y_scale = data_norms
        # v_scale defaults based on x_scale's feature dimension
        if x_scale is not None and isinstance(x_scale, np.ndarray):
            # Assuming v_scale would be an identity matrix if it were for features
            # For simplicity, if it's just a scalar or 1D array, this might need adjustment
            # based on how v_scale is used. Julia uses `I(size(x_scale,2))`.
            # If x_scale is (1, Nf), then size(x_scale,2) is Nf.
            # A simple placeholder might be 1.0 or np.eye(num_features_of_v)
            # This depends on the expected structure of v_scale.
            # For now, let's assume if it's missing, it's not critically used or is scalar 1.
             v_scale = 1.0 # Placeholder, may need np.eye if used for matrix ops
        else:
            v_scale = 1.0 # Default if x_scale is None
    elif len_dn == 5:
        v_scale, x_bias, x_scale, y_bias, y_scale = data_norms
        A_bias, A_scale = 0.0, 1.0 # Default for A
    elif len_dn == 4:
        x_bias, x_scale, y_bias, y_scale = data_norms
        A_bias, A_scale = 0.0, 1.0
        v_scale = 1.0 # Placeholder, as above
    else:
        raise ValueError(f"Length of data_norms tuple must be 4, 5, 6, or 7. Got {len_dn}.")

    # Ensure defaults for any None values if not fully specified by shorter tuples
    if A_bias is None: A_bias = 0.0
    if A_scale is None: A_scale = 1.0
    if v_scale is None: v_scale = 1.0 # Or np.eye if matrix expected
    # x_bias, x_scale, y_bias, y_scale should always be present for len >= 4

    return A_bias, A_scale, v_scale, x_bias, x_scale, y_bias, y_scale


def get_days_in_year(year: Union[int, float]) -> int:
    """
    Get days in year based on (rounded down) year.
    """
    return 366 if int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0) else 365


def get_years(year: Union[int, float], doy: int = 0) -> float:
    """
    Get decimal (fractional) year from year and doy (day of year).
    Day of year (doy) is 1-based.
    """
    if not (1 <= doy <= get_days_in_year(year) or doy == 0): # Allow doy=0 for start of year
        raise ValueError(f"Day of year {doy} is invalid for year {int(year)}.")
    # If doy is 0, it means the very start of the year (Jan 1st, 00:00).
    # The Julia version `round(Int,year) + doy/get_days_in_year(year)` with doy=0
    # would effectively be `int(year)`. If doy is 1-indexed (1 to 365/366),
    # then for Jan 1st, doy=1. (doy-1) makes it 0-indexed for fraction calculation.
    return int(year) + (doy -1 if doy > 0 else 0) / get_days_in_year(year)


def get_lim(data: np.ndarray, frac: float = 0.0) -> Tuple[float, float]:
    """
    Get expanded limits (extrema) of data.
    """
    if data.size == 0:
        return (np.nan, np.nan)
    min_val, max_val = np.nanmin(data), np.nanmax(data)
    data_range = max_val - min_val
    if data_range == 0: # Avoid issues if all data points are the same
        return (min_val, max_val)
    return (min_val - frac * data_range, max_val + frac * data_range)


def expand_range(x: np.ndarray, xlim: Optional[Tuple[float, float]] = None,
                 extra_step: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand range of data that has a constant step size.
    Assumes x is sorted.
    """
    if x.ndim != 1:
        raise ValueError("Input array x must be 1D.")
    if len(x) < 2:
        # Cannot determine step or expand meaningfully
        return x.copy(), np.arange(len(x))

    # dx = get_step(x) # Assumes get_step is available and works for np.ndarray
    # For now, calculate step directly, assuming fairly constant step
    diffs = np.diff(x)
    if not np.allclose(diffs, diffs[0]):
        # Fallback or warning if step is not constant
        # print("Warning: Step size in x is not constant for expand_range. Using mean diff.")
        dx = np.mean(diffs)
    else:
        dx = diffs[0]

    if dx == 0: # Cannot expand if step is zero
        return x.copy(), np.arange(len(x))

    if xlim is None:
        xlim = get_lim(x)

    current_min, current_max = np.min(x), np.max(x)
    
    # Expand towards minimum
    points_to_prepend = []
    val = current_min - dx
    while val >= xlim[0]:
        points_to_prepend.append(val)
        val -= dx
    if extra_step and (not points_to_prepend or points_to_prepend[-1] > xlim[0]):
         points_to_prepend.append(current_min - dx * (len(points_to_prepend) + 1))


    # Expand towards maximum
    points_to_append = []
    val = current_max + dx
    while val <= xlim[1]:
        points_to_append.append(val)
        val += dx
    if extra_step and (not points_to_append or points_to_append[-1] < xlim[1]):
        points_to_append.append(current_max + dx * (len(points_to_append) + 1))
        
    expanded_x = np.concatenate((np.array(points_to_prepend[::-1]), x, np.array(points_to_append)))
    
    # Original indices in the new expanded array
    # If prepended Np points, original data starts at index Np
    original_indices = np.arange(len(points_to_prepend), len(points_to_prepend) + len(x))
    
    return expanded_x, original_indices


def filter_events(flight_name: str, df_event: pd.DataFrame,
                  keyword: str = "",
                  tt_lim: Optional[Tuple[float, float]] = None
                 ) -> pd.DataFrame:
    """
    Filter a DataFrame of in-flight events to only contain relevant events.
    Python equivalent of MagNav.jl's filter_events.

    Args:
        flight_name: Flight name (e.g., "Flt1001").
        df_event: DataFrame with 'flight', 'tt', and 'event' columns.
        keyword: Keyword to search within events (case insensitive).
        tt_lim: Tuple (start_time, end_time) for filtering.

    Returns:
        Filtered DataFrame.
    """
    df_filtered = df_event.copy()

    # Filter by flight name
    # Assuming flight names in DataFrame might be symbols or strings, convert to str for comparison
    if 'flight' not in df_filtered.columns:
        raise ValueError("'flight' column not found in df_event DataFrame.")
    df_filtered['flight_str'] = df_filtered['flight'].astype(str)
    df_filtered = df_filtered[df_filtered['flight_str'] == str(flight_name)] # Compare as strings
    df_filtered.drop(columns=['flight_str'], inplace=True, errors='ignore')


    # Filter by time limits
    if tt_lim:
        if len(tt_lim) != 2:
            raise ValueError("tt_lim must be a tuple of (start_time, end_time).")
        t_start, t_end = tt_lim
        if 'tt' not in df_filtered.columns:
             raise ValueError("'tt' column not found in df_event DataFrame.")
        df_filtered = df_filtered[(df_filtered['tt'] >= t_start) & (df_filtered['tt'] <= t_end)]

    # Filter by keyword
    if keyword:
        if 'event' not in df_filtered.columns:
            raise ValueError("'event' column not found in df_event DataFrame.")
        # Ensure 'event' column is string type before using .str.contains
        df_filtered = df_filtered[df_filtered['event'].astype(str).str.lower().str.contains(keyword.lower())]
        
    return df_filtered
def get_Axy(lines: Union[int, List[int]],
            df_line: pd.DataFrame,
            df_flight: pd.DataFrame,
            df_map: pd.DataFrame,
            features_setup: Optional[List[str]] = None,
            features_no_norm: Optional[List[str]] = None,
            y_type: str = 'd',
            use_mag_str: str = 'mag_1_uc',
            use_mag_c_str: str = 'mag_1_c',
            use_vec_str: str = 'flux_a', # For the "external" A matrix
            terms: Optional[List[str]] = None, # For A matrix within get_x features
            terms_A: Optional[List[str]] = None, # For the "external" A matrix
            sub_diurnal: bool = False,
            sub_igrf: bool = False,
            bpf_mag_data_in_x: bool = False, # Renamed from bpf_mag for clarity
            reorient_vec: bool = False,
            l_window: int = -1,
            mod_TL: bool = False, # If true, create modified "external" TL A matrix with use_mag_str
            map_TL: bool = False, # If true, create map-based "external" TL A matrix
            return_B_comps: bool = False, # Renamed from return_B
            bpf_A_if_y_type_e: bool = True, # Specific BPF for external A if y_type is 'e'
            silent: bool = True
           ) -> tuple:
    """
    Get "external" Tolles-Lawson A matrix, x data matrix, & y target vector
    from multiple flight lines, possibly multiple flights. Optionally return Bt & B_dot
    used to create the "external" Tolles-Lawson A matrix.

    Args:
        lines: Selected line number(s).
        df_line: DataFrame lookup for lines.
        df_flight: DataFrame lookup for flight data.
        df_map: DataFrame lookup for map data.
        features_setup: Features for x matrix (see get_x). Default: ["mag_1_uc", "TL_A_flux_a"]
        features_no_norm: Features not to normalize in x (see get_x). Default: []
        y_type: Target type for y vector (see get_y). Default: 'd'.
        use_mag_str: Uncompensated scalar mag for y (see get_y). Default: 'mag_1_uc'.
        use_mag_c_str: Compensated scalar mag for y (see get_y). Default: 'mag_1_c'.
        use_vec_str: Vector mag for "external" A matrix. Default: 'flux_a'.
        terms: TL terms for A matrix within x features. Default: ["permanent", "induced", "eddy"].
        terms_A: TL terms for "external" A matrix. Default: ["permanent", "induced", "eddy", "bias"].
        sub_diurnal: Subtract diurnal (see get_x, get_y). Default: False.
        sub_igrf: Subtract IGRF (see get_x, get_y). Default: False.
        bpf_mag_data_in_x: BPF scalar mag data in x matrix (see get_x). Default: False.
        reorient_vec: Reorient vector magnetometer (for get_XYZ). Default: False.
        l_window: Windowing for get_ind. Default: -1 (no windowing/trimming by get_ind).
        mod_TL: Use scalar mag (use_mag_str) for Bt in external A. Default: False.
        map_TL: Use map_val for Bt in external A. Default: False.
        return_B_comps: If true, also return Bt & B_dot for external A. Default: False.
        bpf_A_if_y_type_e: If y_type is 'e', apply BPF to the external A matrix. Default: True.
        silent: Suppress info prints. Default: True.

    Returns:
        Tuple containing:
            A_ext (external A matrix), x (feature matrix), y (target vector),
            no_norm_mask (for x), features_names (for x), l_segs (segment lengths).
        If return_B_comps is True, also returns Bt_ext, B_dot_ext.
    """
    if features_setup is None: features_setup = ["mag_1_uc", "TL_A_flux_a"]
    if features_no_norm is None: features_no_norm = []
    if terms is None: terms = ["permanent", "induced", "eddy"]
    if terms_A is None: terms_A = ["permanent", "induced", "eddy", "bias"]

    if isinstance(lines, int):
        lines = [lines]
    
    unique_input_lines = sorted(list(set(lines)))
    valid_lines_from_df = df_line['line'].unique()
    
    processed_lines = []
    for l_num in unique_input_lines:
        if l_num in valid_lines_from_df:
            processed_lines.append(l_num)
        elif not silent:
            print(f"Info: Line {l_num} not in df_line, skipping.")
    
    if not processed_lines: # If no valid lines to process
        # Determine expected number of columns for A and x to return empty arrays of correct shape
        # This is a bit tricky without loading data. For A, it depends on terms_A.
        # For x, it depends on features_setup and the structure of those features.
        # Placeholder:
        num_A_cols = 0
        if "permanent" in terms_A: num_A_cols +=3
        if "induced"   in terms_A: num_A_cols +=6 # Max 5 for symmetric, 6 for general
        if "eddy"      in terms_A: num_A_cols +=9
        if "bias"      in terms_A: num_A_cols +=1

        # For x, this is harder. If features_out was available from a dry run of get_x, use that.
        # For now, returning 0 columns for x if no lines.
        num_x_cols = 0 # Placeholder, ideally infer from features_setup
        
        empty_A = np.empty((0, num_A_cols))
        empty_x = np.empty((0, num_x_cols))
        empty_y = np.array([])
        empty_no_norm = np.array([], dtype=bool)
        empty_features = []
        empty_l_segs = np.array([], dtype=int)
        if return_B_comps:
            return empty_A, np.empty((0,3)), empty_y, empty_x, empty_y, empty_no_norm, empty_features, empty_l_segs
        else:
            return empty_A, empty_x, empty_y, empty_no_norm, empty_features, empty_l_segs

    # Check flight data compatibility (xyz_set)
    line_details_for_set_check = df_line[df_line['line'].isin(processed_lines)]
    flight_names_for_set_check = line_details_for_set_check['flight'].unique()
    flight_details_for_set_check = df_flight[df_flight['flight'].isin(flight_names_for_set_check)]
    if flight_details_for_set_check['xyz_set'].nunique() > 1:
         raise ValueError("Incompatible xyz_sets in df_flight for the selected lines.")

    A_list, Bt_list, B_dot_list = [], [], []
    x_list, y_list, l_segs_list = [], [], []
    no_norm_mask_out, features_out = None, None
    
    current_xyz: Optional[XYZ] = None
    current_flight_name: Optional[str] = None

    get_XYZ_func, get_map_func, get_map_val_func = None, None, None
    try:
        from .get_XYZ import get_XYZ as get_XYZ_imported
        get_XYZ_func = get_XYZ_imported
    except ImportError: pass
    try:
        from .map_utils import get_map as get_map_imported, get_map_val as get_map_val_imported
        get_map_func = get_map_imported
        get_map_val_func = get_map_val_imported
    except ImportError: pass


    for line_num in processed_lines:
        line_info_df = df_line[df_line['line'] == line_num]
        if line_info_df.empty: continue
        line_info = line_info_df.iloc[0]
        flight_name = line_info['flight']

        if flight_name != current_flight_name:
            if get_XYZ_func:
                current_xyz = get_XYZ_func(flight_name, df_flight, reorient_vec=reorient_vec, silent=silent)
            else: # Placeholder XYZ
                if not silent: print(f"Warning: get_XYZ not available. Using placeholder XYZ for flight {flight_name}")
                dummy_traj = Traj(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10), vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10), fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0))
                dummy_ins = INS(N=10, dt=0.1, tt=np.arange(10)*0.1, lat=np.zeros(10), lon=np.zeros(10), alt=np.zeros(10), vn=np.zeros(10), ve=np.zeros(10), vd=np.zeros(10), fn=np.zeros(10), fe=np.zeros(10), fd=np.zeros(10), Cnb=np.array([np.eye(3)]*10).transpose(1,2,0), P=np.zeros((17,17,10)))
                dummy_magv = MagV(x=np.zeros(10),y=np.zeros(10),z=np.zeros(10),t=np.zeros(10))
                current_xyz = XYZ0(info="dummy_Axy", traj=dummy_traj, ins=dummy_ins, flux_a=dummy_magv, flight=np.array([flight_name]*10,dtype=object), line=np.full(10,line_num), year=np.full(10,2020), doy=np.ones(10), diurnal=np.zeros(10), igrf=np.zeros(10), mag_1_c=np.zeros(10), mag_1_uc=np.zeros(10))
            current_flight_name = flight_name
        
        if current_xyz is None: continue

        ind_for_line = get_ind(current_xyz.line, lines=line_num, N_total=current_xyz.traj.N, l_window=l_window, N_seg_min=1, N_buffer=0, silent=silent)
        t_start = line_info.get('t_start', -np.inf); t_end = line_info.get('t_end', np.inf)
        if hasattr(current_xyz.traj, 'tt') and len(current_xyz.traj.tt) == current_xyz.traj.N:
            time_mask = (current_xyz.traj.tt >= t_start) & (current_xyz.traj.tt <= t_end)
            ind_for_line = ind_for_line & time_mask
        
        if not np.any(ind_for_line):
            l_segs_list.append(0)
            continue
        
        l_segs_list.append(np.sum(ind_for_line))

        # X matrix
        xi, no_norm_i, features_i, _ = get_x(current_xyz, ind_for_line,
                                            features_setup=features_setup,
                                            features_no_norm=features_no_norm,
                                            terms=terms, sub_diurnal=sub_diurnal,
                                            sub_igrf=sub_igrf, bpf_mag_data=bpf_mag_data_in_x)
        x_list.append(xi)
        if no_norm_mask_out is None:
            no_norm_mask_out = no_norm_i
            features_out = features_i
        elif not np.array_equal(no_norm_mask_out, no_norm_i) or features_out != features_i:
            raise ValueError("Feature set or no_norm mask mismatch between lines/flights for x matrix.")

        # Map values for Y and potentially for A_external
        map_val_for_line: Union[float, np.ndarray] = -1
        if y_type in ['b', 'c'] or map_TL:
            map_name_series = line_info.get('map_name')
            if map_name_series is None or pd.isna(map_name_series):
                raise ValueError(f"map_name missing in df_line for line {line_num} when y_type is '{y_type}' or map_TL is True.")
            map_name = str(map_name_series)
            if get_map_func and get_map_val_func:
                current_map_data = get_map_func(map_name, df_map)
                map_val_for_line = get_map_val_func(current_map_data, current_xyz.traj, ind_for_line, alpha=200)
            else: # Placeholder map_val
                if not silent: print(f"Warning: get_map/get_map_val not available. Using zeros for map_val for line {line_num}.")
                map_val_for_line = np.zeros(np.sum(ind_for_line))
        
        # Y vector
        fs_for_y = 1.0/current_xyz.traj.dt if hasattr(current_xyz.traj, 'dt') and current_xyz.traj.dt > 0 else None
        yi = get_y(current_xyz, ind_for_line, map_val=map_val_for_line, y_type=y_type,
                   use_mag_str=use_mag_str, use_mag_c_str=use_mag_c_str,
                   sub_diurnal=sub_diurnal, sub_igrf=sub_igrf,
                   bpf_scalar_mag=(y_type=='e'), fs_override=fs_for_y) # bpf_scalar_mag for y_type 'e'
        y_list.append(yi)

        # External A matrix
        if not hasattr(current_xyz, use_vec_str) or not isinstance(getattr(current_xyz, use_vec_str), MagV):
            raise ValueError(f"Vector magnetometer '{use_vec_str}' not found or not MagV type in XYZ for external A.")
        
        vec_mag_data_for_A = getattr(current_xyz, use_vec_str)
        Bt_for_A: Optional[np.ndarray] = None
        if mod_TL:
            if not hasattr(current_xyz, use_mag_str):
                raise ValueError(f"Scalar mag '{use_mag_str}' for mod_TL not in XYZ.")
            Bt_for_A = getattr(current_xyz, use_mag_str)[ind_for_line]
        elif map_TL:
            if not isinstance(map_val_for_line, np.ndarray) or len(map_val_for_line) != np.sum(ind_for_line):
                 raise ValueError("map_val_for_line must be a valid array for map_TL.")
            Bt_for_A = map_val_for_line
        
        # create_TL_A now returns A, Bt_used, B_dot_used
        Ai, Bt_i, B_dot_i = create_TL_A(vec_mag_data_for_A, ind_for_line,
                                        Bt_scalar_override=Bt_for_A,
                                        terms=terms_A,
                                        return_B_components=True) # Always get components
        
        if y_type == 'e' and bpf_A_if_y_type_e:
            fs_for_A = 1.0/current_xyz.traj.dt if hasattr(current_xyz.traj, 'dt') and current_xyz.traj.dt > 0 else None
            if fs_for_A:
                # BPF each column of Ai
                sos_A = get_bpf_sos(fs=fs_for_A) # Default passbands
                for k_col in range(Ai.shape[1]):
                    Ai[:, k_col] = bpf_data(Ai[:, k_col], sos=sos_A)
            elif not silent:
                print(f"Warning: Cannot BPF external A matrix for y_type 'e' on line {line_num} due to missing dt.")

        A_list.append(Ai)
        if return_B_comps:
            Bt_list.append(Bt_i)
            B_dot_list.append(B_dot_i)

    if not x_list: # All lines were skipped or resulted in no data
        num_A_cols = A_list[0].shape[1] if A_list else 0
        num_x_cols = x_list[0].shape[1] if x_list else (len(features_out) if features_out else 0)
        
        final_A = np.empty((0, num_A_cols))
        final_x = np.empty((0, num_x_cols))
        final_y = np.array([])
        final_no_norm = np.array([], dtype=bool) if no_norm_mask_out is None else no_norm_mask_out
        final_features = [] if features_out is None else features_out
        final_l_segs = np.array(l_segs_list, dtype=int)

        if return_B_comps:
            return final_A, np.array([]), np.empty((0,3)), final_x, final_y, final_no_norm, final_features, final_l_segs
        else:
            return final_A, final_x, final_y, final_no_norm, final_features, final_l_segs

    final_A = np.vstack(A_list)
    final_x = np.vstack(x_list)
    final_y = np.concatenate(y_list)
    final_l_segs = np.array(l_segs_list, dtype=int)

    if return_B_comps:
        final_Bt = np.concatenate(Bt_list) if Bt_list else np.array([])
        final_B_dot = np.vstack(B_dot_list) if B_dot_list else np.empty((0,3))
        return final_A, final_Bt, final_B_dot, final_x, final_y, no_norm_mask_out, features_out, final_l_segs
    else:
        return final_A, final_x, final_y, no_norm_mask_out, features_out, final_l_segs