"""
This module is responsible for creating and initializing XYZ data objects,
and handling XYZ text file input/output.
Translated and extended from the Julia MagNav.jl/src/create_XYZ.jl.
"""
import numpy as np
import h5py
from typing import Union, Tuple, List, Optional, Any, Dict
from dataclasses import dataclass, field
import math
import random
import os
import sys # Added for accessing patched constants
import importlib # Added for reloading constants
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from copy import deepcopy
from .common_types import MapS as _ActualMapS, MapSd as _ActualMapSd, MapS3D as _ActualMapS3D
from . import constants # Import constants module

# Define map_check locally as it seems to be missing from analysis_util
# Based on its usage and the fallback definition.
def map_check(m, la, lo) -> bool:
    """Checks if the given latitude/longitude path is valid on the map."""
    # Placeholder implementation, always returns True.
    # TODO: Implement actual map boundary/validity checks if necessary.
    return True

@dataclass
class Traj:
    """Trajectory data class.

    :param N: Number of points
    :param dt: Time step (s)
    :param tt: Time vector (s)
    :param lat: Latitude (rad)
    :param lon: Longitude (rad)
    :param alt: Altitude (m)
    :param vn: North velocity (m/s)
    :param ve: East velocity (m/s)
    :param vd: Down velocity (m/s)
    :param fn: North specific force (m/s^2)
    :param fe: East specific force (m/s^2)
    :param fd: Down specific force (m/s^2)
    :param Cnb: Direction cosine matrix (body to navigation) as 3x3xN array
    """
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
class MagV:
    """Vector magnetometer data.

    :param x: X-component of magnetic field (nT)
    :param y: Y-component of magnetic field (nT)
    :param z: Z-component of magnetic field (nT)
    :param t: Time vector (s)
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray
def utm_zone_from_latlon(lat_deg: float, lon_deg: float) -> tuple[int, bool]:
    """Calculates UTM zone number and hemisphere."""
    # Placeholder implementation from fallback block.
    # TODO: Consider using a more robust UTM library if precision is critical.
    return (int((lon_deg + 180) / 6) + 1, lat_deg >= 0)

def transform_lla_to_utm(lat_rad: float, lon_rad: float, zone: int, is_north: bool) -> tuple[float, float]:
    """Converts LLA to UTM coordinates (placeholder)."""
    # Placeholder implementation from fallback block.
    # TODO: Implement actual LLA to UTM conversion or use a library.
    lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad) # Corrected rad2rad to rad2deg
    return lon_deg * 1000, lat_deg * 1000 # Simplified placeholder

def create_dcm_from_vel(vn: np.ndarray, ve: np.ndarray, dt: float, order: str) -> np.ndarray:
    """Creates an array of DCMs from velocity (placeholder)."""
    # Placeholder implementation from fallback block.
    # TODO: Implement actual DCM creation logic.
    N_samples = len(vn)
    # Stack identity matrices along the third axis
    return np.stack([np.eye(3)] * N_samples, axis=-1) # Returns (3, 3, N)

def get_tolles_lawson_aircraft_field_vector(coeffs, terms, Bt_scale,
                                            flux_c, norm_flux_a, norm_flux_b, norm_flux_c, dcm_data, igrf_bf,
                                            **kwargs): # Added missing args and kwargs
    """Local placeholder for get_tolles_lawson_aircraft_field_vector."""
    print(f"DEBUG_CREATE_XYZ: Using placeholder get_tolles_lawson_aircraft_field_vector")
    # Determine N from dcm_data which should be (N, 3, 3)
    if dcm_data is not None and hasattr(dcm_data, 'shape') and len(dcm_data.shape) == 3:
        N = dcm_data.shape[0]
    # Otherwise try to get from flux_c
    elif flux_c is not None and hasattr(flux_c, 'shape') and len(flux_c.shape) == 1:
        N = flux_c.shape[0]
    else:
        N = 1
    return np.zeros((3, N))
def create_ins_model_matrices(dt: float, std_acc: np.ndarray, std_gyro: np.ndarray,
                                tau_acc: np.ndarray, tau_gyro: np.ndarray,
                                # traj_data_dict: Optional[Dict[str, Any]] = None, # Commented out if not used
                                # ins_options: Optional[Dict[str, Any]] = None,   # Commented out if not used
                                **kwargs) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: # Added **kwargs
    """Creates placeholder INS model matrices P0, Qd, R."""
    print(f"DEBUG: Using placeholder create_ins_model_matrices, received kwargs: {kwargs}") # Log kwargs
    nx = 17 # Changed from 9 to 17
    P0 = np.eye(nx) * 0.1
    Qd = np.eye(nx) * 0.01
    # The third element (R, measurement noise covariance) can remain None if not immediately problematic
    # or be a placeholder like np.eye(num_measurements) if its absence causes issues later.
    # For now, returning None as per the original structure for the third element.
    return (P0, Qd, None) # Return placeholder matrices
def trim_map(map_obj: Any, lat_array: np.ndarray, lon_array: np.ndarray, alt_array: Optional[np.ndarray] = None,
             buffer: float = 0.0, is_lla: bool = True, silent: bool = False) -> Any:
    """
    Placeholder for trim_map function.
    This function is intended to trim a map object to the extents of a given path.
    Currently, it returns the original map object unmodified.
    """
    if not silent:
        print(f"DEBUG: trim_map called with map_obj type: {type(map_obj)}. Returning unmodified map.")
    return map_obj
    return np.eye(17), np.eye(17), np.eye(17) # P0, Qd, R (dummy R)

def get_phi_matrix(*args, **kwargs) -> np.ndarray:
    """Returns a placeholder Phi matrix."""
    # Placeholder implementation from fallback block.
    # TODO: Implement actual Phi matrix calculation.
    return np.eye(17)

def correct_cnb_matrix(cnb, err):
    """Corrects Cnb matrix with given error (placeholder)."""
    # Placeholder implementation, returns original Cnb.
    # TODO: Implement actual Cnb correction if needed.
    return cnb

def upward_fft_map(map_s: Any, alt: float) -> Any: # Based on fallback, added type hints
    """Placeholder for upward_fft_map.
    Performs upward continuation using FFT (currently a placeholder).
    """
    # print(f"DEBUG_CREATE_XYZ: upward_fft_map called with map_s type {type(map_s)}, alt {alt}")
    if hasattr(map_s, 'map') and callable(getattr(map_s, 'copy', None)): # Check if copy is callable
         # Attempt deepcopy if possible, otherwise shallow copy or original
        try:
            new_map_s = deepcopy(map_s)
            # Placeholder: actual upward continuation logic would modify new_map_s.map
            # e.g., new_map_s.map = perform_fft_continuation(new_map_s.map, getattr(new_map_s, 'alt', alt), alt)
            return new_map_s
        except TypeError: # deepcopy might fail for some complex types not designed for it
            try:
                return map_s.copy()
            except AttributeError:
                return map_s # Return original if no copy mechanism
    elif isinstance(map_s, np.ndarray) and hasattr(map_s, 'copy'):
        return map_s.copy() # If it's just a numpy array
    return map_s # Fallback for simple data or if no copy attribute

def get_map_params(map_s: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Placeholder for get_map_params. Returns dummy indices."""
    # Based on fallback: ind0, ind1, _, _
    # print(f"DEBUG_CREATE_XYZ: get_map_params called with map_s type {type(map_s)}")
    return (None, None, None, None)

def fill_map_gaps(map_s: Any) -> Any:
    """Placeholder for fill_map_gaps. Returns the map as is."""
    # print(f"DEBUG_CREATE_XYZ: fill_map_gaps called with map_s type {type(map_s)}")
    # In a real implementation, this would fill gaps in the map data.
    return map_s
def apply_band_pass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """Placeholder for apply_band_pass_filter. Returns original data."""
    print(f"DEBUG: Using placeholder apply_band_pass_filter for data of shape {data.shape}")
    # In a real implementation, this would apply a band-pass filter.
    # For now, it returns the data unmodified.
    return data
# Attempt to import from project modules

# Define generate_fogm_noise locally as a placeholder
def generate_fogm_noise(sigma: float, tau: float, dt: float, N: int) -> np.ndarray:
    """
    Generates placeholder First-Order Gauss-Markov (FOGM) noise.
    This is a simplified placeholder.
    """
    print(f"DEBUG: Using placeholder generate_fogm_noise (sigma={sigma}, tau={tau}, dt={dt}, N={N})")
    # Simple random noise, not true FOGM, for placeholder purposes
    return np.random.randn(N) * sigma

# Define create_tolles_lawson_A_matrix locally as a placeholder
def create_tolles_lawson_A_matrix(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, terms: Optional[List[str]] = None) -> np.ndarray:
    """
    Generates a placeholder Tolles-Lawson A matrix.
    """
    print(f"DEBUG: Using placeholder create_tolles_lawson_A_matrix (Bx_shape={Bx.shape}, terms={terms})")
    # Determine number of terms (columns). Default to 18 if terms not specified or standard.
    # This matches the fallback definition's shape.
    num_terms = 18 # Common number of terms for Tolles-Lawson
    if terms:
        # A more sophisticated approach might adjust num_terms based on the 'terms' list.
        # For a simple placeholder, we can stick to a fixed size or a simple rule.
        # For now, keeping it fixed to 18 as per the original fallback.
        pass
    return np.zeros((len(Bx), num_terms))

def get_igrf_magnetic_field(xyz, ind=None, frame='body', norm_igrf=False, check_xyz=True):
    """Local placeholder for get_igrf_magnetic_field."""
    # This definition is based on the fallback from the except ImportError block.
    num_points = 1 # Default if determination fails
    if hasattr(xyz, 'traj') and hasattr(xyz.traj, 'lat') and xyz.traj.lat is not None:
        try:
            num_points = len(xyz.traj.lat) if ind is None else len(ind)
        except TypeError:
            print(f"Warning: Could not determine num_points from xyz.traj.lat or ind in get_igrf_magnetic_field. Defaulting to {num_points}.")
            pass
    elif ind is not None:
        try:
            num_points = len(ind)
        except TypeError:
            print(f"Warning: Could not determine num_points from ind in get_igrf_magnetic_field. Defaulting to {num_points}.")
            pass
    # If xyz.N is a reliable source for number of points, it could be added as another elif.
    # else:
    #     print(f"Warning: xyz.traj.lat not available and ind is None in get_igrf_magnetic_field. Defaulting to {num_points}.")

    if norm_igrf:
        return np.zeros(num_points)
    else:
        return np.zeros((3, num_points))

# Placeholder for get_trajectory_subset, moved from except block and enhanced
def get_trajectory_subset(traj: 'Traj', ind: Union[List[int], np.ndarray]) -> 'Traj':
    """Simplified placeholder for get_trajectory_subset."""
    # Ensure indices are suitable for numpy array indexing
    if not isinstance(ind, (list, np.ndarray)):
        raise TypeError(f"Indices must be a list or numpy array, got {type(ind)}")
    if not isinstance(ind, np.ndarray): # Convert list to numpy array if necessary
        ind = np.array(ind)
    
    # Placeholder: Actual Traj class might be imported later or defined in except block
    # This structure assumes Traj has these attributes and they are indexable.
    # A more robust placeholder might need to check for attribute existence.
    # For simplicity, direct attribute access is used here.
    # Also, Cnb might need special handling if it can be empty or not always present.
    cnb_data = traj.Cnb[ind] if hasattr(traj, 'Cnb') and traj.Cnb is not None and len(traj.Cnb) > 0 else np.array([])

    # Create a new Traj-like object. If Traj is a dataclass, this is straightforward.
    # If not, this might need to be adjusted based on Traj's actual constructor.
    # Assuming a constructor or that a simple object with attributes is sufficient for placeholder.
    # For now, returning a dictionary, assuming it will be converted to Traj or used as such.
    # This part might need refinement based on how Traj is actually used with the subset.
    # For a true placeholder, we might need to define a dummy Traj class if not available.
    
    # Reverting to direct instantiation if Traj is known (e.g. from .magnav import)
    # The type hint 'Traj' will refer to the imported or dummy Traj class.
    return Traj(N=len(ind), dt=traj.dt, tt=traj.tt[ind], lat=traj.lat[ind], lon=traj.lon[ind], alt=traj.alt[ind],
                vn=traj.vn[ind], ve=traj.ve[ind], vd=traj.vd[ind], fn=traj.fn[ind], fe=traj.fe[ind], fd=traj.fd[ind],
                Cnb=cnb_data)

def tolles_lawson_coeffs_to_matrix(coeffs, terms, Bt_scale):
    """Local placeholder for tolles_lawson_coeffs_to_matrix."""
    print(f"DEBUG_CREATE_XYZ: Using placeholder tolles_lawson_coeffs_to_matrix")
    return (np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))
try:
    from .map_utils import get_map_val, get_map # Correctly import get_map from map_utils
    from .magnav import (XYZ0, Traj, INS, MagV, MapS, MapSd, MapS3D, MapV,
                            Map, Path, add_extension, G_EARTH) # Added G_EARTH
    from .analysis_util import ( # Removed get_tolles_lawson_aircraft_field_vector
        dlat2dn, dlon2de, dn2dlat, # Removed map_check, g_earth, AND get_map
        de2dlon, fdm, # Removed utm_zone_from_latlon, transform_lla_to_utm
        dcm2euler, euler2dcm, # Removed create_dcm_from_vel
        # trim_map, # Removed upward_fft_map, get_map_params, fill_map_gaps - trim_map is defined locally
        # generate_fogm_noise, # Removed import, defined locally
        # create_tolles_lawson_A_matrix, # Removed import, defined locally
        # get_band_pass_filter_coeffs, # Removed import, already defined locally
        # get_tolles_lawson_aircraft_field_vector, # Commented out to resolve ImportError
        # get_trajectory_subset, # Now defined locally as a placeholder
        # approximate_gradient # Removed import, will be defined locally as placeholder
    )
    # Default map identifiers will now be accessed via constants module
    # DEFAULT_SCALAR_MAP_ID = "namad" # Placeholder
    # DEFAULT_VECTOR_MAP_ID = "emm720" # Placeholder

except ImportError as e:
    # Fallback for standalone execution or if modules are structured differently
    print("ERROR_DEBUG: An ImportError occurred in create_xyz.py. Falling back to dummy implementations.")
    print(f"ERROR_DEBUG: Specific import error: {e}")
    print("Warning: Could not import from .magnav or .analysis_util. Placeholder types and functions will be used.")
    # Define dummy classes and functions if needed for linting/testing without full project
    @dataclass
    class BaseMap: pass
    @dataclass
    class MapS(BaseMap): xx: np.ndarray; yy: np.ndarray; map: np.ndarray; alt: Union[float, np.ndarray] = 0.0
    @dataclass
    class MapSd(MapS): mask: Optional[np.ndarray] = None
    @dataclass
    class MapS3D(MapS): zz: Optional[np.ndarray] = None; alt: Optional[np.ndarray] = None # alt might be 3D grid
    @dataclass
    class MapV(BaseMap):pass
    @dataclass
    class INS(Traj): P: np.ndarray # Covariance matrices
    @dataclass
    class XYZ0: info: str; traj: Traj; ins: INS; flux_a: MagV; flights: np.ndarray; lines: np.ndarray; years: np.ndarray; doys: np.ndarray; diurnal: np.ndarray; igrf: np.ndarray; mag_1_c: np.ndarray; mag_1_uc: np.ndarray
    Path = Union[Traj, INS]
    def add_extension(s, ext): return s if s.endswith(ext) else s + ext
    G_EARTH = 9.80665 # Changed to G_EARTH
    # def get_map(map_id, **kwargs): return MapS(xx=np.array([]), yy=np.array([]), map=np.array([])) # Dummy, accepts **kwargs to prevent TypeError - COMMENTED OUT TO USE IMPORTED VERSION
    # DEFAULT_SCALAR_MAP_ID = "namad" # Accessed via constants
    # DEFAULT_VECTOR_MAP_ID = "emm720" # Accessed via constants
    # def map_check(m,la,lo): return True # Moved to main scope
    def dlat2dn(dlat,lat): return dlat * 111000
    def dlon2de(dlon,lat): return dlon * 111000 * np.cos(lat)
    def dn2dlat(dn,lat): return dn / 111000
    def de2dlon(de,lat): return de / (111000 * np.cos(lat))
    def fdm(arr): return np.gradient(arr) if arr.ndim == 1 else np.array([np.gradient(arr[i]) for i in range(arr.shape[0])])
    # def utm_zone_from_latlon(lat_deg, lon_deg): return (int((lon_deg + 180) / 6) + 1, lat_deg >=0) # Moved to main scope
    # def transform_lla_to_utm(lat_rad, lon_rad, zone, is_north): # Moved to main scope
    #     lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad)
    #     return lon_deg * 1000, lat_deg * 1000
    # def create_dcm_from_vel(vn, ve, dt, order): return np.array([np.eye(3)] * len(vn)) # Moved to main scope
    def dcm2euler(dcm, order): return np.zeros((len(dcm), 3)) if dcm.ndim == 3 else np.zeros(3)
    def euler2dcm(roll,pitch,yaw,order): return np.array([np.eye(3)]*len(roll)) if isinstance(roll,np.ndarray) else np.eye(3)
    # def create_ins_model_matrices(*args, **kwargs): return np.eye(17), np.eye(17), np.eye(17) # Moved to main scope
    # def get_phi_matrix(*args, **kwargs): return np.eye(17) # Moved to main scope
    def upward_fft_map(map_s, alt): return map_s
    # def get_map_params(map_s): return (None,None,None,None) # Moved to main scope
    def fill_map_gaps(map_s): return map_s
    def trim_map(map_s): return map_s
    def get_map_val(map_s, lat, lon, alt, alpha=200, return_interpolator=False):
        if return_interpolator: return np.zeros(len(lat)), None
        return np.zeros(len(lat))
    def generate_fogm_noise(sigma, tau, dt, N): return np.random.randn(N) * sigma
    def create_tolles_lawson_A_matrix(Bx,By,Bz,terms=None): return np.zeros((len(Bx), 18)) # Dummy
    def get_igrf_magnetic_field(xyz, ind=None, frame='body', norm_igrf=False, check_xyz=True):
        num_points = len(xyz.traj.lat) if ind is None else len(ind)
        return np.zeros((3, num_points)) if not norm_igrf else np.zeros(num_points)
    def apply_band_pass_filter(data, bpf_coeffs): return data
    def get_band_pass_filter_coeffs(pass1=1e-6,pass2=1): return None
    def tolles_lawson_coeffs_to_matrix(coeffs, terms, Bt_scale): return (np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))
    def get_tolles_lawson_aircraft_field_vector(B_earth, B_earth_dot, c_p, c_i, c_e): return np.zeros_like(B_earth)
    # def get_trajectory_subset(traj, ind): # Simplified # Removed as it's defined globally now
    #     return Traj(N=len(ind), dt=traj.dt, tt=traj.tt[ind], lat=traj.lat[ind], lon=traj.lon[ind], alt=traj.alt[ind],
    #                 vn=traj.vn[ind], ve=traj.ve[ind], vd=traj.vd[ind], fn=traj.fn[ind], fe=traj.fe[ind], fd=traj.fd[ind],
    #                 Cnb=traj.Cnb[ind])
    def approximate_gradient(itp_func, y, x): return np.array([0.0, 0.0]) # Dummy

def approximate_gradient(itp_func: Any, y: Any, x: Any) -> np.ndarray:
    """Local placeholder for approximate_gradient."""
    # This definition is based on the fallback from the except ImportError block,
    # now made globally available as its import is removed from the try block.
    # print(f"DEBUG_CREATE_XYZ: Using global placeholder approximate_gradient")
    return np.array([0.0, 0.0]) # Dummy


def create_xyz0(
    mapS: Optional[Union[MapS, MapSd, MapS3D]] = None,
    alt: float = 1000.0,
    dt: float = 0.1,
    t: float = 300.0,
    v: float = 68.0,
    ll1: Tuple[float, float] = (),
    ll2: Tuple[float, float] = (),
    N_waves: int = 1,
    attempts: int = 10,
    info: str = "Simulated data",
    flight: int = 1,
    line: int = 1,
    year: int = 2023,
    doy: int = 154,
    mapV: Optional[MapV] = None,
    cor_sigma: float = 1.0,
    cor_tau: float = 600.0,
    cor_var: float = 1.0**2,
    cor_drift: float = 0.001,
    cor_perm_mag: float = 5.0,
    cor_ind_mag: float = 5.0,
    cor_eddy_mag: float = 0.5,
    init_pos_sigma: float = 3.0,
    init_alt_sigma: float = 0.001,
    init_vel_sigma: float = 0.01,
    init_att_sigma: float = np.deg2rad(0.01),
    VRW_sigma: float = 0.000238,
    ARW_sigma: float = 0.000000581,
    baro_sigma: float = 1.0,
    ha_sigma: float = 0.001,
    a_hat_sigma: float = 0.01,
    acc_sigma: float = 0.000245,
    gyro_sigma: float = 0.00000000727,
    fogm_sigma: float = 1.0,
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0,
    save_h5: bool = False,
    xyz_h5: str = "xyz_data.h5",
    silent: bool = False,
    default_scalar_map_id_override: Optional[str] = None,
    default_vector_map_id_override: Optional[str] = None
) -> XYZ0:
    """
    Create basic flight data (XYZ0 struct). Assumes constant altitude (2D flight).

    :param mapS: Scalar map (MapS, MapSd, or MapS3D) or None to load default
    :type mapS: Optional[Union[MapS, MapSd, MapS3D]]
    :param alt: Altitude (m) for trajectory, defaults to 1000.0
    :type alt: float, optional
    :param dt: Time step (s), defaults to 0.1
    :type dt: float, optional
    :param t: Total time (s), defaults to 300.0
    :type t: float, optional
    :param v: Velocity (m/s), defaults to 68.0
    :type v: float, optional
    :param ll1: Starting (lat, lon) in radians, defaults to ()
    :type ll1: Tuple[float, float], optional
    :param ll2: Ending (lat, lon) in radians, defaults to ()
    :type ll2: Tuple[float, float], optional
    :param N_waves: Number of sine waves in trajectory, defaults to 1
    :type N_waves: int, optional
    :param attempts: Number of attempts to generate valid trajectory, defaults to 10
    :type attempts: int, optional
    :param info: Information string, defaults to "Simulated data"
    :type info: str, optional
    :param flight: Flight number, defaults to 1
    :type flight: int, optional
    :param line: Line number, defaults to 1
    :type line: int, optional
    :param year: Year, defaults to 2023
    :type year: int, optional
    :param doy: Day of year, defaults to 154
    :type doy: int, optional
    :param mapV: Vector map (MapV) or None to load default, defaults to None
    :type mapV: Optional[MapV], optional
    :param cor_sigma: Standard deviation of correlated noise for magnetometer, defaults to 1.0
    :type cor_sigma: float, optional
    :param cor_tau: Time constant of correlated noise for magnetometer (s), defaults to 600.0
    :type cor_tau: float, optional
    :param cor_var: Variance of white noise for magnetometer, defaults to 1.0**2
    :type cor_var: float, optional
    :param cor_drift: Drift rate for magnetometer (nT/s), defaults to 0.001
    :type cor_drift: float, optional
    :param cor_perm_mag: Permanent magnetism coefficient (nT), defaults to 5.0
    :type cor_perm_mag: float, optional
    :param cor_ind_mag: Induced magnetism coefficient (nT), defaults to 5.0
    :type cor_ind_mag: float, optional
    :param cor_eddy_mag: Eddy current magnetism coefficient (nT), defaults to 0.5
    :type cor_eddy_mag: float, optional
    :param init_pos_sigma: Initial position uncertainty (m), defaults to 3.0
    :type init_pos_sigma: float, optional
    :param init_alt_sigma: Initial altitude uncertainty (m), defaults to 0.001
    :type init_alt_sigma: float, optional
    :param init_vel_sigma: Initial velocity uncertainty (m/s), defaults to 0.01
    :type init_vel_sigma: float, optional
    :param init_att_sigma: Initial attitude uncertainty (rad), defaults to np.deg2rad(0.01)
    :type init_att_sigma: float, optional
    :param VRW_sigma: Velocity random walk (m/s/rtHz), defaults to 0.000238
    :type VRW_sigma: float, optional
    :param ARW_sigma: Angular random walk (rad/rtHz), defaults to 0.000000581
    :type ARW_sigma: float, optional
    :param baro_sigma: Barometer noise standard deviation (m), defaults to 1.0
    :type baro_sigma: float, optional
    :param ha_sigma: Horizontal accelerometer noise (m/s^2), defaults to 0.001
    :type ha_sigma: float, optional
    :param a_hat_sigma: Accelerometer bias noise (m/s^2), defaults to 0.01
    :type a_hat_sigma: float, optional
    :param acc_sigma: Accelerometer noise (m/s^2), defaults to 0.000245
    :type acc_sigma: float, optional
    :param gyro_sigma: Gyroscope noise (rad/s), defaults to 0.00000000727
    :type gyro_sigma: float, optional
    :param fogm_sigma: First-order Gauss-Markov process sigma for magnetometer, defaults to 1.0
    :type fogm_sigma: float, optional
    :param baro_tau: Barometer time constant (s), defaults to 3600.0
    :type baro_tau: float, optional
    :param acc_tau: Accelerometer time constant (s), defaults to 3600.0
    :type acc_tau: float, optional
    :param gyro_tau: Gyroscope time constant (s), defaults to 3600.0
    :type gyro_tau: float, optional
    :param fogm_tau: FOGM time constant for magnetometer (s), defaults to 600.0
    :type fogm_tau: float, optional
    :param save_h5: Save to HDF5 file, defaults to False
    :type save_h5: bool, optional
    :param xyz_h5: HDF5 filename, defaults to "xyz_data.h5"
    :type xyz_h5: str, optional
    :param silent: Suppress output, defaults to False
    :type silent: bool, optional
    :param default_scalar_map_id_override: Override for default scalar map ID, defaults to None
    :type default_scalar_map_id_override: Optional[str], optional
    :param default_vector_map_id_override: Override for default vector map ID, defaults to None
    :type default_vector_map_id_override: Optional[str], optional
    :return: XYZ0 data structure
    :rtype: XYZ0
    """
    # Fetch constants module from sys.modules to access patched values
    constants_module = sys.modules.get('magnavpy.constants', constants) # Fallback to imported if not in sys.modules
    
    # Determine scalar map ID to use
    if default_scalar_map_id_override is not None:
        mapS_name_to_use = default_scalar_map_id_override
        print(f"DEBUG_CREATE_XYZ0: Using default_scalar_map_id_override for scalar map: '{mapS_name_to_use}'")
    else:
        mapS_name_to_use = constants_module.DEFAULT_SCALAR_MAP_ID
        print(f"DEBUG_CREATE_XYZ0: Using constants.DEFAULT_SCALAR_MAP_ID (via sys.modules) for scalar map: '{mapS_name_to_use}'")

    if mapS is None:
        map_kwargs_scalar: Dict[str, Any] = {} # Define map_kwargs for scalar map call
        # Explicitly set map_type="scalar" to avoid ambiguity if ID points to a vector map path
        mapS = get_map(mapS_name_to_use, map_type="scalar", silent=silent, **map_kwargs_scalar)
    
    # Determine the default_map_id_override for create_flux
    # This value will be the potentially patched constants.DEFAULT_VECTOR_MAP_ID
    # mapV_instance_for_flux will hold the MapV object if it's passed directly.
    mapV_instance_for_flux: Optional[MapV] = mapV
    
    # This will be the map_id passed to create_flux if mapV_instance_for_flux is None.
    map_id_to_pass_to_create_flux: Optional[str] = None

    if mapV_instance_for_flux is None:
        # mapV was not provided as an object, so create_flux will need to load it using an ID.
        if default_vector_map_id_override is not None:
            map_id_to_pass_to_create_flux = default_vector_map_id_override
            print(f"DEBUG_CREATE_XYZ0: mapV is None. Using default_vector_map_id_override for create_flux: '{map_id_to_pass_to_create_flux}'")
        else:
            map_id_to_pass_to_create_flux = constants_module.DEFAULT_VECTOR_MAP_ID
            print(f"DEBUG_CREATE_XYZ0: mapV is None. Using constants.DEFAULT_VECTOR_MAP_ID (via sys.modules) for create_flux: '{map_id_to_pass_to_create_flux}'")
        # We do NOT load mapV here; create_flux will handle it using the map_id_to_pass_to_create_flux.
    else:
        print(f"DEBUG_CREATE_XYZ0: mapV (mapV_instance_for_flux) was provided directly. Type: {type(mapV_instance_for_flux)}")
        # If mapV_instance_for_flux is provided, map_id_to_pass_to_create_flux remains None.
        # create_flux will use the mapV_instance_for_flux object.


    xyz_h5 = add_extension(xyz_h5, ".h5")

    # Create trajectory
    traj = create_traj(
        mapS,
        alt=alt, dt=dt, t=t, v=v, ll1=ll1, ll2=ll2, N_waves=N_waves,
        attempts=attempts, save_h5=save_h5, traj_h5=xyz_h5
    )

    # Create INS
    ins = create_ins(
        traj,
        init_pos_sigma=init_pos_sigma, init_alt_sigma=init_alt_sigma,
        init_vel_sigma=init_vel_sigma, init_att_sigma=init_att_sigma,
        VRW_sigma=VRW_sigma, ARW_sigma=ARW_sigma, baro_sigma=baro_sigma,
        ha_sigma=ha_sigma, a_hat_sigma=a_hat_sigma, acc_sigma=acc_sigma,
        gyro_sigma=gyro_sigma, baro_tau=baro_tau, acc_tau=acc_tau,
        gyro_tau=gyro_tau, save_h5=save_h5, ins_h5=xyz_h5
    )

    # Create compensated (clean) scalar magnetometer measurements
    mag_1_c = create_mag_c(
        traj, mapS, # mapS here is the map_object
        meas_var=cor_var,
        fogm_sigma=fogm_sigma,
        fogm_tau=fogm_tau,
        silent=silent
    )

    # Create compensated (clean) vector magnetometer measurements
    # Pass mapV_instance_for_flux (which could be None or a MapV object)
    # and map_id_to_pass_to_create_flux (which is set if mapV_instance_for_flux is None)
    flux_a = create_flux(
        path_or_lat=traj,
        lon_or_mapV=mapV_instance_for_flux, # This can be None
        meas_var=cor_var,
        fogm_sigma=fogm_sigma,
        fogm_tau=fogm_tau,
        silent=silent,
        default_map_id_override=map_id_to_pass_to_create_flux # Pass the override
    )

    # Create uncompensated (corrupted) scalar magnetometer measurements
    mag_1_uc, _, diurnal_effect = corrupt_mag(
        mag_1_c, flux_a,
        traj_ideal=traj, dt=dt,
        cor_sigma=cor_sigma, cor_tau=cor_tau, cor_var=cor_var,
        cor_drift=cor_drift, cor_perm_mag=cor_perm_mag,
        cor_ind_mag=cor_ind_mag, cor_eddy_mag=cor_eddy_mag
    )

    num_points = traj.N
    flights = np.full(num_points, flight, dtype=int)
    lines   = np.full(num_points, line, dtype=int)
    years   = np.full(num_points, year, dtype=int)
    doys    = np.full(num_points, doy, dtype=int)
    diurnal = diurnal_effect

    igrf_initial = np.zeros(num_points)
    
    xyz = XYZ0(info=info, traj=traj, ins=ins, flux_a=flux_a,
               flight=flights, line=lines, year=years, doy=doys, # Changed doys to doy
               diurnal=diurnal, igrf=igrf_initial,
               mag_1_c=mag_1_c, mag_1_uc=mag_1_uc)

    igrf_vector_body = get_igrf_magnetic_field(xyz, frame='body', norm_igrf=False, check_xyz=False)
    # Ensure igrf_vector_body is suitable for np.linalg.norm, converting to float if necessary
    # This addresses potential TypeError if igrf_vector_body has an object dtype
    # or contains elements that np.sqrt (used by norm) cannot handle directly.
    processed_igrf_vector_body = igrf_vector_body
    if isinstance(igrf_vector_body, np.ndarray) and not np.issubdtype(igrf_vector_body.dtype, np.number):
        try:
            processed_igrf_vector_body = igrf_vector_body.astype(float)
            if not silent: # Assuming 'silent' is in scope from create_xyz0 parameters
                print(f"INFO_CREATE_XYZ: Converted igrf_vector_body from dtype {igrf_vector_body.dtype} to float for norm calculation.")
        except ValueError as e:
            if not silent: # Assuming 'silent' is in scope
                print(f"WARNING_CREATE_XYZ: Could not convert igrf_vector_body to float (dtype: {igrf_vector_body.dtype}). Error: {e}. Proceeding with original.")
            # If conversion fails, use original and let np.linalg.norm handle it (it might still error)
    
    # Debugging for Error 1
    if not silent:
        print(f"DEBUG_ERROR_1: type(processed_igrf_vector_body): {type(processed_igrf_vector_body)}")
        if isinstance(processed_igrf_vector_body, np.ndarray):
            print(f"DEBUG_ERROR_1: processed_igrf_vector_body.shape: {processed_igrf_vector_body.shape}")
            print(f"DEBUG_ERROR_1: processed_igrf_vector_body.dtype: {processed_igrf_vector_body.dtype}")
        print(f"DEBUG_ERROR_1: processed_igrf_vector_body: {processed_igrf_vector_body}")

    igrf_scalar = np.linalg.norm(processed_igrf_vector_body, axis=0)
    xyz.igrf = igrf_scalar

    if save_h5:
        with h5py.File(xyz_h5, "a") as file:
            if "flux_a_x" not in file: file.create_dataset("flux_a_x", data=flux_a.x)
            else: file["flux_a_x"][:] = flux_a.x
            if "flux_a_y" not in file: file.create_dataset("flux_a_y", data=flux_a.y)
            else: file["flux_a_y"][:] = flux_a.y
            if "flux_a_z" not in file: file.create_dataset("flux_a_z", data=flux_a.z)
            else: file["flux_a_z"][:] = flux_a.z
            if "flux_a_t" not in file: file.create_dataset("flux_a_t", data=flux_a.t)
            else: file["flux_a_t"][:] = flux_a.t
            
            if "mag_1_uc" not in file: file.create_dataset("mag_1_uc", data=mag_1_uc)
            else: file["mag_1_uc"][:] = mag_1_uc
            if "mag_1_c" not in file: file.create_dataset("mag_1_c", data=mag_1_c)
            else: file["mag_1_c"][:] = mag_1_c
            
            if "flight" not in file: file.create_dataset("flight", data=flights)
            else: file["flight"][:] = flights
            if "line" not in file: file.create_dataset("line", data=lines)
            else: file["line"][:] = lines
            if "year" not in file: file.create_dataset("year", data=years)
            else: file["year"][:] = years
            if "doy" not in file: file.create_dataset("doy", data=doys)
            else: file["doy"][:] = doys
            if "diurnal" not in file: file.create_dataset("diurnal", data=diurnal)
            else: file["diurnal"][:] = diurnal
            if "igrf" not in file: file.create_dataset("igrf", data=xyz.igrf)
            else: file["igrf"][:] = xyz.igrf
    return xyz

def create_traj(
    mapS: Union[MapS, MapSd, MapS3D],
    alt: float = 1000.0,
    dt: float = 0.1,
    t: float = 300.0,
    v: float = 68.0,
    ll1: Tuple[float, float] = (),
    ll2: Tuple[float, float] = (),
    N_waves: int = 1,
    attempts: int = 10,
    save_h5: bool = False,
    traj_h5: str = "traj_data.h5"
) -> Traj:
    """
    Create Traj trajectory struct with a straight or sinusoidal flight path.
    """
    traj_h5 = add_extension(traj_h5, ".h5")

    if isinstance(mapS, MapSd) and hasattr(mapS, 'mask') and mapS.mask is not None:
        valid_alts = mapS.alt[mapS.mask] if mapS.mask.any() else [np.mean(mapS.alt)] # mapS.alt is mapS.alt_matrix for MapSd
        map_altitude_ref = np.median(valid_alts) if len(valid_alts) > 0 else np.mean(mapS.alt)
    elif isinstance(mapS, _ActualMapSd) and hasattr(mapS, 'alt_matrix'): # Explicit check for _ActualMapSd
        valid_alts = mapS.alt_matrix[mapS.mask] if hasattr(mapS, 'mask') and mapS.mask is not None and mapS.mask.any() else mapS.alt_matrix.flatten()
        map_altitude_ref = np.median(valid_alts) if len(valid_alts) > 0 else np.mean(mapS.alt_matrix)
    elif isinstance(mapS.alt, (list, np.ndarray)) and len(mapS.alt) > 0 : # For MapS3D
        map_altitude_ref = mapS.alt[0] if isinstance(mapS.alt, (list,np.ndarray)) else mapS.alt # mapS.alt is mapS.zz for MapS3D
        if isinstance(mapS, (_ActualMapS3D)) and hasattr(mapS, 'zz') and len(mapS.zz)>0: map_altitude_ref = mapS.zz[0]

    elif isinstance(mapS.alt, (float, int)): # For MapS
         map_altitude_ref = mapS.alt
    else:
        raise ValueError("Cannot determine reference altitude from mapS")

    if alt < map_altitude_ref:
        raise ValueError(f"Flight altitude {alt} < map reference altitude {map_altitude_ref}")

    i = 0
    N_pts = 2 
    lat_path = np.zeros(N_pts, dtype=float)
    lon_path = np.zeros(N_pts, dtype=float)
    
    path_found_on_map = False
    while (not path_found_on_map and i < attempts) or i == 0:
        i += 1
        if not ll1:
            lat_min, lat_max = np.min(mapS.yy), np.max(mapS.yy)
            lon_min, lon_max = np.min(mapS.xx), np.max(mapS.xx)
            lat1 = lat_min + (lat_max - lat_min) * (0.25 + 0.50 * random.random())
            lon1 = lon_min + (lon_max - lon_min) * (0.25 + 0.50 * random.random())
        else:
            lat1, lon1 = np.deg2rad(ll1[0]), np.deg2rad(ll1[1])

        if not ll2:
            N_pts = int(round(t / dt + 1))
            dist = v * t
            theta_utm = 2 * math.pi * random.random()
            lat2 = lat1 + dn2dlat(dist * math.sin(theta_utm), lat1)
            lon2 = lon1 + de2dlon(dist * math.cos(theta_utm), lat1)
        else:
            N_pts_est = 1000 * (N_waves + 1)
            lat2, lon2 = np.deg2rad(ll2[0]), np.deg2rad(ll2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        theta_ll = math.atan2(dlat, dlon)

        current_N = N_pts if not ll2 else N_pts_est
        lat_path = np.linspace(lat1, lat2, current_N)
        lon_path = np.linspace(lon1, lon2, current_N)

        if N_waves > 0:
            # Simplified wave application; direct port of Julia's complex wave scaling is involved.
            # This part might need further refinement if precise wave shapes are critical.
            phi_waves = np.linspace(0, N_waves * 2 * math.pi, current_N)
            # Assuming amplitude is relative or needs scaling factor based on path length.
            # For now, a simple perturbation perpendicular to the path.
            # A_wave_rad = 0.001 # Example amplitude in radians, adjust as needed
            # lat_path += A_wave_rad * np.cos(theta_ll) * np.sin(phi_waves) # Perpendicular to path
            # lon_path -= A_wave_rad * np.sin(theta_ll) * np.sin(phi_waves) # Perpendicular to path
            # The Julia code `wav = [ϕ sin.(ϕ)]` and rotation implies a more complex transformation.
            # Replicating Julia's `cor = wav*rot'` logic:
            _phi = np.linspace(0, 1, current_N) # Parametric distance
            # Amplitude of sin wave (relative to total length, scaled by dlat/dlon)
            # This is a simplification of the Julia logic which is non-trivial to map directly
            # without more context on the intended wave amplitude scaling.
            # The Julia code scales `sin.(phi_waves)` by rotation with `theta_ll`
            # and adds it to the base `lat1, lon1`.
            # `wav = [ϕ sin.(ϕ)]` -> `wav = np.column_stack((_phi_scaled_dist, amp_scaled * np.sin(phi_waves)))`
            # `rot_matrix = np.array([[math.cos(theta_ll), -math.sin(theta_ll)],
            #                          [math.sin(theta_ll), math.cos(theta_ll)]])`
            # `cor = np.dot(wav, rot_matrix.T)`
            # `lon_path_wave = (cor[:,0] - cor[0,0]) + lon1`
            # `lat_path_wave = (cor[:,1] - cor[0,1]) + lat1`
            # For now, this part is simplified as the exact scaling in Julia was implicit.
            pass


        dx_m = dlon2de(fdm(lon_path), lat_path)
        dy_m = dlat2dn(fdm(lat_path), lat_path)
        
        valid_segments = (dx_m.shape[0] > 1 and dy_m.shape[0] > 1)
        if valid_segments and len(dx_m) > 1 : # fdm might return N or N-1 points
            segment_distances = np.sqrt(dx_m[1:]**2 + dy_m[1:]**2)
            current_total_dist = np.sum(segment_distances)
        else:
            current_total_dist = 0.0


        if not ll2:
            if current_total_dist > 1e-6:
                scale_factor = dist / current_total_dist
                lat_path = (lat_path - lat_path[0]) * scale_factor + lat_path[0]
                lon_path = (lon_path - lon_path[0]) * scale_factor + lon_path[0]
        else:
            if len(lat_path) > 1 and abs(lat_path[-1] - lat_path[0]) > 1e-9 :
                 scale_factor_lat = (lat2 - lat_path[0]) / (lat_path[-1] - lat_path[0])
                 lat_path = (lat_path - lat_path[0]) * scale_factor_lat + lat_path[0]
            if len(lon_path) > 1 and abs(lon_path[-1] - lon_path[0]) > 1e-9:
                 scale_factor_lon = (lon2 - lon_path[0]) / (lon_path[-1] - lon_path[0])
                 lon_path = (lon_path - lon_path[0]) * scale_factor_lon + lon_path[0]
        
        if ll2:
            dx_m = dlon2de(fdm(lon_path), lat_path)
            dy_m = dlat2dn(fdm(lat_path), lat_path)
            if dx_m.shape[0] > 1 and dy_m.shape[0] > 1 and len(dx_m) > 1:
                segment_distances = np.sqrt(dx_m[1:]**2 + dy_m[1:]**2)
                true_dist = np.sum(segment_distances)
            else:
                true_dist = 0.0
            
            t_flight = true_dist / v if v > 1e-6 else 0.0
            N_pts = int(round(t_flight / dt + 1)) if dt > 1e-6 else 1
            N_pts = max(N_pts, 2) # Ensure at least 2 points for interpolation
            
            if len(lat_path) > 1 :
                current_progression = np.linspace(0, 1, len(lat_path))
                new_progression = np.linspace(0, 1, N_pts)
                interp_lat = interp1d(current_progression, lat_path, kind='linear', fill_value="extrapolate")
                interp_lon = interp1d(current_progression, lon_path, kind='linear', fill_value="extrapolate")
                lat_path = interp_lat(new_progression)
                lon_path = interp_lon(new_progression)
            elif len(lat_path) == 1: # Single point from previous step
                lat_path = np.full(N_pts, lat_path[0])
                lon_path = np.full(N_pts, lon_path[0])
            else: # No points, should not happen if N_pts_est was > 0
                lat_path = np.full(N_pts, lat1) # Fallback
                lon_path = np.full(N_pts, lon1) # Fallback

            t = t_flight

        path_found_on_map = map_check(mapS, lat_path, lon_path)
        if path_found_on_map:
            break
    
    if not path_found_on_map:
        raise RuntimeError(f"Maximum attempts ({attempts}) reached, could not create valid trajectory on map.")

    mean_lat_deg_traj = np.rad2deg(np.mean(lat_path))
    mean_lon_deg_traj = np.rad2deg(np.mean(lon_path))
    utm_zone_num, utm_is_north = utm_zone_from_latlon(mean_lat_deg_traj, mean_lon_deg_traj)
    
    utms_x, utms_y = transform_lla_to_utm(lat_path, lon_path, utm_zone_num, utm_is_north)

    vn = fdm(utms_y) / dt if dt > 1e-6 else np.zeros_like(utms_y)
    ve = fdm(utms_x) / dt if dt > 1e-6 else np.zeros_like(utms_x)
    vd = np.zeros_like(lat_path)
    
    fn = fdm(vn) / dt if dt > 1e-6 else np.zeros_like(vn)
    fe = fdm(ve) / dt if dt > 1e-6 else np.zeros_like(ve)
    fd = fdm(vd) / dt - G_EARTH if dt > 1e-6 else np.full_like(vd, -G_EARTH) # Changed to G_EARTH
    
    tt = np.linspace(0, t, N_pts)

    Cnb_array_wrong_dims = create_dcm_from_vel(vn, ve, dt, order='body2nav') # Shape (N, 3, 3)
    Cnb_array = np.transpose(Cnb_array_wrong_dims, (1, 2, 0)) # Shape (3, 3, N)
    euler_angles = dcm2euler(Cnb_array, order='body2nav') # Now Cnb_array is (3,3,N)
    roll_path  = euler_angles[:,0]
    pitch_path = euler_angles[:,1]
    yaw_path   = euler_angles[:,2]

    alt_path = np.full_like(lat_path, alt)

    if save_h5:
        with h5py.File(traj_h5, "a") as file:
            for key, data_arr in [
                ("tt", tt), ("lat", lat_path), ("lon", lon_path), ("alt", alt_path),
                ("vn", vn), ("ve", ve), ("vd", vd), ("fn", fn), ("fe", fe), ("fd", fd),
                ("roll", roll_path), ("pitch", pitch_path), ("yaw", yaw_path)
            ]:
                if key not in file: file.create_dataset(key, data=data_arr)
                else: file[key][:] = data_arr
                
    return Traj(N=N_pts, dt=dt, tt=tt, lat=lat_path, lon=lon_path, alt=alt_path,
                vn=vn, ve=ve, vd=vd, fn=fn, fe=fe, fd=fd, Cnb=Cnb_array)


def create_ins(
    traj: Traj,
    init_pos_sigma: float = 3.0,
    init_alt_sigma: float = 0.001,
    init_vel_sigma: float = 0.01,
    init_att_sigma: float = np.deg2rad(0.00001),
    VRW_sigma: float = 0.000238,
    ARW_sigma: float = 0.000000581,
    baro_sigma: float = 1.0,
    ha_sigma: float = 0.001,
    a_hat_sigma: float = 0.01,
    acc_sigma: float = 0.000245,
    gyro_sigma: float = 0.00000000727,
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    save_h5: bool = False,
    ins_h5: str = "ins_data.h5"
) -> INS:
    """
    Creates an INS trajectory about a true trajectory using a Pinson error model.
    """
    ins_h5 = add_extension(ins_h5, ".h5")

    N = traj.N
    dt = traj.dt
    nx = 17

    # Define std_acc from the acc_sigma parameter of create_ins
    std_acc = np.array([acc_sigma, acc_sigma, acc_sigma])

    # Placeholder values for missing arguments as per prompt's specific instruction
    placeholder_std_gyro = np.array([0.01, 0.01, 0.01])
    placeholder_tau_acc = np.array([100.0, 100.0, 100.0])
    placeholder_tau_gyro = np.array([100.0, 100.0, 100.0])

    P0, Qd, _ = create_ins_model_matrices(
        dt,
        std_acc, # Correctly use std_acc derived from acc_sigma
        placeholder_std_gyro,
        placeholder_tau_acc,
        placeholder_tau_gyro,
        # Pass through other relevant kwargs from the original call and create_ins parameters
        init_pos_sigma=init_pos_sigma,
        init_alt_sigma=init_alt_sigma, # Was in original kwargs
        init_vel_sigma=init_vel_sigma,
        init_att_sigma=init_att_sigma,
        VRW_sigma=VRW_sigma,       # Was in original kwargs
        ARW_sigma=ARW_sigma,       # Was in original kwargs
        baro_sigma=baro_sigma,     # Was in original kwargs
        ha_sigma=ha_sigma,         # Was in original kwargs
        a_hat_sigma=a_hat_sigma,   # Was in original kwargs
        fogm_state=False           # Was in original kwargs
    )

    P_ins = np.zeros((N, nx, nx))
    err_ins = np.zeros((N, nx))

    P_ins[0,:,:] = P0
    err_ins[0,:] = multivariate_normal.rvs(mean=np.zeros(nx), cov=P0)
    
    try:
        Qd_chol = np.linalg.cholesky(Qd)
    except np.linalg.LinAlgError:
        # If Qd is not positive definite, add small identity to diagonal
        Qd_chol = np.linalg.cholesky(Qd + np.eye(nx) * 1e-12)

    for k in range(N - 1):
        lat_k = traj.lat[0] if len(traj.lat) == 1 else traj.lat[k]
        vn_k  = traj.vn[0]  if len(traj.vn)  == 1 else traj.vn[k]
        ve_k  = traj.ve[0]  if len(traj.ve)  == 1 else traj.ve[k]
        vd_k  = traj.vd[0]  if len(traj.vd)  == 1 else traj.vd[k]
        fn_k = traj.fn[0] if len(traj.fn) == 1 else traj.fn[k]
        fe_k = traj.fe[0] if len(traj.fe) == 1 else traj.fe[k]
        fd_k = traj.fd[0] if len(traj.fd) == 1 else traj.fd[k]
        # Use modulo indexing to safely access DCM array
        dcm_index = k % len(traj.Cnb)
        Cnb_k = traj.Cnb[dcm_index]
        Phi_k = get_phi_matrix(
            nx, lat_k, vn_k, ve_k, vd_k,
            fn_k, fe_k, fd_k, Cnb_k, # Pass Cnb for current step
            baro_tau, acc_tau, gyro_tau, 0, dt, fogm_state=False
        )
        process_noise_k = Qd_chol @ np.random.randn(nx)
        err_ins[k+1,:] = Phi_k @ err_ins[k,:] + process_noise_k
        P_ins[k+1,:,:] = Phi_k @ P_ins[k,:,:] @ Phi_k.T + Qd

    # Create noisy trajectory based on true trajectory and error
    lat_ins = traj.lat + err_ins[:,0] * dn2dlat(1, traj.lat) # Convert northing error to dlat
    lon_ins = traj.lon + err_ins[:,1] * de2dlon(1, traj.lat) # Convert easting error to dlon
    alt_ins = traj.alt - err_ins[:,2] # NED: down is positive, so error in d is subtracted from alt

    vn_ins = traj.vn + err_ins[:,3]
    ve_ins = traj.ve + err_ins[:,4]
    vd_ins = traj.vd + err_ins[:,5]

    # Correct Cnb matrix with attitude error
    # err_ins[:,6:9] are tilt errors (alpha, beta, gamma for small angle approx or specific error def)
    # Assuming these are psi_nb errors (n, e, d components of rotation vector error)
    # Cnb_ins = Cnb_true @ (I - skew(psi_nb_err)) approx
    # Or if err_ins are Euler angle errors, Cnb_ins = Cnb_true @ delta_Cnb(deuler)
    # For simplicity, assuming direct correction or that correct_cnb_matrix handles it.
    # The Julia code uses: Cnb_ins[k,:,:] = correct_cnb_matrix(traj.Cnb[k,:,:], err_ins[k,6:9])
    # Use modulo indexing to safely access DCM array
    # Handle different Cnb array shapes
    if traj.Cnb.size == 1:
        # For scalar or 1-element array, create a 3x3 identity matrix
        Cnb_3d = np.eye(3).reshape(1, 3, 3)
    elif traj.Cnb.size == 9:
        # For 1D array with 9 elements, reshape to 3x3
        Cnb_3d = traj.Cnb.reshape(1, 3, 3)
    elif traj.Cnb.ndim == 2:
        # For 2D array, add time dimension
        Cnb_3d = traj.Cnb.reshape((1, 3, 3))
    else:
        Cnb_3d = traj.Cnb
        
    Cnb_ins_array = np.array([correct_cnb_matrix(Cnb_3d[k % len(Cnb_3d)], err_ins[k,6:9]) for k in range(N)])

    # Specific forces from INS (derived from noisy accels, transformed by noisy Cnb)
    # This part is complex as it depends on how acc_bias (err_ins[:,11:14]) and gyro_bias (err_ins[:,14:17])
    # are defined and how they corrupt true specific force.
    # For now, assume fn, fe, fd are "measured" specific forces in nav frame.
    # A simplified approach: add noise to true specific forces.
    # However, INS specific forces are usually derived from accelerometer outputs.
    # Let's assume for now that the error terms for fn,fe,fd are implicitly handled
    # or that the true fn,fe,fd are used as a base, which is not entirely realistic for INS output.
    # A more accurate model would involve simulating accelerometer outputs with bias/noise,
    # then transforming with Cnb_ins.
    # Using traj.fn, fe, fd as placeholders for what INS would compute before error accumulation in fn/fe/fd states.
    fn_ins = traj.fn # Placeholder - needs more accurate modeling if INS specific force output is critical
    fe_ins = traj.fe # Placeholder
    fd_ins = traj.fd # Placeholder

    ins_trajectory = INS(
        N=N, dt=dt, tt=traj.tt,
        lat=lat_ins, lon=lon_ins, alt=alt_ins,
        vn=vn_ins, ve=ve_ins, vd=vd_ins,
        fn=fn_ins, fe=fe_ins, fd=fd_ins, # These should ideally be from INS processing
        Cnb=Cnb_ins_array,
        P=P_ins
    )

    if save_h5:
        with h5py.File(ins_h5, "a") as file: # Append mode
            # Save all fields from Traj part of INS
            if "tt_ins" not in file: file.create_dataset("tt_ins", data=ins_trajectory.tt)
            else: file["tt_ins"][:] = ins_trajectory.tt # Overwrite if exists
            if "lat_ins" not in file: file.create_dataset("lat_ins", data=ins_trajectory.lat)
            else: file["lat_ins"][:] = ins_trajectory.lat
            if "lon_ins" not in file: file.create_dataset("lon_ins", data=ins_trajectory.lon)
            else: file["lon_ins"][:] = ins_trajectory.lon
            if "alt_ins" not in file: file.create_dataset("alt_ins", data=ins_trajectory.alt)
            else: file["alt_ins"][:] = ins_trajectory.alt
            if "vn_ins" not in file: file.create_dataset("vn_ins", data=ins_trajectory.vn)
            else: file["vn_ins"][:] = ins_trajectory.vn
            if "ve_ins" not in file: file.create_dataset("ve_ins", data=ins_trajectory.ve)
            else: file["ve_ins"][:] = ins_trajectory.ve
            if "vd_ins" not in file: file.create_dataset("vd_ins", data=ins_trajectory.vd)
            else: file["vd_ins"][:] = ins_trajectory.vd
            if "fn_ins" not in file: file.create_dataset("fn_ins", data=ins_trajectory.fn)
            else: file["fn_ins"][:] = ins_trajectory.fn
            if "fe_ins" not in file: file.create_dataset("fe_ins", data=ins_trajectory.fe)
            else: file["fe_ins"][:] = ins_trajectory.fe
            if "fd_ins" not in file: file.create_dataset("fd_ins", data=ins_trajectory.fd)
            else: file["fd_ins"][:] = ins_trajectory.fd
            
            # Save Cnb components (roll, pitch, yaw from INS DCMs)
            euler_ins = dcm2euler(ins_trajectory.Cnb, order='body2nav')
            if "roll_ins" not in file: file.create_dataset("roll_ins", data=euler_ins[:,0])
            else: file["roll_ins"][:] = euler_ins[:,0]
            if "pitch_ins" not in file: file.create_dataset("pitch_ins", data=euler_ins[:,1])
            else: file["pitch_ins"][:] = euler_ins[:,1]
            if "yaw_ins" not in file: file.create_dataset("yaw_ins", data=euler_ins[:,2])
            else: file["yaw_ins"][:] = euler_ins[:,2]

            # Save covariance P (example: diagonal elements for variances)
            # P_diag = np.array([P_ins[k,:,:].diagonal() for k in range(N)])
            # if "P_diag_ins" not in file: file.create_dataset("P_diag_ins", data=P_diag)
            # else: file["P_diag_ins"][:] = P_diag
            # Or save full P if needed, careful with size:
            if "P_ins" not in file: file.create_dataset("P_ins", data=P_ins)
            else: file["P_ins"][:] = P_ins
            
    return ins_trajectory

def create_mag_c(
    path_or_lat: Union[Path, np.ndarray, List[float]],
    lon_or_mapS: Union[np.ndarray, List[float], MapS, MapSd, MapS3D, None] = None,
    alt_or_itp: Union[np.ndarray, List[float], float, str, interp1d, None] = None,
    itp_kwargs: Optional[Dict[str, Any]] = None,
    map_kwargs: Optional[Dict[str, Any]] = None,
    meas_var: float = 1.0,
    fogm_sigma: float = 1.0,
    fogm_tau: float = 600.0,
    dt: Optional[float] = None, # Required if path_or_lat is not Path and fogm is used
    map_resolution_scale: float = 200.0, # Corresponds to alpha in get_map_val
    silent: bool = False
) -> np.ndarray:
    """
    Creates compensated (clean) scalar magnetometer measurements.
    Can take a Path object or explicit lat, lon, alt arrays.

    Args:
        path_or_lat: Path object (Traj or INS) or 1D array of latitudes [rad].
        lon_or_mapS: 1D array of longitudes [rad] OR a MapS/MapSd/MapS3D object.
                     Required if path_or_lat is lat array. If Path is given, this can be a map object
                     to use directly, or None/path_string if map needs to be loaded via alt_or_itp.
        alt_or_itp: 1D array of altitudes [m] OR a scalar altitude OR a map path string OR
                    a precomputed scipy interpolator. Required if path_or_lat is lat array.
                    If Path is given and lon_or_mapS is not a map object, this can be a map_path string.
        itp_kwargs: Dict of keyword arguments for scipy.interpolate.interp1d if creating interpolator.
        map_kwargs: Dict of keyword arguments for get_map if loading map from path.
        meas_var: Measurement variance for white noise [nT^2].
        fogm_sigma: FOGM noise standard deviation [nT].
        fogm_tau: FOGM noise correlation time [s].
        dt: Time step [s], required for FOGM noise if path_or_lat is not a Path object.
        map_resolution_scale: Scaling factor for map resolution, passed as 'alpha' to get_map_val.
        silent: Suppress print statements.

    Returns:
        mag_c: 1D array of compensated scalar magnetometer measurements [nT].
    """
    if itp_kwargs is None:
        itp_kwargs = {}
    if map_kwargs is None:
        map_kwargs = {}

    map_val: np.ndarray
    num_points: int

    if isinstance(path_or_lat, (Traj, INS)): # Path object provided
        num_points = path_or_lat.N
        if dt is None: # dt for FOGM noise
            dt = path_or_lat.dt

        # MODIFIED LOGIC FOR MAP SELECTION
        map_to_use = None # Initialize

        # Define all possible map types for lon_or_mapS.
        _map_types = (_ActualMapS, _ActualMapSd, _ActualMapS3D, MapS, MapSd, MapS3D)

        if isinstance(lon_or_mapS, _map_types):
            map_to_use = lon_or_mapS
        
        if map_to_use is None: # If no map object from lon_or_mapS
            if isinstance(alt_or_itp, str): # Check if alt_or_itp is a path
                _path_to_load_map_from = alt_or_itp
                if not os.path.exists(_path_to_load_map_from):
                     _path_to_load_map_from = get_map(_path_to_load_map_from, return_path=True)
                if not silent: print(f"Loading map from path: {_path_to_load_map_from}")
                final_map_kwargs = {k: v for k, v in map_kwargs.items() if k != 'silent'}
                map_to_use = get_map(_path_to_load_map_from, **final_map_kwargs, silent=silent)
            else: # Not a map object, and not a path string, so use default
                if not silent: print(f"Using default scalar map. No valid map object or path string given.")
                final_map_kwargs = {k: v for k, v in map_kwargs.items() if k != 'silent'}
                map_to_use = get_map(constants.DEFAULT_SCALAR_MAP_ID, **final_map_kwargs, silent=silent)
        
        # Ensure map_to_use is not None before proceeding (critical fallback)
        if map_to_use is None:
            if not silent: print(f"Critical fallback: map_to_use was still None after checks. Using default scalar map.")
            final_map_kwargs = {k: v for k, v in map_kwargs.items() if k != 'silent'}
            map_to_use = get_map(constants.DEFAULT_SCALAR_MAP_ID, **final_map_kwargs, silent=silent)
        # END OF MODIFIED LOGIC
        
        map_val = get_map_val(map_to_use, path_or_lat.lat, path_or_lat.lon, path_or_lat.alt,
                               return_interpolator=False)

    elif isinstance(path_or_lat, (np.ndarray, list)): # lat, lon, alt arrays provided
        _lat = np.asarray(path_or_lat)
        _lon = np.asarray(lon_or_mapS) # Here lon_or_mapS must be lon array
        _alt = np.asarray(alt_or_itp) if not callable(alt_or_itp) and not isinstance(alt_or_itp, str) else alt_or_itp # alt_or_itp can be alt array, scalar, path, or itp

        if not (_lat.ndim == 1 and _lon.ndim == 1):
            raise ValueError("Latitude and longitude must be 1D arrays.")
        if _lat.shape != _lon.shape:
            raise ValueError("Latitude and longitude arrays must have the same shape.")
        num_points = _lat.shape[0]

        if callable(alt_or_itp): # Precomputed interpolator
            if not silent: print("Using precomputed interpolator for map values.")
            points = np.column_stack((_lat, _lon))
            map_val = alt_or_itp(points) 
        elif isinstance(alt_or_itp, str) or isinstance(lon_or_mapS, (_ActualMapS, _ActualMapSd, _ActualMapS3D, MapS, MapSd, MapS3D)):
            map_obj_or_path = lon_or_mapS if isinstance(lon_or_mapS, (_ActualMapS, _ActualMapSd, _ActualMapS3D, MapS, MapSd, MapS3D)) else alt_or_itp
            
            current_map_to_use: Union[_ActualMapS, _ActualMapSd, _ActualMapS3D, MapS, MapSd, MapS3D, None] = None
            if isinstance(map_obj_or_path, (_ActualMapS, _ActualMapSd, _ActualMapS3D, MapS, MapSd, MapS3D)):
                current_map_to_use = map_obj_or_path
            elif isinstance(map_obj_or_path, str):
                _path = map_obj_or_path
                if not os.path.exists(_path): _path = get_map(_path, return_path=True)
                if not silent: print(f"Loading map from path: {_path}")
                current_map_to_use = get_map(_path, map_kwargs=map_kwargs, silent=silent)
            else: 
                if not silent: print(f"Using default map for array input due to unclear map source.")
                current_map_to_use = get_map(DEFAULT_SCALAR_MAP_ID, map_kwargs=map_kwargs, silent=silent)

            if current_map_to_use is None: 
                 raise RuntimeError("Failed to obtain a map object for array input.")

            alt_for_map_val = _alt if isinstance(_alt, (np.ndarray, list)) or isinstance(_alt, (float,int)) else None
            if alt_for_map_val is None: 
                if isinstance(current_map_to_use.alt, (float,int)):
                    alt_for_map_val = current_map_to_use.alt
                    if not silent: print(f"Using map's own altitude ({alt_for_map_val}m) for map value calculation.")
                else: 
                    raise ValueError("Altitude must be provided as array or scalar when lat/lon are arrays and map is 3D/drape without explicit point altitudes.")

            map_val = get_map_val(current_map_to_use, _lat, _lon, alt_for_map_val,
                                  alpha=map_resolution_scale, return_interpolator=False)
        else: 
            if not silent: print("Using default scalar map for array input.")
            default_map = get_map(DEFAULT_SCALAR_MAP_ID, map_kwargs=map_kwargs, silent=silent)
            map_val = get_map_val(default_map, _lat, _lon, _alt, 
                                  alpha=map_resolution_scale, return_interpolator=False)
    else:
        raise TypeError("path_or_lat must be a Path object or a list/array of latitudes.")

    # Add FOGM noise
    if fogm_sigma > 0 and fogm_tau > 0:
        if dt is None:
            raise ValueError("dt must be provided for FOGM noise when not using a Path object.")
        fogm_noise = generate_fogm_noise(fogm_sigma, fogm_tau, dt, num_points)
        map_val += fogm_noise

    # Add white measurement noise
    if meas_var > 0:
        white_noise = np.random.randn(num_points) * np.sqrt(meas_var)
        map_val += white_noise
        
    return map_val


def corrupt_mag(
    mag_c: np.ndarray,
    flux_a: MagV, # True vector field in aircraft body frame
    traj_ideal: Traj,
    dt: float,
    cor_sigma: float = 1.0,
    cor_tau: float = 600.0,
    cor_var: float = 1.0**2,
    cor_drift: float = 0.001,
    cor_perm_mag: float = 5.0,
    cor_ind_mag: float = 5.0,
    cor_eddy_mag: float = 0.5,
    terms: Optional[List[str]] = None,
    Bt_scale: float = 50000.0 # Typical Earth field magnitude for scaling TL coeffs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrupts clean scalar magnetometer data with various noise sources.

    Args:
        mag_c: Clean scalar magnetometer data [nT].
        flux_a: True vector magnetometer measurements (MagV struct) in aircraft body frame [nT].
        dt: Time step [s].
        cor_sigma: FOGM noise std dev for diurnal [nT].
        cor_tau: FOGM noise correlation time for diurnal [s].
        cor_var: Measurement variance for white noise on diurnal [nT^2].
        cor_drift: Constant drift rate for diurnal [nT/s].
        cor_perm_mag: Magnitude of permanent magnetic field coefficients [nT].
        cor_ind_mag: Magnitude of induced magnetic field coefficients [-].
        cor_eddy_mag: Magnitude of eddy current magnetic field coefficients [s].
        terms: Tolles-Lawson terms to use for aircraft field. Defaults to ['permanent', 'induced', 'eddy'].
        Bt_scale: Scaling factor for Tolles-Lawson coefficients, typically Earth's total field strength.


    Returns:
        Tuple containing:
            - mag_uc (np.ndarray): Uncompensated (corrupted) scalar mag data [nT].
            - aircraft_field_scalar (np.ndarray): Scalar aircraft-generated magnetic field [nT].
            - diurnal_effect (np.ndarray): Simulated diurnal variation [nT].
    """
    if terms is None:
        terms = ['permanent', 'induced', 'eddy']

    N = len(mag_c)

    # 1. Simulate diurnal variation
    diurnal_fogm = generate_fogm_noise(cor_sigma, cor_tau, dt, N)
    diurnal_drift = cor_drift * np.arange(N) * dt
    diurnal_white_noise = np.random.randn(N) * np.sqrt(cor_var)
    diurnal_effect = diurnal_fogm + diurnal_drift + diurnal_white_noise

    # 2. Simulate aircraft-generated magnetic field (Tolles-Lawson)
    B_earth_body_x = flux_a.x
    B_earth_body_y = flux_a.y
    B_earth_body_z = flux_a.z
    B_earth_body = np.vstack((B_earth_body_x, B_earth_body_y, B_earth_body_z)) # Shape (3, N)

    B_earth_body_dot_x = fdm(B_earth_body_x) / dt
    B_earth_body_dot_y = fdm(B_earth_body_y) / dt
    B_earth_body_dot_z = fdm(B_earth_body_z) / dt
    B_earth_body_dot = np.vstack((B_earth_body_dot_x, B_earth_body_dot_y, B_earth_body_dot_z))

    c_p_rand = (np.random.rand(3) - 0.5) * 2 * cor_perm_mag
    
    rand_i = (np.random.rand(6) - 0.5) * 2 * cor_ind_mag
    c_i_rand = np.array([
        [rand_i[0], rand_i[1], rand_i[2]],
        [rand_i[1], rand_i[3], rand_i[4]],
        [rand_i[2], rand_i[4], rand_i[5]]
    ])
    
    rand_e = (np.random.rand(9) - 0.5) * 2 * cor_eddy_mag
    c_e_rand = rand_e.reshape((3,3))

    igrf_field_ideal_bf = get_igrf_magnetic_field(traj_ideal) # Pass the Traj object directly
    aircraft_field_vector = get_tolles_lawson_aircraft_field_vector(
        B_earth_body,
        B_earth_body_dot,
        c_p_rand,
        c_i_rand,
        c_e_rand,
        0.0,  # placeholder for norm_flux_b
        0.0,  # placeholder for norm_flux_c
        traj_ideal.Cnb,  # dcm_data
        igrf_field_ideal_bf # igrf_bf
        )

    B_norm = np.linalg.norm(B_earth_body, axis=0, keepdims=True)
    B_norm[B_norm == 0] = 1e-9
    unit_B_earth = B_earth_body / B_norm
    aircraft_field_scalar = np.sum(aircraft_field_vector * unit_B_earth, axis=0)
    
    perturbed_total_field_vector = B_earth_body + aircraft_field_vector
    mag_after_ac_interference = np.linalg.norm(perturbed_total_field_vector, axis=0)
    
    mag_uc = mag_after_ac_interference + diurnal_effect

    return mag_uc, aircraft_field_scalar, diurnal_effect


def create_flux(
    path_or_lat: Union[Path, np.ndarray, List[float]],
    lon_or_mapV: Union[np.ndarray, List[float], MapV, None] = None, # Can be MapV object
    alt_or_itp: Union[np.ndarray, List[float], float, str, interp1d, None] = None, # Can be map path for MapV
    itp_kwargs: Optional[Dict[str, Any]] = None,
    map_kwargs: Optional[Dict[str, Any]] = None,
    meas_var: float = 1.0, # Variance for each component
    fogm_sigma: float = 1.0, # FOGM std dev for each component
    fogm_tau: float = 600.0, # FOGM correlation time
    dt: Optional[float] = None,
    map_resolution_scale: float = 200.0,
    silent: bool = False,
    default_map_id_override: Optional[str] = None # New parameter
) -> MagV:
    """
    Creates compensated (clean) vector magnetometer measurements (MagV struct).
    Similar to create_mag_c but for vector data using MapV.
    """
    if itp_kwargs is None: itp_kwargs = {}
    if map_kwargs is None: map_kwargs = {}

    map_val_x, map_val_y, map_val_z = None, None, None
    num_points: int
    
    # This will hold the actual MapV object to use for getting values.
    # It can be passed directly (as lon_or_mapV), or loaded via path (alt_or_itp as str),
    # or loaded via default_map_id_override, or loaded via constants.DEFAULT_VECTOR_MAP_ID.
    map_object_to_use: Optional[MapV] = None

    # This will hold the name/ID of the map if it needs to be loaded.
    map_id_to_load: Optional[str] = None

    if isinstance(lon_or_mapV, (_ActualMapS, _ActualMapSd, _ActualMapS3D, MapV)): # Check if MapV object is directly passed
        map_object_to_use = lon_or_mapV
        if not silent: print(f"DEBUG_CREATE_FLUX: Using directly provided MapV object: {type(map_object_to_use)}")
    elif isinstance(alt_or_itp, str): # A map path/ID string is provided via alt_or_itp
        map_id_to_load = alt_or_itp
        if not silent: print(f"DEBUG_CREATE_FLUX: Map path/ID provided via alt_or_itp: '{map_id_to_load}'")
    elif default_map_id_override is not None: # An override ID is provided
        map_id_to_load = default_map_id_override
        if not silent: print(f"DEBUG_CREATE_FLUX: Using default_map_id_override: '{map_id_to_load}'")
    else: # Fallback to the global constant
        map_id_to_load = constants.DEFAULT_VECTOR_MAP_ID
        if not silent: print(f"DEBUG_CREATE_FLUX: Using constants.DEFAULT_VECTOR_MAP_ID: '{map_id_to_load}'")

    # Load the map if an ID was determined and no map object was directly passed
    if map_object_to_use is None and map_id_to_load is not None:
        current_map_kwargs = map_kwargs if map_kwargs is not None else {}
        print(f"DEBUG_CREATE_FLUX: Attempting to load map. Name: '{map_id_to_load}', Type: 'vector', kwargs: {current_map_kwargs}, silent: {silent}")
        map_object_to_use = get_map(map_id_to_load, map_type="vector", map_kwargs=current_map_kwargs, silent=silent)
        if map_object_to_use is None:
            raise ValueError(f"Failed to load vector map: {map_id_to_load}")
        if not silent: print(f"DEBUG_CREATE_FLUX: Successfully loaded map: {type(map_object_to_use)} using ID '{map_id_to_load}'")
    elif map_object_to_use is None and not callable(alt_or_itp): # Check if interpolator was passed as alt_or_itp
         # This case means no map object, no ID to load, and alt_or_itp is not an interpolator.
        raise ValueError("create_flux: Could not determine MapV object or a valid map ID to load, and no interpolator provided.")


    # Determine N and dt from path_or_lat
    if isinstance(path_or_lat, (Traj, INS)):
        path_data = path_or_lat
        num_points = path_data.N
        if dt is None: dt = path_data.dt
        lat_arr, lon_arr, alt_arr = path_data.lat, path_data.lon, path_data.alt
    elif isinstance(path_or_lat, (np.ndarray, list)): # lat, lon, alt arrays
        lat_arr = np.asarray(path_or_lat)
        if not isinstance(lon_or_mapV, (np.ndarray, list)) or not isinstance(alt_or_itp, (np.ndarray, list, float, int)):
            if not (map_object_to_use is not None or callable(alt_or_itp)): # if map object not set and alt_or_itp not interpolator
                 raise ValueError("If path_or_lat is lat array, lon_or_mapV (lon) and alt_or_itp (alt/interpolator) must be provided, or a map object determined.")
        
        lon_arr = np.asarray(lon_or_mapV) if isinstance(lon_or_mapV, (np.ndarray, list)) else None # May be None if map_object_to_use is set
        
        # Handle alt_input: can be array, scalar, or taken from map if map_object_to_use is available
        if isinstance(alt_or_itp, (np.ndarray, list)):
            alt_arr = np.asarray(alt_or_itp)
        elif isinstance(alt_or_itp, (float, int)):
            alt_arr = np.full_like(lat_arr, float(alt_or_itp))
        elif map_object_to_use is not None and hasattr(map_object_to_use, 'alt'): # Use map's altitude if alt_or_itp is not alt data
            alt_arr = np.full_like(lat_arr, map_object_to_use.alt) if isinstance(map_object_to_use.alt, (float,int)) else map_object_to_use.alt
        elif callable(alt_or_itp): # If alt_or_itp is an interpolator, alt_arr might not be directly used for get_map_val
            alt_arr = None # Placeholder, actual altitude might be part of interpolator's input
        else:
            raise ValueError("Altitude data (array/scalar) or a map with altitude attribute is required for array input.")

        num_points = len(lat_arr)
        if lon_arr is not None and len(lat_arr) != len(lon_arr): raise ValueError("Lat/lon arrays must have same length.")
        if alt_arr is not None and isinstance(alt_arr, np.ndarray) and len(lat_arr) != len(alt_arr): raise ValueError("Lat/alt arrays must have same length.")
        if dt is None: raise ValueError("dt must be provided if path_or_lat is an array.")
    else:
        raise TypeError("path_or_lat must be a Path object (Traj/INS) or latitude array.")

    # Get map values
    if map_object_to_use is not None:
        if not silent: print(f"DEBUG_CREATE_FLUX: Getting map values from {type(map_object_to_use)}")
        # Ensure lon_arr and alt_arr are valid for get_map_val
        _lon_for_getmap = lon_arr if lon_arr is not None else np.zeros_like(lat_arr) # get_map_val might need some lon
        _alt_for_getmap = alt_arr if alt_arr is not None else (np.full_like(lat_arr, map_object_to_use.alt) if hasattr(map_object_to_use, 'alt') else np.zeros_like(lat_arr))

        map_vals_tuple = get_map_val(map_object_to_use, lat_arr, _lon_for_getmap, _alt_for_getmap,
                                     return_interpolator=False)
        if isinstance(map_vals_tuple, tuple) and len(map_vals_tuple) == 3:
            map_val_x, map_val_y, map_val_z = map_vals_tuple
        else:
            raise ValueError("get_map_val with MapV did not return a 3-component tuple.")
    elif callable(alt_or_itp): # Use interpolator passed via alt_or_itp
        if not silent: print(f"DEBUG_CREATE_FLUX: Using provided interpolator function.")
        # Assumes interpolator takes points (lat, lon, alt) and returns (vx, vy, vz)
        # This requires lat_arr, lon_arr, and alt_arr to be correctly set up for the interpolator.
        if lon_arr is None or alt_arr is None:
            raise ValueError("lon_arr and alt_arr must be available for interpolator.")
        points_for_itp = np.column_stack((lat_arr, lon_arr, alt_arr))
        map_val_x, map_val_y, map_val_z = alt_or_itp(points_for_itp) # Unpack directly
    else:
        raise RuntimeError("Vector map components could not be determined (no map object and no interpolator).")

    if map_val_x is None or map_val_y is None or map_val_z is None:
        raise RuntimeError("Vector map components (map_val_x, y, or z) are None before noise addition.")

    # Add FOGM noise
    if fogm_sigma > 0 and fogm_tau > 0:
        if dt is None: raise ValueError("dt required for FOGM noise.")
        map_val_x += generate_fogm_noise(fogm_sigma, fogm_tau, dt, num_points)
        map_val_y += generate_fogm_noise(fogm_sigma, fogm_tau, dt, num_points)
        map_val_z += generate_fogm_noise(fogm_sigma, fogm_tau, dt, num_points)

    # Add measurement variance
    if meas_var > 0:
        std_dev = np.sqrt(meas_var)
        map_val_x += np.random.randn(num_points) * std_dev
        map_val_y += np.random.randn(num_points) * std_dev
        map_val_z += np.random.randn(num_points) * std_dev
        
    total_mag = np.sqrt(map_val_x**2 + map_val_y**2 + map_val_z**2)

    return MagV(x=map_val_x, y=map_val_y, z=map_val_z, t=total_mag)


def create_dcm_internal(
    roll: Union[float, np.ndarray],
    pitch: Union[float, np.ndarray],
    yaw: Union[float, np.ndarray],
    order: str = 'body2nav'
) -> np.ndarray:
    """
    Internal helper to create DCM matrix/matrices from Euler angles.
    Uses euler2dcm.
    """
    if isinstance(roll, (float, int)) and isinstance(pitch, (float, int)) and isinstance(yaw, (float, int)):
        return euler2dcm(roll, pitch, yaw, order)
    elif isinstance(roll, np.ndarray) and isinstance(pitch, np.ndarray) and isinstance(yaw, np.ndarray):
        if not (roll.shape == pitch.shape == yaw.shape and roll.ndim == 1):
            raise ValueError("If Euler angles are arrays, they must be 1D and have the same shape.")
        return euler2dcm(roll, pitch, yaw, order) 
    else:
        raise TypeError("Euler angles must be all scalars or all 1D numpy arrays of the same length.")


def calculate_imputed_TL_earth(
    xyz: XYZ0, 
    coeffs_TL: np.ndarray, 
    terms_A: Optional[List[str]] = None, 
    Bt_scale: float = 50000.0 
    ) -> np.ndarray:
    """
    Calculate the imputed Earth's magnetic field (scalar) based on uncompensated
    magnetometer data and Tolles-Lawson coefficients.
    B_earth_imputed = B_uc - B_aircraft_TL_modelled
    Args:
        xyz: XYZ0 data structure containing uncompensated mag data and aircraft state.
        coeffs_TL: Array of Tolles-Lawson coefficients.
        terms_A: List of terms defining the structure of the Tolles-Lawson A matrix.
                 Defaults to ['permanent', 'induced', 'eddy', 'bias'].
        Bt_scale: Total field scaling factor used with coeffs_TL.

    Returns:
        Imputed Earth's scalar magnetic field [nT].
    """
    if terms_A is None:
        terms_A = ['permanent', 'induced', 'eddy', 'bias'] 

    mag_uc = xyz.mag_1_uc
    
    igrf_vec_body = get_igrf_magnetic_field(xyz, frame='body', norm_igrf=False, check_xyz=False)
    Bx_earth, By_earth, Bz_earth = igrf_vec_body[0,:], igrf_vec_body[1,:], igrf_vec_body[2,:]

    A_TL = create_tolles_lawson_A_matrix(Bx_earth, By_earth, Bz_earth, terms=terms_A, dt=xyz.traj.dt)

    B_ac_scalar_model = A_TL @ coeffs_TL 
    
    mag_earth_imputed = mag_uc - B_ac_scalar_model
    
    return mag_earth_imputed


def create_informed_xyz(
    xyz_file: str,
    map_file: Optional[str] = None, 
    map_id: Optional[str] = None,   
    map_type: str = "scalar",       
    flight_info: Optional[Dict[str, Any]] = None, 
    comp_coeffs: Optional[np.ndarray] = None, 
    comp_terms_A: Optional[List[str]] = None,
    comp_Bt_scale: float = 50000.0,
    force_recalc_igrf: bool = False,
    force_recalc_mag_c: bool = False, 
    silent: bool = False
) -> XYZ0: 
    """
    Reads an XYZ data file (text or HDF5) and enriches it with map data,
    IGRF data, and potentially compensated magnetometer data if coefficients are provided.

    Args:
        xyz_file: Path to the XYZ data file.
        map_file: Optional path to a map file (e.g., .h5).
        map_id: Optional map identifier (e.g., "namad") if map_file is not given.
        map_type: Type of map ("scalar" or "vector"), relevant if loading map.
        flight_info: Dictionary to set/override 'flight', 'line', 'year', 'doy'.
                     Example: {'flight': 101, 'year': 2022}
        comp_coeffs: Optional Tolles-Lawson coefficients to calculate mag_1_c.
        comp_terms_A: Terms for TL A-matrix if comp_coeffs are used.
        comp_Bt_scale: Bt_scale for TL if comp_coeffs are used.
        force_recalc_igrf: If true, recalculate IGRF even if present.
        force_recalc_mag_c: If true and comp_coeffs given, recalc mag_1_c.
        silent: Suppress print statements.

    Returns:
        An XYZ0 (or similar) data object.
    """
    if not os.path.exists(xyz_file):
        raise FileNotFoundError(f"XYZ file not found: {xyz_file}")

    if xyz_file.endswith(".h5"):
        try:
            xyz_data = h5_to_xyz0(xyz_file) 
            if not silent: print(f"Loaded base data from HDF5: {xyz_file}")
        except Exception as e:
            raise ValueError(f"Could not load base XYZ data from HDF5 file {xyz_file}: {e}")
    else: 
        raise NotImplementedError("Reading from non-HDF5 XYZ files into full structure is not yet fully implemented here.")

    if flight_info:
        for key, value in flight_info.items():
            if hasattr(xyz_data, key) and isinstance(getattr(xyz_data,key), np.ndarray):
                try:
                    current_array = getattr(xyz_data, key)
                    current_array[:] = value 
                    if not silent: print(f"Updated '{key}' in XYZ data.")
                except Exception as e:
                    if not silent: print(f"Warning: Could not update '{key}': {e}")
            elif hasattr(xyz_data, key): 
                 setattr(xyz_data, key, value)
                 if not silent: print(f"Updated '{key}' in XYZ data.")


    if force_recalc_igrf or not hasattr(xyz_data, 'igrf') or xyz_data.igrf is None or np.all(xyz_data.igrf == 0):
        if not silent: print("Calculating IGRF field...")
        igrf_vec_body = get_igrf_magnetic_field(xyz_data, frame='body', norm_igrf=False, check_xyz=True)
        xyz_data.igrf = np.linalg.norm(igrf_vec_body, axis=0)
    
    if not hasattr(xyz_data, 'mag_map') or xyz_data.mag_map is None:
        selected_map_source = map_file if map_file else map_id
        if selected_map_source:
            if not silent: print(f"Calculating map values from source: {selected_map_source}...")
            if map_type == "scalar":
                actual_map_obj = get_map(selected_map_source, map_type="scalar", silent=silent)
                xyz_data.mag_map = create_mag_c(xyz_data.traj, actual_map_obj, None, 
                                                meas_var=0, fogm_sigma=0, silent=silent)
            else:
                if not silent: print(f"Vector map data population to a specific field not implemented in this simplified version.")
        else:
            if not silent: print("No map source provided for 'mag_map' field, skipping.")

    if comp_coeffs is not None:
        if force_recalc_mag_c or not hasattr(xyz_data, 'mag_1_c') or xyz_data.mag_1_c is None or np.all(xyz_data.mag_1_c == 0):
            if not silent: print("Calculating compensated mag data (mag_1_c) using provided coefficients...")
            if not hasattr(xyz_data, 'mag_1_uc') or xyz_data.mag_1_uc is None:
                raise ValueError("mag_1_uc is required in XYZ data to calculate mag_1_c with coefficients.")
            
            xyz_data.mag_1_c = calculate_imputed_TL_earth(
                xyz_data, comp_coeffs, comp_terms_A, comp_Bt_scale
            )
    elif not hasattr(xyz_data, 'mag_1_c') or xyz_data.mag_1_c is None:
        if hasattr(xyz_data, 'mag_map') and xyz_data.mag_map is not None:
            if not silent: print("Using 'mag_map' as 'mag_1_c' (compensated data proxy).")
            xyz_data.mag_1_c = deepcopy(xyz_data.mag_map)
        elif map_file or map_id : 
             actual_map_obj = get_map(map_file if map_file else map_id, map_type="scalar", silent=silent)
             if not silent: print(f"Calculating 'mag_1_c' from map {map_file if map_file else map_id} as clean data proxy.")
             xyz_data.mag_1_c = create_mag_c(xyz_data.traj, actual_map_obj, None,
                                             meas_var=0, fogm_sigma=0, silent=silent)


    if not hasattr(xyz_data, 'flux_a') or xyz_data.flux_a is None:
        if hasattr(xyz_data, 'igrf') and xyz_data.igrf is not None and hasattr(xyz_data.traj, 'N'):
            igrf_vec_body_for_flux = get_igrf_magnetic_field(xyz_data, frame='body', norm_igrf=False, check_xyz=False)
            xyz_data.flux_a = MagV(x=igrf_vec_body_for_flux[0,:],
                                   y=igrf_vec_body_for_flux[1,:],
                                   z=igrf_vec_body_for_flux[2,:],
                                   t=xyz_data.igrf) 
            if not silent: print("Populated flux_a using IGRF body vector components.")
        else:
            if not silent: print("Warning: flux_a is missing and could not be derived from IGRF.")
            dummy_field = np.zeros(xyz_data.traj.N if hasattr(xyz_data,'traj') else 1)
            xyz_data.flux_a = MagV(x=dummy_field,y=dummy_field,z=dummy_field,t=dummy_field)

    if hasattr(xyz_data, 'traj') and hasattr(xyz_data.traj, 'N'):
        default_len = xyz_data.traj.N
        for field_name in ['flight', 'line', 'year', 'doy', 'diurnal', 'igrf', 'mag_1_c', 'mag_1_uc']:
            if not hasattr(xyz_data, field_name) or getattr(xyz_data, field_name) is None:
                if not silent: print(f"Warning: XYZ field '{field_name}' missing. Initializing with zeros/defaults.")
                setattr(xyz_data, field_name, np.zeros(default_len))
    
    return xyz_data


# --- Filename generation ---
def xyz_file_name(
    project: str,
    flight: Union[int, str],
    line: Union[int, str, None] = None,
    xyz_type: str = "xyz", # "xyz", "xyz0", "xyz1", "xyz20", "xyz21"
    ext: str = ".h5"
) -> str:
    """
    Generates a standardized XYZ file name.
    Example: project_Flt1003_L10.xyz0.h5
    """
    name = f"{project}_Flt{flight}"
    if line is not None:
        name += f"_L{line}"
    name += f".{xyz_type}"
    name = add_extension(name, ext)
    return name

# --- XYZ File I/O ---
def write_xyz(
    xyz_data: Union[XYZ0, Any], 
    file_path: str,
    overwrite: bool = False,
    silent: bool = False
):
    """
    Writes an XYZ data structure to an HDF5 file.
    This is a basic example; a full implementation would handle all XYZ types
    and their specific fields. For now, focuses on XYZ0-like structure.
    """
    file_path = add_extension(file_path, ".h5")

    if os.path.exists(file_path) and not overwrite:
        if not silent: print(f"File {file_path} already exists. Use overwrite=True to replace.")
        return

    mode = "w" if overwrite else "w-" 

    try:
        with h5py.File(file_path, mode) as f:
            if hasattr(xyz_data, 'info'):
                f.attrs['info'] = str(xyz_data.info)
            
            if hasattr(xyz_data, 'traj'):
                traj_group = f.create_group("traj")
                for key, value in xyz_data.traj.__dict__.items():
                    if isinstance(value, np.ndarray):
                        traj_group.create_dataset(key, data=value)
                    elif value is not None:
                         traj_group.attrs[key] = value


            if hasattr(xyz_data, 'ins'):
                ins_group = f.create_group("ins")
                for key, value in xyz_data.ins.__dict__.items():
                    if isinstance(value, np.ndarray):
                        ins_group.create_dataset(key, data=value)
                    elif value is not None:
                        ins_group.attrs[key] = value
            
            if hasattr(xyz_data, 'flux_a') and isinstance(xyz_data.flux_a, MagV):
                flux_a_group = f.create_group("flux_a")
                for key, value in xyz_data.flux_a.__dict__.items():
                     if isinstance(value, np.ndarray):
                        flux_a_group.create_dataset(key, data=value)

            direct_fields = ['flight', 'line', 'year', 'doy', 'diurnal', 'igrf', 'mag_1_c', 'mag_1_uc']
            for field_name in direct_fields:
                if hasattr(xyz_data, field_name):
                    value = getattr(xyz_data, field_name)
                    if isinstance(value, np.ndarray):
                        f.create_dataset(field_name, data=value)
                    elif value is not None: 
                        f.attrs[field_name] = value
            
            if not silent: print(f"XYZ data successfully written to {file_path}")

    except Exception as e:
        if not silent: print(f"Error writing XYZ data to {file_path}: {e}")
        if os.path.exists(file_path) and mode == "w-":
            try:
                os.remove(file_path)
            except OSError:
                pass 
        raise


def read_xyz_file(file_path: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Reads a CSV-like XYZ file into a pandas DataFrame.
    This is a basic reader; column names and types might need adjustment.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XYZ text file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, skipinitialspace=True)
        return df
    except Exception as e:
        raise ValueError(f"Error reading XYZ text file {file_path}: {e}")


def h5_to_xyz0(file_path: str, silent: bool = False) -> XYZ0:
    """
    Loads data from an HDF5 file into an XYZ0 structure.
    This is a simplified loader, assuming a specific HDF5 structure created by `write_xyz` or similar.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    data_dict = {}
    traj_data = {}
    ins_data = {}
    flux_a_data = {}

    with h5py.File(file_path, 'r') as f:
        data_dict['info'] = f.attrs.get('info', "Loaded from HDF5")

        if "traj" in f:
            for key in f["traj"].keys(): traj_data[key] = f["traj"][key][:]
            for key in f["traj"].attrs.keys(): traj_data[key] = f["traj"].attrs[key]
        if "ins" in f:
            for key in f["ins"].keys(): ins_data[key] = f["ins"][key][:]
            for key in f["ins"].attrs.keys(): ins_data[key] = f["ins"].attrs[key]
        if "flux_a" in f:
            for key in f["flux_a"].keys(): flux_a_data[key] = f["flux_a"][key][:]
        
        direct_fields = ['flight', 'line', 'year', 'doy', 'diurnal', 'igrf', 'mag_1_c', 'mag_1_uc']
        for field_name in direct_fields:
            if field_name in f: data_dict[field_name] = f[field_name][:]
            elif field_name in f.attrs: data_dict[field_name] = f.attrs[field_name]


    traj_obj = Traj(**{k: v for k, v in traj_data.items() if k in Traj.__annotations__})
    
    ins_fields_from_traj = {k: v for k,v in traj_data.items() if k in INS.__annotations__ and k not in ins_data}
    ins_obj = INS(**ins_fields_from_traj, **{k: v for k, v in ins_data.items() if k in INS.__annotations__})

    flux_a_obj = MagV(**{k: v for k, v in flux_a_data.items() if k in MagV.__annotations__})

    xyz_data_obj = XYZ0(
        info=data_dict.get('info', "Info missing"),
        traj=traj_obj,
        ins=ins_obj,
        flux_a=flux_a_obj,
        flight=data_dict.get('flight', np.array([])),
        line=data_dict.get('line', np.array([])),
        year=data_dict.get('year', np.array([])),
        doy=data_dict.get('doy', np.array([])),
        diurnal=data_dict.get('diurnal', np.array([])),
        igrf=data_dict.get('igrf', np.array([])),
        mag_1_c=data_dict.get('mag_1_c', np.array([])),
        mag_1_uc=data_dict.get('mag_1_uc', np.array([]))
    )
    if not silent: print(f"Data loaded into XYZ0 structure from {file_path}")
    return xyz_data_obj


def create_xyz(
    xyz_file: Optional[str] = None,
    map_file: Optional[str] = None,
    map_id: Optional[str] = None,
    alt: float = 1000.0, dt: float = 0.1, t: float = 300.0, v: float = 68.0, 
    flight_info: Optional[Dict[str, Any]] = None,
    comp_coeffs: Optional[np.ndarray] = None,
    silent: bool = False,
    **kwargs 
) -> XYZ0:
    """
    Master function to create or load and inform XYZ data.
    - If xyz_file is provided, it attempts to load and enrich it.
    - If xyz_file is None, it generates new data using create_xyz0 and other params.
    """
    if xyz_file:
        if not os.path.exists(xyz_file):
            raise FileNotFoundError(f"Specified xyz_file {xyz_file} not found for loading.")
        
        return create_informed_xyz(
            xyz_file,
            map_file=map_file,
            map_id=map_id,
            flight_info=flight_info,
            comp_coeffs=comp_coeffs,
            silent=silent
        )
    else:
        if not silent: print("No xyz_file provided, generating new data using create_xyz0...")
        
        mapS_for_create = None
        if map_file:
            mapS_for_create = get_map(map_file, map_type="scalar", silent=silent)
        elif map_id:
            mapS_for_create = get_map(map_id, map_type="scalar", silent=silent)
            
        xyz_data = create_xyz0(
            mapS=mapS_for_create, 
            alt=alt, dt=dt, t=t, v=v, 
            **kwargs,
            silent=silent
        )
        
        if comp_coeffs is not None and hasattr(xyz_data, 'mag_1_uc'):
            if not silent: print("Applying compensation coefficients to newly generated data...")
            xyz_data.mag_1_c = calculate_imputed_TL_earth(
                xyz_data, comp_coeffs 
            )
        return xyz_data