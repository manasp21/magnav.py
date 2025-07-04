# -*- coding: utf-8 -*-
"""
Utility functions for map handling, including interpolation, trimming, filling,
and other manipulations.
"""
import numpy as np
import os
import scipy.io
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.spatial import KDTree as SciPyKDTree
import pickle
from typing import Union, Tuple, Any, List, Optional, Callable
import dataclasses # Used for dataclasses.replace

# Attempt to import MapS, MapV, MAP_S_NULL from .common_types
try:
    from .common_types import MapS, MapV, MapS3D, MapSd, MAP_S_NULL, MapCache
except ImportError:
    print("Warning: Could not import types from .common_types in map_utils.py. Using placeholder types.")
    from dataclasses import dataclass, field

    @dataclass
    class MapS:
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([])) # Meshgrid lat
        lon: np.ndarray = field(default_factory=lambda: np.array([])) # Meshgrid lon
        alt: Union[np.ndarray, float] = field(default_factory=lambda: np.array([]))
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([])) # Unique lon coords
        yy: np.ndarray = field(default_factory=lambda: np.array([])) # Unique lat coords
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))

    @dataclass
    class MapV: # Placeholder for vector map
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: Union[np.ndarray, float] = field(default_factory=lambda: np.array([]))
        mapX: np.ndarray = field(default_factory=lambda: np.array([]))
        mapY: np.ndarray = field(default_factory=lambda: np.array([]))
        mapZ: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([]))
        yy: np.ndarray = field(default_factory=lambda: np.array([]))
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))

    @dataclass
    class MapS3D: # Placeholder for 3D scalar map
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([])) # Meshgrid lat
        lon: np.ndarray = field(default_factory=lambda: np.array([])) # Meshgrid lon
        alt: np.ndarray = field(default_factory=lambda: np.array([])) # Array of altitudes
        map: np.ndarray = field(default_factory=lambda: np.array([])) # 3D: (yy, xx, alt_levels)
        xx: np.ndarray = field(default_factory=lambda: np.array([])) # Unique lon coords
        yy: np.ndarray = field(default_factory=lambda: np.array([])) # Unique lat coords
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([])) # 3D mask

    @dataclass
    class MapSd: # Placeholder for drape scalar map
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: np.ndarray = field(default_factory=lambda: np.array([])) # 2D altitude grid
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([]))
        yy: np.ndarray = field(default_factory=lambda: np.array([]))
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))

    MAP_S_NULL = MapS(info="null_map_placeholder")

    @dataclass
    class MapCacheBase: # Renamed to avoid conflict if common_types.MapCache exists
        maps: List[MapS] = field(default_factory=list)
        fallback_map: Optional[MapS] = None
        dz: float = 100.0 # Default altitude spacing for cache layers if generated
        # Store interpolators directly, keyed by a map identifier or altitude
        _interpolators: dict = field(default_factory=dict)
        _fallback_interpolator: Optional[Callable] = None

        def __post_init__(self):
            self.update_interpolators()

        def add_map(self, map_obj: MapS, make_interpolator: bool = True):
            self.maps.append(map_obj)
            if make_interpolator:
                try:
                    # Key by map info and altitude for uniqueness
                    key = (map_obj.info, float(np.mean(map_obj.alt)) if isinstance(map_obj.alt, np.ndarray) else float(map_obj.alt))
                    self._interpolators[key] = map_interpolate(map_obj)
                except Exception as e:
                    print(f"Warning: Could not create interpolator for map {map_obj.info}: {e}")
            # TODO: Add logic for sorting maps by altitude if needed (like map_sort_ind in Julia)

        def update_interpolators(self):
            self._interpolators.clear()
            for map_obj in self.maps:
                try:
                    key = (map_obj.info, float(np.mean(map_obj.alt)) if isinstance(map_obj.alt, np.ndarray) else float(map_obj.alt))
                    self._interpolators[key] = map_interpolate(map_obj)
                except Exception as e:
                    print(f"Warning: Could not create interpolator for map {map_obj.info} during update: {e}")
            if self.fallback_map:
                try:
                    self._fallback_interpolator = map_interpolate(self.fallback_map)
                except Exception as e:
                    print(f"Warning: Could not create interpolator for fallback map {self.fallback_map.info}: {e}")
            else:
                self._fallback_interpolator = None
        
        def get_interpolator(self, alt_query: float) -> Optional[Callable]:
            """Selects the best interpolator based on altitude."""
            if not self._interpolators:
                return self._fallback_interpolator

            best_interp = None
            min_alt_diff = float('inf')

            for (map_info, map_alt_scalar), interp_func in self._interpolators.items():
                alt_diff = abs(map_alt_scalar - alt_query)
                if alt_diff < min_alt_diff:
                    min_alt_diff = alt_diff
                    best_interp = interp_func
            
            if best_interp is None: # No suitable primary map, use fallback
                return self._fallback_interpolator
            return best_interp

# Helper functions (ported from Julia or new)
def get_step(arr: np.ndarray) -> float:
    """
    Get the median step size (spacing) of elements in a 1D array.
    Returns 0.0 if array has fewer than 2 elements.
    """
    if arr is None or arr.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(arr))))

def map_params(map_data: np.ndarray, map_xx: Optional[np.ndarray] = None, map_yy: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Get basic map parameters: zero/non-zero indices and dimensions.
    Assumes map_data is 2D. Zeros and NaNs are treated as invalid/gap data.
    """
    if map_data is None:
        return np.array([]), np.array([]), 0, 0
    
    map_copy = map_data.copy() # Avoid modifying original
    nan_mask = np.isnan(map_copy)
    map_copy[nan_mask] = 0 # Replace NaNs with 0 for consistent gap identification

    ind0 = (map_copy == 0) # Indices of gaps (originally zero or NaN)
    ind1 = ~ind0           # Indices of valid data

    ny, nx = map_data.shape
    
    if map_xx is not None and nx != map_xx.size:
        raise ValueError(f"xx map dimension ({map_xx.size}) inconsistent with map_data x-dim ({nx})")
    if map_yy is not None and ny != map_yy.size:
        raise ValueError(f"yy map dimension ({map_yy.size}) inconsistent with map_data y-dim ({ny})")
            
    # Ensure a 4-tuple is always returned
    return (ind0, ind1, int(nx), int(ny))

def map_lims(map_obj: Union[MapS, MapV, MapS3D, MapSd], buffer_percent: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Calculates the latitude and longitude limits of a map object with an optional buffer.
    Returns ((lat_min, lat_max), (lon_min, lon_max)).
    """
    if map_obj.xx is None or map_obj.yy is None or map_obj.xx.size == 0 or map_obj.yy.size == 0:
        return (np.nan, np.nan), (np.nan, np.nan)

    lon_min, lon_max = np.min(map_obj.xx), np.max(map_obj.xx)
    lat_min, lat_max = np.min(map_obj.yy), np.max(map_obj.yy)

    if buffer_percent > 0:
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_buffer = lon_range * buffer_percent / 100.0
        lat_buffer = lat_range * buffer_percent / 100.0
        lon_min -= lon_buffer
        lon_max += lon_buffer
        lat_min -= lat_buffer
        lat_max += lat_buffer
        
    return (lat_min, lat_max), (lon_min, lon_max)

def map_res(map_obj: Union[MapS, MapV, MapS3D, MapSd]) -> Tuple[float, float]:
    """
    Returns the approximate median resolution (dy, dx) of the map in the units of yy, xx.
    """
    if map_obj.xx is None or map_obj.yy is None or map_obj.xx.size < 1 or map_obj.yy.size < 1: # Allow single point maps
        return (0.0, 0.0)
    # For single point maps, resolution is undefined, return 0
    dx = get_step(map_obj.xx) if map_obj.xx.size > 1 else 0.0
    dy = get_step(map_obj.yy) if map_obj.yy.size > 1 else 0.0
    return dy, dx

def map_shape(map_obj: Union[MapS, MapV, MapS3D, MapSd]) -> Optional[Tuple[int, ...]]:
    """Returns the shape of the primary map data."""
    if hasattr(map_obj, 'map') and map_obj.map is not None:
        return map_obj.map.shape
    elif hasattr(map_obj, 'mapX') and map_obj.mapX is not None: # For MapV
        return map_obj.mapX.shape
    return None

# Core map functions
def map_interpolate(map_obj: Union[MapS, MapS3D],
                    method: str = "linear",
                    bounds_error: bool = False,
                    fill_value: float = np.nan,
                    return_vert_deriv: bool = False) -> Union[Optional[Callable], Tuple[Optional[Callable], Optional[Callable]]]:
    """
    Creates and returns an interpolator function for a MapS or MapS3D object.
    For MapS3D, interpolates through the altitude dimension as well if alt_query is provided to the interpolator.

    Args:
        map_obj: MapS or MapS3D object.
        method: Interpolation method ("linear", "nearest"). "cubic" requires interpn.
        bounds_error: If True, raise error for out-of-bounds. Else, use fill_value.
        fill_value: Value for out-of-bounds points.

    Returns:
        If return_vert_deriv is False:
            A callable interpolator function, or None if interpolation setup fails.
            The interpolator expects points as (lat, lon) for MapS, or (lat, lon, alt) for MapS3D.
        If return_vert_deriv is True:
            A tuple (interpolator, derivative_interpolator).
            Currently, derivative_interpolator will be None.
    """
    print(f"DEBUG_ERROR_6: map_interpolate from map_utils.py called. map_obj type: {type(map_obj)}, return_vert_deriv: {return_vert_deriv}") # For Error 6
    if not (hasattr(map_obj, 'yy') and hasattr(map_obj, 'xx') and hasattr(map_obj, 'map')):
        raise ValueError("Map object is missing 'yy', 'xx', or 'map' attributes.")
    if map_obj.map is None or map_obj.yy is None or map_obj.xx is None:
        print(f"Warning: Map '{map_obj.info}' has None for map, yy, or xx. Cannot create interpolator.")
        if return_vert_deriv:
            return None, None
        return None

    map_grid_lat = np.asarray(map_obj.yy).squeeze()
    map_grid_lon = np.asarray(map_obj.xx).squeeze()
    map_values = np.asarray(map_obj.map)

    if map_grid_lat.ndim == 0: map_grid_lat = np.array([map_grid_lat.item()])
    if map_grid_lon.ndim == 0: map_grid_lon = np.array([map_grid_lon.item()])

    if map_grid_lat.size == 0 or map_grid_lon.size == 0 or map_values.size == 0:
        print(f"Warning: Map '{map_obj.info}' has empty coordinates or map data. Cannot create interpolator.")
        if return_vert_deriv:
            return None, None
        return None

    if map_grid_lat.ndim != 1 or map_grid_lon.ndim != 1:
        raise ValueError("Map grid coordinates (yy, xx) must be 1D for RegularGridInterpolator.")

    points_coords = (map_grid_lat, map_grid_lon)
    
    if isinstance(map_obj, MapS3D):
        if not hasattr(map_obj, 'alt') or map_obj.alt is None or map_obj.alt.size == 0:
            raise ValueError("MapS3D object must have 'alt' coordinates for 3D interpolation.")
        map_grid_alt = np.asarray(map_obj.alt).squeeze()
        if map_grid_alt.ndim == 0: map_grid_alt = np.array([map_grid_alt.item()])
        if map_grid_alt.ndim != 1:
            raise ValueError("MapS3D.alt coordinates must be 1D for RegularGridInterpolator.")
        points_coords = (map_grid_lat, map_grid_lon, map_grid_alt)
        if map_values.ndim != 3 or map_values.shape != (map_grid_lat.size, map_grid_lon.size, map_grid_alt.size):
            raise ValueError(f"MapS3D data shape {map_values.shape} mismatch with grid dimensions.")
    elif isinstance(map_obj, MapS):
        if map_values.ndim != 2 or map_values.shape != (map_grid_lat.size, map_grid_lon.size):
            raise ValueError(f"MapS data shape {map_values.shape} mismatch with grid dimensions.")
    else:
        raise TypeError("map_obj must be MapS or MapS3D.")

    try:
        if method.lower() == "cubic" and map_values.ndim > 1 : # interpn supports cubic for >1D
             # For interpn, points are (coords_dim1, coords_dim2, ...)
             # and xi is a tuple of new coordinate arrays or a (N, D_coord) array of points
             # To make it behave like RegularGridInterpolator, we wrap it.
            def interpn_callable(query_points_array):
                # query_points_array is (N, D_coord)
                # interpn expects xi as a tuple of arrays for each dimension if method is not 'linear' or 'nearest' for grid.
                # Or, if xi is (M, ndim), it's treated as M points.
                return interpn(points_coords, map_values, query_points_array, method="cubic", bounds_error=bounds_error, fill_value=fill_value)
            
            if return_vert_deriv:
                # TODO: Implement actual vertical derivative calculation and its interpolator
                return interpn_callable, None
            return interpn_callable
        else:
            interpolator = RegularGridInterpolator(
                points_coords,
                map_values,
                method=method,
                bounds_error=bounds_error,
                fill_value=fill_value
            )
            if return_vert_deriv:
                # TODO: Implement actual vertical derivative calculation and its interpolator
                return interpolator, None
            return interpolator
    except ValueError as e:
        print(f"Error creating interpolator for map '{map_obj.info}': {e}. Check if coordinates are sorted and unique.")
        if return_vert_deriv:
            return None, None
        return None


def get_map_val(
    map_data: Union[MapS, MapS3D, MapCache], # Extended to MapS3D
    lat_query: Union[float, np.ndarray],
    lon_query: Union[float, np.ndarray],
    alt_query: Optional[Union[float, np.ndarray]] = None,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float = np.nan,
    return_interpolator: bool = False
) -> Union[np.ndarray, float, Tuple[Union[np.ndarray, float], Any]]:
    """
    Get magnetic anomaly map value(s) at specified geographic coordinate(s).
    Interpolates from MapS, MapS3D, or uses MapCacheBase.
    For MapS3D, alt_query is required for 3D interpolation.
    If MapS (2D) is given and alt_query is provided, it's currently ignored by the interpolator itself
    unless specific logic for upward/downward continuation is added here (complex).
    """
    interpolator_object = None

    if isinstance(map_data, MapCache):
        # Determine query altitude for cache selection
        alt_q_scalar = 0.0 # Default
        if alt_query is not None:
            alt_q_scalar = np.mean(alt_query) if isinstance(alt_query, np.ndarray) else float(alt_query)
        elif map_data.maps: # Use first map's altitude from cache if alt_query is None
            first_map_alt = map_data.maps[0].alt
            alt_q_scalar = np.mean(first_map_alt) if isinstance(first_map_alt, np.ndarray) else float(first_map_alt)
        
        interpolator_object = map_data.get_interpolator(alt_q_scalar)
        if interpolator_object is None:
            # print("Warning: No suitable interpolator in MapCache. Returning NaNs.")
            nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
            return (nan_vals, None) if return_interpolator else nan_vals
    
    elif isinstance(map_data, (MapS, MapS3D)):
        interpolator_object = map_interpolate(map_data, method=method, bounds_error=bounds_error, fill_value=fill_value)
        if interpolator_object is None: # Interpolator creation failed
            nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
            return (nan_vals, None) if return_interpolator else nan_vals
    elif isinstance(map_data, MapV):
        # Check for essential attributes and non-None map components
        if not (hasattr(map_data, 'yy') and hasattr(map_data, 'xx') and
                hasattr(map_data, 'x') and hasattr(map_data, 'y') and hasattr(map_data, 'z') and
                map_data.yy is not None and map_data.xx is not None and
                map_data.x is not None and map_data.y is not None and map_data.z is not None):
            # print(f"Warning: MapV '{getattr(map_data, 'info', 'N/A')}' is missing attributes or has None for coordinates/component maps. Cannot interpolate.")
            # Determine shape for NaN values based on original inputs
            original_input_shape_for_nans = np.broadcast(lat_query, lon_query).shape
            if not original_input_shape_for_nans: # scalar inputs
                nan_val_to_use = fill_value # Already a float
            else: # array inputs
                nan_val_to_use = np.full(original_input_shape_for_nans, fill_value, dtype=float)

            nan_result_tuple = (nan_val_to_use, nan_val_to_use, nan_val_to_use)
            return (nan_result_tuple, [None, None, None]) if return_interpolator else nan_result_tuple

        map_grid_lat = np.asarray(map_data.yy).squeeze()
        map_grid_lon = np.asarray(map_data.xx).squeeze()

        if map_grid_lat.ndim == 0: map_grid_lat = np.array([map_grid_lat.item()])
        if map_grid_lon.ndim == 0: map_grid_lon = np.array([map_grid_lon.item()])

        if map_grid_lat.size == 0 or map_grid_lon.size == 0:
            # print(f"Warning: MapV '{getattr(map_data, 'info', 'N/A')}' has empty coordinates. Cannot interpolate.")
            original_input_shape_for_nans = np.broadcast(lat_query, lon_query).shape
            if not original_input_shape_for_nans: # scalar inputs
                nan_val_to_use = fill_value
            else: # array inputs
                nan_val_to_use = np.full(original_input_shape_for_nans, fill_value, dtype=float)
            nan_result_tuple = (nan_val_to_use, nan_val_to_use, nan_val_to_use)
            return (nan_result_tuple, [None, None, None]) if return_interpolator else nan_result_tuple

        if map_grid_lat.ndim != 1 or map_grid_lon.ndim != 1:
            raise ValueError("MapV grid coordinates (yy, xx) must be 1D.")

        points_coords_2d = (map_grid_lat, map_grid_lon)

        # Prepare query points, ensuring they can be broadcasted or match
        lat_q_arr_orig = np.atleast_1d(lat_query) # Keep original for broadcast shape determination
        lon_q_arr_orig = np.atleast_1d(lon_query)

        # Attempt to broadcast query points for processing
        try:
            b = np.broadcast(lat_q_arr_orig, lon_q_arr_orig)
            # Create query arrays based on the broadcast output shape
            # This ensures lat_q_arr and lon_q_arr are compatible for column_stack
            # and correctly represent the query points over the broadcasted grid.
            if b.ndim > 0 : # If broadcast result is not scalar
                query_grid = np.empty(b.shape + (2,))
                query_grid[..., 0] = lat_q_arr_orig # Broadcasting happens here
                query_grid[..., 1] = lon_q_arr_orig # And here
                lat_q_arr = query_grid[..., 0].ravel()
                lon_q_arr = query_grid[..., 1].ravel()
            else: # Scalar broadcast
                lat_q_arr = lat_q_arr_orig.ravel()
                lon_q_arr = lon_q_arr_orig.ravel()

        except ValueError: # If broadcasting fails
            raise ValueError("lat_query, lon_query shapes are incompatible for MapV interpolation.")


        query_points_2d = np.column_stack((lat_q_arr, lon_q_arr))
        original_input_shape = np.broadcast(lat_query, lon_query).shape # Shape of the final output

        results_xyz_components = []
        interpolators_list = [] if return_interpolator else None

        for component_idx, component_data_array in enumerate([map_data.x, map_data.y, map_data.z]):
            if component_data_array.shape != (map_grid_lat.size, map_grid_lon.size):
                comp_name = ['x', 'y', 'z'][component_idx]
                raise ValueError(f"MapV component '{comp_name}' data shape {component_data_array.shape} mismatch with grid dimensions "
                                 f"({map_grid_lat.size}, {map_grid_lon.size}) for map '{getattr(map_data, 'info', 'N/A')}'.")
            
            current_interpolator = None
            try:
                # Ensure map grid coordinates are sorted for RegularGridInterpolator
                if not np.all(np.diff(map_grid_lat) > 0) and not np.all(np.diff(map_grid_lat) < 0):
                    # Attempt to sort if not monotonic, this is a bit risky if map_data is not aligned
                    # print(f"Warning: MapV latitude coordinates for map '{getattr(map_data, 'info', 'N/A')}' are not monotonic. Attempting to sort.")
                    # sort_idx_lat = np.argsort(map_grid_lat)
                    # map_grid_lat_sorted = map_grid_lat[sort_idx_lat]
                    # component_data_array_sorted_lat = component_data_array[sort_idx_lat, :]
                    pass # For now, assume they should be sorted by the caller or map creation
                if not np.all(np.diff(map_grid_lon) > 0) and not np.all(np.diff(map_grid_lon) < 0):
                    # print(f"Warning: MapV longitude coordinates for map '{getattr(map_data, 'info', 'N/A')}' are not monotonic. Attempting to sort.")
                    # sort_idx_lon = np.argsort(map_grid_lon)
                    # map_grid_lon_sorted = map_grid_lon[sort_idx_lon]
                    # component_data_array_final = (component_data_array_sorted_lat if 'component_data_array_sorted_lat' in locals() else component_data_array)[:, sort_idx_lon]
                    pass # For now, assume they should be sorted

                interpolator_comp = RegularGridInterpolator(
                    (map_grid_lat, map_grid_lon), # Use potentially sorted coords
                    component_data_array, # Use potentially sorted data
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=fill_value
                )
                interpolated_values_comp_flat = interpolator_comp(query_points_2d)
                current_interpolator = interpolator_comp
            except ValueError as ve: # Catch specific errors from RGI if coords not strictly monotonic
                # print(f"ValueError during MapV component interpolation for map '{getattr(map_data, 'info', 'N/A')}': {ve}. Check coordinate monotonicity.")
                interpolated_values_comp_flat = np.full(query_points_2d.shape[0], fill_value, dtype=float)
            except Exception as e:
                # print(f"Error during MapV component interpolation for map '{getattr(map_data, 'info', 'N/A')}': {e}")
                interpolated_values_comp_flat = np.full(query_points_2d.shape[0], fill_value, dtype=float)
            
            if return_interpolator and interpolators_list is not None:
                interpolators_list.append(current_interpolator)

            # Reshape component to match original query structure
            if not original_input_shape: # All original inputs were scalar
                reshaped_component_value = interpolated_values_comp_flat.item() if interpolated_values_comp_flat.size == 1 else interpolated_values_comp_flat
            else:
                reshaped_component_value = interpolated_values_comp_flat.reshape(original_input_shape)
                # Ensure scalar output for scalar inputs even if original_input_shape is e.g. (1,) but inputs were float
                if reshaped_component_value.size == 1 and \
                   isinstance(lat_query, (int, float)) and \
                   isinstance(lon_query, (int, float)): # Check original types
                     reshaped_component_value = reshaped_component_value.item()
            results_xyz_components.append(reshaped_component_value)

        final_tuple_result = tuple(results_xyz_components)

        if return_interpolator:
            return final_tuple_result, interpolators_list
        else:
            return final_tuple_result
    else:
        raise TypeError("map_data must be a MapS, MapS3D, MapV, or MapCache object.")

    # Prepare query points
    lat_q_arr = np.atleast_1d(lat_query)
    lon_q_arr = np.atleast_1d(lon_query)

    if isinstance(map_data, MapS3D) or (isinstance(map_data, MapCache) and alt_query is not None):
        if alt_query is None:
            raise ValueError("alt_query must be provided for 3D interpolation with MapS3D or relevant MapCache.")
        alt_q_arr = np.atleast_1d(alt_query)
        if not (lat_q_arr.shape == lon_q_arr.shape == alt_q_arr.shape):
             # If shapes differ, attempt to broadcast if one is scalar and others are arrays.
             # This case needs careful handling if general broadcasting is desired.
             # For now, assume they should match or be scalar.
            if not (lat_q_arr.size==1 or lon_q_arr.size==1 or alt_q_arr.size==1):
                 if not(lat_q_arr.shape == lon_q_arr.shape and lat_q_arr.shape == alt_q_arr.shape):
                    raise ValueError("lat_query, lon_query, alt_query must have same shape for 3D interpolation or be scalar.")
            # Basic broadcasting for scalar + array
            if lat_q_arr.size == 1 and lon_q_arr.size > 1: lat_q_arr = np.full_like(lon_q_arr, lat_q_arr[0])
            if lon_q_arr.size == 1 and lat_q_arr.size > 1: lon_q_arr = np.full_like(lat_q_arr, lon_q_arr[0])
            if alt_q_arr.size == 1 and lat_q_arr.size > 1: alt_q_arr = np.full_like(lat_q_arr, alt_q_arr[0])
            # Recheck after basic broadcasting attempt
            if not (lat_q_arr.shape == lon_q_arr.shape == alt_q_arr.shape):
                 raise ValueError("lat_query, lon_query, alt_query shapes are incompatible after basic broadcasting.")


        query_points = np.column_stack((lat_q_arr.ravel(), lon_q_arr.ravel(), alt_q_arr.ravel()))
    else: # 2D interpolation
        if lat_q_arr.shape != lon_q_arr.shape:
            if not (lat_q_arr.size==1 or lon_q_arr.size==1):
                raise ValueError("lat_query, lon_query must have same shape for 2D interpolation or be scalar.")
            if lat_q_arr.size == 1 and lon_q_arr.size > 1: lat_q_arr = np.full_like(lon_q_arr, lat_q_arr[0])
            if lon_q_arr.size == 1 and lat_q_arr.size > 1: lon_q_arr = np.full_like(lat_q_arr, lon_q_arr[0])

        query_points = np.column_stack((lat_q_arr.ravel(), lon_q_arr.ravel()))

    try:
        interpolated_values = interpolator_object(query_points)
    except Exception as e:
        # print(f"Error during interpolation: {e}")
        # This can happen if query points are outside strict bounds and bounds_error=True for RegularGridInterpolator
        # or other interpolator issues.
        nan_vals = np.full(query_points.shape[0], fill_value, dtype=float)
        interpolated_values = nan_vals


    # Reshape to match original query shape if input was array, otherwise squeeze to scalar
    output_shape = np.broadcast(lat_query, lon_query).shape # Get shape from original inputs
    if not output_shape: # If all inputs were scalar
        result_values = interpolated_values.item() if interpolated_values.size == 1 else interpolated_values
    else:
        result_values = interpolated_values.reshape(output_shape)
        if result_values.size == 1 and isinstance(lat_query, (float,int)): # Squeezed back if original was scalar-like
             result_values = result_values.item()


    if return_interpolator:
        return result_values, interpolator_object
    return result_values


def map_gradient(map_obj: Union[MapS, MapS3D], on_interpolated: bool = False, query_points=None, method="linear", **kwargs) -> Optional[List[np.ndarray]]:
    """
    Computes the gradient of the map.
    If on_interpolated is False (default), uses np.gradient on the raw map_obj.map data.
    Spacing is taken from map_obj.yy, map_obj.xx (and map_obj.alt for 3D).
    If on_interpolated is True, query_points must be provided, and the gradient of the
    interpolated field at these points is computed using finite differences. (This part is simplified).
    """
    if map_obj.map is None or map_obj.map.size == 0:
        return None

    if not on_interpolated:
        coords = [map_obj.yy, map_obj.xx]
        if isinstance(map_obj, MapS3D):
            coords.append(map_obj.alt)
        
        # Check for uniform spacing, np.gradient prefers this or explicit coordinates
        # For simplicity, we pass the coordinate vectors directly if supported,
        # otherwise, user must be aware of scaling.
        # np.gradient accepts coordinate vectors from version 1.26
        try:
            grad = np.gradient(map_obj.map, *coords)
        except TypeError: # Older NumPy might not support list of coordinates
            print("Warning: np.gradient with coordinate vectors failed (likely older NumPy). Calculating unscaled gradient.")
            grad = np.gradient(map_obj.map)
            # Manual scaling would be needed here:
            # dy = get_step(map_obj.yy); dx = get_step(map_obj.xx)
            # grad[0] /= dy; grad[1] /= dx
            # if isinstance(map_obj, MapS3D): d_alt = get_step(map_obj.alt); grad[2] /= d_alt
        return grad
    else:
        # Gradient of interpolated field (simplified: user should do this carefully)
        print("Placeholder: map_gradient on_interpolated=True. User should interpolate at query_points and neighbors, then apply finite differences.")
        # itp = map_interpolate(map_obj, method=method)
        # if itp is None or query_points is None: return None
        # ... logic to compute gradient at query_points using itp ...
        return None

def map_hessian(map_obj: Union[MapS, MapS3D], on_interpolated: bool = False, **kwargs) -> Optional[List[List[np.ndarray]]]:
    """
    Computes the Hessian matrix of the map. Placeholder.
    If on_interpolated is False, uses np.gradient twice on raw map_obj.map data.
    """
    print("Placeholder: map_hessian called. Implement using np.gradient twice on map_obj.map or interpolated field.")
    # grad_list = map_gradient(map_obj, on_interpolated=on_interpolated, **kwargs)
    # if grad_list is None: return None
    # hessian_matrix = []
    # for grad_component in grad_list:
    #     # This needs careful handling of coordinates for np.gradient again
    #     coords = [map_obj.yy, map_obj.xx]
    #     if isinstance(map_obj, MapS3D): coords.append(map_obj.alt)
    #     try:
    #         hessian_row = np.gradient(grad_component, *coords)
    #     except TypeError:
    #         hessian_row = np.gradient(grad_component) # Unscaled
    #     hessian_matrix.append(hessian_row)
    # return hessian_matrix # This will be list of lists of arrays
    return None


def map_trim(map_obj: Union[MapS, MapSd, MapS3D, MapV],
             lat_lim: Optional[Tuple[float, float]] = None,
             lon_lim: Optional[Tuple[float, float]] = None,
             alt_lim: Optional[Tuple[float, float]] = None, # For MapS3D
             pad_cells: int = 0,
             trim_to_data: bool = True,
             copy: bool = True) -> Union[MapS, MapSd, MapS3D, MapV]:
    """
    Trims a map object to specified latitude/longitude limits and/or to data bounds.

    Args:
        map_obj: The map object to trim.
        lat_lim: Optional (min_lat, max_lat) tuple.
        lon_lim: Optional (min_lon, max_lon) tuple.
        alt_lim: Optional (min_alt, max_alt) tuple (for MapS3D).
        pad_cells: Number of cells to pad around the data bounds if trim_to_data is True.
        trim_to_data: If True, also trim to the bounding box of non-gap data.
        copy: If True, returns a new trimmed map object. Otherwise, modifies in place if possible.

    Returns:
        The trimmed map object.
    """
    if map_obj.map is None or map_obj.xx is None or map_obj.yy is None:
        return map_obj

    if copy:
        # Create a new object with copied underlying arrays
        map_obj = dataclasses.replace(map_obj) # Shallow copy of dataclass structure
        for attr_name in ['map', 'xx', 'yy', 'alt', 'mask', 'mapX', 'mapY', 'mapZ', 'lat', 'lon']:
            if hasattr(map_obj, attr_name) and getattr(map_obj, attr_name) is not None:
                # Only call .copy() on array objects, not floats
                attr_val = getattr(map_obj, attr_name)
                if isinstance(attr_val, np.ndarray):
                    setattr(map_obj, attr_name, attr_val.copy())
                else:
                    # For non-array values, use the original value
                    setattr(map_obj, attr_name, attr_val)
    
    current_map_data = map_obj.map if hasattr(map_obj, 'map') else map_obj.mapX # Use mapX for MapV

    # Determine initial slicing from data presence if trim_to_data
    min_r, max_r, min_c, max_c = 0, current_map_data.shape[0], 0, current_map_data.shape[1]

    if trim_to_data:
        _, valid_data_mask, _, _ = map_params(current_map_data)
        if np.any(valid_data_mask):
            rows, cols = np.where(valid_data_mask)
            min_r_data, max_r_data = np.min(rows), np.max(rows) + 1
            min_c_data, max_c_data = np.min(cols), np.max(cols) + 1
            
            min_r = max(min_r, min_r_data - pad_cells)
            max_r = min(max_r, max_r_data + pad_cells)
            min_c = max(min_c, min_c_data - pad_cells)
            max_c = min(max_c, max_c_data + pad_cells)

    # Apply lat/lon limits
    if lat_lim is not None:
        min_r = max(min_r, np.searchsorted(map_obj.yy, lat_lim[0], side='left'))
        max_r = min(max_r, np.searchsorted(map_obj.yy, lat_lim[1], side='right'))
    if lon_lim is not None:
        min_c = max(min_c, np.searchsorted(map_obj.xx, lon_lim[0], side='left'))
        max_c = min(max_c, np.searchsorted(map_obj.xx, lon_lim[1], side='right'))

    # Ensure slices are valid
    min_r = np.clip(min_r, 0, map_obj.yy.size)
    max_r = np.clip(max_r, min_r, map_obj.yy.size)
    min_c = np.clip(min_c, 0, map_obj.xx.size)
    max_c = np.clip(max_c, min_c, map_obj.xx.size)
    
    yy_slice = slice(min_r, max_r)
    xx_slice = slice(min_c, max_c)

    map_obj.yy = map_obj.yy[yy_slice]
    map_obj.xx = map_obj.xx[xx_slice]

    if hasattr(map_obj, 'lat') and map_obj.lat is not None and map_obj.lat.ndim == 2:
         map_obj.lat = map_obj.lat[yy_slice, xx_slice]
    if hasattr(map_obj, 'lon') and map_obj.lon is not None and map_obj.lon.ndim == 2:
         map_obj.lon = map_obj.lon[yy_slice, xx_slice]

    alt_slice = slice(None) # For 3rd dimension of MapS3D
    if isinstance(map_obj, MapS3D) and alt_lim is not None and map_obj.alt is not None:
        min_a = np.searchsorted(map_obj.alt, alt_lim[0], side='left')
        max_a = np.searchsorted(map_obj.alt, alt_lim[1], side='right')
        min_a = np.clip(min_a, 0, map_obj.alt.size)
        max_a = np.clip(max_a, min_a, map_obj.alt.size)
        alt_slice = slice(min_a, max_a)
        map_obj.alt = map_obj.alt[alt_slice]

    if hasattr(map_obj, 'map') and map_obj.map is not None:
        map_obj.map = map_obj.map[yy_slice, xx_slice, alt_slice] if isinstance(map_obj, MapS3D) else map_obj.map[yy_slice, xx_slice]
    if hasattr(map_obj, 'mask') and map_obj.mask is not None:
        map_obj.mask = map_obj.mask[yy_slice, xx_slice, alt_slice] if isinstance(map_obj, MapS3D) and map_obj.mask.ndim==3 else map_obj.mask[yy_slice, xx_slice]
    
    if isinstance(map_obj, MapV):
        if map_obj.mapX is not None: map_obj.mapX = map_obj.mapX[yy_slice, xx_slice]
        if map_obj.mapY is not None: map_obj.mapY = map_obj.mapY[yy_slice, xx_slice]
        if map_obj.mapZ is not None: map_obj.mapZ = map_obj.mapZ[yy_slice, xx_slice]
    
    if isinstance(map_obj, MapSd) and hasattr(map_obj, 'alt') and isinstance(map_obj.alt, np.ndarray) and map_obj.alt.ndim == 2:
        map_obj.alt = map_obj.alt[yy_slice, xx_slice]
        
    return map_obj


def map_fill_gaps(map_obj: Union[MapS, MapSd, MapS3D], k: int = 3, copy: bool = True) -> Union[MapS, MapSd, MapS3D]:
    """
    Fills NaN or zero gaps in a map object using k-nearest neighbors.
    For MapS3D, fills slice by slice along the altitude dimension.
    """
    if map_obj.map is None:
        return map_obj

    if copy:
        map_obj = dataclasses.replace(map_obj)
        map_obj.map = map_obj.map.copy()
        if map_obj.mask is not None: map_obj.mask = map_obj.mask.copy()
        if isinstance(map_obj, MapSd) and map_obj.alt is not None and isinstance(map_obj.alt, np.ndarray):
            map_obj.alt = map_obj.alt.copy()


    def fill_2d_slice(data_slice: np.ndarray, yy_coords_vec: np.ndarray, xx_coords_vec: np.ndarray, k_val: int):
        nan_mask = np.isnan(data_slice)
        zero_mask = (data_slice == 0) # Assuming 0 is also a gap
        gap_mask = nan_mask | zero_mask

        if not np.any(gap_mask):
            return data_slice, np.ones_like(data_slice, dtype=bool) # Return original mask if no gaps

        yy_grid, xx_grid = np.meshgrid(yy_coords_vec, xx_coords_vec, indexing='ij')
        
        valid_points_mask = ~gap_mask
        if not np.any(valid_points_mask):
            print("Warning: No valid data points in slice for map_fill_gaps.")
            return data_slice, gap_mask # Return original data and gap mask

        valid_yy = yy_grid[valid_points_mask]
        valid_xx = xx_grid[valid_points_mask]
        valid_values = data_slice[valid_points_mask]
        
        gap_yy = yy_grid[gap_mask]
        gap_xx = xx_grid[gap_mask]

        if gap_yy.size == 0: return data_slice, ~gap_mask # Should be caught by np.any(gap_mask)

        tree_data = np.column_stack((valid_yy.ravel(), valid_xx.ravel()))
        if tree_data.shape[0] == 0:
             print("Warning: No valid data points to build KDTree in slice.")
             return data_slice, ~gap_mask

        kdtree = SciPyKDTree(tree_data)
        query_points = np.column_stack((gap_yy.ravel(), gap_xx.ravel()))
        
        actual_k = min(k_val, tree_data.shape[0])
        if actual_k == 0: return data_slice, ~gap_mask

        _, indices = kdtree.query(query_points, k=actual_k)
        
        filled_slice = data_slice.copy()
        if actual_k == 1:
            filled_values = valid_values[indices]
        else:
            filled_values = np.mean(valid_values[indices], axis=1)
            
        filled_slice[gap_mask] = filled_values
        new_mask_slice = ~gap_mask # Initially, then update based on what was filled
        new_mask_slice[gap_mask] = True # Mark filled gaps as valid in the new mask
        return filled_slice, new_mask_slice

    if isinstance(map_obj, MapS3D):
        new_3d_map = map_obj.map.copy()
        new_3d_mask = map_obj.mask.copy() if map_obj.mask is not None else np.ones_like(map_obj.map, dtype=bool)
        for i in range(map_obj.map.shape[2]): # Iterate over altitude slices
            filled_slice, mask_slice = fill_2d_slice(map_obj.map[:,:,i], map_obj.yy, map_obj.xx, k)
            new_3d_map[:,:,i] = filled_slice
            new_3d_mask[:,:,i] = mask_slice
        map_obj.map = new_3d_map
        map_obj.mask = new_3d_mask
    elif isinstance(map_obj, MapSd): # Has 2D altitude map as well
        map_obj.map, new_mask = fill_2d_slice(map_obj.map, map_obj.yy, map_obj.xx, k)
        if map_obj.mask is not None: map_obj.mask = new_mask # Update mask
        # Optionally fill altitude map if it also has gaps
        # map_obj.alt, _ = fill_2d_slice(map_obj.alt, map_obj.yy, map_obj.xx, k)
    else: # MapS
        map_obj.map, new_mask = fill_2d_slice(map_obj.map, map_obj.yy, map_obj.xx, k)
        if map_obj.mask is not None: map_obj.mask = new_mask # Update mask
        
    return map_obj


def map_resample(map_obj: Union[MapS, MapSd, MapS3D],
                 new_yy: np.ndarray,
                 new_xx: np.ndarray,
                 new_alt: Optional[np.ndarray] = None, # For MapS3D
                 method: str = "linear",
                 copy: bool = True) -> Union[MapS, MapSd, MapS3D]:
    """
    Resamples a map object to a new grid (new_yy, new_xx, and optionally new_alt for MapS3D).
    """
    if map_obj.map is None: return map_obj

    if copy:
        resampled_map_obj = dataclasses.replace(map_obj) # Shallow copy
    else:
        resampled_map_obj = map_obj
    
    itp = map_interpolate(map_obj, method=method, fill_value=0) # Fill with 0 for areas outside original
    if itp is None:
        print("Warning: Interpolator creation failed in map_resample. Returning original object.")
        return map_obj

    if isinstance(map_obj, MapS3D):
        if new_alt is None: raise ValueError("new_alt must be provided for resampling MapS3D.")
        yy_grid_new, xx_grid_new, alt_grid_new = np.meshgrid(new_yy, new_xx, new_alt, indexing='ij')
        query_points = np.column_stack((yy_grid_new.ravel(), xx_grid_new.ravel(), alt_grid_new.ravel()))
        resampled_map_data = itp(query_points).reshape(len(new_yy), len(new_xx), len(new_alt))
        resampled_map_obj.alt = new_alt.copy()
    else: # MapS, MapSd
        yy_grid_new, xx_grid_new = np.meshgrid(new_yy, new_xx, indexing='ij')
        query_points = np.column_stack((yy_grid_new.ravel(), xx_grid_new.ravel()))
        resampled_map_data = itp(query_points).reshape(len(new_yy), len(new_xx))
        if isinstance(map_obj, MapSd) and map_obj.alt is not None: # Resample altitude map for MapSd
            itp_alt = map_interpolate(dataclasses.replace(map_obj, map=map_obj.alt), method=method, fill_value=0)
            if itp_alt:
                resampled_map_obj.alt = itp_alt(query_points).reshape(len(new_yy), len(new_xx))

    resampled_map_obj.map = resampled_map_data
    resampled_map_obj.yy = new_yy.copy()
    resampled_map_obj.xx = new_xx.copy()
    
    # Create new meshgrid lat/lon if they existed
    if hasattr(resampled_map_obj, 'lat') and resampled_map_obj.lat is not None:
        resampled_map_obj.lat, resampled_map_obj.lon = np.meshgrid(new_yy, new_xx, indexing='ij')

    # Create a new mask (True where resampled data is not the fill_value, assuming fill_value means gap)
    # This is a simple mask; a more sophisticated one might consider original mask
    resampled_map_obj.mask = (resampled_map_data != 0) 
    
    return resampled_map_obj


def upward_fft(map_in: Union[MapS, MapS3D, MapV],
               alt_out: Union[float, np.ndarray],
               alpha: float = 0.0, # Regularization for downward continuation
               expand: bool = True) -> Union[MapS, MapS3D, MapV]:
    """
    Upward/downward continuation of a potential field (i.e., magnetic anomaly field) map.
    Uses the Fast Fourier Transform (FFT) to convert the map to the frequency
    domain, applies an upward continuation filter, and uses the inverse FFT to
    convert the map back to the spatial domain.

    Args:
        map_in: Input MapS, MapS3D, or MapV object.
        alt_out: Target altitude or array of altitudes [m].
        alpha: Regularization parameter for downward continuation (dz < 0).
        expand: If true, expand map temporarily to reduce edge effects.

    Returns:
        A new map object of the same type as map_in, at alt_out.
    """
    try:
        # Import the FFT-based implementation from map_fft module
        from .map_fft import upward_fft as map_fft_upward_fft
        
        # Use the FFT-based implementation
        return map_fft_upward_fft(map_in, alt_out, expand=expand, alpha=alpha)
    
    except ImportError:
        print(f"WARNING: map_fft module not found. Using placeholder implementation for upward_fft.")
        
        # Fallback to placeholder implementation
        map_out = dataclasses.replace(map_in) # Shallow copy for structure
        
        # Deep copy mutable fields
        if hasattr(map_out, 'map') and map_out.map is not None: map_out.map = map_out.map.copy()
        if hasattr(map_out, 'xx') and map_out.xx is not None: map_out.xx = map_out.xx.copy()
        if hasattr(map_out, 'yy') and map_out.yy is not None: map_out.yy = map_out.yy.copy()
        if hasattr(map_out, 'mask') and map_out.mask is not None: map_out.mask = map_out.mask.copy()
        if hasattr(map_out, 'lat') and map_out.lat is not None: map_out.lat = map_out.lat.copy()
        if hasattr(map_out, 'lon') and map_out.lon is not None: map_out.lon = map_out.lon.copy()

        if isinstance(map_in, MapV):
            if map_out.mapX is not None: map_out.mapX = map_out.mapX.copy()
            if map_out.mapY is not None: map_out.mapY = map_out.mapY.copy()
            if map_out.mapZ is not None: map_out.mapZ = map_out.mapZ.copy()
        
        current_alt_scalar = 0.0
        if isinstance(map_in.alt, np.ndarray):
            current_alt_scalar = np.mean(map_in.alt) if map_in.alt.size > 0 else 0.0
        elif isinstance(map_in.alt, (float, int)):
            current_alt_scalar = float(map_in.alt)

        dz = (np.mean(alt_out) if isinstance(alt_out, np.ndarray) else alt_out) - current_alt_scalar
        if dz < 0 and alpha == 0:
            print("WARNING: Downward continuation (dz < 0) attempted without regularization (alpha=0).")
            print("         This is unstable. Results will be unreliable.")

        if isinstance(map_in, (MapS, MapSd, MapV)) and isinstance(alt_out, (float, int, np.floating, np.integer)):
            map_out.alt = float(alt_out)
        elif isinstance(map_in, MapS) and isinstance(alt_out, np.ndarray): # Convert MapS to MapS3D
            print("INFO: upward_fft converting MapS to MapS3D for multiple output altitudes.")
            num_levels = len(alt_out)
            original_map_data = map_in.map
            new_3d_map_data = np.stack([original_map_data.copy()] * num_levels, axis=-1)
            new_3d_mask_data = None
            if map_in.mask is not None:
                new_3d_mask_data = np.stack([map_in.mask.copy()] * num_levels, axis=-1)
            
            return MapS3D(info=f"{map_in.info}_to_3D",
                          map=new_3d_map_data,
                          xx=map_in.xx.copy(),
                          yy=map_in.yy.copy(),
                          alt=alt_out.copy(),
                          mask=new_3d_mask_data)
        elif isinstance(map_in, MapS3D) and isinstance(alt_out, np.ndarray):
            # Placeholder: For MapS3D output, we'd need to actually compute each layer.
            # For now, just update the alt array and keep the first layer's data replicated.
            print("INFO: upward_fft for MapS3D with multiple altitudes will replicate first data layer.")
            num_levels = len(alt_out)
            first_layer_map = map_in.map[:,:,0].copy()
            first_layer_mask = map_in.mask[:,:,0].copy() if map_in.mask is not None else None
            
            map_out.map = np.stack([first_layer_map] * num_levels, axis=-1)
            if first_layer_mask is not None:
                map_out.mask = np.stack([first_layer_mask] * num_levels, axis=-1)
            else:
                map_out.mask = None
            map_out.alt = alt_out.copy()
        elif isinstance(map_in, MapS3D) and isinstance(alt_out, (float, int)):
            # Output a MapS from a MapS3D if single altitude is requested
            print("INFO: upward_fft converting MapS3D to MapS for single output altitude (using first layer data).")
            return MapS(info=f"{map_in.info}_to_S",
                        map=map_in.map[:,:,0].copy(), # Placeholder: use first layer
                        xx=map_in.xx.copy(),
                        yy=map_in.yy.copy(),
                        alt=float(alt_out),
                        mask=map_in.mask[:,:,0].copy() if map_in.mask is not None else None)
        else:
            map_out.alt = alt_out # This might be an array for MapSd if alt_out is array

        return map_out


# Map Caching
def map_cache_filename(map_name: str, alt: float, cache_dir: str = ".map_cache") -> str:
    """Generates a standard filename for a cached map."""
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            print(f"Warning: Could not create cache directory {cache_dir}: {e}")
            # Fallback to current directory if cache dir creation fails
            cache_dir = "."
            
    safe_map_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in map_name)
    # Ensure alt is a scalar for filename
    alt_scalar = np.mean(alt) if isinstance(alt, np.ndarray) else alt
    return os.path.join(cache_dir, f"{safe_map_name}_alt{alt_scalar:.0f}.pkl")

def save_map_to_cache(map_obj: Union[MapS, MapS3D, MapSd, MapV], filename: str):
    """Saves a map object to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(map_obj, f)
        # print(f"Map '{map_obj.info}' saved to cache: {filename}")
    except Exception as e:
        print(f"Error saving map to cache {filename}: {e}")

def load_map_from_cache(filename: str) -> Optional[Union[MapS, MapS3D, MapSd, MapV]]:
    """Loads a map object from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                map_obj = pickle.load(f)
            # print(f"Map loaded from cache: {filename}")
            return map_obj
        except Exception as e:
            print(f"Error loading map from cache {filename}: {e}")
            return None
    return None

def get_cached_map(map_cache_obj: MapCache, lat_query: float, lon_query: float, alt_query: float) -> Optional[Callable]:
    """
    Gets an interpolator from the map cache, prioritizing maps whose altitude is closest to alt_query
    and whose bounds contain (lat_query, lon_query).
    This is a simplified version of Julia's get_cached_map. Full version would involve
    on-the-fly upward_fft if a map at the exact dz interval isn't found.
    """
    best_interp = None
    min_alt_diff = float('inf')
    found_suitable_map = False

    # Check primary maps
    for (map_info, map_alt_scalar), interp_func in map_cache_obj._interpolators.items():
        # Find the original map object to check bounds (this is a bit inefficient)
        original_map = next((m for m in map_cache_obj.maps if m.info == map_info and (np.mean(m.alt) if isinstance(m.alt,np.ndarray) else m.alt) == map_alt_scalar), None)
        if original_map:
            (lat_min, lat_max), (lon_min, lon_max) = map_lims(original_map)
            if not (lat_min <= lat_query <= lat_max and lon_min <= lon_query <= lon_max):
                continue # Query point outside this map's bounds

        alt_diff = abs(map_alt_scalar - alt_query)
        if alt_diff < min_alt_diff:
            min_alt_diff = alt_diff
            best_interp = interp_func
            found_suitable_map = True
        elif alt_diff == min_alt_diff: # Prefer maps closer to query altitude if multiple have same diff
            # This part could be more sophisticated, e.g. smaller map if multiple exact matches
            pass


    if found_suitable_map:
        return best_interp
    
    # Check fallback map if no primary map was suitable
    if map_cache_obj._fallback_interpolator and map_cache_obj.fallback_map:
        (lat_min, lat_max), (lon_min, lon_max) = map_lims(map_cache_obj.fallback_map)
        if lat_min <= lat_query <= lat_max and lon_min <= lon_query <= lon_max:
            return map_cache_obj._fallback_interpolator
            
    return None


# Functions for IGRF correction
def map_correct_igrf(map_obj: Union[MapS, MapSd, MapS3D], sub_igrf_date: Optional[float] = None, add_igrf_date: Optional[float] = None,
                    zone_utm: int = 18, is_north: bool = True, map_units: str = 'rad', **kwargs) -> Union[MapS, MapSd, MapS3D]:
    """
    Correct the International Geomagnetic Reference Field (IGRF), i.e., core field,
    of a map by subtracting and/or adding the IGRF on specified date(s).
    
    Args:
        map_obj: MapS, MapSd, or MapS3D scalar magnetic anomaly map struct
        sub_igrf_date: Date of IGRF core field to subtract [yr], None to ignore
        add_igrf_date: Date of IGRF core field to add [yr], None to ignore
        zone_utm: UTM zone, only used if map_units = 'utm'
        is_north: If true, map is in northern hemisphere, only used if map_units = 'utm'
        map_units: Map xx/yy units {'rad','deg','utm'}
        
    Returns:
        Map object with IGRF corrected map data
    """
    # Create a copy of the input map to avoid modifying the original
    map_result = dataclasses.replace(map_obj)
    
    # Deep copy mutable fields
    if hasattr(map_result, 'map') and map_result.map is not None:
        map_result.map = map_result.map.copy()
    if hasattr(map_result, 'mask') and map_result.mask is not None:
        map_result.mask = map_result.mask.copy()
    
    # Check if we need to do any IGRF correction
    if sub_igrf_date is None and add_igrf_date is None:
        return map_result
    
    try:
        # Try to import pyIGRF for IGRF calculations
        import pyIGRF
    except ImportError:
        print(f"Warning: pyIGRF module not found. IGRF correction not applied to map {map_obj.info}.")
        return map_result
    
    # Process different map types
    if isinstance(map_obj, (MapS, MapSd)):
        # Get map parameters
        _, ind1, _, _ = map_params(map_obj.map, map_obj.xx, map_obj.yy)
        
        # Handle altitude
        if isinstance(map_obj, MapSd):
            # For drape maps, altitude is a 2D grid
            map_alt = map_obj.alt
        else:
            # For scalar maps, altitude is a single value
            if isinstance(map_obj.alt, np.ndarray) and map_obj.alt.size > 0:
                map_alt = np.full_like(map_obj.map, np.mean(map_obj.alt))
            else:
                map_alt = np.full_like(map_obj.map, map_obj.alt)
        
        # Process each point in the map
        for i in range(map_obj.xx.size):
            for j in range(map_obj.yy.size):
                if ind1[j, i]:  # Only process valid data points
                    # Get coordinates based on map units
                    if map_units == 'utm':
                        # Convert UTM to LLA
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs(
                                f"+proj=utm +zone={zone_utm} +{'north' if is_north else 'south'} +ellps=WGS84",
                                "+proj=longlat +ellps=WGS84 +datum=WGS84",
                                always_xy=True
                            )
                            lon, lat = transformer.transform(map_obj.xx[i], map_obj.yy[j])
                            alt = map_alt[j, i]
                        except ImportError:
                            print("Warning: pyproj not available for UTM to LLA conversion. Skipping IGRF correction.")
                            return map_result
                    elif map_units == 'rad':
                        lat = np.rad2deg(map_obj.yy[j])
                        lon = np.rad2deg(map_obj.xx[i])
                        alt = map_alt[j, i]
                    elif map_units == 'deg':
                        lat = map_obj.yy[j]
                        lon = map_obj.xx[i]
                        alt = map_alt[j, i]
                    else:
                        raise ValueError(f"Unsupported map_units: {map_units}")
                    
                    # Subtract IGRF if requested
                    if sub_igrf_date is not None:
                        # Calculate IGRF values at this point
                        igrf_values = pyIGRF.igrf_value(lat, lon, alt/1000, sub_igrf_date)  # alt in km for pyIGRF
                        # Extract total field intensity (F)
                        igrf_total = igrf_values[6]  # F is the 7th value (index 6)
                        # Subtract from map
                        map_result.map[j, i] -= igrf_total
                    
                    # Add IGRF if requested
                    if add_igrf_date is not None:
                        # Calculate IGRF values at this point
                        igrf_values = pyIGRF.igrf_value(lat, lon, alt/1000, add_igrf_date)  # alt in km for pyIGRF
                        # Extract total field intensity (F)
                        igrf_total = igrf_values[6]  # F is the 7th value (index 6)
                        # Add to map
                        map_result.map[j, i] += igrf_total
    
    elif isinstance(map_obj, MapS3D):
        # Process each altitude level separately
        for i in range(len(map_obj.alt)):
            # Create a MapS for this altitude level
            level_map = MapS(
                info=map_obj.info,
                map=map_obj.map[:, :, i],
                xx=map_obj.xx,
                yy=map_obj.yy,
                alt=map_obj.alt[i],
                mask=map_obj.mask[:, :, i] if map_obj.mask is not None else None
            )
            
            # Apply IGRF correction to this level
            corrected_level = map_correct_igrf(
                level_map,
                sub_igrf_date=sub_igrf_date,
                add_igrf_date=add_igrf_date,
                zone_utm=zone_utm,
                is_north=is_north,
                map_units=map_units,
                **kwargs
            )
            
            # Update the result map with the corrected level
            map_result.map[:, :, i] = corrected_level.map
    
    return map_result

def map_combine(map1: Union[MapS, MapSd], map2: Union[MapS, MapSd],
                method: str = "average_overlap",
                feather_width: float = 0.1,
                resample_to: str = "map1",
                copy: bool = True) -> Union[MapS, MapSd]:
    """
    Combines two map objects (MapS or MapSd) using various methods.
    
    Args:
        map1: First map object
        map2: Second map object
        method: Combination method:
            - "average_overlap": Average values in overlapping regions
            - "feather": Smooth transition in overlap using distance-based weighting
            - "priority_map1": Prioritize map1, use map2 only in gaps
            - "priority_map2": Prioritize map2, use map1 only in gaps
        feather_width: Width of feathering transition zone as fraction of overlap region (0-1)
        resample_to: Grid to use for the result ("map1", "map2", or "common")
        copy: If True, create a new map object; otherwise modify map1 in-place
    
    Returns:
        Combined map object of the same type as inputs
    """
    # Validate inputs
    if not isinstance(map1, (MapS, MapSd)) or not isinstance(map2, (MapS, MapSd)):
        raise TypeError("map_combine requires MapS or MapSd objects")
    
    if not isinstance(map1, type(map2)):
        raise TypeError(f"Cannot combine maps of different types: {type(map1)} and {type(map2)}")
    
    # Handle altitude differences
    if not np.allclose(map1.alt, map2.alt):
        print(f"Maps at different altitudes: {map1.alt} vs {map2.alt}. Continuing map2 to map1's altitude.")
        try:
            # Use upward_fft to continue map2 to map1's altitude
            map2 = upward_fft(map2, map1.alt)
        except Exception as e:
            print(f"Warning: Failed to continue map2 to map1's altitude: {e}")
            print("Proceeding with original altitudes, results may be inaccurate.")
    
    # Determine target grid based on resample_to parameter
    if resample_to == "map1":
        target_xx, target_yy = map1.xx, map1.yy
    elif resample_to == "map2":
        target_xx, target_yy = map2.xx, map2.yy
    elif resample_to == "common":
        # Create a common grid that encompasses both maps
        xx_min = min(np.min(map1.xx), np.min(map2.xx))
        xx_max = max(np.max(map1.xx), np.max(map2.xx))
        yy_min = min(np.min(map1.yy), np.min(map2.yy))
        yy_max = max(np.max(map1.yy), np.max(map2.yy))
        
        # Use the finer resolution of the two maps
        dx1 = get_step(map1.xx)
        dx2 = get_step(map2.xx)
        dy1 = get_step(map1.yy)
        dy2 = get_step(map2.yy)
        
        dx = min(dx1, dx2) if min(dx1, dx2) > 0 else max(dx1, dx2)
        dy = min(dy1, dy2) if min(dy1, dy2) > 0 else max(dy1, dy2)
        
        # Create new grid
        target_xx = np.arange(xx_min, xx_max + dx/2, dx)
        target_yy = np.arange(yy_min, yy_max + dy/2, dy)
    else:
        raise ValueError(f"Invalid resample_to value: {resample_to}. Must be 'map1', 'map2', or 'common'")
    
    # Resample both maps to the target grid if needed
    map1_resampled = map1
    if not (np.array_equal(map1.xx, target_xx) and np.array_equal(map1.yy, target_yy)):
        map1_resampled = map_resample(map1, target_yy, target_xx, copy=True)
    
    map2_resampled = map2
    if not (np.array_equal(map2.xx, target_xx) and np.array_equal(map2.yy, target_yy)):
        map2_resampled = map_resample(map2, target_yy, target_xx, copy=True)
    
    # Create output map object
    if copy:
        combined_map_obj = dataclasses.replace(map1_resampled)
        combined_map_obj.map = map1_resampled.map.copy()
        if map1_resampled.mask is not None:
            combined_map_obj.mask = map1_resampled.mask.copy()
        else:
            combined_map_obj.mask = np.ones_like(map1_resampled.map, dtype=bool)
    else:
        combined_map_obj = map1_resampled
        if combined_map_obj.mask is None:
            combined_map_obj.mask = np.ones_like(combined_map_obj.map, dtype=bool)
    
    # Identify valid data regions in both maps
    map1_valid = ~np.isnan(map1_resampled.map) & (map1_resampled.map != 0)
    if map1_resampled.mask is not None:
        map1_valid = map1_valid & map1_resampled.mask
    
    map2_valid = ~np.isnan(map2_resampled.map) & (map2_resampled.map != 0)
    if map2_resampled.mask is not None:
        map2_valid = map2_valid & map2_resampled.mask
    
    # Identify overlap and gap regions
    overlap = map1_valid & map2_valid
    map1_only = map1_valid & ~map2_valid
    map2_only = map2_valid & ~map1_valid
    
    # Apply the selected combination method
    if method == "average_overlap":
        # Use map1 in map1-only regions
        combined_map_obj.map[map1_only] = map1_resampled.map[map1_only]
        combined_map_obj.mask[map1_only] = True
        
        # Use map2 in map2-only regions
        combined_map_obj.map[map2_only] = map2_resampled.map[map2_only]
        combined_map_obj.mask[map2_only] = True
        
        # Average in overlap regions
        combined_map_obj.map[overlap] = (map1_resampled.map[overlap] + map2_resampled.map[overlap]) / 2
        combined_map_obj.mask[overlap] = True
    
    elif method == "feather":
        # Use map1 in map1-only regions
        combined_map_obj.map[map1_only] = map1_resampled.map[map1_only]
        combined_map_obj.mask[map1_only] = True
        
        # Use map2 in map2-only regions
        combined_map_obj.map[map2_only] = map2_resampled.map[map2_only]
        combined_map_obj.mask[map2_only] = True
        
        # For overlap regions, compute distance-based weights
        if np.any(overlap):
            try:
                # Create distance transforms from the edges of each valid region
                from scipy.ndimage import distance_transform_edt
                
                # Get distance from edge of map1 valid region
                map1_edge_dist = distance_transform_edt(map1_valid)
                # Normalize to [0, 1] within overlap region
                if np.max(map1_edge_dist[overlap]) > 0:
                    map1_edge_dist[overlap] /= np.max(map1_edge_dist[overlap])
                
                # Get distance from edge of map2 valid region
                map2_edge_dist = distance_transform_edt(map2_valid)
                # Normalize to [0, 1] within overlap region
                if np.max(map2_edge_dist[overlap]) > 0:
                    map2_edge_dist[overlap] /= np.max(map2_edge_dist[overlap])
                
                # Compute relative weights
                w1 = map1_edge_dist[overlap]
                w2 = map2_edge_dist[overlap]
                sum_w = w1 + w2
                
                # Avoid division by zero
                sum_w[sum_w == 0] = 1.0
                
                # Apply weighted average in overlap region
                combined_map_obj.map[overlap] = (
                    (w1 * map1_resampled.map[overlap] + w2 * map2_resampled.map[overlap]) / sum_w
                )
                combined_map_obj.mask[overlap] = True
                
            except ImportError:
                print("Warning: scipy.ndimage not available for feathering. Using simple average instead.")
                combined_map_obj.map[overlap] = (map1_resampled.map[overlap] + map2_resampled.map[overlap]) / 2
                combined_map_obj.mask[overlap] = True
    
    elif method == "priority_map1":
        # Use map1 wherever it's valid
        combined_map_obj.map[map1_valid] = map1_resampled.map[map1_valid]
        combined_map_obj.mask[map1_valid] = True
        
        # Use map2 only where map1 is invalid
        combined_map_obj.map[map2_only] = map2_resampled.map[map2_only]
        combined_map_obj.mask[map2_only] = True
    
    elif method == "priority_map2":
        # Use map2 wherever it's valid
        combined_map_obj.map[map2_valid] = map2_resampled.map[map2_valid]
        combined_map_obj.mask[map2_valid] = True
        
        # Use map1 only where map2 is invalid
        combined_map_obj.map[map1_only] = map1_resampled.map[map1_only]
        combined_map_obj.mask[map1_only] = True
    
    else:
        raise ValueError(f"Invalid combination method: {method}")
    
    # Update map info
    combined_map_obj.info = f"Combined({method}): {map1.info} & {map2.info}"
    
    # Handle special case for MapSd - combine altitude grids if needed
    if isinstance(combined_map_obj, MapSd) and isinstance(combined_map_obj.alt, np.ndarray):
        if isinstance(map1.alt, np.ndarray) and isinstance(map2.alt, np.ndarray):
            # Use the same combination method for altitude grids
            alt_combined = np.zeros_like(combined_map_obj.map)
            
            if method in ["priority_map1", "average_overlap", "feather"]:
                alt_combined[map1_valid] = map1_resampled.alt[map1_valid]
                alt_combined[map2_only] = map2_resampled.alt[map2_only]
                if method == "average_overlap" and np.any(overlap):
                    alt_combined[overlap] = (map1_resampled.alt[overlap] + map2_resampled.alt[overlap]) / 2
            else:  # priority_map2
                alt_combined[map2_valid] = map2_resampled.alt[map2_valid]
                alt_combined[map1_only] = map1_resampled.alt[map1_only]
            
            combined_map_obj.alt = alt_combined
    
    return combined_map_obj


def map_combine_multi(maps: List[Union[MapS, MapSd]],
                      target_alt: Optional[float] = None,
                      method: str = "average_overlap",
                      resample_to: str = "common",
                      copy: bool = True) -> Union[MapS, MapSd]:
    """
    Combines multiple maps at potentially different altitudes into a single map.
    
    Args:
        maps: List of MapS or MapSd objects to combine
        target_alt: Target altitude for the combined map. If None, uses altitude of first map.
        method: Combination method (see map_combine for options)
        resample_to: Grid to use for the result ("common" or index of map to use as reference)
        copy: If True, create a new map object; otherwise modify first map in-place
    
    Returns:
        Combined map object of the same type as inputs
    """
    if not maps:
        raise ValueError("No maps provided to map_combine_multi")
    
    if len(maps) == 1:
        return maps[0] if not copy else dataclasses.replace(maps[0])
    
    # Determine target altitude
    if target_alt is None:
        target_alt = maps[0].alt
        if isinstance(target_alt, np.ndarray):
            target_alt = np.mean(target_alt)
    
    # Continue all maps to target altitude
    continued_maps = []
    for i, map_obj in enumerate(maps):
        if np.allclose(map_obj.alt, target_alt):
            continued_maps.append(map_obj)
        else:
            try:
                continued_map = upward_fft(map_obj, target_alt)
                continued_maps.append(continued_map)
            except Exception as e:
                print(f"Warning: Failed to continue map {i} to target altitude: {e}")
                print("Skipping this map in the combination.")
    
    if not continued_maps:
        raise ValueError("No maps could be continued to target altitude")
    
    # Determine target grid
    if resample_to == "common":
        # Find bounds that encompass all maps
        xx_min = min(np.min(m.xx) for m in continued_maps)
        xx_max = max(np.max(m.xx) for m in continued_maps)
        yy_min = min(np.min(m.yy) for m in continued_maps)
        yy_max = max(np.max(m.yy) for m in continued_maps)
        
        # Use the finest resolution among all maps
        dx_values = [get_step(m.xx) for m in continued_maps]
        dy_values = [get_step(m.yy) for m in continued_maps]
        
        dx = min(dx for dx in dx_values if dx > 0) if any(dx > 0 for dx in dx_values) else dx_values[0]
        dy = min(dy for dy in dy_values if dy > 0) if any(dy > 0 for dy in dy_values) else dy_values[0]
        
        # Create common grid
        target_xx = np.arange(xx_min, xx_max + dx/2, dx)
        target_yy = np.arange(yy_min, yy_max + dy/2, dy)
    else:
        # Use the grid of the specified map
        try:
            idx = int(resample_to)
            if idx < 0 or idx >= len(continued_maps):
                raise ValueError(f"Invalid resample_to index: {idx}")
            target_xx = continued_maps[idx].xx
            target_yy = continued_maps[idx].yy
        except (ValueError, TypeError):
            # Default to first map if invalid
            print(f"Warning: Invalid resample_to value: {resample_to}. Using first map's grid.")
            target_xx = continued_maps[0].xx
            target_yy = continued_maps[0].yy
    
    # Resample all maps to target grid
    resampled_maps = []
    for i, map_obj in enumerate(continued_maps):
        if np.array_equal(map_obj.xx, target_xx) and np.array_equal(map_obj.yy, target_yy):
            resampled_maps.append(map_obj)
        else:
            try:
                resampled_map = map_resample(map_obj, target_yy, target_xx, copy=True)
                resampled_maps.append(resampled_map)
            except Exception as e:
                print(f"Warning: Failed to resample map {i} to target grid: {e}")
                print("Skipping this map in the combination.")
    
    if not resampled_maps:
        raise ValueError("No maps could be resampled to target grid")
    
    # Combine maps sequentially
    result = resampled_maps[0]
    for i in range(1, len(resampled_maps)):
        result = map_combine(result, resampled_maps[i], method=method, copy=copy)
    
    # Update info
    map_names = [m.info for m in maps]
    result.info = f"Multi-combined({method}): {', '.join(map_names)}"
    
    return result

# Functions from the original map_utils.py that are being kept or slightly adapted
def get_map(map_name: str, variable_name: str = "map_data", *args, map_type: Optional[str] = None, **kwargs) -> Union[MapS, MapV, None]:
    print(f"DEBUG_MAP_UTILS_GET_MAP: ID of this get_map: {id(get_map)}") # New print
    print(f"IMMEDIATE_GET_MAP_ENTRY: map_name='{map_name}', var_name='{variable_name}', args={args}, map_type_param='{map_type}', kwargs={kwargs}")
    print(f"DEBUG_GET_MAP_ENTRY: map_name='{map_name}', variable_name='{variable_name}', map_type='{map_type}', args={args}, kwargs={kwargs}")
    """
    Retrieves a specific magnetic map, attempting to load from a .mat file if map_name is a path.
    If map_name is a known ID (e.g., "namad", "emm720"), it returns a placeholder.
    This function is a placeholder and primarily for loading test .mat files.
    """
    if os.path.exists(map_name) and map_name.endswith(".mat"):
        try:
            mat_content = scipy.io.loadmat(map_name)
            if variable_name in mat_content:
                data_struct = mat_content[variable_name]
                
                # Common structure for .mat files from MagNav test data:
                # data_struct is often a struct array, so access fields via [0,0] and then field name.
                # Field names might be 'map', 'xx', 'yy', 'alt', 'info', 'mask'
                # Or nested like data_struct['map'][0,0]
                
                map_values = data_struct['map'][0,0] if 'map' in data_struct.dtype.names else None
                map_xx_deg = data_struct['xx'][0,0].ravel() if 'xx' in data_struct.dtype.names else None
                map_yy_deg = data_struct['yy'][0,0].ravel() if 'yy' in data_struct.dtype.names else None
                map_alt_m = data_struct['alt'][0,0] if 'alt' in data_struct.dtype.names else None
                map_info_str = str(data_struct['info'][0,0][0]) if 'info' in data_struct.dtype.names and data_struct['info'][0,0].size > 0 else f"Loaded from {map_name}"
                map_mask_bool = data_struct['mask'][0,0].astype(bool) if 'mask' in data_struct.dtype.names else np.ones_like(map_values, dtype=bool)

                if map_values is None or map_xx_deg is None or map_yy_deg is None or map_alt_m is None:
                    print(f"Warning: Essential map fields (map, xx, yy, alt) not found in {map_name} under variable {variable_name}. Returning MAP_S_NULL.")
                    return MAP_S_NULL

                # Convert degrees to radians for xx, yy as per common convention in MagNav
                map_xx_rad = np.deg2rad(map_xx_deg)
                map_yy_rad = np.deg2rad(map_yy_deg)
                
                # Ensure alt is scalar if it's supposed to be (e.g. for MapS)
                if isinstance(map_alt_m, np.ndarray) and map_alt_m.size == 1:
                    map_alt_m_scalar = map_alt_m.item()
                elif isinstance(map_alt_m, (float, int)):
                    map_alt_m_scalar = float(map_alt_m)
                else: # If alt is an array (e.g. for MapSd or if map is 3D)
                    # This loader is simplified for MapS, so we might take the mean or first value
                    print(f"Warning: Altitude from .mat file is an array. Taking mean for MapS.alt: {map_alt_m}")
                    map_alt_m_scalar = np.mean(map_alt_m) if isinstance(map_alt_m, np.ndarray) and map_alt_m.size > 0 else 0.0


                # Create lat, lon meshgrids if needed by MapS type (though interpolator uses xx, yy vectors)
                # yy_mesh, xx_mesh = np.meshgrid(map_yy_rad, map_xx_rad, indexing='ij')

                return MapS(info=map_info_str,
                            map=map_values,
                            xx=map_xx_rad,
                            yy=map_yy_rad,
                            alt=map_alt_m_scalar,
                            mask=map_mask_bool,
                            lat=np.array([]), # Placeholder, not directly from typical .mat
                            lon=np.array([])  # Placeholder
                            )
            else:
                print(f"Warning: Variable '{variable_name}' not found in {map_name}. Returning MAP_S_NULL.")
                return MAP_S_NULL
        except Exception as e:
            print(f"Error loading map from {map_name}: {e}. Returning MAP_S_NULL.")
            return MAP_S_NULL
    
    # Fallback for known map IDs or if file not found/not .mat
    if map_name.lower() == "namad":
        print(f"Placeholder: get_map called for NAMAD. Returning MAP_S_NULL. Load actual NAMAD data separately.")
    elif map_name.lower() == "emm720" or "emm720" in map_name.lower():
        with open("debug_get_map.txt", "a") as f_debug:
            f_debug.write(f"--- New get_map emm720 entry ---\n")
            f_debug.write(f"EMM720_BLOCK_ENTERED: map_name='{map_name}'\n")
            f_debug.write(f"KWARGS: {kwargs}\n")
            # Using direct map_type parameter now for f_debug
            f_debug.write(f"FDEBUG_MAP_TYPE_PARAM_VAL: {map_type}\n")
            f_debug.write(f"FDEBUG_TYPE_MAP_TYPE_PARAM: {type(map_type)}\n")
            is_vector_robust_check_fdebug = map_type and map_type.lower() == 'vector'
            f_debug.write(f"FDEBUG_IS_VECTOR_ROBUST_CHECK_RESULT: {is_vector_robust_check_fdebug}\n")

        print(f"DEBUG: In get_map emm720 block. map_name='{map_name}'")
        print(f"DEBUG: kwargs received: {kwargs}")
        # Using direct map_type parameter now for print
        print(f"DEBUG: map_type parameter value: {map_type}")
        print(f"DEBUG: type of map_type parameter: {type(map_type)}")
        print(f"DEBUG: Condition (map_type and map_type.lower() == 'vector') is: {map_type and map_type.lower() == 'vector'}")
        
        if map_type and map_type.lower() == "vector": # Use the direct parameter, robustly
            with open("debug_get_map.txt", "a") as f_debug:
                f_debug.write("EMM720_BLOCK_VECTOR_PATH_TAKEN\n")
            print(f"Placeholder: get_map called for EMM720 (or similar) with map_type='vector'. Returning DUMMY MapV. Load actual EMM720 data separately.")
            # Create a minimal valid MapV object to prevent ValueError in create_flux
            # Ensure xx and yy are 1D and have at least one element.
            # Ensure x, y, z components match dimensions of xx, yy.
            dummy_xx = np.array([0.0, 1.0]) # Min 2 points for some interpolators if not careful
            dummy_yy = np.array([0.0, 1.0])
            dummy_data_shape = (dummy_yy.size, dummy_xx.size)
            return MapV(info="Dummy EMM720 Vector Map",
                        alt=0.0,
                        x=np.zeros(dummy_data_shape),
                        y=np.zeros(dummy_data_shape),
                        z=np.zeros(dummy_data_shape),
                        xx=dummy_xx,
                        yy=dummy_yy,
                        mask=np.ones(dummy_data_shape, dtype=bool),
                        lat=np.array([]), # Placeholder
                        lon=np.array([]) # Placeholder
                        )
        else: # Corresponds to: if kwargs.get('map_type') == "vector"
            with open("debug_get_map.txt", "a") as f_debug:
                f_debug.write("EMM720_BLOCK_SCALAR_PATH_TAKEN\n")
            print(f"Placeholder: get_map called for EMM720 (scalar, or similar). Returning MAP_S_NULL. Load actual EMM720 data separately.")
            return MAP_S_NULL # Original behavior for scalar
    else: # Corresponds to: if map_name.lower() == "namad": / elif map_name.lower() == "emm720":
        print(f"Placeholder: get_map called for '{map_name}'. File not found or not a .mat, or unknown ID. Returning MAP_S_NULL.")
        return MAP_S_NULL

# Placeholders for specific map loaders mentioned in original map_utils or task
def ottawa_area_maps(map_name: str = "", *args, **kwargs) -> Optional[MapS]:
    print(f"Placeholder: ottawa_area_maps called for {map_name}. Not implemented.")
    return MAP_S_NULL

def namad(*args, **kwargs) -> Optional[MapS]:
    print("Placeholder: namad map loader called. Not implemented. Use get_map with path to NAMAD data.")
    return MAP_S_NULL

def emag2(*args, **kwargs) -> Optional[MapS]:
    print("Placeholder: emag2 map loader called. Not implemented.")
    return MAP_S_NULL

def emm720(*args, **kwargs) -> Optional[MapS]:
    print("Placeholder: emm720 map loader called. Not implemented. Use get_map with path to EMM720 data.")
    return MAP_S_NULL

# --- End of map_utils.py content ---