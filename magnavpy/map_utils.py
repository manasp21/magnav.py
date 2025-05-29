# -*- coding: utf-8 -*-
"""
Utility functions for map handling, including interpolation.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Union, Tuple, Any

# Attempt to import MapS, MapV, MAP_S_NULL from .common_types
# This creates a dependency, but it's a lower-level one than analysis_util or magnav.
try:
    from .common_types import MapS, MapV, MAP_S_NULL, MapCache
except ImportError:
    # Fallback for standalone execution or if common_types is not found initially
    # This is primarily for robustness during development/testing of this module in isolation.
    # In a full package context, the .common_types import should work.
    print("Warning: Could not import MapS, MapV, MAP_S_NULL, MapCache from .common_types in map_utils.py. Using placeholder types.")
    # Define placeholder dataclasses if import fails, to allow type hinting and basic structure.
    from dataclasses import dataclass, field
    @dataclass
    class MapS:
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: Union[np.ndarray, float] = field(default_factory=lambda: np.array([])) # Can be scalar or array
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([])) # longitude grid
        yy: np.ndarray = field(default_factory=lambda: np.array([])) # latitude grid

    @dataclass
    class MapV: # Placeholder
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: Union[np.ndarray, float] = field(default_factory=lambda: np.array([]))
        map: np.ndarray = field(default_factory=lambda: np.array([])) # Placeholder, might be components
        x: np.ndarray = field(default_factory=lambda: np.array([]))
        y: np.ndarray = field(default_factory=lambda: np.array([]))
        z: np.ndarray = field(default_factory=lambda: np.array([]))

    MAP_S_NULL = MapS(info="null_placeholder")

    class MapCache: # Placeholder
        def __init__(self, maps, fallback=None, dz=100):
            self.maps = maps
            self.fallback_map = fallback
            self.dz = dz
            self.interpolators = [] # List of interpolator functions or None


def get_map_val(
    map_data: Union[MapS, MapCache],
    lat_query: Union[float, np.ndarray],
    lon_query: Union[float, np.ndarray],
    alt_query: Union[float, np.ndarray] = None, # Optional, for future use with 3D maps or MapCache
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float = np.nan,
    return_interpolator: bool = False
) -> Union[np.ndarray, float, Tuple[Union[np.ndarray, float], Any]]:
    """
    Get magnetic anomaly map value(s) at specified geographic coordinate(s).

    This function interpolates values from a gridded magnetic anomaly map (MapS)
    or uses a MapCache.

    Args:
        map_data: A MapS object containing the map grid and data, or a MapCache object.
        lat_query: Latitude(s) of query point(s) [radians].
        lon_query: Longitude(s) of query point(s) [radians].
        alt_query: Altitude(s) of query point(s) [m]. (Currently used with MapCache,
                   or for selecting map from cache if applicable. For a single MapS,
                   interpolation is done on the map's intrinsic altitude plane).
        method: Interpolation method (e.g., "linear", "nearest").
        bounds_error: If True, an error is raised if query points are outside map bounds.
                      If False, fill_value is used.
        fill_value: Value to use for points outside the interpolation bounds if bounds_error is False.
        return_interpolator: If True, returns a tuple (values, interpolator_object).
                             The interpolator object is None if MapCache is used and no single
                             best interpolator can be determined, or if interpolation fails.

    Returns:
        Interpolated magnetic anomaly value(s) [nT]. Returns np.nan for points
        outside map bounds if bounds_error is False.
        If return_interpolator is True, returns a tuple (values, interpolator_object).
    """
    interpolator_to_return = None # Initialize

    if isinstance(map_data, MapCache):
        # Use MapCache's interpolation logic if available and suitable
        # This is a simplified interaction; MapCache might have more complex selection.
        # For now, try the first available interpolator if any.
        # A more robust approach would select the interpolator based on alt_query.
        if map_data.interpolators:
            # Find the best interpolator based on altitude.
            # This is a simplified selection logic.
            best_interp = None
            min_alt_diff = float('inf')

            if alt_query is None and map_data.maps: # If no alt_query, use first map's alt
                alt_q_scalar = map_data.maps[0].alt
                if isinstance(alt_q_scalar, np.ndarray): # If map alt is array, take mean
                    alt_q_scalar = np.mean(alt_q_scalar) if alt_q_scalar.size > 0 else 0
            elif isinstance(alt_query, np.ndarray):
                alt_q_scalar = np.mean(alt_query) # Use mean of query altitudes for selection
            else:
                alt_q_scalar = alt_query # Use scalar query altitude

            for i, interp_func in enumerate(map_data.interpolators):
                if interp_func is not None and i < len(map_data.maps):
                    map_obj = map_data.maps[i]
                    map_alt = map_obj.alt
                    if isinstance(map_alt, np.ndarray): # If map alt is array, take mean
                        map_alt = np.mean(map_alt) if map_alt.size > 0 else float('inf')

                    alt_diff = abs(map_alt - alt_q_scalar)
                    if alt_diff < min_alt_diff:
                        min_alt_diff = alt_diff
                        best_interp = interp_func
            
            interpolator_to_return = best_interp # This might be None if no suitable one is found

            if best_interp:
                points = np.column_stack((np.atleast_1d(lat_query), np.atleast_1d(lon_query)))
                interpolated_values = best_interp(points).squeeze()
                if return_interpolator:
                    return interpolated_values, interpolator_to_return
                return interpolated_values
            elif map_data.fallback_map and map_data.fallback_map.map.size > 0:
                # print("Warning: No suitable interpolator in MapCache, using fallback map.")
                map_s_object = map_data.fallback_map # Proceed to interpolate with fallback
            else:
                # print("Warning: No suitable interpolator in MapCache and no fallback. Returning NaNs.")
                nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
                if return_interpolator:
                    return nan_vals, None
                return nan_vals
        else: # No interpolators in cache
            if map_data.fallback_map and map_data.fallback_map.map.size > 0:
                # print("Warning: No interpolators in MapCache, using fallback map.")
                map_s_object = map_data.fallback_map # Proceed to interpolate with fallback
            else:
                # print("Warning: No interpolators in MapCache and no fallback. Returning NaNs.")
                nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
                if return_interpolator:
                    return nan_vals, None
                return nan_vals

    elif isinstance(map_data, MapS):
        map_s_object = map_data
    else:
        raise TypeError("map_data must be a MapS object or MapCache object.")

    if not (hasattr(map_s_object, 'yy') and hasattr(map_s_object, 'xx') and hasattr(map_s_object, 'map')):
        raise ValueError("MapS object is missing 'yy', 'xx', or 'map' attributes.")

    if map_s_object.map is None or map_s_object.yy is None or map_s_object.xx is None:
        # print(f"Warning: Map '{map_s_object.info}' has None for map, yy, or xx. Returning NaNs.")
        nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
        if return_interpolator:
            return nan_vals, None
        return nan_vals
        
    map_grid_lat = np.asarray(map_s_object.yy).squeeze()
    map_grid_lon = np.asarray(map_s_object.xx).squeeze()
    map_values = np.asarray(map_s_object.map)

    if map_grid_lat.ndim == 0: map_grid_lat = np.array([map_grid_lat.item()])
    if map_grid_lon.ndim == 0: map_grid_lon = np.array([map_grid_lon.item()])

    if map_grid_lat.size == 0 or map_grid_lon.size == 0 or map_values.size == 0:
        # print(f"Warning: Map '{map_s_object.info}' has empty coordinates or map data. Returning NaNs.")
        nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
        if return_interpolator:
            return nan_vals, None
        return nan_vals

    # Ensure map grid coordinates are 1D for RegularGridInterpolator
    if map_grid_lat.ndim != 1 or map_grid_lon.ndim != 1:
        raise ValueError("Map grid coordinates (yy, xx) must be 1D arrays.")

    # Handle map_values dimensions
    # For MapS, map_values should be 2D (lat, lon)
    if map_values.ndim != 2:
        # If it's 1D and matches a grid dimension, it might be a profile - not typical for MapS.
        # If it's 3D, it might be a MapS3D, which this basic get_map_val doesn't fully handle yet without alt_query.
        raise ValueError(f"MapS.map data must be 2D. Got {map_values.ndim}D for map '{map_s_object.info}'.")

    if map_values.shape != (map_grid_lat.size, map_grid_lon.size):
        raise ValueError(
            f"Map data shape {map_values.shape} does not match grid coordinate "
            f"dimensions ({map_grid_lat.size}, {map_grid_lon.size}) for map '{map_s_object.info}'."
        )

    # Create interpolator
    # Note: MapS.lat and MapS.lon are often the meshgrid results,
    # while MapS.yy and MapS.xx are the unique coordinate vectors.
    try:
        interpolator = RegularGridInterpolator(
            (map_grid_lat, map_grid_lon),
            map_values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        interpolator_to_return = interpolator
    except ValueError as e:
        # This can happen if coordinates are not strictly monotonic increasing
        # print(f"Error creating interpolator for map '{map_s_object.info}': {e}. Check if yy and xx are sorted and unique.")
        # Fallback to NaN or re-raise depending on desired strictness
        nan_vals = np.full_like(np.atleast_1d(lat_query), fill_value, dtype=float).squeeze()
        if return_interpolator:
            return nan_vals, None
        return nan_vals

    # Prepare query points: (lat, lon)
    # RegularGridInterpolator expects points as (N, D_coord) where D_coord is 2 for (lat,lon)
    query_points = np.column_stack((np.atleast_1d(lat_query), np.atleast_1d(lon_query)))

    # Interpolate
    interpolated_values = interpolator_to_return(query_points)

    result_values = interpolated_values.squeeze()

    if return_interpolator:
        return result_values, interpolator_to_return
    return result_values

# Placeholder for a function that might be needed for vector maps if get_map_val is extended
# def get_map_val_vector(...):
#     # Similar logic but would interpolate Bx, By, Bz components from a MapV object
#     pass
# Placeholder functions moved from the (missing) get_map.py
# These would need their actual implementations.

def get_map(map_name: str, *args, **kwargs) -> Union[MapS, MapV, None]:
    """
    Placeholder for get_map. Retrieves a specific magnetic map.
    Actual implementation would load from file or database.
    """
    print(f"Placeholder: get_map called for {map_name}")
    # Return a dummy MapS object or MAP_S_NULL
    return MAP_S_NULL

def ottawa_area_maps(map_name: str = "", *args, **kwargs) -> Union[MapS, None]:
    """Placeholder for ottawa_area_maps."""
    print(f"Placeholder: ottawa_area_maps called for {map_name}")
    return MAP_S_NULL

def namad(*args, **kwargs) -> Union[MapS, None]:
    """Placeholder for namad map loader."""
    print("Placeholder: namad called")
    return MAP_S_NULL

def emag2(*args, **kwargs) -> Union[MapS, None]:
    """Placeholder for emag2 map loader."""
    print("Placeholder: emag2 called")
    return MAP_S_NULL

def emm720(*args, **kwargs) -> Union[MapS, None]:
    """Placeholder for emm720 map loader."""
    print("Placeholder: emm720 called")
    return MAP_S_NULL

def upward_fft(map_in: MapS, alt_out: float, *args, **kwargs) -> MapS:
    """Placeholder for upward_fft continuation."""
    print(f"Placeholder: upward_fft called for map {map_in.info} to alt {alt_out}")
    # Return a modified version of map_in or a new MapS
    return map_in # Simplistic placeholder

def map_interpolate(map_obj: MapS, lat: np.ndarray, lon: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Placeholder for map_interpolate. This is largely superseded by get_map_val."""
    print(f"Placeholder: map_interpolate called for map {map_obj.info}. Consider using get_map_val.")
    # Simple passthrough or call get_map_val
    return get_map_val(map_obj, lat, lon)