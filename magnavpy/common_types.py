# This file will contain common type definitions for MagNavPy.
from typing import List, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys # For print statements to stderr, if uncommented (currently all commented out)

from abc import ABC # For abstract base classes
from dataclasses import dataclass, field

@dataclass
class MagV:
    """Vector magnetometer measurement struct."""
    x: np.ndarray  # 1D array x-direction magnetic field [nT]
    y: np.ndarray  # 1D array y-direction magnetic field [nT]
    z: np.ndarray  # 1D array z-direction magnetic field [nT]
    t: np.ndarray  # 1D array total magnetic field [nT]

# --- Data Structures (Map related classes) ---
class Map(ABC):
    """Abstract type for a magnetic anomaly map.
    Base fields are inherited by subclasses.
    """
    info: str       # map information
    lat: np.ndarray # latitude [rad] - often general bounds or not used if xx/yy present
    lon: np.ndarray # longitude [rad] - often general bounds or not used if xx/yy present
    alt: np.ndarray # altitude [m] - type varies in subclasses, shadowed if needed
    map: np.ndarray # magnetic anomaly map [nT] - structure varies, shadowed if needed

@dataclass
class MapS(Map):
    """Scalar magnetic anomaly map struct. Based on Julia's MapS.
    Inherits lat, lon from Map ABC. Defines its own alt (scalar) and map (2D).
    """
    info: str       # map information
    lat: np.ndarray # Explicitly define to include in __init__, inherited from Map
    lon: np.ndarray # Explicitly define to include in __init__, inherited from Map
    map: np.ndarray # ny x nx scalar magnetic anomaly map [nT] (2D)
    alt: float      # map altitude [m] (scalar)
    # Default fields must come after non-default fields
    xx: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # nx map x-direction (longitude) coordinates [rad] (1D)
    yy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # ny map y-direction (latitude)  coordinates [rad] (1D)
    mask: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=bool)) # ny x nx mask for valid (not filled-in) map data (2D boolean)

    # lat, lon are inherited from Map ABC. If they are needed for __init__ (i.e. no default in ABC),
    # they must be provided or Map ABC must be adjusted.
    # For dataclass __init__ generation, fields in MapS shadow those in Map.
    # Unmentioned fields from Map (like lat, lon if they lack defaults) would be required by __init__.
    # Assuming lat, lon from Map ABC are handled (e.g. have defaults or are optional in practice).

@dataclass
class MapSd(Map): # Inherits directly from Map, not MapS, for alt_matrix clarity
    """Scalar magnetic anomaly map struct for drape maps (altitude varies per grid point).
    Based on Julia's MapSd.
    """
    info: str
    map: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=float))     # ny x nx scalar magnetic anomaly map [nT] (2D)
    xx: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # nx map x-direction (longitude) coordinates [rad] (1D)
    yy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # ny map y-direction (latitude)  coordinates [rad] (1D)
    alt_matrix: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=float)) # ny x nx altitude map [m] (2D). Corresponds to Julia MapSd.alt.
    mask: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=bool))    # ny x nx mask (2D boolean)
    # This class does not use the 'alt' field from the Map ABC in the same way as MapS.
    # lat, lon are inherited from Map ABC.

@dataclass
class MapS3D(Map):
    """3D (multi-level) scalar magnetic anomaly map struct. Based on Julia's MapS3D."""
    info: str
    map: np.ndarray = field(default_factory=lambda: np.empty((0,0,0), dtype=float)) # ny x nx x nz 3D map [nT]
    xx: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # nx longitude coordinates [rad] (1D)
    yy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # ny latitude coordinates [rad] (1D)
    zz: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))      # nz altitude levels [m] (1D). Corresponds to Julia MapS3D.alt.
    mask: np.ndarray = field(default_factory=lambda: np.empty((0,0,0), dtype=bool)) # ny x nx x nz 3D mask
    # lat, lon are inherited from Map ABC. 'alt' from Map ABC is not directly used by zz.

@dataclass
class MapV(Map):
    """Vector magnetic anomaly map struct. Based on Julia's MapV."""
    info: str
    alt: float    # map altitude [m] (scalar)
    # mapX, mapY, mapZ from Julia are x, y, z here.
    x: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=float)) # x-direction magnetic anomaly [nT] (2D)
    y: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=float)) # y-direction magnetic anomaly [nT] (2D)
    z: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=float)) # z-direction magnetic anomaly [nT] (2D)
    xx: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # nx longitude coordinates [rad] (1D)
    yy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # ny latitude coordinates [rad] (1D)
    mask: np.ndarray = field(default_factory=lambda: np.empty((0,0), dtype=bool)) # ny x nx mask (2D boolean)
    # 'map' field from Map ABC is not directly used by components x,y,z.
    # lat, lon are inherited. 'alt' from Map ABC is shadowed by float alt here.

# Null map constant, updated for the new MapS definition.
# Assumes MapS constructor will handle inherited lat, lon from Map ABC if they are required.
# For this to work without error, lat/lon in Map ABC should ideally have defaults or be Optional.
# If lat/lon from Map ABC are mandatory non-default, they must be provided here.
MAP_S_NULL = MapS(
    info="Null map",
    # Provide lat/lon if they are mandatory from Map ABC and lack defaults
    lat=np.array([], dtype=float), # Assuming these are needed by MapS effective __init__
    lon=np.array([], dtype=float), # Assuming these are needed by MapS effective __init__
    map=np.zeros((1,1), dtype=float),
    xx=np.array([0.0], dtype=float),
    yy=np.array([0.0], dtype=float),
    alt=0.0,
    mask=np.ones((1,1), dtype=bool)
)

class MapCache:
    """Map cache struct, mutable."""
    def __init__(self, maps: List[MapS], fallback: MapS = MAP_S_NULL, dz: Union[int, float] = 100):
        if not isinstance(maps, list):
            raise TypeError("Input 'maps' must be a list.")

        self.maps = sorted(
            [
                m for m in maps
                if isinstance(m, MapS) and
                   hasattr(m, 'map') and isinstance(m.map, np.ndarray) and m.map.ndim == 2 and
                   hasattr(m, 'alt') # alt is used for sorting
            ],
            key=lambda m: m.alt
        )
        self.fallback_map = fallback if isinstance(fallback, MapS) else MAP_S_NULL
        self.dz = dz

        self.interpolators = []
        for m_obj in self.maps:
            if hasattr(m_obj, 'yy') and m_obj.yy is not None and \
               hasattr(m_obj, 'xx') and m_obj.xx is not None and \
               m_obj.map is not None:

                yy_coords = np.asarray(m_obj.yy).squeeze()
                xx_coords = np.asarray(m_obj.xx).squeeze()

                if yy_coords.ndim == 0: yy_coords = np.array([yy_coords.item()])
                if xx_coords.ndim == 0: xx_coords = np.array([xx_coords.item()])

                if yy_coords.size == 0 or xx_coords.size == 0 or \
                   m_obj.map.shape[0] != yy_coords.size or m_obj.map.shape[1] != xx_coords.size:
                    self.interpolators.append(None)
                    # map_info = getattr(m_obj, 'info', 'N/A')
                    # print(f"Warning: Skipping interpolator for map '{map_info}' due to coordinate/map mismatch or empty coords.", file=sys.stderr)
                    continue

                try:
                    interp_func = RegularGridInterpolator((yy_coords, xx_coords), m_obj.map,
                                                          bounds_error=False, fill_value=np.nan)
                    self.interpolators.append(interp_func)
                except ValueError as e:
                    self.interpolators.append(None)
                    # map_info = getattr(m_obj, 'info', 'N/A')
                    # print(f"Warning: ValueError creating interpolator for map '{map_info}': {e}. Ensure coordinates are monotonic.", file=sys.stderr)
                except Exception as e:
                    self.interpolators.append(None)
                    # map_info = getattr(m_obj, 'info', 'N/A')
                    # print(f"Warning: Unexpected error creating interpolator for map '{map_info}': {e}", file=sys.stderr)
            else:
                self.interpolators.append(None)
                # map_info = getattr(m_obj, 'info', 'N/A')
                # print(f"Warning: Map '{map_info}' missing map, yy, or xx data for interpolator. Skipping.", file=sys.stderr)
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