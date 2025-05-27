# This file will contain common type definitions for MagNavPy.
from typing import List, Union
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys # For print statements to stderr, if uncommented (currently all commented out)

# This import is the critical point. If it causes a new circular import,
# then MapS (and likely other Map related types) also need to be moved here.
from .magnav import MapS

class MapCache:
    """Map cache struct, mutable."""
    def __init__(self, maps: List[MapS], fallback: MapS = None, dz: Union[int, float] = 100):
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
        self.fallback_map = fallback if isinstance(fallback, MapS) else None
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

# Note: If a __call__ method or other methods for MapCache exist in magnav.py for this class,
# they MUST be moved here as well. This operation is based on the __init__ part
# identified from the file snippet and list_code_definition_names.