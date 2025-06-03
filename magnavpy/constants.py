# magnav.py/magnavpy/constants.py

import os
import numpy as np

# Attempt to import MapS.
# If MapS is defined in common_types.py and common_types.py itself imports
# constants.py, this could lead to a circular import.
# A placeholder is used if the actual class is not found.
_MAP_S_IMPORTED = False
try:
    # Assuming common_types.py is in the same directory (magnavpy)
    from .common_types import MapS
    _MAP_S_IMPORTED = True
except ImportError:
    # Define a placeholder if MapS is not available.
    # This allows constants.py to be imported even if common_types.py is not fully ready
    # or if there's a temporary circular dependency during porting.
    class MapS:
        """Minimal placeholder for MapS if the actual class is not available."""
        def __init__(self, info: str, map_data, xx, yy, alt: float, mask):
            self.info = info
            self.map = map_data
            self.xx = xx
            self.yy = yy
            self.alt = alt
            self.mask = mask

        def __repr__(self):
            return (f"MapS(info='{self.info}', alt={self.alt}, "
                    f"map_shape={getattr(self.map, 'shape', None)})")
    # Note: If this placeholder is used, a message can be logged when constants.py is imported,
    # or users should be aware that MAP_S_NULL uses this placeholder.


# Version
# In Julia: const magnav_version = VersionNumber(TOML.parse(open(project_toml))["version"])
# Python equivalent can be complex. For constants.py, a string is often used,
# potentially updated by a build process or defined in a central __init__.py.
MAGNAV_VERSION = "0.0.0"  # Placeholder: Set manually or by build process/package metadata

# Numerical Constants from MagNav.jl (lines 47-79 in MagNav.jl)
NUM_MAG_MAX = 6             # Maximum number of scalar & vector magnetometers (each)
E_EARTH = 0.0818191908426   # First eccentricity of Earth [-]
G_EARTH = 9.80665           # Gravity of Earth [m/s^2]
R_EARTH = 6378137           # WGS-84 radius of Earth [m] (original was Int)
OMEGA_EARTH = 7.2921151467e-5 # Rotation rate of Earth [rad/s]

# Paths to artifact-like data files (lines 89, 96, 189, 202, 213 in MagNav.jl)
# In Julia, these are resolved using the artifact system.
# In Python, these would typically be paths relative to a data directory.
# Here, they are defined as string constants representing conventional relative paths
# from a conceptual project 'data' directory.
# os.path.join is used for OS-independent path construction.
_DATA_DIR_NAME = "data" # Assumed data directory at the project root

USGS_PATH = os.path.join(_DATA_DIR_NAME, "util_files", "util_files", "color_scale_usgs.csv")
ICON_CIRCLE_PATH = os.path.join(_DATA_DIR_NAME, "util_files", "util_files", "icon_circle.dae")
EMAG2_H5_FILE = os.path.join(_DATA_DIR_NAME, "EMAG2", "EMAG2.h5")
EMM720_H5_FILE = os.path.join(_DATA_DIR_NAME, "EMM720_World", "EMM720_World.h5")
NAMAD_H5_FILE = os.path.join(_DATA_DIR_NAME, "NAMAD_305", "NAMAD_305.h5")

# Boolean Constants (line 103 in MagNav.jl)
SILENT_DEBUG = True         # Internal flag for verbose print outs

# Complex Constants (line 395 in MagNav.jl)
# Default Map IDs
DEFAULT_VECTOR_MAP_ID = "default_vector_map_id_placeholder"
DEFAULT_SCALAR_MAP_ID = "default_scalar_map_id_placeholder"
# Null scalar magnetic anomaly map (MapS instance)
# Julia: const mapS_null = MapS("Null map",zeros(1,1),[0.0],[0.0],0.0,trues(1,1))
# Requires numpy and a MapS class definition (actual or placeholder).
MAP_S_NULL = MapS(
    info="Null map",
    lat=np.array([], dtype=float),
    lon=np.array([], dtype=float),
    map=np.zeros((1, 1), dtype=float),  # Changed from map_data, Julia zeros(1,1) -> 1x1 Matrix{Float64}
    xx=np.array([0.0], dtype=float),         # Julia [0.0] -> Vector{Float64}
    yy=np.array([0.0], dtype=float),         # Julia [0.0] -> Vector{Float64}
    alt=0.0,                                 # Julia 0.0 -> Float64
    mask=np.ones((1, 1), dtype=bool)         # Julia trues(1,1) -> 1x1 BitMatrix
)

# End of constant definitions from MagNav.jl

if not _MAP_S_IMPORTED:
    # This section is primarily for developers' information if the placeholder is active.
    # It could be a print statement during import for debugging, but that's often discouraged in libraries.
    # A comment suffices for the source code.
    # print(
    #     "MagNavPy constants INFO: MAP_S_NULL initialized using a placeholder for MapS. "
    #     "Ensure common_types.MapS is available for full functionality."
    # )
    pass