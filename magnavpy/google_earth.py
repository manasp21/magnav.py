"""
Functions for generating KML/KMZ files for Google Earth.
Ported from MagNav.jl/src/google_earth.jl
"""
import simplekml
import numpy as np
import os
import zipfile
from typing import Union, Tuple, List, Any

# Placeholder type definitions. Ideally, import from magnavpy.common_types
# from .common_types import Traj, INS, FILTout, MapS, MapSd, MapS3D

class _BaseNavData:
    def __init__(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
        self.lat = lat  # Expected in radians
        self.lon = lon  # Expected in radians
        self.alt = alt  # Expected in meters

class Traj(_BaseNavData):
    pass

class INS(_BaseNavData):
    pass

class FILTout(_BaseNavData):
    pass

Path = Union[Traj, INS, FILTout]

class _BaseMapData:
    def __init__(self, map_data: np.ndarray, xx: np.ndarray, yy: np.ndarray, mask: np.ndarray = None):
        # map_data typically (ny, nx) or (ny, nx, nz)
        # xx (longitude), yy (latitude) expected in radians
        self.map = map_data
        self.xx = xx
        self.yy = yy
        self.mask = mask

class MapS(_BaseMapData):
    pass

class MapSd(_BaseMapData): # Assuming similar structure for simplicity
    pass

class MapS3D(_BaseMapData): # Assuming map[:,:,0] is used as in Julia
    pass

MapTypes = Union[MapS, MapSd, MapS3D]

# Default icon for point placemarks if not specified
DEFAULT_POINT_ICON = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

def _ensure_extension(filename: str, ext: str) -> str:
    """Ensures the filename has the given extension."""
    if not ext.startswith('.'):
        ext = '.' + ext
    if not filename.lower().endswith(ext.lower()):
        return filename + ext
    return filename

def _remove_extension(filename: str, ext: str) -> str:
    """Removes the given extension from the filename if present."""
    if not ext.startswith('.'):
        ext = '.' + ext
    if filename.lower().endswith(ext.lower()):
        return filename[:-len(ext)]
    return filename

def path_to_kml(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    path_kml: str = "path.kml",
    path_units: str = "rad",
    width: int = 3,
    color1: str = "ff000000",  # KML color AABBGGRR (Black)
    color2: str = "80000000",  # KML color AABBGGRR (Semi-transparent Black for poly fill)
    points: bool = False,
    point_icon_href: str = DEFAULT_POINT_ICON,
    point_icon_scale: float = 1.0
) -> None:
    """
    Create KML file of a flight path for use with Google Earth.

    Args:
        lat: Latitude array.
        lon: Longitude array.
        alt: Altitude array [m].
        path_kml: Path/name of the KML file to save.
        path_units: Units of lat/lon {'rad', 'deg'}.
        width: Line width for path.
        color1: Path color (AABBGGRR format, e.g., 'ff0000ff' for red).
        color2: Below-path color for LineString extrusion (AABBGGRR format).
        points: If True, create points instead of a line.
        point_icon_href: URL or path to the icon for points.
        point_icon_scale: Scale factor for the point icon.
    """
    if not isinstance(lat, np.ndarray): lat = np.array(lat)
    if not isinstance(lon, np.ndarray): lon = np.array(lon)
    if not isinstance(alt, np.ndarray): alt = np.array(alt)

    num_coords = len(lat)
    # Google Earth point limits (approximate from Julia code)
    limit = 1000 if points else 30000
    step = int(np.ceil(num_coords / limit)) if num_coords > limit else 1

    if path_units == "rad":
        lat_deg = np.rad2deg(lat)
        lon_deg = np.rad2deg(lon)
    elif path_units == "deg":
        lat_deg = lat
        lon_deg = lon
    else:
        raise ValueError(f"Unsupported path_units: {path_units}. Use 'rad' or 'deg'.")

    path_kml = _ensure_extension(path_kml, ".kml")

    kml = simplekml.Kml()
    kml.document.name = _remove_extension(os.path.basename(path_kml), ".kml")

    if points:
        folder = kml.newfolder(name="Path Points")
        for i in range(0, num_coords, step):
            pnt = folder.newpoint(name=f"Point {i//step + 1}")
            pnt.coords = [(lon_deg[i], lat_deg[i], alt[i])]
            pnt.altitudemode = simplekml.AltitudeMode.relativetoground
            pnt.style.iconstyle.icon.href = point_icon_href
            pnt.style.iconstyle.scale = point_icon_scale
            # KML color is AABBGGRR, simplekml color is also AABBGGRR
            pnt.style.iconstyle.color = color1 # Apply main color to icon
            # The Julia version uses a <Model> for points, which is more complex.
            # This uses standard KML point styling.
    else:
        ls = kml.newlinestring(name="Aircraft Path")
        coords = []
        for i in range(0, num_coords, step):
            coords.append((lon_deg[i], lat_deg[i], alt[i]))
        ls.coords = coords
        ls.extrude = 1
        ls.tessellate = 1
        ls.altitudemode = simplekml.AltitudeMode.relativetoground

        ls.style.linestyle.width = width
        ls.style.linestyle.color = color1
        ls.style.polystyle.color = color2 # For the extruded part below the line
        # To match Julia's StyleMap for normal/highlight, simplekml requires StyleMap object
        # For simplicity, applying a single style here.
        # stylemap = kml.newstylemap()
        # stylemap.normalstyle.linestyle.color = color1
        # stylemap.normalstyle.linestyle.width = width
        # stylemap.normalstyle.polystyle.color = color2
        # stylemap.highlightstyle.linestyle.color = color1 # Or a different highlight color
        # stylemap.highlightstyle.linestyle.width = width * 1.5
        # stylemap.highlightstyle.polystyle.color = color2
        # ls.stylemap = stylemap

    kml.save(path_kml)

def path_obj_to_kml(
    path_obj: Path,
    path_kml: str = "path.kml",
    width: int = 3,
    color1: str = "",  # Auto-selected based on type if empty
    color2: str = "00ffffff",  # Transparent white for PolyStyle fill
    points: bool = False,
    point_icon_href: str = DEFAULT_POINT_ICON,
    point_icon_scale: float = 1.0
) -> None:
    """
    Create KML file of flight path from a Path object (Traj, INS, FILTout).

    Args:
        path_obj: Path object with .lat, .lon, .alt attributes (in radians).
        path_kml: Path/name of the KML file to save.
        width: Line width.
        color1: Path color (AABBGGRR). Auto-set if empty.
        color2: Below-path color (AABBGGRR).
        points: If True, create points instead of a line.
        point_icon_href: URL or path to the icon for points.
        point_icon_scale: Scale factor for the point icon.
    """
    _color1 = color1
    if not _color1:
        if isinstance(path_obj, Traj):
            _color1 = "ff0085ff"  # Orange (ABGR: ffff8500)
        elif isinstance(path_obj, INS):
            _color1 = "ffec502b"  # Blue (ABGR: ff2b50ec)
        elif isinstance(path_obj, FILTout):
            _color1 = "ff009b31"  # Green (ABGR: ff319b00)
        else:
            _color1 = "ff000000"  # Default to Black

    # Handle common color names from Julia version
    if _color1.lower() in ["black", "k"]: _color1 = "ff000000"
    if color2.lower() in ["black", "k"]: color2 = "80000000"


    path_to_kml(path_obj.lat, path_obj.lon, path_obj.alt, path_kml,
                  path_units="rad", width=width, color1=_color1, color2=color2,
                  points=points, point_icon_href=point_icon_href,
                  point_icon_scale=point_icon_scale)

def map_to_kmz(
    map_map: np.ndarray,
    map_xx: np.ndarray,
    map_yy: np.ndarray,
    map_kmz: str = "map.kmz",
    map_units: str = "rad",
    plot_alt: float = 0.0,
    opacity: float = 0.75,
    clims: Tuple[float, float] = (),
    map_png_path: str = None # Path to the pre-generated map image
) -> None:
    """
    Create KMZ file of a map for use with Google Earth.
    This function assumes the map image (PNG) is already generated or
    will be generated externally and its path provided via `map_png_path`.
    If `map_png_path` is None, it will derive a name but not create the image.

    Args:
        map_map: 2D gridded map data (ny, nx).
        map_xx: Map x-direction (longitude) coordinates.
        map_yy: Map y-direction (latitude) coordinates.
        map_kmz: Path/name of the map KMZ file to save.
        map_units: Map xx/yy units {'rad', 'deg'}.
        plot_alt: Map altitude in Google Earth [m].
        opacity: Map opacity {0:1}.
        clims: Length-2 map colorbar limits (cmin, cmax). Used by external plotting.
        map_png_path: (Optional) Path to an existing PNG image of the map.
                      If None, a name is derived (e.g., 'map.png'), but the
                      image file itself must be created by other means.
    """
    if map_units == "rad":
        map_west_deg = np.rad2deg(np.min(map_xx))
        map_east_deg = np.rad2deg(np.max(map_xx))
        map_south_deg = np.rad2deg(np.min(map_yy))
        map_north_deg = np.rad2deg(np.max(map_yy))
    elif map_units == "deg":
        map_west_deg = np.min(map_xx)
        map_east_deg = np.max(map_xx)
        map_south_deg = np.min(map_yy)
        map_north_deg = np.max(map_yy)
    else:
        raise ValueError(f"Unsupported map_units: {map_units}. Use 'rad' or 'deg'.")

    map_kmz_abs = os.path.abspath(_ensure_extension(map_kmz, ".kmz"))
    base_name = _remove_extension(os.path.basename(map_kmz_abs), ".kmz")
    
    # Define paths for KML and PNG components
    # These files will be created in the same directory as the KMZ by default
    # or in a temporary location if preferred.
    # For simplicity, creating them next to the KMZ output.
    output_dir = os.path.dirname(map_kmz_abs)
    if not output_dir: # If map_kmz is just a filename, use current dir
        output_dir = "."

    kml_file_name_in_zip = base_name + ".kml"
    kml_file_path = os.path.join(output_dir, kml_file_name_in_zip)

    png_file_name_in_zip = base_name + ".png"
    _map_png_path = map_png_path
    if _map_png_path is None:
        _map_png_path = os.path.join(output_dir, png_file_name_in_zip)
        # IMPORTANT: The actual PNG file generation is NOT handled here.
        # It's assumed that `_map_png_path` will point to a valid image
        # created by a separate plotting function using map_map, map_xx, map_yy, clims.
        # e.g., call a function like:
        # generate_map_image(map_map, map_xx, map_yy, _map_png_path, clims=clims, ...)
        print(f"Warning: Map image '{_map_png_path}' is expected to be generated externally.")
    else:
        # If map_png_path is provided, ensure its basename matches what KML expects
        png_file_name_in_zip = os.path.basename(_map_png_path)


    # KML color is AABBGGRR. Opacity is applied to white (ffffff).
    alpha_hex = format(int(round(opacity * 255)), '02x')
    kml_overlay_color = alpha_hex + "ffffff"  # White with alpha

    kml = simplekml.Kml()
    kml.document.name = base_name
    
    ground = kml.newgroundoverlay(name=base_name + " Overlay")
    ground.icon.href = png_file_name_in_zip # Relative path within KMZ
    ground.latlonbox.north = map_north_deg
    ground.latlonbox.south = map_south_deg
    ground.latlonbox.east = map_east_deg
    ground.latlonbox.west = map_west_deg
    ground.color = kml_overlay_color # For opacity

    if plot_alt > 0:
        ground.altitude = plot_alt
        ground.altitudemode = simplekml.AltitudeMode.absolute
    else:
        # Default is clampToGround, explicitly set if needed or rely on simplekml default
        ground.altitudemode = simplekml.AltitudeMode.clamptoground
        ground.altitude = 0 # Explicitly set altitude to 0 for clampToGround

    kml.save(kml_file_path)

    # Create KMZ archive
    try:
        with zipfile.ZipFile(map_kmz_abs, 'w', zipfile.ZIP_DEFLATED) as kmz_file:
            kmz_file.write(kml_file_path, arcname=kml_file_name_in_zip)
            if os.path.exists(_map_png_path):
                kmz_file.write(_map_png_path, arcname=png_file_name_in_zip)
            else:
                # This case should be handled by user ensuring _map_png_path exists
                # or by the image generation step if integrated.
                print(f"Warning: PNG file '{_map_png_path}' not found. KMZ will be incomplete.")
    finally:
        # Clean up temporary KML file
        if os.path.exists(kml_file_path):
            os.remove(kml_file_path)
        # The original PNG (_map_png_path) is NOT removed here, as it might be
        # a user-provided file or generated by a process that expects to manage it.
        # Julia code removes its self-generated PNG. If this function were to
        # generate the PNG internally, it should clean it up.

def map_obj_to_kmz(
    map_obj: MapTypes,
    map_kmz: str = "map.kmz",
    use_mask: bool = True,
    plot_alt: float = 0.0,
    opacity: float = 0.75,
    clims: Tuple[float, float] = (),
    map_png_path: str = None
) -> None:
    """
    Create KMZ file from a MapS-like object.

    Args:
        map_obj: Map object with .map, .xx, .yy, and optionally .mask attributes.
        map_kmz: Path/name of the map KMZ file to save.
        use_mask: If True and map_obj.mask exists, apply it to map_obj.map.
        plot_alt: Map altitude in Google Earth [m].
        opacity: Map opacity {0:1}.
        clims: Map colorbar limits.
        map_png_path: Path to an existing PNG image of the map.
    """
    current_map_data = map_obj.map
    if isinstance(map_obj, MapS3D):
        print("Info: 3D map provided, using map at lowest altitude (first slice).")
        current_map_data = map_obj.map[:, :, 0]
        if hasattr(map_obj, 'mask') and map_obj.mask is not None:
            current_mask = map_obj.mask[:, :, 0]
        else:
            current_mask = None
    else: # MapS, MapSd
        if hasattr(map_obj, 'mask') and map_obj.mask is not None:
            current_mask = map_obj.mask
        else:
            current_mask = None

    if use_mask and current_mask is not None:
        # Ensure mask is boolean for multiplication if necessary, or use np.where
        # Assuming map_data is numeric and mask is boolean or 0/1
        # If mask is boolean, convert to float for multiplication
        display_map = current_map_data * current_mask.astype(current_map_data.dtype)
    else:
        display_map = current_map_data

    map_to_kmz(display_map, map_obj.xx, map_obj.yy, map_kmz,
                 map_units="rad", plot_alt=plot_alt, opacity=opacity,
                 clims=clims, map_png_path=map_png_path)