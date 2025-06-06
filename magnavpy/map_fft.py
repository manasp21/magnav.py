# -*- coding: utf-8 -*-
"""
FFT-based map operations for upward/downward continuation and other transformations.

This module implements the FFT-based upward/downward continuation algorithm for
magnetic anomaly maps, as well as other FFT-based operations like vector field
calculations and power spectral density analysis.

Reference: Blakely, Potential Theory in Gravity and Magnetic Applications,
2009, Chapter 12 & Appendix B (pg. 315-317 & 402).
"""
import numpy as np
from typing import Tuple, Union, Optional, List, Any
import dataclasses

# Import map types from common_types
try:
    from .common_types import MapS, MapV, MapS3D, MapSd
except ImportError:
    print("Warning: Could not import types from .common_types in map_fft.py. Using placeholder types.")
    from dataclasses import dataclass, field
    
    @dataclass
    class MapS:
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: Union[np.ndarray, float] = field(default_factory=lambda: np.array([]))
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([]))
        yy: np.ndarray = field(default_factory=lambda: np.array([]))
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
    
    @dataclass
    class MapV:
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
    class MapS3D:
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: np.ndarray = field(default_factory=lambda: np.array([]))
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([]))
        yy: np.ndarray = field(default_factory=lambda: np.array([]))
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))
    
    @dataclass
    class MapSd:
        info: str = "placeholder"
        lat: np.ndarray = field(default_factory=lambda: np.array([]))
        lon: np.ndarray = field(default_factory=lambda: np.array([]))
        alt: np.ndarray = field(default_factory=lambda: np.array([]))
        map: np.ndarray = field(default_factory=lambda: np.array([]))
        xx: np.ndarray = field(default_factory=lambda: np.array([]))
        yy: np.ndarray = field(default_factory=lambda: np.array([]))
        mask: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))

# Import utility functions
try:
    from .map_utils import get_step, dlon2de, dlat2dn
except ImportError:
    # Define placeholder functions if imports fail
    def get_step(arr):
        """Get the median step size of elements in a 1D array."""
        if arr is None or arr.size < 2:
            return 0.0
        return float(np.median(np.abs(np.diff(arr))))
    
    def dlon2de(dlon, lat):
        """Convert longitude difference to east distance."""
        return dlon * 111320.0 * np.cos(lat)
    
    def dlat2dn(dlat, lat=None):
        """Convert latitude difference to north distance."""
        return dlat * 110540.0

def create_k(dx: float, dy: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create radial wavenumber (spatial frequency) grid.
    
    Args:
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
        nx: x-direction map dimension [-]
        ny: y-direction map dimension [-]
    
    Returns:
        k: ny x nx radial wavenumber (i.e., magnitude of wave vector)
        kx: ny x nx x-direction radial wavenumber
        ky: ny x nx y-direction radial wavenumber
    """
    # DFT sample frequencies [rad/m], 1/dx & 1/dy are sampling rates [1/m]
    if nx * dx == 0:
        kx = np.zeros((ny, nx), dtype=float)
    else:
        kx = np.repeat(2 * np.pi * np.fft.fftfreq(nx, 1/dx)[np.newaxis, :], ny, axis=0)
    
    if ny * dy == 0:
        ky = np.zeros((ny, nx), dtype=float)
    else:
        ky = np.repeat(2 * np.pi * np.fft.fftfreq(ny, 1/dy)[:, np.newaxis], nx, axis=1)
    
    k = np.sqrt(kx**2 + ky**2)
    return k, kx, ky

def smooth7(x: int) -> int:
    """
    Find the lowest 7-smooth number y >= x.
    
    A 7-smooth number is a positive integer whose prime factors are all <= 7.
    This is used to optimize FFT performance by ensuring dimensions are products
    of small primes (2, 3, 5, 7).
    
    Args:
        x: Input integer
    
    Returns:
        y: Smallest 7-smooth number >= x
    """
    y = 2 * x
    for i in range(int(np.ceil(np.log(x) / np.log(7))) + 1):
        for j in range(int(np.ceil(np.log(x) / np.log(5))) + 1):
            for k in range(int(np.ceil(np.log(x) / np.log(3))) + 1):
                z = 7**i * 5**j * 3**k
                if z < 2 * x:
                    y = min(y, 2**int(np.ceil(np.log(x/z) / np.log(2))) * z)
    return y

def map_expand(map_map: np.ndarray, pad: int = 1) -> Tuple[np.ndarray, int, int]:
    """
    Expand a map with padding on each edge to eliminate discontinuities in the
    discrete Fourier transform. The map is "wrapped around" to make it periodic.
    Padding expands the map to 7-smooth dimensions, allowing for a faster Fast
    Fourier Transform algorithm to be used during upward/downward continuation.
    
    Args:
        map_map: ny x nx 2D gridded map data
        pad: minimum padding (grid cells) along map edges
    
    Returns:
        map_map: ny x nx 2D gridded map data, expanded (padded)
        padx: x-direction padding (grid cells) applied on first edge
        pady: y-direction padding (grid cells) applied on first edge
    """
    map_ = map_map.astype(float)
    
    ny, nx = map_.shape  # original map size
    Ny, Nx = [smooth7(dim + 2*pad) for dim in (ny, nx)]  # map size with 7-smooth padding
    
    # padding on each edge
    padx = (int(np.floor((Nx-nx)/2)), int(np.ceil((Nx-nx)/2)))
    pady = (int(np.floor((Ny-ny)/2)), int(np.ceil((Ny-ny)/2)))
    
    # place original map in middle of new map
    x1, x2 = 1 + padx[0], nx + padx[0]
    y1, y2 = 1 + pady[0], ny + pady[0]
    map_expanded = np.zeros((Ny, Nx), dtype=float)
    map_expanded[y1:y2+1, x1:x2+1] = map_
    
    # fill row edges (right/left)
    for j in range(y1, y2+1):
        vals = np.linspace(map_expanded[j, x1], map_expanded[j, x2], Nx-nx+2)[1:-1]
        map_expanded[j, 0:x1] = vals[:padx[0]][::-1]
        map_expanded[j, x2+1:] = vals[padx[0]:][::-1]
    
    # fill column edges (top/bottom)
    for i in range(Nx):
        vals = np.linspace(map_expanded[y1, i], map_expanded[y2, i], Ny-ny+2)[1:-1]
        map_expanded[0:y1, i] = vals[:pady[0]][::-1]
        map_expanded[y2+1:, i] = vals[pady[0]:][::-1]
    
    return map_expanded, padx[0], pady[0]

def map_up_fft(map_map: np.ndarray, dx: float, dy: float, dz: Union[float, np.ndarray], 
               expand: bool = True, alpha: float = 0) -> np.ndarray:
    """
    Upward continuation of a potential field (i.e., magnetic anomaly field) map.
    Uses the Fast Fourier Transform (FFT) to convert the map to the frequency
    domain, applies an upward continuation filter, and uses the inverse FFT to
    convert the map back to the spatial domain. Optionally expands the map
    temporarily with periodic padding. Downward continuation may be performed to a
    limited degree as well, but be careful, as this is generally unstable and
    amplifies high frequencies (i.e., noise).
    
    Reference: Blakely, Potential Theory in Gravity and Magnetic Applications,
    2009, Chapter 12 & Appendix B (pg. 315-317 & 402).
    
    Args:
        map_map: ny x nx 2D gridded map data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
        dz: z-direction upward/downward continuation distance(s) [m]
        expand: if true, expand map temporarily to reduce edge effects
        alpha: regularization parameter for downward continuation
    
    Returns:
        map_map: ny x nx (x nz) 2D or 3D gridded map data, upward/downward continued
    """
    ny, nx = map_map.shape
    
    if expand:
        pad = min(int(np.ceil(10 * np.max(np.abs(dz)) / min(dx, dy))), 5000)  # set pad > 10*dz
        map_, px, py = map_expand(map_map, pad)  # expand with pad
        Ny, Nx = map_.shape
    else:
        map_ = map_map
        Ny, Nx, px, py = ny, nx, 0, 0
    
    # Convert dz to array if it's a scalar
    if np.isscalar(dz):
        dz = np.array([dz])
    
    nz = len(dz)
    if nz == 1:
        result_map = map_map.astype(float)
    else:
        result_map = np.repeat(map_map[:, :, np.newaxis], nz, axis=2).astype(float)
    
    # Skip computation if all dz values are approximately zero
    if np.all(np.isclose(dz, 0)):
        return result_map
    
    # Create radial wavenumber grid
    k, _, _ = create_k(dx, dy, Nx, Ny)
    
    # Apply FFT
    map_fft = np.fft.fft2(map_)
    
    # Process each dz value
    for i, dz_i in enumerate(dz):
        # Set alpha to 0 for upward continuation (dz > 0)
        alpha_i = 0 if dz_i > 0 else alpha
        
        # Create filter
        H_temp = np.exp(-k * dz_i)
        H = H_temp / (1 + alpha_i * k**2 * H_temp)
        
        # Apply filter and inverse FFT
        result = np.real(np.fft.ifft2(map_fft * H))
        
        # Extract the relevant portion from the expanded map
        if nz == 1:
            result_map = result[py:py+ny, px:px+nx]
        else:
            result_map[:, :, i] = result[py:py+ny, px:px+nx]
    
    return result_map

def upward_fft(map_map: Union[MapS, MapSd, MapS3D, MapV], 
               alt: Union[float, np.ndarray], 
               expand: bool = True, 
               alpha: float = 0) -> Union[MapS, MapSd, MapS3D, MapV]:
    """
    Upward continuation of a potential field (i.e., magnetic anomaly field) map.
    Uses the Fast Fourier Transform (FFT) to convert the map to the frequency
    domain, applies an upward continuation filter, and uses the inverse FFT to
    convert the map back to the spatial domain.
    
    This function handles different map types (MapS, MapSd, MapS3D, MapV) and
    can convert between them as needed based on the input altitude(s).
    
    Args:
        map_map: Map object (MapS, MapSd, MapS3D, or MapV)
        alt: Target upward continuation altitude(s) [m]
        expand: If true, expand map temporarily to reduce edge effects
        alpha: Regularization parameter for downward continuation
    
    Returns:
        Map object of appropriate type, upward/downward continued to target altitude(s)
    """
    # Convert alt to array if it's a scalar
    if np.isscalar(alt):
        alt_array = np.array([alt])
    else:
        alt_array = np.sort(alt)
    
    N_alt = len(alt_array)
    
    # Check if multiple altitudes are provided for a scalar map
    if N_alt > 1 and isinstance(map_map, (MapS, MapSd)):
        if not isinstance(map_map, (MapS, MapS3D)):
            raise ValueError("Multiple upward continuation altitudes only allowed for MapS or MapS3D")
    
    # Convert altitude to the same type as map_map.alt
    if hasattr(map_map, 'alt') and hasattr(map_map.alt, 'dtype'):
        alt_array = alt_array.astype(map_map.alt.dtype)
    
    # Calculate step sizes in meters
    dx = dlon2de(get_step(map_map.xx), np.mean(map_map.yy))
    dy = dlat2dn(get_step(map_map.yy), np.mean(map_map.yy))
    
    # Process different map types
    if isinstance(map_map, (MapS, MapSd, MapV)) and (np.all(alt_array >= np.median(map_map.alt)) or alpha > 0):
        # Calculate altitude difference
        dz = alt_array - np.median(map_map.alt)
        
        if isinstance(map_map, (MapS, MapSd)):  # scalar map
            if N_alt > 1:  # 3D map output
                continued_map = map_up_fft(map_map.map, dx, dy, dz, expand=expand, alpha=alpha)
                # Create mask for all altitude levels
                if map_map.mask is not None:
                    mask_3d = np.repeat(map_map.mask[:, :, np.newaxis], N_alt, axis=2)
                else:
                    mask_3d = None
                
                return MapS3D(
                    info=map_map.info,
                    map=continued_map,
                    xx=map_map.xx,
                    yy=map_map.yy,
                    alt=alt_array,
                    mask=mask_3d
                )
            else:  # 2D map output
                continued_map = map_up_fft(map_map.map, dx, dy, dz, expand=expand, alpha=alpha)
                return MapS(
                    info=map_map.info,
                    map=continued_map,
                    xx=map_map.xx,
                    yy=map_map.yy,
                    alt=alt_array[0],
                    mask=map_map.mask
                )
        elif isinstance(map_map, MapV):  # vector map
            # Process each component separately
            mapX = map_up_fft(map_map.mapX, dx, dy, dz, expand=expand, alpha=alpha)
            mapY = map_up_fft(map_map.mapY, dx, dy, dz, expand=expand, alpha=alpha)
            mapZ = map_up_fft(map_map.mapZ, dx, dy, dz, expand=expand, alpha=alpha)
            
            return MapV(
                info=map_map.info,
                mapX=mapX,
                mapY=mapY,
                mapZ=mapZ,
                xx=map_map.xx,
                yy=map_map.yy,
                alt=alt_array[0] if N_alt == 1 else alt_array,
                mask=map_map.mask
            )
    
    elif isinstance(map_map, MapS3D) and (np.all(alt_array >= map_map.alt[0]) or alpha > 0):
        ny, nx, _ = map_map.map.shape
        
        # Check altitude step size consistency
        dalt = get_step(map_map.alt)
        if not np.all(np.isclose(np.mod(alt_array - map_map.alt[0], dalt), 0)):
            raise ValueError("alt must have same step size as in MapS3D alt")
        
        # Separate altitudes that need downward or upward continuation
        alt_down = alt_array[alt_array < map_map.alt[0]]
        alt_up = alt_array[alt_array > map_map.alt[-1]]
        N_down = len(alt_down)
        N_up = len(alt_up)
        
        # Process downward continuation if needed
        if N_down > 0:
            dz = alt_down - map_map.alt[0]
            map_down = map_up_fft(map_map.map[:, :, 0], dx, dy, dz, expand=expand, alpha=alpha)
            if map_down.ndim == 2:
                map_down = map_down[:, :, np.newaxis]
        else:
            map_down = np.empty((ny, nx, 0), dtype=map_map.map.dtype)
        
        # Process upward continuation if needed
        if N_up > 0:
            dz = alt_up - map_map.alt[-1]
            map_up = map_up_fft(map_map.map[:, :, -1], dx, dy, dz, expand=expand, alpha=alpha)
            if map_up.ndim == 2:
                map_up = map_up[:, :, np.newaxis]
        else:
            map_up = np.empty((ny, nx, 0), dtype=map_map.map.dtype)
        
        # Combine maps
        combined_map = np.concatenate([map_down, map_map.map, map_up], axis=2)
        
        # Combine masks
        if map_map.mask is not None:
            mask_down = np.repeat(map_map.mask[:, :, 0:1], N_down, axis=2) if N_down > 0 else np.empty((ny, nx, 0), dtype=bool)
            mask_up = np.repeat(map_map.mask[:, :, -1:], N_up, axis=2) if N_up > 0 else np.empty((ny, nx, 0), dtype=bool)
            combined_mask = np.concatenate([mask_down, map_map.mask, mask_up], axis=2)
        else:
            combined_mask = None
        
        # Combine altitudes
        combined_alt = np.concatenate([alt_down, map_map.alt, alt_up])
        
        return MapS3D(
            info=map_map.info,
            map=combined_map,
            xx=map_map.xx,
            yy=map_map.yy,
            alt=combined_alt,
            mask=combined_mask
        )
    
    else:
        print("α must be specified for downward continuation, returning original map")
        return map_map

def vector_fft(map_map: np.ndarray, dx: float, dy: float, D: float, I: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get potential field (i.e., magnetic anomaly field) map vector components
    using declination and inclination.
    
    Args:
        map_map: ny x nx 2D gridded map data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
        D: map declination (Earth core field) [rad]
        I: map inclination (Earth core field) [rad]
    
    Returns:
        Bx, By, Bz: map vector components
    """
    ny, nx = map_map.shape
    k, u, v = create_k(dx, dy, nx, ny)
    
    l = np.cos(I) * np.cos(D)
    m = np.cos(I) * np.sin(D)
    n = np.sin(I)
    
    F = np.fft.fft2(map_map)
    
    # Create filters for each component
    Hx = 1j * u / (1j * (u * l + m * v) + n * k)
    Hy = 1j * v / (1j * (u * l + m * v) + n * k)
    Hz = k / (1j * (u * l + m * v) + n * k)
    
    # Handle singularity at DC component
    Hx[0, 0] = 1
    Hy[0, 0] = 1
    Hz[0, 0] = 1
    
    # Apply filters and inverse FFT
    Bx = np.real(np.fft.ifft2(Hx * F))
    By = np.real(np.fft.ifft2(Hy * F))
    Bz = np.real(np.fft.ifft2(Hz * F))
    
    return Bx, By, Bz

def downward_L(map_map: Union[np.ndarray, MapS, MapSd, MapS3D], 
               alt: Union[float, np.ndarray], 
               alphas: List[float], 
               expand: bool = True,
               map_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Downward continuation using a sequence of regularization parameters to create
    a characteristic L-curve. The optimal regularization parameter is at a local
    minimum on the L-curve, which is a local maximum of curvature.
    
    Args:
        map_map: Map data or map object
        alt: Target downward continuation altitude [m]
        alphas: Sequence of regularization parameters
        expand: If true, expand map temporarily to reduce edge effects
        map_mask: Optional mask for valid (not filled-in) map data
    
    Returns:
        norms: L-infinity norm of difference between sequential D.C. solutions
    """
    # Handle different input types
    if isinstance(map_map, (MapS, MapSd, MapS3D)):
        # Extract map data and parameters from map object
        dx = dlon2de(get_step(map_map.xx), np.mean(map_map.yy))
        dy = dlat2dn(get_step(map_map.yy), np.mean(map_map.yy))
        
        if isinstance(map_map, MapSd):
            alt_ = np.median(map_map.alt[map_map.mask]) if map_map.mask is not None else np.median(map_map.alt)
        else:
            alt_ = map_map.alt[0] if isinstance(map_map.alt, np.ndarray) else map_map.alt
        
        dz = alt - alt_
        
        if isinstance(map_map, MapS3D):
            print("3D map provided, using map at lowest altitude")
            map_data = map_map.map[:, :, 0]
            mask_data = map_map.mask[:, :, 0] if map_map.mask is not None else None
        else:
            map_data = map_map.map
            mask_data = map_map.mask
    else:
        # Assume map_map is a numpy array and other parameters are provided directly
        map_data = map_map
        mask_data = map_mask
        dx = dy = 1.0  # Default values if not provided
        dz = 0.0  # Default value if not provided
    
    ny, nx = map_data.shape
    norms = np.zeros(len(alphas) - 1, dtype=float)
    
    if expand:
        pad = min(int(np.ceil(10 * abs(dz) / min(dx, dy))), 5000)  # set pad > 10*dz
        map_expanded, px, py = map_expand(map_data, pad)
        Ny, Nx = map_expanded.shape
    else:
        map_expanded = map_data
        Ny, Nx, px, py = ny, nx, 0, 0
    
    # Create radial wavenumber grid
    k, _, _ = create_k(dx, dy, Nx, Ny)
    
    # Apply FFT
    map_fft = np.fft.fft2(map_expanded)
    
    # Calculate filter for first alpha
    H_temp = np.exp(-k * dz)
    H = H_temp / (1 + alphas[0] * k**2 * H_temp)
    
    # Apply filter and inverse FFT
    map_old = np.real(np.fft.ifft2(map_fft * H))
    map_old = map_old[py:py+ny, px:px+nx]
    
    # Apply mask if provided
    if mask_data is not None:
        map_old = map_old[mask_data]
    
    # Process remaining alphas
    for i in range(1, len(alphas)):
        H = H_temp / (1 + alphas[i] * k**2 * H_temp)
        map_new = np.real(np.fft.ifft2(map_fft * H))
        map_new = map_new[py:py+ny, px:px+nx]
        
        # Apply mask if provided
        if mask_data is not None:
            map_new = map_new[mask_data]
        
        # Calculate L-infinity norm of difference
        norms[i-1] = np.linalg.norm(map_new - map_old, np.inf)
        map_old = map_new
    
    return norms

def psd_calc(map_map: Union[np.ndarray, MapS, MapSd, MapS3D], 
          dx: Optional[float] = None, 
          dy: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Power spectral density of a potential field (i.e., magnetic anomaly field) map.
    Uses the Fast Fourier Transform to determine the spectral energy distribution
    across the radial wavenumbers (spatial frequencies) in the Fourier transform.
    
    Args:
        map_map: Map data or map object
        dx: x-direction map step size [m], required if map_map is ndarray
        dy: y-direction map step size [m], required if map_map is ndarray
    
    Returns:
        map_psd: ny x nx power spectral density of 2D gridded map data
        kx: ny x nx x-direction radial wavenumber
        ky: ny x nx y-direction radial wavenumber
    """
    # Handle different input types
    if isinstance(map_map, (MapS, MapSd, MapS3D)):
        # Extract map data and parameters from map object
        dx = dlon2de(get_step(map_map.xx), np.mean(map_map.yy))
        dy = dlat2dn(get_step(map_map.yy), np.mean(map_map.yy))
        
        if isinstance(map_map, MapS3D):
            print("3D map provided, using map at lowest altitude")
            if map_map.mask is not None:
                map_data = map_map.map[:, :, 0] * map_map.mask[:, :, 0]
            else:
                map_data = map_map.map[:, :, 0]
        else:
            if map_map.mask is not None:
                map_data = map_map.map * map_map.mask
            else:
                map_data = map_map.map
    else:
        # Assume map_map is a numpy array
        if dx is None or dy is None:
            raise ValueError("dx and dy must be provided when map_map is a numpy array")
        map_data = map_map
    
    ny, nx = map_data.shape
    _, kx, ky = create_k(dx, dy, nx, ny)
    
    # Calculate power spectral density
    map_psd = np.abs(np.fft.fft2(map_data))**2
    
    return map_psd, kx, ky

def map_fft(map_data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the 2D FFT of map data.
    
    Args:
        map_data: ny x nx 2D gridded map data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
    
    Returns:
        fft_result: ny x nx complex FFT result
    """
    return np.fft.fft2(map_data)

def map_ifft(fft_data: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the inverse 2D FFT of frequency domain data.
    
    Args:
        fft_data: ny x nx complex frequency domain data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
    
    Returns:
        map_data: ny x nx real spatial domain data
    """
    return np.real(np.fft.ifft2(fft_data))

def map_filt_fft(map_data: np.ndarray, dx: float, dy: float,
                filter_type: str = "lowpass", cutoff_wavelength: float = None) -> np.ndarray:
    """
    Apply a frequency domain filter to map data.
    
    Args:
        map_data: ny x nx 2D gridded map data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
        filter_type: Type of filter to apply ("lowpass", "highpass", "bandpass")
        cutoff_wavelength: Wavelength cutoff for the filter [m]
                          For bandpass, provide a tuple of (low_cutoff, high_cutoff)
    
    Returns:
        filtered_map: ny x nx filtered map data
    """
    ny, nx = map_data.shape
    k, kx, ky = create_k(dx, dy, nx, ny)
    
    # Apply FFT
    map_fft_data = np.fft.fft2(map_data)
    
    # Create filter based on filter_type
    if cutoff_wavelength is None:
        cutoff_wavelength = 5 * max(dx, dy)  # Default cutoff
    
    if filter_type.lower() == "lowpass":
        # Low-pass filter: keep low frequencies (long wavelengths)
        cutoff_k = 2 * np.pi / cutoff_wavelength
        H = np.exp(-k**2 / (2 * cutoff_k**2))
    elif filter_type.lower() == "highpass":
        # High-pass filter: keep high frequencies (short wavelengths)
        cutoff_k = 2 * np.pi / cutoff_wavelength
        H = 1 - np.exp(-k**2 / (2 * cutoff_k**2))
    elif filter_type.lower() == "bandpass":
        # Band-pass filter: keep frequencies between low and high cutoffs
        if not isinstance(cutoff_wavelength, (list, tuple)) or len(cutoff_wavelength) != 2:
            raise ValueError("cutoff_wavelength must be a tuple of (low_cutoff, high_cutoff) for bandpass filter")
        low_cutoff, high_cutoff = cutoff_wavelength
        low_k = 2 * np.pi / high_cutoff  # Note: longer wavelength = lower frequency
        high_k = 2 * np.pi / low_cutoff   # Note: shorter wavelength = higher frequency
        H = np.exp(-k**2 / (2 * high_k**2)) - np.exp(-k**2 / (2 * low_k**2))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter and inverse FFT
    filtered_fft = map_fft_data * H
    filtered_map = np.real(np.fft.ifft2(filtered_fft))
    
    return filtered_map

def map_grad_fft(map_data: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gradient of a map using FFT.
    
    Args:
        map_data: ny x nx 2D gridded map data
        dx: x-direction map step size [m]
        dy: y-direction map step size [m]
    
    Returns:
        grad_x: ny x nx x-component of gradient
        grad_y: ny x nx y-component of gradient
    """
    ny, nx = map_data.shape
    _, kx, ky = create_k(dx, dy, nx, ny)
    
    # Apply FFT
    map_fft_data = np.fft.fft2(map_data)
    
    # Calculate gradient in frequency domain
    # Multiplication by i*kx gives the x-derivative
    # Multiplication by i*ky gives the y-derivative
    grad_x_fft = 1j * kx * map_fft_data
    grad_y_fft = 1j * ky * map_fft_data
    
    # Apply inverse FFT
    grad_x = np.real(np.fft.ifft2(grad_x_fft))
    grad_y = np.real(np.fft.ifft2(grad_y_fft))
    
    return grad_x, grad_y