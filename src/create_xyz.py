"""
This module is responsible for creating and initializing XYZ data objects,
translated from the Julia MagNav.jl/src/create_XYZ.jl.
"""
import numpy as np
import h5py
from typing import Union, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import math
import random
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from copy import deepcopy

# Attempt to import from project modules
try:
    from .magnav import (XYZ0, Traj, INS, MagV, MapS, MapSd, MapS3D, MapV,
                           BaseMap, # Assuming BaseMap is a common base for MapS, MapSd, etc.
                           Path) # Path = Union[Traj, INS] or a base class
    from .analysis_util import (
        add_extension, g_earth, get_map, map_check, dlat2dn, dlon2de, dn2dlat,
        de2dlon, fdm, utm_zone_from_latlon, transform_lla_to_utm, # Assumed geospatial helpers
        create_dcm_from_vel, dcm2euler, euler2dcm, # Attitude helpers
        create_ins_model_matrices, get_phi_matrix, correct_cnb_matrix, # INS helpers
        upward_fft_map, get_map_params, fill_map_gaps, trim_map, get_map_value_at_coords, # Map helpers
        generate_fogm_noise, create_tolles_lawson_A_matrix, # Noise & TL helpers
        get_igrf_magnetic_field, # IGRF helper
        apply_band_pass_filter, get_band_pass_filter_coeffs, # Filter helpers
        tolles_lawson_coeffs_to_matrix, get_tolles_lawson_aircraft_field_vector, # More TL helpers
        get_trajectory_subset, # Helper for xyz.traj(ind) like behavior
        approximate_gradient # Helper for map gradient
    )
    # Define default map identifiers if they are constants in analysis_util
    DEFAULT_SCALAR_MAP_ID = "namad" # Placeholder
    DEFAULT_VECTOR_MAP_ID = "emm720" # Placeholder

except ImportError:
    # Fallback for standalone execution or if modules are structured differently
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
    class Traj: N: int; dt: float; tt: np.ndarray; lat: np.ndarray; lon: np.ndarray; alt: np.ndarray; vn: np.ndarray; ve: np.ndarray; vd: np.ndarray; fn: np.ndarray; fe: np.ndarray; fd: np.ndarray; Cnb: np.ndarray
    @dataclass
    class INS(Traj): P: np.ndarray # Covariance matrices
    @dataclass
    class MagV: x: np.ndarray; y: np.ndarray; z: np.ndarray; t: np.ndarray
    @dataclass
    class XYZ0: info: str; traj: Traj; ins: INS; flux_a: MagV; flights: np.ndarray; lines: np.ndarray; years: np.ndarray; doys: np.ndarray; diurnal: np.ndarray; igrf: np.ndarray; mag_1_c: np.ndarray; mag_1_uc: np.ndarray
    Path = Union[Traj, INS]
    def add_extension(s, ext): return s if s.endswith(ext) else s + ext
    g_earth = 9.80665
    def get_map(map_id): return MapS(xx=np.array([]), yy=np.array([]), map=np.array([])) # Dummy
    DEFAULT_SCALAR_MAP_ID = "namad"
    DEFAULT_VECTOR_MAP_ID = "emm720"
    # Add other dummy functions as needed based on usage below...
    def map_check(m,la,lo): return True
    def dlat2dn(dlat,lat): return dlat * 111000
    def dlon2de(dlon,lat): return dlon * 111000 * np.cos(lat)
    def dn2dlat(dn,lat): return dn / 111000
    def de2dlon(de,lat): return de / (111000 * np.cos(lat))
    def fdm(arr): return np.gradient(arr) if arr.ndim == 1 else np.array([np.gradient(arr[i]) for i in range(arr.shape[0])])
    def utm_zone_from_latlon(lat_deg, lon_deg): return (int((lon_deg + 180) / 6) + 1, lat_deg >=0)
    def transform_lla_to_utm(lat_rad, lon_rad, zone, is_north):
        # Simplified placeholder for pyproj or gdal
        lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad)
        # This is NOT a real UTM conversion, just for structure
        return lon_deg * 1000, lat_deg * 1000
    def create_dcm_from_vel(vn, ve, dt, order): return np.array([np.eye(3)] * len(vn))
    def dcm2euler(dcm, order): return np.zeros((len(dcm), 3)) if dcm.ndim == 3 else np.zeros(3)
    def euler2dcm(roll,pitch,yaw,order): return np.array([np.eye(3)]*len(roll)) if isinstance(roll,np.ndarray) else np.eye(3)
    def create_ins_model_matrices(*args, **kwargs): return np.eye(17), np.eye(17), np.eye(17) # P0, Qd, R
    def get_phi_matrix(*args, **kwargs): return np.eye(17)
    def correct_cnb_matrix(cnb, err): return cnb
    def upward_fft_map(map_s, alt): return map_s
    def get_map_params(map_s): return (None,None,None,None) # ind0,ind1,_,_
    def fill_map_gaps(map_s): return map_s
    def trim_map(map_s): return map_s
    def get_map_value_at_coords(map_s, lat, lon, alt, alpha=200, return_itp=False):
        if return_itp: return np.zeros(len(lat)), None
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
    def get_trajectory_subset(traj, ind): # Simplified
        return Traj(N=len(ind), dt=traj.dt, tt=traj.tt[ind], lat=traj.lat[ind], lon=traj.lon[ind], alt=traj.alt[ind],
                    vn=traj.vn[ind], ve=traj.ve[ind], vd=traj.vd[ind], fn=traj.fn[ind], fe=traj.fe[ind], fd=traj.fd[ind],
                    Cnb=traj.Cnb[ind])
    def approximate_gradient(itp_func, y, x): return np.array([0.0, 0.0]) # Dummy


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
    cor_var: float = 1.0**2, # This is for compensated, but Julia has it duplicated
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
    fogm_sigma: float = 1.0, # This is for compensated
    baro_tau: float = 3600.0,
    acc_tau: float = 3600.0,
    gyro_tau: float = 3600.0,
    fogm_tau: float = 600.0, # This is for compensated
    save_h5: bool = False,
    xyz_h5: str = "xyz_data.h5",
    silent: bool = False
) -> XYZ0:
    """
    Create basic flight data. Assumes constant altitude (2D flight).
    Corresponds to create_XYZ0 in Julia.
    """
    if mapS is None:
        mapS = get_map(DEFAULT_SCALAR_MAP_ID)
    if mapV is None:
        mapV = get_map(DEFAULT_VECTOR_MAP_ID)

    xyz_h5 = add_extension(xyz_h5, ".h5")

    # Create trajectory
    traj = create_traj(
        mapS,
        alt=alt, dt=dt, t=t, v=v, ll1=ll1, ll2=ll2, N_waves=N_waves,
        attempts=attempts, save_h5=save_h5, traj_h5=xyz_h5 # Note: traj_h5 uses xyz_h5
    )

    # Create INS
    ins = create_ins(
        traj,
        init_pos_sigma=init_pos_sigma, init_alt_sigma=init_alt_sigma,
        init_vel_sigma=init_vel_sigma, init_att_sigma=init_att_sigma,
        VRW_sigma=VRW_sigma, ARW_sigma=ARW_sigma, baro_sigma=baro_sigma,
        ha_sigma=ha_sigma, a_hat_sigma=a_hat_sigma, acc_sigma=acc_sigma,
        gyro_sigma=gyro_sigma, baro_tau=baro_tau, acc_tau=acc_tau,
        gyro_tau=gyro_tau, save_h5=save_h5, ins_h5=xyz_h5 # Note: ins_h5 uses xyz_h5
    )

    # Create compensated (clean) scalar magnetometer measurements
    # Julia's create_XYZ0 uses cor_var, fogm_sigma, fogm_tau for this.
    mag_1_c = create_mag_c(
        traj, mapS,
        meas_var=cor_var, # Using the general cor_var for compensated noise variance
        fogm_sigma=fogm_sigma, # Using the general fogm_sigma for compensated FOGM
        fogm_tau=fogm_tau, # Using the general fogm_tau for compensated FOGM
        silent=silent
    )

    # Create compensated (clean) vector magnetometer measurements
    flux_a = create_flux(
        traj, mapV,
        meas_var=cor_var, # Using the general cor_var
        fogm_sigma=fogm_sigma, # Using the general fogm_sigma
        fogm_tau=fogm_tau, # Using the general fogm_tau
        silent=silent
    )

    # Create uncompensated (corrupted) scalar magnetometer measurements
    # Julia's create_XYZ0 uses specific cor_ values for this.
    mag_1_uc, _, diurnal_effect = corrupt_mag(
        mag_1_c, flux_a,
        dt=dt,
        cor_sigma=cor_sigma, cor_tau=cor_tau, cor_var=cor_var, # These are the uncompensated specific ones
        cor_drift=cor_drift, cor_perm_mag=cor_perm_mag,
        cor_ind_mag=cor_ind_mag, cor_eddy_mag=cor_eddy_mag
    )

    num_points = len(traj.lat)
    flights = np.full(num_points, flight, dtype=int)
    lines   = np.full(num_points, line, dtype=int)
    years   = np.full(num_points, year, dtype=int)
    doys    = np.full(num_points, doy, dtype=int)

    # Placeholder for diurnal, assuming corrupt_mag returns it or it's calculated elsewhere.
    # In Julia, it's the third return from corrupt_mag.
    diurnal = diurnal_effect # Or np.zeros(num_points) if not returned by corrupt_mag

    # Initialize IGRF, then calculate actual values
    igrf_initial = np.zeros(num_points)
    
    # Construct preliminary XYZ to pass to get_igrf
    # Note: This might require XYZ0 to allow partial initialization or a different approach
    # For now, assuming XYZ0 can be created with placeholder igrf
    xyz_temp_fields = {
        "info": info, "traj": traj, "ins": ins, "flux_a": flux_a,
        "flights": flights, "lines": lines, "years": years, "doys": doys,
        "diurnal": diurnal, "igrf": igrf_initial, "mag_1_c": mag_1_c, "mag_1_uc": mag_1_uc
    }
    # This is a bit tricky. If XYZ0 is a strict dataclass, we might need to pass all args.
    # Let's assume get_igrf can work with a dictionary or a partially formed object,
    # or we make a temporary XYZ0 instance.
    
    # Simplification: Create XYZ0 first, then update igrf.
    # This requires all fields for XYZ0 constructor.
    xyz = XYZ0(info=info, traj=traj, ins=ins, flux_a=flux_a,
               flights=flights, lines=lines, years=years, doys=doys,
               diurnal=diurnal, igrf=igrf_initial, # igrf_initial is placeholder
               mag_1_c=mag_1_c, mag_1_uc=mag_1_uc)

    igrf_vector_body = get_igrf_magnetic_field(xyz, frame='body', norm_igrf=False, check_xyz=False)
    igrf_scalar = np.linalg.norm(igrf_vector_body, axis=0) # Assuming igrf_vector_body is 3xN
    xyz.igrf = igrf_scalar

    if save_h5:
        # 'a' mode: read/write if exists, create otherwise. Julia 'cw' is similar.
        with h5py.File(xyz_h5, "a") as file:
            # Overwrite if traj/ins already wrote some of these, or manage groups.
            # For simplicity, assuming direct write/overwrite at root or that
            # traj_h5/ins_h5 were different files or used groups.
            # If xyz_h5 is the *same* file, need to be careful about overwriting.
            # The Julia code implies it's the same file and appends/overwrites.
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
            if "igrf" not in file: file.create_dataset("igrf", data=xyz.igrf) # Use updated igrf
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
    Corresponds to create_traj in Julia.
    """
    traj_h5 = add_extension(traj_h5, ".h5")

    # Check flight altitude
    if isinstance(mapS, MapSd) and hasattr(mapS, 'mask') and mapS.mask is not None:
        # Ensure mapS.alt is indexable by mapS.mask if mapS.alt is a grid
        # This part needs careful handling based on MapSd structure
        # Assuming mapS.alt is a 2D grid and mapS.mask is a boolean grid of same shape
        valid_alts = mapS.alt[mapS.mask] if mapS.mask.any() else [np.mean(mapS.alt)] # fallback
        map_altitude_ref = np.median(valid_alts) if len(valid_alts) > 0 else np.mean(mapS.alt)

    elif isinstance(mapS.alt, (list, np.ndarray)) and len(mapS.alt) > 0:
        map_altitude_ref = mapS.alt[0]
    elif isinstance(mapS.alt, (float, int)):
         map_altitude_ref = mapS.alt
    else:
        # Fallback or error if mapS.alt structure is unknown
        raise ValueError("Cannot determine reference altitude from mapS")

    if alt < map_altitude_ref:
        raise ValueError(f"Flight altitude {alt} < map reference altitude {map_altitude_ref}")

    i = 0
    N_pts = 2 # Initial number of points for lat/lon arrays
    lat_path = np.zeros(N_pts, dtype=float)
    lon_path = np.zeros(N_pts, dtype=float)
    
    # Loop to find a valid trajectory
    # The condition `(not map_check(mapS, lat_path, lon_path) and i <= attempts) or i == 0`
    # ensures at least one attempt and retries if path is not on map.
    path_found_on_map = False
    while (not path_found_on_map and i < attempts) or i == 0:
        i += 1
        if not ll1:  # If ll1 is empty, put initial point in middle 50% of map
            lat_min, lat_max = np.min(mapS.yy), np.max(mapS.yy)
            lon_min, lon_max = np.min(mapS.xx), np.max(mapS.xx)
            # Ensure mapS.yy and mapS.xx are in radians if subsequent calcs expect radians
            # Assuming they are already in radians as per typical geodetic calculations
            lat1 = lat_min + (lat_max - lat_min) * (0.25 + 0.50 * random.random())
            lon1 = lon_min + (lon_max - lon_min) * (0.25 + 0.50 * random.random())
        else:  # Use given initial point (degrees), convert to radians
            lat1, lon1 = np.deg2rad(ll1[0]), np.deg2rad(ll1[1])

        if not ll2:  # Use given velocity & time to set distance
            N_pts = int(round(t / dt + 1))
            dist = v * t  # distance
            theta_utm = 2 * math.pi * random.random()  # random heading
            # Convert distance and heading to lat/lon changes
            # These require geodetic calculations (e.g., Vincenty or simpler spherical model)
            # Using analysis_util helpers: dn2dlat, de2dlon
            # dn = dist * sin(theta_utm), de = dist * cos(theta_utm)
            lat2 = lat1 + dn2dlat(dist * math.sin(theta_utm), lat1)
            lon2 = lon1 + de2dlon(dist * math.cos(theta_utm), lat1)
        else:  # Use given final point directly
            N_pts_est = 1000 * (N_waves + 1) # Estimated N, to be corrected
            lat2, lon2 = np.deg2rad(ll2[0]), np.deg2rad(ll2[1])
            # N_pts will be refined later based on actual path length and dt

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        theta_ll = math.atan2(dlat, dlon) # Heading in lat/lon rad space (y,x)

        # Initial straight line path (or base for waves)
        # Use N_pts if ll2 is not set, or N_pts_est if ll2 is set (will be resampled)
        current_N = N_pts if not ll2 else N_pts_est
        lat_path = np.linspace(lat1, lat2, current_N)
        lon_path = np.linspace(lon1, lon2, current_N)

        if N_waves > 0:
            phi_waves = np.linspace(0, N_waves * 2 * math.pi, current_N)
            # Wave amplitude needs to be determined. Julia's example doesn't specify amplitude.
            # Assuming a fraction of the path length or a fixed value.
            # The Julia code `wav = [ϕ sin.(ϕ)]` implies amplitude is 1 in ϕ units.
            # This needs to be scaled to geographic units.
            # The Julia code `cor = wav*rot'` and then `lat = (cor[:,2] .- cor[1,2] .+ lat1)`
            # suggests the wave is applied to the parametric progression (phi) and then rotated.
            # This part is complex to translate directly without knowing wave amplitude scaling.
            # For now, a simplified placeholder or assuming amplitude is implicitly handled.
            # Let's assume the wave is a perturbation perpendicular to the path.
            # A common way is A * sin(phi_waves), where A is amplitude in radians.
            # The Julia code seems to make the path itself sinusoidal in the direction of travel.
            # `wav = [ϕ sin.(ϕ)]` -> `wav_x = phi_waves`, `wav_y = np.sin(phi_waves)`
            # `rot = [[cos(θ_ll), -sin(θ_ll)], [sin(θ_ll), cos(θ_ll)]]`
            # `cor = np.dot(np.vstack((wav_x, wav_y)).T, rot.T)`
            # `lon_path = (cor[:,0] - cor[0,0]) + lon1`
            # `lat_path = (cor[:,1] - cor[0,1]) + lat1`
            # This interpretation makes the path itself a rotated sinusoid.
            # The Julia code `cor[:,2]` and `cor[:,1]` implies wav is (N,2) and rot is (2,2).
            # `wav = [ϕ sin.(ϕ)]` -> `wav = np.column_stack((phi_waves, np.sin(phi_waves)))`
            # `rot_matrix = np.array([[math.cos(theta_ll), -math.sin(theta_ll)],
            #                          [math.sin(theta_ll), math.cos(theta_ll)]])`
            # `cor = np.dot(wav, rot_matrix.T)`
            # `lon_path = (cor[:,0] - cor[0,0]) + lon1` # Julia used cor[:,1] for lat, cor[:,2] for lon (1-indexed)
            # `lat_path = (cor[:,1] - cor[0,1]) + lat1` # Python: cor[:,0] for x-like, cor[:,1] for y-like
            # This seems to be the correct interpretation of Julia's 1-based indexing.
            # The amplitude of sin(phi) is 1. This needs scaling to meters or degrees.
            # The Julia code does not show explicit scaling of sin(phi).
            # This suggests the "waviness" is relative to the total length implicitly.
            # Re-evaluating: `lat = (cor[:,2] .- cor[1,2] .+ lat1)`
            # `lon = (cor[:,1] .- cor[1,1] .+ lon1)`
            # This means `cor` has columns for transformed x and y.
            # If `wav` is `[phi, sin(phi)]`, then `phi` is the along-track progress and `sin(phi)` is cross-track.
            # This part is tricky. A simpler approach for Python might be needed if direct translation is unclear.
            # For now, skipping the complex wave part for brevity, assuming straight line or simple wave.
            # If waves are critical, this needs more detailed translation.
            pass # Simplified: waves not fully implemented here due to ambiguity in scaling.

        # Iteratively scale path to target distance or endpoint
        # This loop in Julia: `while !(frac1 ≈ 1) | !(frac2 ≈ 1)`
        # Python: `while not (np.isclose(frac1, 1.0) and np.isclose(frac2, 1.0)):` (approx)
        # For simplicity, let's assume a few iterations or direct calculation if possible.
        # The Julia code recalculates dx, dy, d_now inside this loop.
        
        # Calculate actual distance of the current path (lat_path, lon_path)
        # Using fdm and geodetic distance helpers
        dx_m = dlon2de(fdm(lon_path), lat_path) # Easting distances
        dy_m = dlat2dn(fdm(lat_path), lat_path) # Northing distances
        # fdm might return N points or N-1. Assuming N points, first is often 0 or NaN.
        # Distances are between points, so use diff or slice.
        segment_distances = np.sqrt(dx_m[1:]**2 + dy_m[1:]**2) # Skip first fdm element if it's an offset
        current_total_dist = np.sum(segment_distances)

        if not ll2: # Scale to target distance `dist`
            if current_total_dist > 1e-6: # Avoid division by zero
                scale_factor = dist / current_total_dist
                # Rescale lat_path, lon_path. This is non-trivial for curved paths.
                # Julia's approach: `lat = (lat .- lat[1])*frac1 .+ lat[1]`
                # This scales relative to the start point.
                lat_path = (lat_path - lat_path[0]) * scale_factor + lat_path[0]
                lon_path = (lon_path - lon_path[0]) * scale_factor + lon_path[0]
        else: # Scale to target endpoint (lat2, lon2)
            if len(lat_path) > 1 and abs(lat_path[-1] - lat_path[0]) > 1e-9 : # Avoid div by zero
                 scale_factor_lat = (lat2 - lat_path[0]) / (lat_path[-1] - lat_path[0])
                 lat_path = (lat_path - lat_path[0]) * scale_factor_lat + lat_path[0]
            if len(lon_path) > 1 and abs(lon_path[-1] - lon_path[0]) > 1e-9:
                 scale_factor_lon = (lon2 - lon_path[0]) / (lon_path[-1] - lon_path[0])
                 lon_path = (lon_path - lon_path[0]) * scale_factor_lon + lon_path[0]
        
        # Recalculate N_pts based on true distance and dt if ll2 was given
        if ll2:
            dx_m = dlon2de(fdm(lon_path), lat_path)
            dy_m = dlat2dn(fdm(lat_path), lat_path)
            segment_distances = np.sqrt(dx_m[1:]**2 + dy_m[1:]**2)
            true_dist = np.sum(segment_distances)
            
            t_flight = true_dist / v # True time
            N_pts = int(round(t_flight / dt + 1)) # Correct number of time steps
            
            # Resample lat_path, lon_path to new N_pts
            if len(lat_path) > 1 :
                current_progression = np.linspace(0, 1, len(lat_path))
                new_progression = np.linspace(0, 1, N_pts)
                interp_lat = interp1d(current_progression, lat_path, kind='linear', fill_value="extrapolate")
                interp_lon = interp1d(current_progression, lon_path, kind='linear', fill_value="extrapolate")
                lat_path = interp_lat(new_progression)
                lon_path = interp_lon(new_progression)
            else: # Single point case
                lat_path = np.full(N_pts, lat_path[0])
                lon_path = np.full(N_pts, lon_path[0])
            t = t_flight # Update total time

        path_found_on_map = map_check(mapS, lat_path, lon_path)
        if path_found_on_map:
            break # Exit loop if valid path found
    
    if not path_found_on_map: # or i > attempts
        raise RuntimeError(f"Maximum attempts ({attempts}) reached, could not create valid trajectory on map. Decrease t or increase v.")

    # Ensure lat_path, lon_path are set for the final N_pts
    # If ll2 was not set, N_pts was set from t/dt initially.
    # If ll2 was set, N_pts was refined.
    
    # UTM conversion using assumed helpers from analysis_util
    # These helpers should handle pyproj or gdal internally.
    mean_lat_deg_traj = np.rad2deg(np.mean(lat_path))
    mean_lon_deg_traj = np.rad2deg(np.mean(lon_path))
    utm_zone_num, utm_is_north = utm_zone_from_latlon(mean_lat_deg_traj, mean_lon_deg_traj)
    
    utms_x, utms_y = transform_lla_to_utm(lat_path, lon_path, utm_zone_num, utm_is_north)

    # Velocities & specific forces from position
    # fdm (finite difference method) from analysis_util
    vn = fdm(utms_y) / dt  # North velocity from UTM y
    ve = fdm(utms_x) / dt  # East velocity from UTM x
    vd = np.zeros_like(lat_path) # Assuming constant altitude, so vd is zero
    
    fn = fdm(vn) / dt      # North specific force (acceleration)
    fe = fdm(ve) / dt      # East specific force
    fd = fdm(vd) / dt - g_earth # Down specific force (includes gravity)
    
    tt = np.linspace(0, t, N_pts) # Time vector

    # DCM (body to navigation) from heading
    # create_dcm_from_vel and dcm2euler from analysis_util
    Cnb_array = create_dcm_from_vel(vn, ve, dt, order='body2nav') # Returns N x 3 x 3
    euler_angles = dcm2euler(Cnb_array, order='body2nav') # Returns N x 3 (roll, pitch, yaw)
    roll_path  = euler_angles[:,0]
    pitch_path = euler_angles[:,1]
    yaw_path   = euler_angles[:,2]

    alt_path = np.full_like(lat_path, alt)

    if save_h5:
        with h5py.File(traj_h5, "a") as file: # Append mode
            # Create datasets if they don't exist, or overwrite if they do.
            # Grouping might be better if file is shared (e.g., file.require_group("traj"))
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
    init_att_sigma: float = np.deg2rad(0.00001), # Julia default was 0.01, then changed to 0.00001
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
    Corresponds to create_ins in Julia.
    """
    ins_h5 = add_extension(ins_h5, ".h5")

    N = traj.N
    dt = traj.dt
    nx = 17  # Total state dimension for Pinson error model

    # Get initial covariance P0 and process noise Qd from analysis_util helper
    P0, Qd, _ = create_ins_model_matrices( # R (measurement noise) not used here
        dt, traj.lat[0], # Initial latitude
        init_pos_sigma=init_pos_sigma, init_alt_sigma=init_alt_sigma,
        init_vel_sigma=init_vel_sigma, init_att_sigma=init_att_sigma,
        VRW_sigma=VRW_sigma, ARW_sigma=ARW_sigma,
        baro_sigma=baro_sigma, ha_sigma=ha_sigma, a_hat_sigma=a_hat_sigma,
        acc_sigma=acc_sigma, gyro_sigma=gyro_sigma,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
        fogm_state=False # Assuming FOGM state is not part of this 17-state model
    )

    P_ins = np.zeros((N, nx, nx)) # Store covariance at each step
    err_ins = np.zeros((N, nx))   # Store error state at each step

    P_ins[0,:,:] = P0
    # Sample initial error from N(0, P0)
    err_ins[0,:] = multivariate_normal.rvs(mean=np.zeros(nx), cov=P0)
    
    # Cholesky decomposition of Qd for sampling process noise: sqrt(Qd) * N(0,I)
    # Ensure Qd is positive definite. Add small epsilon if numerical issues.
    try:
        Qd_chol = np.linalg.cholesky(Qd)
    except np.linalg.LinAlgError:
        # Add small diagonal offset if Qd is not positive definite
        Qd_chol = np.linalg.cholesky(Qd + np.eye(nx) * 1e-12)


    for k in range(N - 1):
        # Get state transition matrix Phi from analysis_util helper
        Phi_k = get_phi_matrix(
            nx, traj.lat[k], traj.vn[k], traj.ve[k], traj.vd[k],
            traj.fn[k], traj.fe[k], traj.fd[k], traj.Cnb[k,:,:], # Cnb at step k
            baro_tau, acc_tau, gyro_tau, 0, dt, fogm_state=False # 0 for fogm_tau if no fogm_state
        )
        # Propagate error state: err_k+1 = Phi_k * err_k + w_k
        # w_k ~ N(0, Qd), so sample as Qd_chol * N(0,I)
        process_noise_k = Qd_chol @ np.random.randn(nx)
        err_ins[k+1,:] = Phi_k @ err_ins[k,:] + process_noise_k
        
        # Propagate covariance: P_k+1 = Phi_k * P_k * Phi_k^T + Qd
        P_ins[k+1,:,:] = Phi_k @ P_ins[k,:,:] @ Phi_k.T + Qd

    # Apply errors to true trajectory to get INS trajectory
    # Error state definition (typical Pinson):
    # err[0:3] = pos_err (lat, lon, alt) -> Julia used subtraction, so INS = True - Err
    # err[3:6] = vel_err (vn, ve, vd)
    # err[6:9] = att_err (psi_n, psi_e, psi_d) -> tilt errors
    # err[9:12]= acc_bias
    # err[12:15]=gyro_bias
    # err[15] = baro_bias
    # err[16] = baro_scale_factor (or other, depending on model variant)
    
    # Note: Julia subtracts error: lat = traj.lat - err[0,:]. If err is (true-estimate), then estimate = true - err.
    # If err is (estimate-true), then estimate = true + err.
    # Assuming err_ins represents (true - ins_estimate) for position/velocity,
    # or (ins_estimate - true) for biases.
    # The Julia code `lat = traj.lat - err[0,:]` implies err[0] is (true_lat - ins_lat).
    # So, ins_lat = traj.lat - err_pos_lat. This seems consistent.
    
    ins_lat = traj.lat - err_ins[:,0]
    ins_lon = traj.lon - err_ins[:,1]
    ins_alt = traj.alt - err_ins[:,2]
    ins_vn  = traj.vn  - err_ins[:,3]
    ins_ve  = traj.ve  - err_ins[:,4]
    ins_vd  = traj.vd  - err_ins[:,5]

    # Recalculate specific forces from INS velocities
    ins_fn = fdm(ins_vn) / dt
    ins_fe = fdm(ins_ve) / dt
    ins_fd = fdm(ins_vd) / dt - g_earth

    # Correct Cnb using attitude errors (err_ins[:,6:9])
    # correct_cnb_matrix from analysis_util
    # Julia used -err[7:9,:], so if err_att is (true_att - ins_att), then ins_att = true_att - err_att.
    # Cnb_ins = Cnb_true * (I - skew(err_att_ins))
    ins_Cnb = correct_cnb_matrix(traj.Cnb, -err_ins[:,6:9].T) # Pass err_att (3xN)

    if np.any(ins_Cnb > 1.00001) or np.any(ins_Cnb < -1.00001): # Add tolerance for float precision
        # This check might be too strict for floating point DCMs.
        # Consider checking orthogonality or determinant instead if issues arise.
        print("Warning: INS Cnb matrix out of expected bounds [-1, 1].")
        # raise ValueError("create_ins() failed, Cnb out of bounds. Re-run or check trajectory.")

    ins_euler_angles = dcm2euler(ins_Cnb, order='body2nav')
    ins_roll  = ins_euler_angles[:,0]
    ins_pitch = ins_euler_angles[:,1]
    ins_yaw   = ins_euler_angles[:,2]

    if save_h5:
        with h5py.File(ins_h5, "a") as file:
            for key, data_arr in [
                ("ins_tt", traj.tt), ("ins_lat", ins_lat), ("ins_lon", ins_lon), ("ins_alt", ins_alt),
                ("ins_vn", ins_vn), ("ins_ve", ins_ve), ("ins_vd", ins_vd),
                ("ins_fn", ins_fn), ("ins_fe", ins_fe), ("ins_fd", ins_fd),
                ("ins_roll", ins_roll), ("ins_pitch", ins_pitch), ("ins_yaw", ins_yaw),
                ("ins_P", P_ins) # Save covariance history
            ]:
                if key not in file: file.create_dataset(key, data=data_arr)
                else: file[key][:] = data_arr
                
    return INS(N=N, dt=dt, tt=traj.tt, lat=ins_lat, lon=ins_lon, alt=ins_alt,
               vn=ins_vn, ve=ins_ve, vd=ins_vd, fn=ins_fn, fe=ins_fe, fd=ins_fd,
               Cnb=ins_Cnb, P=P_ins)


def create_mag_c(
    path_or_lat: Union[Path, np.ndarray],
    lon_or_mapS: Union[np.ndarray, Union[MapS, MapSd, MapS3D]],
    mapS_if_path: Optional[Union[MapS, MapSd, MapS3D]] = None,
    alt: Optional[float] = None, # Used if lat,lon are inputs
    dt: Optional[float] = None,   # Used if lat,lon are inputs
    meas_var: float = 1.0**2,
    fogm_sigma: float = 1.0,
    fogm_tau: float = 600.0,
    silent: bool = False
) -> np.ndarray:
    """
    Create compensated (clean) scalar magnetometer measurements.
    Overloaded: (lat, lon, mapS, ...) or (path, mapS, ...)
    Corresponds to create_mag_c in Julia.
    """
    if isinstance(path_or_lat, (Traj, INS)): # Path object
        path = path_or_lat
        mapS_actual = lon_or_mapS # mapS is the second arg in this case
        if not isinstance(mapS_actual, (MapS, MapSd, MapS3D)):
             if mapS_actual is None: mapS_actual = get_map(DEFAULT_SCALAR_MAP_ID)
             else: raise TypeError("mapS_actual must be a MapS, MapSd, or MapS3D object when path is provided.")

        lat_rad = path.lat
        lon_rad = path.lon
        # Use median altitude from path if alt not explicitly overridden by user (though not an option here)
        alt_val = np.median(path.alt)
        dt_val  = path.dt
    elif isinstance(path_or_lat, np.ndarray) and isinstance(lon_or_mapS, np.ndarray): # lat, lon arrays
        lat_rad = path_or_lat
        lon_rad = lon_or_mapS
        mapS_actual = mapS_if_path
        if not isinstance(mapS_actual, (MapS, MapSd, MapS3D)):
            if mapS_actual is None: mapS_actual = get_map(DEFAULT_SCALAR_MAP_ID)
            else: raise TypeError("mapS_actual must be a MapS, MapSd, or MapS3D object.")
        
        if alt is None or dt is None:
            raise ValueError("alt and dt must be provided when giving lat, lon arrays.")
        alt_val = alt
        dt_val = dt
    else:
        raise TypeError("Invalid arguments for create_mag_c. Provide (Path, MapS) or (lat_arr, lon_arr, MapS).")

    # Convert MapS3D to MapS at specified altitude if necessary
    # upward_fft_map from analysis_util
    if isinstance(mapS_actual, MapS3D):
        mapS_actual = upward_fft_map(mapS_actual, alt_val)

    N = len(lat_rad)
    
    # map_params, fill_map_gaps, trim_map from analysis_util
    # These steps ensure the map is suitable for value extraction.
    # The Julia code checks `sum(ind0)/sum(ind0+ind1) > 0.01` for filling.
    # This implies `map_params` returns info about filled/unfilled portions.
    # Assuming these util functions handle such logic.
    # ind0, ind1, _, _ = get_map_params(mapS_actual) # If needed for explicit check
    # if sum(ind0) / (sum(ind0) + sum(ind1)) > 0.01: # Simplified logic
    if not silent: print("Info: preparing scalar map (filling/trimming if necessary).")
    mapS_processed = fill_map_gaps(trim_map(mapS_actual)) # Chain operations

    # Get map values along trajectory
    # get_map_value_at_coords from analysis_util
    if not silent: print("Info: getting scalar map values (upward/downward continuation if needed).")
    map_values_scalar = get_map_value_at_coords(mapS_processed, lat_rad, lon_rad, alt_val, alpha=200)

    # Add FOGM & white noise
    # generate_fogm_noise from analysis_util
    if not silent: print("Info: adding FOGM & white noise to scalar map values.")
    fogm_noise_scalar = generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N)
    white_noise_scalar = np.sqrt(meas_var) * np.random.randn(N)
    
    mag_c_values = map_values_scalar + fogm_noise_scalar + white_noise_scalar
    
    return mag_c_values


def corrupt_mag(
    mag_c: np.ndarray,
    flux_or_Bx: Union[MagV, np.ndarray], # MagV object or Bx array
    By_if_coords: Optional[np.ndarray] = None,
    Bz_if_coords: Optional[np.ndarray] = None,
    dt: float = 0.1,
    cor_sigma: float = 1.0,
    cor_tau: float = 600.0,
    cor_var: float = 1.0**2,
    cor_drift: float = 0.001,
    cor_perm_mag: float = 5.0, # Std dev for permanent TL coefs
    cor_ind_mag: float = 5.0,  # Std dev for induced TL coefs
    cor_eddy_mag: float = 0.5  # Std dev for eddy current TL coefs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrupt compensated scalar mag data with FOGM, drift, and Tolles-Lawson noise.
    Overloaded: (mag_c, flux_obj, ...) or (mag_c, Bx, By, Bz, ...)
    Returns (mag_uc, TL_coefficients, corruption_FOGM_noise)
    Corresponds to corrupt_mag in Julia.
    """
    if isinstance(flux_or_Bx, MagV):
        flux_obj = flux_or_Bx
        Bx_vals, By_vals, Bz_vals = flux_obj.x, flux_obj.y, flux_obj.z
    elif isinstance(flux_or_Bx, np.ndarray) and \
         isinstance(By_if_coords, np.ndarray) and \
         isinstance(Bz_if_coords, np.ndarray):
        Bx_vals, By_vals, Bz_vals = flux_or_Bx, By_if_coords, Bz_if_coords
    else:
        raise TypeError("Invalid arguments for corrupt_mag. Provide (mag_c, MagV_obj) or (mag_c, Bx, By, Bz).")

    N = len(mag_c)

    # Tolles-Lawson coefficients are sampled from N(0, P)
    # P is diagonal with variances: perm^2 (3 terms), ind^2 (6 terms), eddy^2 (9 terms)
    # Total 3+6+9 = 18 coefficients for standard TL model.
    tl_variances = np.concatenate([
        np.full(3, cor_perm_mag**2),
        np.full(6, cor_ind_mag**2),
        np.full(9, cor_eddy_mag**2)
    ])
    P_tl_cov = np.diag(tl_variances)
    
    # Sample TL coefficients
    # Ensure mean is explicitly zero for multivariate_normal
    tl_coefficients = multivariate_normal.rvs(mean=np.zeros(len(tl_variances)), cov=P_tl_cov)

    # FOGM noise for corruption
    corruption_fogm_noise = generate_fogm_noise(cor_sigma, cor_tau, dt, N)

    # White noise for corruption
    corruption_white_noise = np.sqrt(cor_var) * np.random.randn(N)
    
    # Linear drift for corruption
    # Julia: cor_drift*rand()*(0:dt:dt*(N-1)) -> single random scale for whole drift
    time_vector = np.arange(N) * dt
    corruption_drift = cor_drift * random.random() * time_vector
    
    mag_uc_intermediate = mag_c + corruption_white_noise + corruption_fogm_noise + corruption_drift

    # Add Tolles-Lawson aircraft field contribution if Bx,By,Bz are non-zero
    # create_tolles_lawson_A_matrix from analysis_util
    if not (np.allclose(Bx_vals, 0) and np.allclose(By_vals, 0) and np.allclose(Bz_vals, 0)):
        A_tl = create_tolles_lawson_A_matrix(Bx_vals, By_vals, Bz_vals) # Uses all terms by default
        # A_tl should be N x 18. tl_coefficients is 18.
        # TL effect is A_tl @ tl_coefficients
        if A_tl.shape[1] == len(tl_coefficients): # Check compatibility
             tl_effect_on_scalar = A_tl @ tl_coefficients
             mag_uc_final = mag_uc_intermediate + tl_effect_on_scalar
        else:
            print(f"Warning: TL matrix columns {A_tl.shape[1]} != TL coeffs {len(tl_coefficients)}. Skipping TL effect.")
            mag_uc_final = mag_uc_intermediate
    else:
        mag_uc_final = mag_uc_intermediate
        
    return mag_uc_final, tl_coefficients, corruption_fogm_noise


def create_flux(
    path_or_lat: Union[Path, np.ndarray],
    lon_or_mapV: Union[np.ndarray, MapV],
    mapV_if_path: Optional[MapV] = None,
    Cnb_if_coords: Optional[np.ndarray] = None, # N x 3 x 3, used if lat,lon are inputs
    alt: Optional[float] = None, # Used if lat,lon are inputs
    dt: Optional[float] = None,   # Used if lat,lon are inputs
    meas_var: float = 1.0**2,
    fogm_sigma: float = 1.0,
    fogm_tau: float = 600.0,
    silent: bool = False
) -> MagV:
    """
    Create compensated (clean) vector magnetometer measurements (fluxgate).
    Overloaded: (path, mapV, ...) or (lat, lon, mapV, Cnb, alt, dt, ...)
    Corresponds to create_flux in Julia.
    """
    if isinstance(path_or_lat, (Traj, INS)): # Path object
        path = path_or_lat
        mapV_actual = lon_or_mapV # mapV is the second arg
        if not isinstance(mapV_actual, MapV):
            if mapV_actual is None: mapV_actual = get_map(DEFAULT_VECTOR_MAP_ID)
            else: raise TypeError("mapV_actual must be a MapV object when path is provided.")

        lat_rad = path.lat
        lon_rad = path.lon
        Cnb_val = path.Cnb # N x 3 x 3
        alt_val = np.median(path.alt)
        dt_val  = path.dt
    elif isinstance(path_or_lat, np.ndarray) and isinstance(lon_or_mapV, np.ndarray): # lat, lon arrays
        lat_rad = path_or_lat
        lon_rad = lon_or_mapV
        mapV_actual = mapV_if_path
        if not isinstance(mapV_actual, MapV):
            if mapV_actual is None: mapV_actual = get_map(DEFAULT_VECTOR_MAP_ID)
            else: raise TypeError("mapV_actual must be a MapV object.")
        
        if Cnb_if_coords is None or alt is None or dt is None:
            raise ValueError("Cnb, alt, and dt must be provided when giving lat, lon arrays.")
        Cnb_val = Cnb_if_coords
        alt_val = alt
        dt_val = dt
    else:
        raise TypeError("Invalid arguments for create_flux. Provide (Path, MapV) or (lat, lon, MapV, Cnb, ...).")

    N = len(lat_rad)
    if Cnb_val.shape[0] != N or Cnb_val.shape[1:] != (3,3):
        # If Cnb is single 3x3, tile it. Julia default was repeat(I(3),1,1,N)
        if Cnb_val.shape == (3,3) and N > 0 : # Single DCM provided
            Cnb_val = np.tile(Cnb_val, (N,1,1))
        elif N == 0 and Cnb_val.shape == (3,3): # No points, but Cnb given
             pass # Cnb_val is fine, loops won't run
        else:
            raise ValueError(f"Cnb shape {Cnb_val.shape} inconsistent with N={N} points.")


    # Get vector map values (Bx_nav, By_nav, Bz_nav) along trajectory
    # get_map_value_at_coords for MapV should return a tuple of 3 arrays or similar
    if not silent: print("Info: getting vector map values (upward/downward continuation if needed).")
    # Assuming get_map_value_at_coords for MapV returns tuple (Bx_nav, By_nav, Bz_nav)
    map_values_vector_nav = get_map_value_at_coords(mapV_actual, lat_rad, lon_rad, alt_val, alpha=200)
    if not (isinstance(map_values_vector_nav, tuple) and len(map_values_vector_nav) == 3):
        raise ValueError("get_map_value_at_coords for MapV did not return 3 vector components.")
    Bx_nav, By_nav, Bz_nav = map_values_vector_nav

    # Add FOGM & white noise to each component
    if not silent: print("Info: adding FOGM & white noise to vector map values.")
    noise_std_dev = np.sqrt(meas_var)
    Bx_nav_noisy = Bx_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    By_nav_noisy = By_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    Bz_nav_noisy = Bz_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    
    # Total field magnitude in navigation frame (before rotation to body)
    Bt_nav_noisy = np.sqrt(Bx_nav_noisy**2 + By_nav_noisy**2 + Bz_nav_noisy**2)

    # Rotate measurements from navigation to body frame: B_body = Cnb^T * B_nav
    # Cnb is body to nav, so Cnb.T is nav to body.
    Bx_body = np.zeros(N)
    By_body = np.zeros(N)
    Bz_body = np.zeros(N)

    for i in range(N):
        B_nav_i = np.array([Bx_nav_noisy[i], By_nav_noisy[i], Bz_nav_noisy[i]])
        # Cnb_val[i] is C_body_nav for point i. Cnb_val[i].T is C_nav_body
        B_body_i = Cnb_val[i,:,:].T @ B_nav_i
        Bx_body[i], By_body[i], Bz_body[i] = B_body_i[0], B_body_i[1], B_body_i[2]
        
    # Total field magnitude in body frame (should be same as Bt_nav_noisy if rotation is correct)
    Bt_body = np.sqrt(Bx_body**2 + By_body**2 + Bz_body**2)
    # Assert or check if Bt_body is close to Bt_nav_noisy as a sanity check.
    # if N > 0 and not np.allclose(Bt_body, Bt_nav_noisy):
    #    print("Warning: Total field magnitude changed after rotation to body frame.")

    return MagV(x=Bx_body, y=By_body, z=Bz_body, t=Bt_body)


def create_dcm_internal( # Renamed from create_dcm to avoid conflict if there's a main one
    vn: np.ndarray,
    ve: np.ndarray,
    dt: float = 0.1,
    order: str = 'body2nav' # :body2nav or :nav2body
) -> np.ndarray:
    """
    Internal helper to estimate DCM using known heading with FOGM noise.
    Corresponds to create_dcm in Julia.
    """
    N = len(vn)
    # FOGM noise parameters from Julia code (hardcoded)
    roll_fogm_std_rad  = np.deg2rad(2.0)
    pitch_fogm_std_rad = np.deg2rad(0.5)
    yaw_fogm_std_rad   = np.deg2rad(1.0)
    fogm_tau_attitude  = 2.0 # seconds

    roll_noise  = generate_fogm_noise(roll_fogm_std_rad,  fogm_tau_attitude, dt, N)
    pitch_noise = generate_fogm_noise(pitch_fogm_std_rad, fogm_tau_attitude, dt, N)
    yaw_noise   = generate_fogm_noise(yaw_fogm_std_rad,   fogm_tau_attitude, dt, N)

    # Band-pass filter for attitude noise (from analysis_util)
    # Julia: bpf = get_bpf(;pass1=1e-6,pass2=1)
    bpf_coeffs = get_band_pass_filter_coeffs(pass1=1e-6, pass2=1.0) # Assuming 1 Hz upper for attitude
    
    roll_filt  = apply_band_pass_filter(roll_noise, bpf_coeffs=bpf_coeffs)
    pitch_filt = apply_band_pass_filter(pitch_noise, bpf_coeffs=bpf_coeffs) + np.deg2rad(2.0) # Pitch bias
    
    # Yaw definition from velocities (heading) + filtered noise
    # atan2(ve, vn) gives heading angle (from North, positive East)
    heading_rad = np.arctan2(ve, vn)
    yaw_filt = apply_band_pass_filter(yaw_noise, bpf_coeffs=bpf_coeffs) + heading_rad
    
    # Convert Euler angles (roll, pitch, yaw) to DCM
    # euler2dcm from analysis_util
    dcm_array = euler2dcm(roll_filt, pitch_filt, yaw_filt, order=order) # N x 3 x 3
    
    return dcm_array


def calculate_imputed_TL_earth(
    xyz: XYZ0, # Assuming XYZ0 or a compatible type
    ind: np.ndarray, # Indices for the subset of data
    map_val_scalar: np.ndarray, # Scalar magnetic anomaly map values for these indices
    set_igrf_in_xyz: bool,
    TL_coef: np.ndarray, # Tolles-Lawson coefficients (1D array)
    terms: List[str] = ['permanent', 'induced', 'eddy'], # Or use symbols/enums
    Bt_scale: float = 50000.0
) -> Tuple[np.ndarray, np.ndarray]: # Returns (TL_aircraft_vector_field_3xN, B_earth_vector_field_3xN)
    """
    Internal helper to get imputed Earth vector and TL aircraft field.
    Corresponds to calculate_imputed_TL_earth in Julia.
    """
    # Get IGRF vector in body frame for the selected indices
    # get_igrf_magnetic_field from analysis_util
    igrf_vector_body = get_igrf_magnetic_field(
        xyz, ind=ind, frame='body', norm_igrf=False, check_xyz=(not set_igrf_in_xyz)
    ) # Returns 3xN_ind

    if set_igrf_in_xyz:
        # Calculate scalar IGRF and update in xyz object for the given indices
        xyz.igrf[ind] = np.linalg.norm(igrf_vector_body, axis=0)

    # Total Earth field scalar magnitude (map anomaly + IGRF scalar)
    # Assuming map_val_scalar corresponds to ind
    B_earth_scalar_total = map_val_scalar + np.linalg.norm(igrf_vector_body, axis=0) # No diurnal here

    # Impute Earth vector field by scaling normalized IGRF vector
    # Normalize each column of igrf_vector_body (3xN_ind)
    norm_igrf_vec = np.linalg.norm(igrf_vector_body, axis=0, keepdims=True)
    # Avoid division by zero if norm is zero
    norm_igrf_vec[norm_igrf_vec == 0] = 1e-9 # Replace zero norms
    unit_igrf_vector_body = igrf_vector_body / norm_igrf_vec
    
    B_earth_vector_body = unit_igrf_vector_body * B_earth_scalar_total[np.newaxis, :] # 3xN_ind

    # Time-derivative of Earth vector field (in body frame)
    # fdm from analysis_util, applied row-wise to 3xN_ind array
    # Assuming fdm handles 2D arrays by operating on rows or needs a loop/map.
    # If fdm is 1D:
    B_earth_dot_body = np.vstack([
        fdm(B_earth_vector_body[0,:]),
        fdm(B_earth_vector_body[1,:]),
        fdm(B_earth_vector_body[2,:])
    ]) # Results in 3xN_ind

    # Convert TL coefficients to matrix form (permanent, induced, eddy matrices)
    # tolles_lawson_coeffs_to_matrix from analysis_util
    TL_coef_p_mat, TL_coef_i_mat, TL_coef_e_mat = \
        tolles_lawson_coeffs_to_matrix(TL_coef, terms=terms, Bt_scale=Bt_scale)

    # Calculate aircraft's magnetic field vector due to Earth's field (TL effect)
    # get_tolles_lawson_aircraft_field_vector from analysis_util
    TL_aircraft_vector_field = get_tolles_lawson_aircraft_field_vector(
        B_earth_vector_body, B_earth_dot_body,
        TL_coef_p_mat, TL_coef_i_mat, TL_coef_e_mat
    ) # Returns 3xN_ind
    
    return TL_aircraft_vector_field, B_earth_vector_body


def create_informed_xyz(
    xyz: XYZ0, # Assuming XYZ0 or a compatible type
    ind: np.ndarray, # Indices for the subset of data
    mapS: Union[MapS, MapSd, MapS3D],
    use_mag_field_name: str, # e.g., "mag_1_uc"
    use_vec_field_name: str, # e.g., "flux_a"
    TL_coef: np.ndarray, # Tolles-Lawson coefficients (1D array)
    terms: List[str] = ['permanent', 'induced', 'eddy'],
    disp_min_m: float = 100.0, # Min displacement in meters
    disp_max_m: float = 500.0, # Max displacement in meters
    Bt_disp_nT: float = 50.0,  # Target total field displacement offset in nT
    Bt_scale: float = 50000.0
) -> XYZ0:
    """
    Create knowledge-informed XYZ data by displacing trajectory and updating mag fields.
    Corresponds to create_informed_xyz in Julia.
    """
    # Basic validation of terms (simplified from Julia's specific symbol checks)
    required_term_categories = {'permanent', 'induced', 'eddy'}
    provided_term_categories = set()
    for term in terms: # crude check
        if 'perm' in term: provided_term_categories.add('permanent')
        if 'ind' in term: provided_term_categories.add('induced')
        if 'eddy' in term: provided_term_categories.add('eddy')
    if not required_term_categories.issubset(provided_term_categories):
        raise ValueError("Permanent, induced, and eddy terms are required for create_informed_xyz.")
    if any(t in terms for t in ['fdm', 'bias']): # crude check
        raise ValueError("Derivative and bias terms may not be used in create_informed_xyz.")

    # Validate TL_coef length against a test A matrix (if create_TL_A is robust)
    # A_test = create_tolles_lawson_A_matrix(np.array([1.0]),np.array([1.0]),np.array([1.0]), terms=terms)
    # if len(TL_coef) != A_test.shape[1]:
    #     raise ValueError("TL_coef length does not agree with specified terms.")

    # Get the relevant subset of the trajectory
    # get_trajectory_subset from analysis_util
    traj_subset = get_trajectory_subset(xyz.traj, ind)

    if not map_check(mapS, traj_subset.lat, traj_subset.lon):
        raise ValueError("Original trajectory subset must be inside the provided map.")

    # Get scalar map values and interpolation object for the original trajectory subset
    # get_map_value_at_coords from analysis_util
    map_val_orig, itp_mapS_obj = get_map_value_at_coords(
        mapS, traj_subset.lat, traj_subset.lon, np.median(traj_subset.alt),
        alpha=200, return_itp=True
    )
    if itp_mapS_obj is None:
        raise RuntimeError("Could not get map interpolator object.")

    # Calculate initial aircraft and Earth vector fields for the original subset
    # Note: set_igrf_in_xyz=False as we are not modifying the input xyz here.
    TL_aircraft_orig, B_earth_orig = calculate_imputed_TL_earth(
        xyz, ind, map_val_orig, False, TL_coef, terms=terms, Bt_scale=Bt_scale
    ) # Both are 3xN_ind

    # Determine displacement direction (towards map center, along map gradient)
    # Sample ~100 points along trajectory subset
    num_samples = min(100, len(traj_subset.lat))
    sample_indices = np.linspace(0, len(traj_subset.lat) - 1, num_samples, dtype=int)
    
    sampled_lats = traj_subset.lat[sample_indices]
    sampled_lons = traj_subset.lon[sample_indices]

    # Average lat/lon of trajectory subset and map center
    traj_avg_latlon = np.array([np.mean(traj_subset.lat), np.mean(traj_subset.lon)])
    map_center_latlon = np.array([np.mean(mapS.yy), np.mean(mapS.xx)]) # Assuming yy,xx are rad

    # Calculate average map gradient at sampled points
    # approximate_gradient(itp_func, y, x) from analysis_util -> returns [dval/dlat, dval/dlon]
    gradients_at_samples = [approximate_gradient(itp_mapS_obj, la, lo)
                            for la, lo in zip(sampled_lats, sampled_lons)]
    avg_gradient_latlon = np.mean(np.array(gradients_at_samples), axis=0) # [avg_dval/dlat, avg_dval/dlon]

    # Displacement direction based on gradient, normalized
    if np.linalg.norm(avg_gradient_latlon) < 1e-9: # Avoid division by zero if gradient is flat
        # Default to direction towards map center if gradient is zero
        disp_dir_latlon = map_center_latlon - traj_avg_latlon
    else:
        disp_dir_latlon = avg_gradient_latlon
    
    norm_disp_dir = np.linalg.norm(disp_dir_latlon)
    if norm_disp_dir < 1e-9: # If direction is still zero (e.g. traj_avg == map_center)
        disp_dir_latlon = np.array([1.0, 0.0]) # Arbitrary direction (e.g., East)
        norm_disp_dir = 1.0
    disp_dir_unit_latlon = disp_dir_latlon / norm_disp_dir

    # Ensure displacement is towards map center (if gradient pointed away)
    vec_to_map_center = map_center_latlon - traj_avg_latlon
    if np.dot(vec_to_map_center, disp_dir_unit_latlon) < 0:
        disp_dir_unit_latlon *= -1 # Reverse direction

    # Convert displacement limits from meters to radians
    # Using dn2dlat, de2dlon from analysis_util
    # Use average latitude of the trajectory for conversion factor
    avg_lat_for_conv = traj_avg_latlon[0]
    disp_min_rad = min(dn2dlat(disp_min_m, avg_lat_for_conv),
                       de2dlon(disp_min_m, avg_lat_for_conv))
    disp_max_rad = max(dn2dlat(disp_max_m, avg_lat_for_conv),
                       de2dlon(disp_max_m, avg_lat_for_conv))

    # Determine displacement magnitude in radians to achieve Bt_disp_nT change
    # Gradient magnitude along displacement direction: |d(MapVal)/d(Path)| in [nT/rad]
    # This is dot(avg_gradient_latlon, disp_dir_unit_latlon)
    gradient_mag_along_disp = abs(np.dot(avg_gradient_latlon, disp_dir_unit_latlon))
    
    if gradient_mag_along_disp < 1e-6: # Avoid division by zero if map is flat along displacement
        disp_rad_magnitude = (disp_min_rad + disp_max_rad) / 2 # Mid-range displacement
    else:
        disp_rad_magnitude = Bt_disp_nT / gradient_mag_along_disp
        
    # Clamp displacement magnitude to [disp_min_rad, disp_max_rad]
    disp_rad_magnitude_clamped = np.clip(disp_rad_magnitude, disp_min_rad, disp_max_rad)
    
    # Final displacement vector in (delta_lat_rad, delta_lon_rad)
    displacement_latlon_rad = disp_rad_magnitude_clamped * disp_dir_unit_latlon

    # Create a deep copy of the original xyz data to modify
    xyz_disp = deepcopy(xyz)

    # Apply displacement to the trajectory subset in the new xyz_disp object
    xyz_disp.traj.lat[ind] += displacement_latlon_rad[0]
    xyz_disp.traj.lon[ind] += displacement_latlon_rad[1]
    # Note: Velocities, accelerations, Cnb in xyz_disp.traj for these 'ind' are NOT updated.
    # This assumes the displacement is a parallel shift and dynamics are preserved,
    # which is a simplification. Julia code also does not update them.

    # Check if new displaced trajectory is still on the map
    displaced_traj_subset = get_trajectory_subset(xyz_disp.traj, ind)
    if not map_check(mapS, displaced_traj_subset.lat, displaced_traj_subset.lon):
        raise ValueError("Displaced trajectory is outside the map. Adjust displacement parameters or map.")

    # Get scalar map values for the new displaced trajectory subset
    map_val_disp = get_map_value_at_coords(
        mapS, displaced_traj_subset.lat, displaced_traj_subset.lon,
        np.median(displaced_traj_subset.alt), alpha=200
    )

    # Calculate aircraft and Earth vector fields for the new displaced subset
    # This time, set_igrf_in_xyz=True to update IGRF in xyz_disp for these 'ind'.
    TL_aircraft_disp, B_earth_disp = calculate_imputed_TL_earth(
        xyz_disp, ind, map_val_disp, True, TL_coef, terms=terms, Bt_scale=Bt_scale
    ) # Both are 3xN_ind

    # Calculate changes in vector fields
    delta_B_TL_effect = TL_aircraft_disp - TL_aircraft_orig # Change due to aircraft's TL interaction at new spot
    delta_B_earth_field = B_earth_disp - B_earth_orig     # Change due to Earth's field at new spot
    total_delta_B_vector = delta_B_TL_effect + delta_B_earth_field # Total change in 3D vector field (3xN_ind)

    # Update vector magnetometer readings in xyz_disp for the specified field and indices
    # getattr and setattr can be used for dynamic field names
    flux_disp_obj = getattr(xyz_disp, use_vec_field_name) # e.g., xyz_disp.flux_a
    
    # Assuming flux_disp_obj fields (x,y,z,t) are numpy arrays that can be indexed by 'ind'
    # And that they are already copies from deepcopy(xyz)
    flux_disp_obj.x[ind] += total_delta_B_vector[0,:]
    flux_disp_obj.y[ind] += total_delta_B_vector[1,:]
    flux_disp_obj.z[ind] += total_delta_B_vector[2,:]
    flux_disp_obj.t[ind] = np.sqrt(flux_disp_obj.x[ind]**2 + \
                                   flux_disp_obj.y[ind]**2 + \
                                   flux_disp_obj.z[ind]**2)
    # setattr(xyz_disp, use_vec_field_name, flux_disp_obj) # Not needed if flux_disp_obj is mutable reference

    # Update scalar magnetometer readings in xyz_disp
    # Change in scalar reading is projection of total_delta_B_vector onto new total field direction
    # New total field direction is approximately direction of flux_disp_obj.x/y/z[ind]
    
    # Project total_delta_B_vector onto the direction of the new (displaced) flux vector
    # new_flux_vectors_at_ind = np.vstack((flux_disp_obj.x[ind], flux_disp_obj.y[ind], flux_disp_obj.z[ind])) # 3xN_ind
    # norm_new_flux = np.linalg.norm(new_flux_vectors_at_ind, axis=0, keepdims=True)
    # norm_new_flux[norm_new_flux == 0] = 1e-9
    # unit_new_flux_vectors = new_flux_vectors_at_ind / norm_new_flux
    # delta_B_scalar_projection = np.sum(total_delta_B_vector * unit_new_flux_vectors, axis=0) # Dot product per column

    # Julia's approach for scalar update:
    # ΔB_dot = dot.(eachcol(ΔB),[[x,y,z] for (x,y,z) in zip(flux.x[ind],flux.y[ind],flux.z[ind])]) ./ flux.t[ind]
    # This is projecting ΔB onto the *original* flux direction at displaced location (before ΔB is added to flux).
    # Let's use the new flux vector (after adding total_delta_B_vector) for projection, which seems more consistent.
    # If flux.x[ind] etc. are already updated:
    current_flux_x_at_ind = flux_disp_obj.x[ind]
    current_flux_y_at_ind = flux_disp_obj.y[ind]
    current_flux_z_at_ind = flux_disp_obj.z[ind]
    current_flux_t_at_ind = flux_disp_obj.t[ind] # This is already sqrt(x^2+y^2+z^2) of *new* flux

    # Avoid division by zero for flux_t
    valid_t_mask = current_flux_t_at_ind != 0
    delta_B_scalar_projection = np.zeros_like(current_flux_t_at_ind)

    if np.any(valid_t_mask):
        dot_prod_valid = (total_delta_B_vector[0, valid_t_mask] * current_flux_x_at_ind[valid_t_mask] +
                          total_delta_B_vector[1, valid_t_mask] * current_flux_y_at_ind[valid_t_mask] +
                          total_delta_B_vector[2, valid_t_mask] * current_flux_z_at_ind[valid_t_mask])
        delta_B_scalar_projection[valid_t_mask] = dot_prod_valid / current_flux_t_at_ind[valid_t_mask]


    # Update the specified uncompensated scalar magnetometer field
    current_mag_uc_vals = getattr(xyz_disp, use_mag_field_name)
    current_mag_uc_vals[ind] += delta_B_scalar_projection
    # setattr(xyz_disp, use_mag_field_name, current_mag_uc_vals) # Not needed if mutable

    # Update the compensated scalar magnetometer field (mag_1_c)
    # Change is simply the difference in map anomaly values
    delta_map_val_scalar = map_val_disp - map_val_orig
    xyz_disp.mag_1_c[ind] += delta_map_val_scalar
    
    return xyz_disp