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
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from copy import deepcopy

# Attempt to import from project modules
try:
    from .map_utils import get_map_val
    from .magnav import (XYZ0, Traj, INS, MagV, MapS, MapSd, MapS3D, MapV,
                           BaseMap, Path)
    from .analysis_util import (
        add_extension, g_earth, get_map, map_check, dlat2dn, dlon2de, dn2dlat,
        de2dlon, fdm, utm_zone_from_latlon, transform_lla_to_utm,
        create_dcm_from_vel, dcm2euler, euler2dcm,
        create_ins_model_matrices, get_phi_matrix, correct_cnb_matrix,
        upward_fft_map, get_map_params, fill_map_gaps, trim_map,
        generate_fogm_noise, create_tolles_lawson_A_matrix,
        get_igrf_magnetic_field,
        apply_band_pass_filter, get_band_pass_filter_coeffs,
        tolles_lawson_coeffs_to_matrix, get_tolles_lawson_aircraft_field_vector,
        get_trajectory_subset,
        approximate_gradient
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
    def map_check(m,la,lo): return True
    def dlat2dn(dlat,lat): return dlat * 111000
    def dlon2de(dlon,lat): return dlon * 111000 * np.cos(lat)
    def dn2dlat(dn,lat): return dn / 111000
    def de2dlon(de,lat): return de / (111000 * np.cos(lat))
    def fdm(arr): return np.gradient(arr) if arr.ndim == 1 else np.array([np.gradient(arr[i]) for i in range(arr.shape[0])])
    def utm_zone_from_latlon(lat_deg, lon_deg): return (int((lon_deg + 180) / 6) + 1, lat_deg >=0)
    def transform_lla_to_utm(lat_rad, lon_rad, zone, is_north):
        lat_deg, lon_deg = np.rad2deg(lat_rad), np.rad2deg(lon_rad)
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
    silent: bool = False
) -> XYZ0:
    """
    Create basic flight data (XYZ0 struct). Assumes constant altitude (2D flight).
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
        traj, mapS,
        meas_var=cor_var,
        fogm_sigma=fogm_sigma,
        fogm_tau=fogm_tau,
        silent=silent
    )

    # Create compensated (clean) vector magnetometer measurements
    flux_a = create_flux(
        traj, mapV,
        meas_var=cor_var,
        fogm_sigma=fogm_sigma,
        fogm_tau=fogm_tau,
        silent=silent
    )

    # Create uncompensated (corrupted) scalar magnetometer measurements
    mag_1_uc, _, diurnal_effect = corrupt_mag(
        mag_1_c, flux_a,
        dt=dt,
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
               flights=flights, lines=lines, years=years, doys=doys,
               diurnal=diurnal, igrf=igrf_initial,
               mag_1_c=mag_1_c, mag_1_uc=mag_1_uc)

    igrf_vector_body = get_igrf_magnetic_field(xyz, frame='body', norm_igrf=False, check_xyz=False)
    igrf_scalar = np.linalg.norm(igrf_vector_body, axis=0)
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
        valid_alts = mapS.alt[mapS.mask] if mapS.mask.any() else [np.mean(mapS.alt)]
        map_altitude_ref = np.median(valid_alts) if len(valid_alts) > 0 else np.mean(mapS.alt)
    elif isinstance(mapS.alt, (list, np.ndarray)) and len(mapS.alt) > 0:
        map_altitude_ref = mapS.alt[0]
    elif isinstance(mapS.alt, (float, int)):
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
    fd = fdm(vd) / dt - g_earth if dt > 1e-6 else np.full_like(vd, -g_earth)
    
    tt = np.linspace(0, t, N_pts)

    Cnb_array = create_dcm_from_vel(vn, ve, dt, order='body2nav')
    euler_angles = dcm2euler(Cnb_array, order='body2nav')
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

    P0, Qd, _ = create_ins_model_matrices(
        dt, traj.lat[0],
        init_pos_sigma=init_pos_sigma, init_alt_sigma=init_alt_sigma,
        init_vel_sigma=init_vel_sigma, init_att_sigma=init_att_sigma,
        VRW_sigma=VRW_sigma, ARW_sigma=ARW_sigma,
        baro_sigma=baro_sigma, ha_sigma=ha_sigma, a_hat_sigma=a_hat_sigma,
        acc_sigma=acc_sigma, gyro_sigma=gyro_sigma,
        baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
        fogm_state=False
    )

    P_ins = np.zeros((N, nx, nx))
    err_ins = np.zeros((N, nx))

    P_ins[0,:,:] = P0
    err_ins[0,:] = multivariate_normal.rvs(mean=np.zeros(nx), cov=P0)
    
    try:
        Qd_chol = np.linalg.cholesky(Qd)
    except np.linalg.LinAlgError:
        Qd_chol = np.linalg.cholesky(Qd + np.eye(nx) * 1e-12)

    for k in range(N - 1):
        Phi_k = get_phi_matrix(
            nx, traj.lat[k], traj.vn[k], traj.ve[k], traj.vd[k],
            traj.fn[k], traj.fe[k], traj.fd[k], traj.Cnb[k,:,:], # Pass Cnb for current step
            baro_tau, acc_tau, gyro_tau, 0, dt, fogm_state=False
        )
        process_noise_k = Qd_chol @ np.random.randn(nx)
        err_ins[k+1,:] = Phi_k @ err_ins[k,:] + process_noise_k
        P_ins[k+1,:,:] = Phi_k @ P_ins[k,:,:] @ Phi_k.T + Qd

    ins_lat = traj.lat - err_ins[:,0]
    ins_lon = traj.lon - err_ins[:,1]
    ins_alt = traj.alt - err_ins[:,2]
    ins_vn  = traj.vn  - err_ins[:,3]
    ins_ve  = traj.ve  - err_ins[:,4]
    ins_vd  = traj.vd  - err_ins[:,5]

    ins_fn = fdm(ins_vn) / dt if dt > 1e-6 else np.zeros_like(ins_vn)
    ins_fe = fdm(ins_ve) / dt if dt > 1e-6 else np.zeros_like(ins_ve)
    ins_fd = fdm(ins_vd) / dt - g_earth if dt > 1e-6 else np.full_like(ins_vd, -g_earth)

    ins_Cnb_list = []
    for k_cnb in range(N):
      ins_Cnb_list.append(correct_cnb_matrix(traj.Cnb[k_cnb,:,:], -err_ins[k_cnb,6:9]))
    ins_Cnb = np.array(ins_Cnb_list)


    if np.any(np.abs(ins_Cnb) > 1.00001):
        print("Warning: INS Cnb matrix out of expected bounds [-1, 1]. Values might be clamped or indicate instability.")
        ins_Cnb = np.clip(ins_Cnb, -1.0, 1.0) # Basic stabilization

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
                ("ins_P", P_ins)
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
    alt: Optional[float] = None,
    dt: Optional[float] = None,
    meas_var: float = 1.0**2,
    fogm_sigma: float = 1.0,
    fogm_tau: float = 600.0,
    silent: bool = False
) -> np.ndarray:
    """
    Create compensated (clean) scalar magnetometer measurements.
    """
    if isinstance(path_or_lat, (Traj, INS)):
        path = path_or_lat
        mapS_actual = lon_or_mapS
        if not isinstance(mapS_actual, (MapS, MapSd, MapS3D)):
             if mapS_actual is None: mapS_actual = get_map(DEFAULT_SCALAR_MAP_ID)
             else: raise TypeError("mapS_actual must be a MapS, MapSd, or MapS3D object when path is provided.")
        lat_rad, lon_rad, alt_val, dt_val = path.lat, path.lon, np.median(path.alt), path.dt
    elif isinstance(path_or_lat, np.ndarray) and isinstance(lon_or_mapS, np.ndarray):
        lat_rad, lon_rad = path_or_lat, lon_or_mapS
        mapS_actual = mapS_if_path
        if not isinstance(mapS_actual, (MapS, MapSd, MapS3D)):
            if mapS_actual is None: mapS_actual = get_map(DEFAULT_SCALAR_MAP_ID)
            else: raise TypeError("mapS_actual must be a MapS, MapSd, or MapS3D object.")
        if alt is None or dt is None: raise ValueError("alt and dt must be provided for lat, lon arrays.")
        alt_val, dt_val = alt, dt
    else:
        raise TypeError("Invalid arguments for create_mag_c.")

    if isinstance(mapS_actual, MapS3D):
        mapS_actual = upward_fft_map(mapS_actual, alt_val)

    N = len(lat_rad)
    if not silent: print("Info: preparing scalar map (filling/trimming if necessary).")
    mapS_processed = fill_map_gaps(trim_map(mapS_actual))

    if not silent: print("Info: getting scalar map values (upward/downward continuation if needed).")
    map_values_scalar = get_map_val(mapS_processed, lat_rad, lon_rad, alt_val)

    if not silent: print("Info: adding FOGM & white noise to scalar map values.")
    fogm_noise_scalar = generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N)
    white_noise_scalar = np.sqrt(meas_var) * np.random.randn(N)
    
    return map_values_scalar + fogm_noise_scalar + white_noise_scalar


def corrupt_mag(
    mag_c: np.ndarray,
    flux_or_Bx: Union[MagV, np.ndarray],
    By_if_coords: Optional[np.ndarray] = None,
    Bz_if_coords: Optional[np.ndarray] = None,
    dt: float = 0.1,
    cor_sigma: float = 1.0,
    cor_tau: float = 600.0,
    cor_var: float = 1.0**2,
    cor_drift: float = 0.001,
    cor_perm_mag: float = 5.0,
    cor_ind_mag: float = 5.0,
    cor_eddy_mag: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrupt compensated scalar mag data with FOGM, drift, and Tolles-Lawson noise.
    Returns (mag_uc, TL_coefficients, corruption_FOGM_noise)
    """
    if isinstance(flux_or_Bx, MagV):
        Bx_vals, By_vals, Bz_vals = flux_or_Bx.x, flux_or_Bx.y, flux_or_Bx.z
    elif isinstance(flux_or_Bx, np.ndarray) and \
         isinstance(By_if_coords, np.ndarray) and \
         isinstance(Bz_if_coords, np.ndarray):
        Bx_vals, By_vals, Bz_vals = flux_or_Bx, By_if_coords, Bz_if_coords
    else:
        raise TypeError("Invalid arguments for corrupt_mag.")

    N = len(mag_c)
    tl_variances = np.concatenate([
        np.full(3, cor_perm_mag**2), np.full(6, cor_ind_mag**2), np.full(9, cor_eddy_mag**2)
    ])
    P_tl_cov = np.diag(tl_variances)
    tl_coefficients = multivariate_normal.rvs(mean=np.zeros(len(tl_variances)), cov=P_tl_cov)

    corruption_fogm_noise = generate_fogm_noise(cor_sigma, cor_tau, dt, N)
    corruption_white_noise = np.sqrt(cor_var) * np.random.randn(N)
    time_vector = np.arange(N) * dt
    corruption_drift = cor_drift * random.random() * time_vector
    
    mag_uc_intermediate = mag_c + corruption_white_noise + corruption_fogm_noise + corruption_drift

    if not (np.allclose(Bx_vals, 0) and np.allclose(By_vals, 0) and np.allclose(Bz_vals, 0)):
        A_tl = create_tolles_lawson_A_matrix(Bx_vals, By_vals, Bz_vals)
        if A_tl.shape[0] > 0 and A_tl.shape[1] == len(tl_coefficients): # Check compatibility
             tl_effect_on_scalar = A_tl @ tl_coefficients
             mag_uc_final = mag_uc_intermediate + tl_effect_on_scalar
        else:
            print(f"Warning: TL matrix columns {A_tl.shape[1]} != TL coeffs {len(tl_coefficients)} or A_tl is empty. Skipping TL effect.")
            mag_uc_final = mag_uc_intermediate
    else:
        mag_uc_final = mag_uc_intermediate
        
    return mag_uc_final, tl_coefficients, corruption_fogm_noise


def create_flux(
    path_or_lat: Union[Path, np.ndarray],
    lon_or_mapV: Union[np.ndarray, MapV],
    mapV_if_path: Optional[MapV] = None,
    Cnb_if_coords: Optional[np.ndarray] = None,
    alt: Optional[float] = None,
    dt: Optional[float] = None,
    meas_var: float = 1.0**2,
    fogm_sigma: float = 1.0,
    fogm_tau: float = 600.0,
    silent: bool = False
) -> MagV:
    """
    Create compensated (clean) vector magnetometer measurements (fluxgate).
    """
    if isinstance(path_or_lat, (Traj, INS)):
        path = path_or_lat
        mapV_actual = lon_or_mapV
        if not isinstance(mapV_actual, MapV):
            if mapV_actual is None: mapV_actual = get_map(DEFAULT_VECTOR_MAP_ID)
            else: raise TypeError("mapV_actual must be a MapV object when path is provided.")
        lat_rad, lon_rad, Cnb_val, alt_val, dt_val = path.lat, path.lon, path.Cnb, np.median(path.alt), path.dt
    elif isinstance(path_or_lat, np.ndarray) and isinstance(lon_or_mapV, np.ndarray):
        lat_rad, lon_rad = path_or_lat, lon_or_mapV
        mapV_actual = mapV_if_path
        if not isinstance(mapV_actual, MapV):
            if mapV_actual is None: mapV_actual = get_map(DEFAULT_VECTOR_MAP_ID)
            else: raise TypeError("mapV_actual must be a MapV object.")
        if Cnb_if_coords is None or alt is None or dt is None:
            raise ValueError("Cnb, alt, and dt must be provided for lat, lon arrays.")
        Cnb_val, alt_val, dt_val = Cnb_if_coords, alt, dt
    else:
        raise TypeError("Invalid arguments for create_flux.")

    N = len(lat_rad)
    if Cnb_val.ndim == 2 and Cnb_val.shape == (3,3) and N > 0: # Single DCM provided
        Cnb_val = np.tile(Cnb_val, (N,1,1))
    elif Cnb_val.shape[0] != N or Cnb_val.shape[1:] != (3,3):
        if not (N == 0 and Cnb_val.shape == (3,3)): # Allow if N=0 and Cnb is single
             raise ValueError(f"Cnb shape {Cnb_val.shape} inconsistent with N={N} points.")

    if not silent: print("Info: getting vector map values.")
    map_values_vector_nav = get_map_val(mapV_actual, lat_rad, lon_rad, alt_val) # Expected (Bx,By,Bz) tuple
    if not (isinstance(map_values_vector_nav, tuple) and len(map_values_vector_nav) == 3):
        raise ValueError("get_map_val for MapV did not return 3 vector components.")
    Bx_nav, By_nav, Bz_nav = map_values_vector_nav

    if not silent: print("Info: adding FOGM & white noise to vector map values.")
    noise_std_dev = np.sqrt(meas_var)
    Bx_nav_noisy = Bx_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    By_nav_noisy = By_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    Bz_nav_noisy = Bz_nav + generate_fogm_noise(fogm_sigma, fogm_tau, dt_val, N) + noise_std_dev * np.random.randn(N)
    
    Bx_body, By_body, Bz_body = np.zeros(N), np.zeros(N), np.zeros(N)
    if N > 0: # Ensure Cnb_val is ready for loop if N=0
        for i in range(N):
            B_nav_i = np.array([Bx_nav_noisy[i], By_nav_noisy[i], Bz_nav_noisy[i]])
            B_body_i = Cnb_val[i,:,:].T @ B_nav_i
            Bx_body[i], By_body[i], Bz_body[i] = B_body_i[0], B_body_i[1], B_body_i[2]
        
    Bt_body = np.sqrt(Bx_body**2 + By_body**2 + Bz_body**2)
    return MagV(x=Bx_body, y=By_body, z=Bz_body, t=Bt_body)


def create_dcm_internal(
    vn: np.ndarray,
    ve: np.ndarray,
    dt: float = 0.1,
    order: str = 'body2nav'
) -> np.ndarray:
    """
    Internal helper to estimate DCM using known heading with FOGM noise.
    """
    N = len(vn)
    roll_fogm_std_rad  = np.deg2rad(2.0)
    pitch_fogm_std_rad = np.deg2rad(0.5)
    yaw_fogm_std_rad   = np.deg2rad(1.0)
    fogm_tau_attitude  = 2.0

    roll_noise  = generate_fogm_noise(roll_fogm_std_rad,  fogm_tau_attitude, dt, N)
    pitch_noise = generate_fogm_noise(pitch_fogm_std_rad, fogm_tau_attitude, dt, N)
    yaw_noise   = generate_fogm_noise(yaw_fogm_std_rad,   fogm_tau_attitude, dt, N)

    bpf_coeffs = get_band_pass_filter_coeffs(pass1=1e-6, pass2=1.0)
    
    roll_filt  = apply_band_pass_filter(roll_noise, bpf_coeffs=bpf_coeffs)
    pitch_filt = apply_band_pass_filter(pitch_noise, bpf_coeffs=bpf_coeffs) + np.deg2rad(2.0)
    
    heading_rad = np.arctan2(ve, vn)
    yaw_filt = apply_band_pass_filter(yaw_noise, bpf_coeffs=bpf_coeffs) + heading_rad
    
    return euler2dcm(roll_filt, pitch_filt, yaw_filt, order=order)


def calculate_imputed_TL_earth(
    xyz: XYZ0,
    ind: np.ndarray,
    map_val_scalar: np.ndarray,
    set_igrf_in_xyz: bool,
    TL_coef: np.ndarray,
    terms: List[str] = ['permanent', 'induced', 'eddy'],
    Bt_scale: float = 50000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper to get imputed Earth vector and TL aircraft field.
    Returns (TL_aircraft_vector_field_3xN, B_earth_vector_field_3xN)
    """
    igrf_vector_body = get_igrf_magnetic_field(
        xyz, ind=ind, frame='body', norm_igrf=False, check_xyz=(not set_igrf_in_xyz)
    )

    if set_igrf_in_xyz:
        xyz.igrf[ind] = np.linalg.norm(igrf_vector_body, axis=0)

    B_earth_scalar_total = map_val_scalar + np.linalg.norm(igrf_vector_body, axis=0)

    norm_igrf_vec = np.linalg.norm(igrf_vector_body, axis=0, keepdims=True)
    norm_igrf_vec[norm_igrf_vec == 0] = 1e-9
    unit_igrf_vector_body = igrf_vector_body / norm_igrf_vec
    
    B_earth_vector_body = unit_igrf_vector_body * B_earth_scalar_total[np.newaxis, :]

    B_earth_dot_body = np.vstack([
        fdm(B_earth_vector_body[0,:]), fdm(B_earth_vector_body[1,:]), fdm(B_earth_vector_body[2,:])
    ]) / xyz.traj.dt # Assuming dt is consistent for the subset

    TL_coef_p_mat, TL_coef_i_mat, TL_coef_e_mat = \
        tolles_lawson_coeffs_to_matrix(TL_coef, terms=terms, Bt_scale=Bt_scale)

    TL_aircraft_vector_field = get_tolles_lawson_aircraft_field_vector(
        B_earth_vector_body, B_earth_dot_body,
        TL_coef_p_mat, TL_coef_i_mat, TL_coef_e_mat
    )
    
    return TL_aircraft_vector_field, B_earth_vector_body


def create_informed_xyz(
    xyz: XYZ0,
    ind: np.ndarray,
    mapS: Union[MapS, MapSd, MapS3D],
    use_mag_field_name: str,
    use_vec_field_name: str,
    TL_coef: np.ndarray,
    terms: List[str] = ['permanent', 'induced', 'eddy'],
    disp_min_m: float = 100.0,
    disp_max_m: float = 500.0,
    Bt_disp_nT: float = 50.0,
    Bt_scale: float = 50000.0
) -> XYZ0:
    """
    Create knowledge-informed XYZ data by displacing trajectory and updating mag fields.
    """
    required_term_categories = {'permanent', 'induced', 'eddy'}
    provided_term_categories = set()
    for term_group in terms: # terms might be like 'p', 'i', 'e' or full names
        if 'perm' in term_group or term_group == 'p': provided_term_categories.add('permanent')
        if 'ind' in term_group or term_group == 'i': provided_term_categories.add('induced')
        if 'eddy' in term_group or term_group == 'e': provided_term_categories.add('eddy')
    if not required_term_categories.issubset(provided_term_categories):
        raise ValueError("Permanent, induced, and eddy terms are required.")
    if any(t in terms for t in ['fdm', 'f', 'bias', 'b']):
        raise ValueError("Derivative and bias terms may not be used.")

    traj_subset = get_trajectory_subset(xyz.traj, ind)
    if not map_check(mapS, traj_subset.lat, traj_subset.lon):
        raise ValueError("Original trajectory subset must be inside the provided map.")

    map_val_orig, itp_mapS_obj = get_map_val(
        mapS, traj_subset.lat, traj_subset.lon, np.median(traj_subset.alt),
        return_interpolator=True
    )
    if itp_mapS_obj is None: raise RuntimeError("Could not get map interpolator object.")

    TL_aircraft_orig, B_earth_orig = calculate_imputed_TL_earth(
        xyz, ind, map_val_orig, False, TL_coef, terms=terms, Bt_scale=Bt_scale
    )

    num_samples = min(100, len(traj_subset.lat))
    sample_indices = np.linspace(0, len(traj_subset.lat) - 1, num_samples, dtype=int) if num_samples > 0 else []
    
    if not sample_indices.size: # Handle empty trajectory subset
        print("Warning: Empty trajectory subset for informed XYZ, returning copy.")
        return deepcopy(xyz)

    sampled_lats = traj_subset.lat[sample_indices]
    sampled_lons = traj_subset.lon[sample_indices]

    traj_avg_latlon = np.array([np.mean(traj_subset.lat), np.mean(traj_subset.lon)])
    map_center_latlon = np.array([np.mean(mapS.yy), np.mean(mapS.xx)])

    gradients_at_samples = [approximate_gradient(itp_mapS_obj, la, lo)
                            for la, lo in zip(sampled_lats, sampled_lons)]
    avg_gradient_latlon = np.mean(np.array(gradients_at_samples), axis=0)

    disp_dir_latlon = avg_gradient_latlon if np.linalg.norm(avg_gradient_latlon) >= 1e-9 else map_center_latlon - traj_avg_latlon
    norm_disp_dir = np.linalg.norm(disp_dir_latlon)
    disp_dir_unit_latlon = disp_dir_latlon / norm_disp_dir if norm_disp_dir >= 1e-9 else np.array([1.0, 0.0])
    
    vec_to_map_center = map_center_latlon - traj_avg_latlon
    if np.dot(vec_to_map_center, disp_dir_unit_latlon) < 0: disp_dir_unit_latlon *= -1

    avg_lat_for_conv = traj_avg_latlon[0]
    disp_min_rad = min(dn2dlat(disp_min_m, avg_lat_for_conv), de2dlon(disp_min_m, avg_lat_for_conv))
    disp_max_rad = max(dn2dlat(disp_max_m, avg_lat_for_conv), de2dlon(disp_max_m, avg_lat_for_conv))

    gradient_mag_along_disp = abs(np.dot(avg_gradient_latlon, disp_dir_unit_latlon))
    disp_rad_magnitude = Bt_disp_nT / gradient_mag_along_disp if gradient_mag_along_disp >= 1e-6 else (disp_min_rad + disp_max_rad) / 2
    disp_rad_magnitude_clamped = np.clip(disp_rad_magnitude, disp_min_rad, disp_max_rad)
    displacement_latlon_rad = disp_rad_magnitude_clamped * disp_dir_unit_latlon

    xyz_disp = deepcopy(xyz)
    xyz_disp.traj.lat[ind] += displacement_latlon_rad[0]
    xyz_disp.traj.lon[ind] += displacement_latlon_rad[1]

    displaced_traj_subset = get_trajectory_subset(xyz_disp.traj, ind)
    if not map_check(mapS, displaced_traj_subset.lat, displaced_traj_subset.lon):
        raise ValueError("Displaced trajectory is outside the map.")

    map_val_disp = get_map_val(
        mapS, displaced_traj_subset.lat, displaced_traj_subset.lon,
        np.median(displaced_traj_subset.alt), alpha=200
    )

    TL_aircraft_disp, B_earth_disp = calculate_imputed_TL_earth(
        xyz_disp, ind, map_val_disp, True, TL_coef, terms=terms, Bt_scale=Bt_scale
    )

    delta_B_TL_effect = TL_aircraft_disp - TL_aircraft_orig
    delta_B_earth_field = B_earth_disp - B_earth_orig
    total_delta_B_vector = delta_B_TL_effect + delta_B_earth_field

    flux_disp_obj = getattr(xyz_disp, use_vec_field_name)
    flux_disp_obj.x[ind] += total_delta_B_vector[0,:]
    flux_disp_obj.y[ind] += total_delta_B_vector[1,:]
    flux_disp_obj.z[ind] += total_delta_B_vector[2,:]
    flux_disp_obj.t[ind] = np.sqrt(flux_disp_obj.x[ind]**2 + flux_disp_obj.y[ind]**2 + flux_disp_obj.z[ind]**2)

    current_flux_x_at_ind = flux_disp_obj.x[ind]
    current_flux_y_at_ind = flux_disp_obj.y[ind]
    current_flux_z_at_ind = flux_disp_obj.z[ind]
    current_flux_t_at_ind = flux_disp_obj.t[ind]
    valid_t_mask = current_flux_t_at_ind != 0
    delta_B_scalar_projection = np.zeros_like(current_flux_t_at_ind)
    if np.any(valid_t_mask):
        dot_prod_valid = (total_delta_B_vector[0, valid_t_mask] * current_flux_x_at_ind[valid_t_mask] +
                          total_delta_B_vector[1, valid_t_mask] * current_flux_y_at_ind[valid_t_mask] +
                          total_delta_B_vector[2, valid_t_mask] * current_flux_z_at_ind[valid_t_mask])
        delta_B_scalar_projection[valid_t_mask] = dot_prod_valid / current_flux_t_at_ind[valid_t_mask]

    current_mag_uc_vals = getattr(xyz_disp, use_mag_field_name)
    current_mag_uc_vals[ind] += delta_B_scalar_projection
    
    delta_map_val_scalar = map_val_disp - map_val_orig
    xyz_disp.mag_1_c[ind] += delta_map_val_scalar
    
    return xyz_disp

# --- New functions for text-based XYZ file handling ---

def xyz_file_name(
    flight: Union[int, str],
    line: Union[int, str],
    output_dir: str = ".",
    prefix: str = "flight",
    suffix: str = "",
    ext: str = ".xyz"
) -> str:
    """
    Generates a standardized XYZ file name.
    Example: flight_1001_line_1.xyz
    """
    base_name = f"{prefix}_{flight}_line_{line}{suffix}{ext}"
    return os.path.join(output_dir, base_name)

def write_xyz(
    xyz_obj: XYZ0,
    file_path: str,
    columns: Optional[List[str]] = None,
    na_rep: str = "NaN",
    float_format: str = "%.3f",
    include_header: bool = True,
    delimiter: str = ","
) -> None:
    """
    Writes data from an XYZ0 object to a text-based XYZ file (typically CSV).

    Args:
        xyz_obj: The XYZ0 data object.
        file_path: Path to the output XYZ file.
        columns: Optional list of column names to write. If None, a default set is used.
                 Available sources: 'traj', 'ins', 'flux_a', 'scalar_mags', 'meta'.
                 Specific columns can be like 'traj.lat', 'mag_1_uc', etc.
        na_rep: Representation for missing values.
        float_format: Format string for floating point numbers.
        include_header: Whether to write the header row.
        delimiter: Delimiter for the output file.
    """
    data_to_write = {}
    N = xyz_obj.traj.N

    # Default columns if none specified
    if columns is None:
        columns = [
            'tt', 'lat', 'lon', 'alt', 'vn', 've', 'vd', 'roll', 'pitch', 'yaw', # from traj/ins
            'mag_1_uc', 'mag_1_c', 'diurnal', 'igrf', # scalar mags
            'flux_a_x', 'flux_a_y', 'flux_a_z', 'flux_a_t', # vector mags
            'flight', 'line', 'year', 'doy' # metadata
        ]

    # Populate data_to_write dictionary
    # Trajectory data (prefer INS if available, fallback to TRAJ)
    source_traj = xyz_obj.ins if hasattr(xyz_obj, 'ins') and xyz_obj.ins is not None else xyz_obj.traj
    
    # Euler angles might not be directly in Traj, but in INS or calculated
    # For simplicity, assume roll, pitch, yaw are accessible if requested
    # This part might need adjustment based on how Euler angles are stored/derived for XYZ0.traj
    # If using xyz_obj.ins, it has roll, pitch, yaw via dcm2euler from its Cnb.
    # If using xyz_obj.traj, its Cnb can be used.
    
    # Get Euler from source_traj.Cnb
    if source_traj.Cnb is not None and source_traj.Cnb.ndim == 3 and source_traj.Cnb.shape[0] == N:
        euler_angles_rad = dcm2euler(source_traj.Cnb, order='body2nav') # N x 3 (roll, pitch, yaw)
        traj_roll_deg = np.rad2deg(euler_angles_rad[:, 0])
        traj_pitch_deg = np.rad2deg(euler_angles_rad[:, 1])
        traj_yaw_deg = np.rad2deg(euler_angles_rad[:, 2])
    else: # Fallback if Cnb is not as expected
        traj_roll_deg = np.full(N, np.nan)
        traj_pitch_deg = np.full(N, np.nan)
        traj_yaw_deg = np.full(N, np.nan)


    possible_data = {
        'tt': source_traj.tt,
        'lat': np.rad2deg(source_traj.lat), # Convert to degrees for XYZ file
        'lon': np.rad2deg(source_traj.lon), # Convert to degrees for XYZ file
        'alt': source_traj.alt,
        'vn': source_traj.vn,
        've': source_traj.ve,
        'vd': source_traj.vd,
        'roll': traj_roll_deg,
        'pitch': traj_pitch_deg,
        'yaw': traj_yaw_deg,
        'mag_1_uc': xyz_obj.mag_1_uc,
        'mag_1_c': xyz_obj.mag_1_c,
        'diurnal': xyz_obj.diurnal,
        'igrf': xyz_obj.igrf,
        'flux_a_x': xyz_obj.flux_a.x,
        'flux_a_y': xyz_obj.flux_a.y,
        'flux_a_z': xyz_obj.flux_a.z,
        'flux_a_t': xyz_obj.flux_a.t,
        'flight': xyz_obj.flights,
        'line': xyz_obj.lines,
        'year': xyz_obj.years,
        'doy': xyz_obj.doys,
        # Add more direct fields from traj if needed, e.g., fn, fe, fd
        'fn': source_traj.fn,
        'fe': source_traj.fe,
        'fd': source_traj.fd,
    }

    for col_name in columns:
        if col_name in possible_data:
            data_to_write[col_name] = possible_data[col_name]
        else:
            print(f"Warning: Column '{col_name}' not found in XYZ0 data sources, skipping.")

    df = pd.DataFrame(data_to_write)
    df.to_csv(file_path, index=False, na_rep=na_rep, float_format=float_format, header=include_header, sep=delimiter)
    print(f"XYZ data written to {file_path}")


def read_xyz_file(file_path: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Reads a text-based XYZ file into a pandas DataFrame.

    Args:
        file_path: Path to the input XYZ file.
        delimiter: Delimiter used in the file.

    Returns:
        A pandas DataFrame containing the XYZ data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XYZ file not found: {file_path}")
    
    df = pd.read_csv(file_path, sep=delimiter)
    print(f"XYZ data read from {file_path}")
    return df

def create_xyz(
    output_dir: str = ".",
    flight: int = 1,
    line: int = 1,
    save_text_xyz: bool = True,
    save_h5_xyz0: bool = False, # Consistent with create_xyz0's save_h5
    xyz0_params: Optional[Dict[str, Any]] = None,
    xyz_text_columns: Optional[List[str]] = None
) -> Optional[XYZ0]:
    """
    Orchestrator function to create XYZ0 data object and optionally write it
    to a text-based XYZ file and/or an HDF5 file.

    Args:
        output_dir: Directory to save output files.
        flight: Flight number.
        line: Line number.
        save_text_xyz: If True, saves data to a text XYZ file.
        save_h5_xyz0: If True, saves XYZ0 data to an HDF5 file (via create_xyz0).
        xyz0_params: Dictionary of parameters to pass to create_xyz0.
        xyz_text_columns: Specific columns to write to the text XYZ file.

    Returns:
        The created XYZ0 object, or None if creation fails.
    """
    if xyz0_params is None:
        xyz0_params = {}

    # Ensure flight, line, and save_h5 are passed to create_xyz0 if they are relevant
    xyz0_params.setdefault('flight', flight)
    xyz0_params.setdefault('line', line)
    xyz0_params.setdefault('save_h5', save_h5_xyz0) # For HDF5 saving by create_xyz0
    
    if save_h5_xyz0 and 'xyz_h5' not in xyz0_params:
         # Use a default HDF5 filename if saving to HDF5 and not specified
        xyz0_params['xyz_h5'] = os.path.join(output_dir, f"xyz0_data_f{flight}_l{line}.h5")


    xyz_object = create_xyz0(**xyz0_params)

    if xyz_object and save_text_xyz:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        txt_file = xyz_file_name(flight, line, output_dir=output_dir)
        write_xyz(xyz_object, txt_file, columns=xyz_text_columns)
        
    return xyz_object


# --- Placeholder functions from original Python file (not found in Julia's create_XYZ.jl) ---
def get_xyz20(*args, **kwargs):
    """Placeholder for get_xyz20."""
    raise NotImplementedError("get_xyz20 is not yet implemented in this module.")

def get_XYZ(*args, **kwargs):
    """Placeholder for get_XYZ. This might load data from files."""
    raise NotImplementedError("get_XYZ is not yet implemented in this module. Consider using read_xyz_file for text XYZ files.")

def sgl_2020_train(*args, **kwargs):
    """Placeholder for sgl_2020_train."""
    raise NotImplementedError("sgl_2020_train is not yet implemented in this module.")

def sgl_2021_train(*args, **kwargs):
    """Placeholder for sgl_2021_train."""
    raise NotImplementedError("sgl_2021_train is not yet implemented in this module.")