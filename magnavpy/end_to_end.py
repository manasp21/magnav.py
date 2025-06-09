'''
Created on Jun 9, 2025

@author: Admin
'''
# End-to-End Machine Learning for Aeromagnetic Compensation Notebook
# This notebook provides a background on machine learning with an application toward aeromagnetic compensation

# Machine learning projects entail much more than just training neural networks. The steps of an end-to-end machine learning project:
# 1. Look at the big picture
# 2. Get the data
# 3. Discover and visualize the data to gain insights
# 4. Prepare the data for machine learning algorithms
# 5. Select a model and train it
# 6. Fine-tune the model
# 7. Present the solution
# 8. Launch, monitor, and maintain the system

# ## 0. Import packages and DataFrames

# The DataFrames listed below provide useful information about the flight data collected by Sander Geophysics Ltd. (SGL) and magnetic anomaly maps.

# Dataframe  | Description
# :--------- | :----------
# `df_map`   | map files relevant for SGL flights
# `df_cal`   | SGL calibration flight lines
# `df_flight`| SGL flight files
# `df_all`   | all flight lines
# `df_nav`   | all *navigation-capable* flight lines
# `df_event` | pilot-recorded in-flight events

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import seed

sys.path.append(r"C:\Users\Admin\eclipse-workspace\magnav.py")
import magnavpy.magnav as MagNav

"""
imported julia code from ai
"""

import h5py
import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, Union, Optional
import os
from pathlib import Path

# Constants
g_earth = 9.80665  # Earth's gravity in m/s²

@dataclass
class MagV:
    """Magnetometer vector data"""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray

@dataclass
class Traj:
    """Trajectory data structure"""
    N: int
    dt: float
    tt: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    utm_z: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    fn: np.ndarray
    fe: np.ndarray
    fd: np.ndarray
    Cnb: np.ndarray

@dataclass
class INS:
    """INS data structure"""
    N: int
    dt: float
    tt: np.ndarray
    ins_lat: np.ndarray
    ins_lon: np.ndarray
    ins_alt: np.ndarray
    ins_vn: np.ndarray
    ins_ve: np.ndarray
    ins_vd: np.ndarray
    ins_fn: np.ndarray
    ins_fe: np.ndarray
    ins_fd: np.ndarray
    ins_Cnb: np.ndarray
    ins_P: np.ndarray

@dataclass
class XYZ20:
    """XYZ20 data structure containing all flight data"""
    info: Any
    traj: Traj
    ins: INS
    flux_a: MagV
    flux_b: MagV
    flux_c: MagV
    flux_d: MagV
    flight: np.ndarray
    line: np.ndarray
    year: np.ndarray
    doy: np.ndarray
    utm_x: np.ndarray
    utm_y: np.ndarray
    utm_z: np.ndarray
    msl: np.ndarray
    baro: np.ndarray
    diurnal: np.ndarray
    igrf: np.ndarray
    mag_1_c: np.ndarray
    mag_1_lag: np.ndarray
    mag_1_dc: np.ndarray
    mag_1_igrf: np.ndarray
    mag_1_uc: np.ndarray
    mag_2_uc: np.ndarray
    mag_3_uc: np.ndarray
    mag_4_uc: np.ndarray
    mag_5_uc: np.ndarray
    mag_6_uc: np.ndarray
    ogs_mag: np.ndarray
    ogs_alt: np.ndarray
    ins_wander: np.ndarray
    ins_roll: np.ndarray
    ins_pitch: np.ndarray
    ins_yaw: np.ndarray
    roll_rate: np.ndarray
    pitch_rate: np.ndarray
    yaw_rate: np.ndarray
    ins_acc_x: np.ndarray
    ins_acc_y: np.ndarray
    ins_acc_z: np.ndarray
    lgtl_acc: np.ndarray
    ltrl_acc: np.ndarray
    nrml_acc: np.ndarray
    pitot_p: np.ndarray
    static_p: np.ndarray
    total_p: np.ndarray
    cur_com_1: np.ndarray
    cur_ac_hi: np.ndarray
    cur_ac_lo: np.ndarray
    cur_tank: np.ndarray
    cur_flap: np.ndarray
    cur_strb: np.ndarray
    cur_srvo_o: np.ndarray
    cur_srvo_m: np.ndarray
    cur_srvo_i: np.ndarray
    cur_heat: np.ndarray
    cur_acpwr: np.ndarray
    cur_outpwr: np.ndarray
    cur_bat_1: np.ndarray
    cur_bat_2: np.ndarray
    vol_acpwr: np.ndarray
    vol_outpwr: np.ndarray
    vol_bat_1: np.ndarray
    vol_bat_2: np.ndarray
    vol_res_p: np.ndarray
    vol_res_n: np.ndarray
    vol_back_p: np.ndarray
    vol_back_n: np.ndarray
    vol_gyro_1: np.ndarray
    vol_gyro_2: np.ndarray
    vol_acc_p: np.ndarray
    vol_acc_n: np.ndarray
    vol_block: np.ndarray
    vol_back: np.ndarray
    vol_srvo: np.ndarray
    vol_cabt: np.ndarray
    vol_fan: np.ndarray
    aux_1: np.ndarray
    aux_2: np.ndarray
    aux_3: np.ndarray

def add_extension(filename: str, extension: str) -> str:
    """Add extension to filename if not already present"""
    if not filename.endswith(extension):
        return filename + extension
    return filename

def xyz_fields(fields_type: str) -> list:
    """Return list of field names for XYZ20 data"""
    # This represents the :fields20 symbol from Julia
    if fields_type == "fields20":
        return [
            'flight', 'line', 'year', 'doy', 'tt', 'utm_x', 'utm_y', 'utm_z', 'msl',
            'lat', 'lon', 'baro', 'diurnal', 'mag_1_c', 'mag_1_lag', 'mag_1_dc',
            'mag_1_igrf', 'mag_1_uc', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc',
            'mag_6_uc', 'ogs_mag', 'ogs_alt', 'ins_wander', 'ins_lat', 'ins_lon',
            'ins_alt', 'ins_roll', 'ins_pitch', 'ins_yaw', 'ins_vn', 'ins_vw', 'ins_vu',
            'roll_rate', 'pitch_rate', 'yaw_rate', 'ins_acc_x', 'ins_acc_y', 'ins_acc_z',
            'lgtl_acc', 'ltrl_acc', 'nrml_acc', 'pitot_p', 'static_p', 'total_p',
            'cur_com_1', 'cur_ac_hi', 'cur_ac_lo', 'cur_tank', 'cur_flap', 'cur_strb',
            'cur_srvo_o', 'cur_srvo_m', 'cur_srvo_i', 'cur_heat', 'cur_acpwr',
            'cur_outpwr', 'cur_bat_1', 'cur_bat_2', 'vol_acpwr', 'vol_outpwr',
            'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n', 'vol_back_p',
            'vol_back_n', 'vol_gyro_1', 'vol_gyro_2', 'vol_acc_p', 'vol_acc_n',
            'vol_block', 'vol_back', 'vol_srvo', 'vol_cabt', 'vol_fan',
            'flux_a_x', 'flux_a_y', 'flux_a_z', 'flux_a_t',
            'flux_b_x', 'flux_b_y', 'flux_b_z', 'flux_b_t',
            'flux_c_x', 'flux_c_y', 'flux_c_z', 'flux_c_t',
            'flux_d_x', 'flux_d_y', 'flux_d_z', 'flux_d_t'
        ]
    return []

def read_check(h5_file: h5py.File, field: str, N_or_default: Union[int, Any], silent: bool = False) -> np.ndarray:
    """Read field from HDF5 file with error checking"""
    try:
        if field in h5_file.keys():
            data = h5_file[field][:]
            if isinstance(N_or_default, int):
                # If expecting array of length N, pad or truncate as needed
                N = N_or_default
                if len(data) < N:
                    # Pad with last value or zeros
                    if len(data) > 0:
                        padded = np.full(N, data[-1] if np.isscalar(data[-1]) else 0)
                        padded[:len(data)] = data
                        return padded
                    else:
                        return np.zeros(N)
                elif len(data) > N:
                    return data[:N]
                else:
                    return data
            else:
                # Return scalar or default value
                return data if len(data) > 0 else N_or_default
        else:
            if not silent:
                logging.warning(f"Field '{field}' not found in HDF5 file")
            if isinstance(N_or_default, int):
                return np.zeros(N_or_default)
            else:
                return N_or_default
    except Exception as e:
        if not silent:
            logging.warning(f"Error reading field '{field}': {e}")
        if isinstance(N_or_default, int):
            return np.zeros(N_or_default)
        else:
            return N_or_default

def fdm(x: np.ndarray) -> np.ndarray:
    """Finite difference method (forward differences)"""
    if len(x) <= 1:
        return np.zeros_like(x)
    
    result = np.zeros_like(x)
    # Forward difference for all but last point
    result[:-1] = x[1:] - x[:-1]
    # Backward difference for last point
    result[-1] = result[-2] if len(x) > 1 else 0
    return result

def deg2rad(degrees: np.ndarray) -> np.ndarray:
    """Convert degrees to radians"""
    return np.deg2rad(degrees)

def euler2dcm(roll: Union[float, np.ndarray], pitch: Union[float, np.ndarray], 
              yaw: Union[float, np.ndarray], convention: str) -> np.ndarray:
    """Convert Euler angles to direction cosine matrix"""
    # Handle scalar inputs
    if np.isscalar(roll):
        roll = np.array([roll])
        pitch = np.array([pitch])
        yaw = np.array([yaw])
        scalar_input = True
    else:
        scalar_input = False
    
    N = len(roll)
    
    if convention == "body2nav":
        # Body to navigation frame transformation
        dcm = np.zeros((3, 3, N))
        
        for i in range(N):
            r, p, y = roll[i], pitch[i], yaw[i]
            
            # Rotation matrices
            R_x = np.array([[1, 0, 0],
                           [0, np.cos(r), -np.sin(r)],
                           [0, np.sin(r), np.cos(r)]])
            
            R_y = np.array([[np.cos(p), 0, np.sin(p)],
                           [0, 1, 0],
                           [-np.sin(p), 0, np.cos(p)]])
            
            R_z = np.array([[np.cos(y), -np.sin(y), 0],
                           [np.sin(y), np.cos(y), 0],
                           [0, 0, 1]])
            
            # Combined rotation (Z-Y-X order for body to nav)
            dcm[:, :, i] = R_z @ R_y @ R_x
    
    if scalar_input and N == 1:
        return dcm[:, :, 0]
    else:
        return dcm


import sys
from typing import BinaryIO


def get_XYZ20(xyz_h5: str, 
              info: str = None,
              tt_sort: bool = True,
              silent: bool = False) -> XYZ20:
    """
    Load XYZ20 data from HDF5 file
    
    Parameters:
    xyz_h5: path to HDF5 file
    info: information string (defaults to filename)
    tt_sort: whether to sort by time
    silent: suppress info messages
    """
    
    if info is None:
        info = Path(xyz_h5).name
        xyz_h5 = add_extension(xyz_h5, ".h5")
    else:
        xyz_h5 = info.loc[info['flight']==xyz_h5]['xyz_file'].values[0]
    fields = "fields20"
    
    if not silent:
        logging.info(f"reading in XYZ20 data: {xyz_h5}")
    
    # Open HDF5 file for reading
    with h5py.File(xyz_h5, "r") as xyz:
        # Find maximum length across all datasets
        print('xyz of h5', xyz)
        N = max([xyz[k].shape[0] for k in xyz.keys() if xyz[k].shape != ()]) #max([len(read(xyz, k)) for k in xyz.keys()])
        d = {}
        
        # Sort by time if requested
        if tt_sort:
            tt_data = read_check(xyz, 'tt', N, silent)
            ind = np.argsort(tt_data)
        else:
            ind = np.arange(N)
        
        # Read all fields
        for field in xyz_fields(fields):
            if field != 'ignore':
                d[field] = read_check(xyz, field, N, silent)[ind]
        
        # Read info field
        field = 'info'
        info_data = read_check(xyz, field, info)
        d[field] = info_data
        
        # Read auxiliary fields
        for field in ['aux_1', 'aux_2', 'aux_3']:
            d[field] = read_check(xyz, field, N, True)
    
    # Calculate time step
    dt = round(d['tt'][1] - d['tt'][0], 9) if N > 1 else 0.1
    
    # Convert degrees to radians for angular measurements
    for field in ['lat', 'lon', 'ins_roll', 'ins_pitch', 'ins_yaw',
                  'roll_rate', 'pitch_rate', 'yaw_rate']:
        d[field] = deg2rad(d[field])
    
    # Calculate IGRF difference for convenience
    d['igrf'] = d['mag_1_dc'] - d['mag_1_igrf']
    
    # Calculate trajectory velocities & specific forces from position
    d['vn'] = fdm(d['utm_y']) / dt
    d['ve'] = fdm(d['utm_x']) / dt
    d['vd'] = -fdm(d['utm_z']) / dt
    d['fn'] = fdm(d['vn']) / dt
    d['fe'] = fdm(d['ve']) / dt
    d['fd'] = fdm(d['vd']) / dt - g_earth
    
    # Direction cosine matrix (body to navigation) from roll, pitch, yaw
    d['Cnb'] = np.zeros((3, 3, N))  # unknown
    d['ins_Cnb'] = euler2dcm(d['ins_roll'], d['ins_pitch'], d['ins_yaw'], 'body2nav')
    d['ins_P'] = np.zeros((1, 1, N))  # unknown
    
    # INS velocities in NED direction
    d['ins_ve'] = -d['ins_vw']
    d['ins_vd'] = -d['ins_vu']
    
    # INS specific forces from measurements, rotated by wander angle (CW for NED)
    ins_f = np.zeros((N, 3))
    for i in range(N):
        wander_dcm = euler2dcm(0, 0, -d['ins_wander'][i], 'body2nav')
        acc_vector = np.array([d['ins_acc_x'][i], -d['ins_acc_y'][i], -d['ins_acc_z'][i]])
        ins_f[i, :] = wander_dcm @ acc_vector
    
    d['ins_fn'] = ins_f[:, 0]
    d['ins_fe'] = ins_f[:, 1] 
    d['ins_fd'] = ins_f[:, 2]
    
    # Alternative INS specific forces from finite differences (commented out)
    # d['ins_fn'] = fdm(-d['ins_vn']) / dt
    # d['ins_fe'] = fdm(-d['ins_ve']) / dt
    # d['ins_fd'] = fdm(-d['ins_vd']) / dt - g_earth
    
    return XYZ20(
        d['info'],
        Traj(N, dt, d['tt'], d['lat'], d['lon'], d['utm_z'], d['vn'],
             d['ve'], d['vd'], d['fn'], d['fe'], d['fd'], d['Cnb']),
        INS(N, dt, d['tt'], d['ins_lat'], d['ins_lon'], d['ins_alt'],
            d['ins_vn'], d['ins_ve'], d['ins_vd'], d['ins_fn'],
            d['ins_fe'], d['ins_fd'], d['ins_Cnb'], d['ins_P']),
        MagV(d['flux_a_x'], d['flux_a_y'], d['flux_a_z'], d['flux_a_t']),
        MagV(d['flux_b_x'], d['flux_b_y'], d['flux_b_z'], d['flux_b_t']),
        MagV(d['flux_c_x'], d['flux_c_y'], d['flux_c_z'], d['flux_c_t']),
        MagV(d['flux_d_x'], d['flux_d_y'], d['flux_d_z'], d['flux_d_t']),
        d['flight'], d['line'], d['year'], d['doy'],
        d['utm_x'], d['utm_y'], d['utm_z'], d['msl'],
        d['baro'], d['diurnal'], d['igrf'], d['mag_1_c'],
        d['mag_1_lag'], d['mag_1_dc'], d['mag_1_igrf'], d['mag_1_uc'],
        d['mag_2_uc'], d['mag_3_uc'], d['mag_4_uc'], d['mag_5_uc'],
        d['mag_6_uc'], d['ogs_mag'], d['ogs_alt'], d['ins_wander'],
        d['ins_roll'], d['ins_pitch'], d['ins_yaw'], d['roll_rate'],
        d['pitch_rate'], d['yaw_rate'], d['ins_acc_x'], d['ins_acc_y'],
        d['ins_acc_z'], d['lgtl_acc'], d['ltrl_acc'], d['nrml_acc'],
        d['pitot_p'], d['static_p'], d['total_p'], d['cur_com_1'],
        d['cur_ac_hi'], d['cur_ac_lo'], d['cur_tank'], d['cur_flap'],
        d['cur_strb'], d['cur_srvo_o'], d['cur_srvo_m'], d['cur_srvo_i'],
        d['cur_heat'], d['cur_acpwr'], d['cur_outpwr'], d['cur_bat_1'],
        d['cur_bat_2'], d['vol_acpwr'], d['vol_outpwr'], d['vol_bat_1'],
        d['vol_bat_2'], d['vol_res_p'], d['vol_res_n'], d['vol_back_p'],
        d['vol_back_n'], d['vol_gyro_1'], d['vol_gyro_2'], d['vol_acc_p'],
        d['vol_acc_n'], d['vol_block'], d['vol_back'], d['vol_srvo'],
        d['vol_cabt'], d['vol_fan'], d['aux_1'], d['aux_2'],
        d['aux_3']
    )



"""
end imported code from ai
"""

seed(33)  # for reproducibility



"""
setup dataframes begin
"""
import pandas as pd
from pathlib import Path

# setup DataFrames for use with examples

## SGL calibration flight lines
df_cal = pd.read_csv("dataframes/df_cal.csv")
df_cal['flight'] = df_cal['flight'].apply(lambda x: x if isinstance(x, str) else str(x))
df_cal['map_name'] = df_cal['map_name'].apply(lambda x: x if isinstance(x, str) else str(x))

## SGL flight data files
df_flight = pd.read_csv("dataframes/df_flight.csv")
df_flight['flight'] = df_flight['flight'].apply(lambda x: x if isinstance(x, str) else str(x))
df_flight['xyz_type'] = df_flight['xyz_type'].apply(lambda x: x if isinstance(x, str) else str(x))
df_flight['xyz_file'] = df_flight['xyz_file'].astype(str)

# to store/load the data locally uncomment the for loop below
# and make sure the file locations match up with the xyz_file column
for i, flight in enumerate(df_flight['flight']):
    if df_flight.at[i, 'xyz_type'] == 'XYZ20':
        df_flight.at[i, 'xyz_file'] = str(Path(MagNav.sgl_2020_train_path()) / f"{flight}_train.h5")
    if df_flight.at[i, 'xyz_type'] == 'XYZ21':
        df_flight.at[i, 'xyz_file'] = str(Path(MagNav.sgl_2021_train_path()) / f"{flight}_train.h5")

## map data files (associated with SGL flights)
df_map = pd.read_csv("dataframes/df_map.csv")
df_map['map_name'] = df_map['map_name'].apply(lambda x: x if isinstance(x, str) else str(x))
df_map['map_type'] = df_map['map_type'].apply(lambda x: x if isinstance(x, str) else str(x))
df_map['map_file'] = df_map['map_file'].astype(str)

# to store/load the maps locally, uncomment the for loop below
# and make sure the file locations match up with the map_file column
for i, map_name in enumerate(df_map['map_name']):
    df_map.at[i, 'map_file'] = str(Path(MagNav.ottawa_area_maps_path()) / f"{map_name}.h5")

## all flight lines
df_all = pd.read_csv("dataframes/df_all.csv")
df_all['flight'] = df_all['flight'].apply(lambda x: x if isinstance(x, str) else str(x))

## navigation-capable flight lines
df_nav = pd.read_csv("dataframes/df_nav.csv")
df_nav['flight'] = df_nav['flight'].apply(lambda x: x if isinstance(x, str) else str(x))
df_nav['map_name'] = df_nav['map_name'].apply(lambda x: x if isinstance(x, str) else str(x))
df_nav['map_type'] = df_nav['map_type'].apply(lambda x: x if isinstance(x, str) else str(x))

## in-flight events
df_event = pd.read_csv("dataframes/df_event.csv")
df_event['flight'] = df_event['flight'].apply(lambda x: x if isinstance(x, str) else str(x))

## all flight lines for flights 3-6, except 1003.05
# 1003.05 not used because of ~1.5 min data anomaly in mag_4_uc & mag_5_uc
# between times 59641 & 59737, causing ~25x NN comp errors
flts = ['Flt1003', 'Flt1004', 'Flt1005', 'Flt1006']
df_all_3456 = df_all[(df_all['flight'].isin(flts)) & (df_all['line'] != 1003.05)].copy()

## navigation-capable flight lines for flights 3-6
flts = ['Flt1003', 'Flt1004', 'Flt1005', 'Flt1006']
df_nav_3456 = df_nav[df_nav['flight'].isin(flts)].copy()
"""
setup dataframe ends
"""


# Include dataframes setup (equivalent to dataframes_setup.jl)
# This would be implemented separately


# ## 1. Look at the big picture

# Airborne magnetic anomaly navigation (MagNav) is an emerging technology that can be used for aerial navigation in case GPS is not available. 
# MagNav uses maps of variations in the magnetic field originating from the crust of the Earth. A navigation algorithm compares onboard 
# measurements of the magnetic field with a magnetic anomaly map, which (combined with inertial measurements), produces an estimate of the aircraft position.

# But there's a catch! The magnetometers on the aircraft measure the total magnetic field, which is comprised of multiple magnetic fields 
# arising from not only the crust, but also the Earth's core, diurnal variations, and the aircraft itself. In order to use the crustal anomaly 
# field for MagNav, the other contributions to the total magnetic field must be removed. Magnetic models and base station measurements suffice 
# to remove the core and diurnal fields, which leaves the aircraft field to remove. Unlike the other contributions, the aircraft field is difficult 
# to isolate. Aeromagnetic compensation is used to identify and remove the aircraft field.

# The standard approach for aeromagnetic compensation, known as the Tolles-Lawson model, uses a physics-based linear model of the aircraft field 
# combined with data taken during a specific flight pattern designed to maximize the contribution arising from the aircraft. Tolles-Lawson works 
# well when the aircraft field is small compared to the Earth's core field, for example when the magnetometer is located on a boom (stinger) behind 
# the aircraft (Mag 1) but falls short for magnetometers located in the cabin (Mags 2-5).

# The goal here is to perform aeromagnetic compensation using the in-cabin sensors. In addition to the scalar magnetometers (Mags 2-5), which detect 
# the magnitude of the total magnetic field, there are measurements from vector magnetometers (Flux A-D), which detect the three cartesian components 
# of the total magnetic field. There are also measurements available from additional sensors, notably current sensors. Performance is measured using 
# the standard deviation of the error between the predicted values and the professionally-compensated stinger magnetometer.

# ## 2. Get the data

# For Tolles-Lawson and testing, we select Flight 1006 and gather the data structure which contains the GPS-based trajectory, inertial navigation system, 
# flight information, magnetometer readings, and auxilliary sensor data.



import pandas as pd
from typing import Union

def get_XYZ(flight: str, df_flight: pd.DataFrame,
            tt_sort: bool = True,
            reorient_vec: bool = False,
            silent: bool = False):
    """
    Get XYZ data for a specific flight from a DataFrame.
    
    Parameters:
    -----------
    flight : str
        Flight identifier
    df_flight : pandas.DataFrame
        DataFrame containing flight data
    tt_sort : bool, optional
        Whether to sort by time (default: True)
    reorient_vec : bool, optional
        Whether to reorient vectors (default: False)
    silent : bool, optional
        Whether to suppress output (default: False)
    
    Returns:
    --------
    xyz : object
        XYZ data object
    """
    
    # Find first index where flight column matches the input flight
    flight_symbols = df_flight['flight'].astype(str)
    ind = None
    for i, f in enumerate(flight_symbols):
        if f == flight:
            ind = i
            break
    
    if ind is None:
        raise ValueError(f"Flight '{flight}' not found in DataFrame")
    
    xyz_file = str(df_flight.iloc[ind]['xyz_file'])
    xyz_type = str(df_flight.iloc[ind]['xyz_type'])
    
    # Call appropriate get_XYZ function based on xyz_type
    if xyz_type == 'XYZ0':
        xyz = get_XYZ0(xyz_file, tt_sort=tt_sort, silent=silent)
    elif xyz_type == 'XYZ1':
        xyz = get_XYZ1(xyz_file, tt_sort=tt_sort, silent=silent)
    elif xyz_type == 'XYZ20':
        xyz = get_XYZ20(xyz_file, tt_sort=tt_sort, silent=silent)
    elif xyz_type == 'XYZ21':
        xyz = get_XYZ21(xyz_file, tt_sort=tt_sort, silent=silent)
    else:
        raise ValueError(f"{xyz_type} xyz_type not defined")
    
    # Optionally reorient vectors
    if reorient_vec:
        xyz_reorient_vec_(xyz)
    
    return xyz



flight = 'Flt1006'  # select flight, full list in df_flight

xyz = get_XYZ(flight, df_flight)  # load flight data (equivalent Julia function)

# The `xyz` flight data struct contains all the flight data from the HDF5 file, but Boolean indices can be used as a mask to return specific portion(s) of flight data.

print(type(xyz))

# Print fieldnames (equivalent to fieldnames(MagNav.XYZ20) in Julia)
# In Python, we would use dir() or list the attributes if it's a class
# For demonstration, showing the equivalent fields:
fields = ['info', 'traj', 'ins', 'flux_a', 'flux_b', 'flux_c', 'flux_d', 'flight', 'line', 'year', 
          'doy', 'utm_x', 'utm_y', 'utm_z', 'msl', 'baro', 'diurnal', 'igrf', 'mag_1_c', 'mag_1_lag', 
          'mag_1_dc', 'mag_1_igrf', 'mag_1_uc', 'mag_2_uc', 'mag_3_uc', 'mag_4_uc', 'mag_5_uc', 
          'mag_6_uc', 'ogs_mag', 'ogs_alt', 'ins_wander', 'ins_roll', 'ins_pitch', 'ins_yaw', 
          'roll_rate', 'pitch_rate', 'yaw_rate', 'ins_acc_x', 'ins_acc_y', 'ins_acc_z', 'lgtl_acc', 
          'ltrl_acc', 'nrml_acc', 'pitot_p', 'static_p', 'total_p', 'cur_com_1', 'cur_ac_hi', 
          'cur_ac_lo', 'cur_tank', 'cur_flap', 'cur_strb', 'cur_srvo_o', 'cur_srvo_m', 'cur_srvo_i', 
          'cur_heat', 'cur_acpwr', 'cur_outpwr', 'cur_bat_1', 'cur_bat_2', 'vol_acpwr', 'vol_outpwr', 
          'vol_bat_1', 'vol_bat_2', 'vol_res_p', 'vol_res_n', 'vol_back_p', 'vol_back_n', 'vol_gyro_1', 
          'vol_gyro_2', 'vol_acc_p', 'vol_acc_n', 'vol_block', 'vol_back', 'vol_srvo', 'vol_cabt', 
          'vol_fan', 'aux_1', 'aux_2', 'aux_3']
print(fields)

# For the Tolles-Lawson calibration, flight line 1006.04 is selected, which occurred at a higher altitude. 
# This is the first calibration box of this flight line. `TL_ind` holds the Boolean indices (mask) just for this portion of the calibration flight line.



import numpy as np
import pandas as pd
import copy
import logging
from typing import Union, Tuple, List, Any, Optional

def get_ind_vectors(tt: List[float], line: List[int], 
                   ind: Optional[Union[List[bool], np.ndarray]] = None,
                   lines: Tuple = (),
                   tt_lim: Tuple = (),
                   splits: Tuple[float, ...] = (1,)) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Get BitVector of indices for further analysis from specified indices (subset),
    lines, and/or time range. Any or all of these may be used. Defaults to use all
    indices, lines, and times.

    Arguments:
    - tt:     time [s]
    - line:   line number(s)
    - ind:    (optional) selected data indices
    - lines:  (optional) selected line number(s)
    - tt_lim: (optional) end time limit or length-2 start & end time limits (inclusive) [s]
    - splits: (optional) data splits, must sum to 1

    Returns:
    - ind: BitVector (or tuple of BitVector) of selected data indices
    """
    
    assert abs(sum(splits) - 1) < 1e-10, f"sum of splits = {sum(splits)} ≠ 1"
    assert len(tt_lim) <= 2, f"length of tt_lim = {len(tt_lim)} > 2"
    assert len(splits) <= 3, f"number of splits = {len(splits)} > 3"

    if ind is None:
        ind = np.ones(len(tt), dtype=bool)
    
    tt = np.array(tt)
    line = np.array(line)
    
    if isinstance(ind, (list, np.ndarray)) and len(ind) > 0 and isinstance(ind[0], bool):
        ind_ = copy.deepcopy(ind)
        ind_ = np.array(ind_, dtype=bool)
    else:
        ind_ = np.isin(np.arange(len(tt)), ind)

    N = len(tt)

    if len(lines) > 0:
        ind = np.zeros(N, dtype=bool)
        for l in lines:
            ind = ind | (line == l)
    else:
        ind = np.ones(N, dtype=bool)

    if len(tt_lim) == 1:
        ind = ind & (tt <= min(tt_lim[0]))
    elif len(tt_lim) == 2:
        print('ind=', ind)
        print('tt_lim', tt_lim)
        print('tt', tt)
        ind = ind & (tt >= tt_lim[0]) & (tt <= tt_lim[1])

    # ind = ind & ind_

    if np.sum(ind) == 0:
        logging.info("ind contains all falses")

    if len(splits) == 1:
        return ind
    elif len(splits) == 2:
        ind1 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * splits[0])))
        ind2 = ind & ~ind1
        return (ind1, ind2)
    elif len(splits) == 3:
        ind1 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * splits[0])))
        ind2 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * (splits[0] + splits[1])))) & ~ind1
        ind3 = ind & ~(ind1 | ind2)
        return (ind1, ind2, ind3)


def get_ind_xyz(xyz, 
               ind: Optional[np.ndarray] = None,
               lines: Tuple = (),
               tt_lim: Tuple = (),
               splits: Tuple[float, ...] = (1,)) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Get BitVector of indices for further analysis from specified indices (subset),
    lines, and/or time range. Any or all of these may be used. Defaults to use all
    indices, lines, and times.

    Arguments:
    - xyz:    XYZ flight data struct
    - ind:    (optional) selected data indices
    - lines:  (optional) selected line number(s)
    - tt_lim: (optional) end time limit or length-2 start & end time limits (inclusive) [s]
    - splits: (optional) data splits, must sum to 1

    Returns:
    - ind: BitVector (or tuple of BitVector) of selected data indices
    """
    if ind is None:
        ind = np.ones(xyz.traj.N, dtype=bool)
    
    # Check if xyz has line attribute
    if hasattr(xyz, 'line'):
        line_ = xyz.line
    else:
        line_ = np.ones(len(xyz.traj.tt[ind]))
    
    return get_ind_vectors(xyz.traj.tt, line_,
                          ind=ind,
                          lines=lines,
                          tt_lim=tt_lim,
                          splits=splits)


def get_ind_xyz_line_df(xyz, line: float, df_line: pd.DataFrame,
                       splits: Tuple[float, ...] = (1,),
                       l_window: int = -1) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Get BitVector of indices for further analysis via DataFrame lookup.

    Arguments:
    - xyz:     XYZ flight data struct
    - line:    line number
    - df_line: lookup table (DataFrame) of lines
    |**Field**|**Type**|**Description**
    |:--|:--|:--
    `flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
    `line`     |`Real`  | line number, i.e., segments within `flight`
    `t_start`  |`Real`  | start time of `line` to use [s]
    `t_end`    |`Real`  | end   time of `line` to use [s]
    `map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`
    - splits:   (optional) data splits, must sum to 1
    - l_window: (optional) trim data by N % l_window, -1 to ignore

    Returns:
    - ind: BitVector (or tuple of BitVector) of selected data indices
    """
    
    line_mask = df_line['line'] == line
    tt_lim = [df_line.loc[line_mask, 't_start'].iloc[0],
              df_line.loc[line_mask, 't_end'].iloc[-1]]
    
    # Check if xyz has line attribute
    if hasattr(xyz, 'line'):
        line_ = xyz.line
    else:
        # Note: this creates a dummy ind for the ones() call, but it's not used properly in original
        # Following the original logic exactly
        ind_dummy = np.ones(xyz.traj.N, dtype=bool)  # This matches the original ind reference
        line_ = np.ones(len(xyz.traj.tt[ind_dummy]))
    
    inds = get_ind_vectors(xyz.traj.tt, line_,
                          lines=[line],
                          tt_lim=tt_lim,
                          splits=splits)

    if l_window > 0:
        if isinstance(inds, tuple):
            for ind in inds:
                N_trim = np.sum(xyz.traj.lat[ind]) % l_window
                for _ in range(N_trim):
                    last_true_idx = np.where(ind == 1)[0][-1] if np.any(ind == 1) else None
                    if last_true_idx is not None:
                        ind[last_true_idx] = 0
        else:
            N_trim = np.sum(inds) % l_window
            for _ in range(N_trim):
                last_true_idx = np.where(inds == 1)[0][-1] if np.any(inds == 1) else None
                if last_true_idx is not None:
                    inds[last_true_idx] = 0

    return inds


def get_ind_xyz_lines_df(xyz, lines, df_line: pd.DataFrame,
                        splits: Tuple[float, ...] = (1,),
                        l_window: int = -1) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Get BitVector of selected data indices for further analysis via DataFrame lookup.

    Arguments:
    - xyz:     XYZ flight data struct
    - lines:   selected line number(s)
    - df_line: lookup table (DataFrame) of lines
    |**Field**|**Type**|**Description**
    |:--|:--|:--
    `flight`   |`Symbol`| flight name (e.g., `:Flt1001`)
    `line`     |`Real`  | line number, i.e., segments within `flight`
    `t_start`  |`Real`  | start time of `line` to use [s]
    `t_end`    |`Real`  | end   time of `line` to use [s]
    `map_name` |`Symbol`| (optional) name of magnetic anomaly map relevant to `line`
    - splits:   (optional) data splits, must sum to 1
    - l_window: (optional) trim data by N % l_window, -1 to ignore

    Returns:
    - ind: BitVector (or tuple of BitVector) of selected data indices
    """
    
    assert abs(sum(splits) - 1) < 1e-10, f"sum of splits = {sum(splits)} ≠ 1"
    assert len(splits) <= 3, f"number of splits = {len(splits)} > 3"

    ind = np.zeros(xyz.traj.N, dtype=bool)
    for line in lines:
        if line in df_line['line'].values:
            line_ind = get_ind_xyz_line_df(xyz, line, df_line, l_window=l_window)
            ind = ind | line_ind

    if len(splits) == 1:
        return ind
    elif len(splits) == 2:
        ind1 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * splits[0])))
        ind2 = ind & ~ind1
        return (ind1, ind2)
    elif len(splits) == 3:
        ind1 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * splits[0])))
        ind2 = ind & (np.cumsum(ind) <= int(round(np.sum(ind) * (splits[0] + splits[1])))) & ~ind1
        ind3 = ind & ~(ind1 | ind2)
        return (ind1, ind2, ind3)



TL_i = 6  # select first calibration box of 1006.04
TL_ind = get_ind_xyz(xyz, tt_lim=[df_cal.t_start[TL_i], df_cal.t_end[TL_i]])  # equivalent Julia function

# Here `df_all` is filtered into `df_options` to ensure that the selected flight line(s) for testing correspond with the selected flight (`:Flt1006`).

df_options = df_all[df_all.flight == flight]
print(df_options)

# For testing, we use Boolean indices (mask) corresponding to flight line 1006.08 in `df_options`.

line = 1006.08  # select flight line (row) from df_options
ind = get_ind_xyz_line_df(xyz, line, df_options)  # get Boolean indices

# For training, we select all available flight data from Flights 1003-1006 into `lines_train`, except the held-out flight `line`.

flts = ['Flt1003', 'Flt1004', 'Flt1005', 'Flt1006']  # select flights for training
df_train = df_all[(df_all.flight.isin(flts)) & (df_all.line != line)]  # use all flight data except held-out line
lines_train = df_train.line  # training lines
print(lines_train)

# ## 3. Discover and visualize the data to gain insights

# As noted in the datasheet, the full 2020 SGL training dataset has 753573 total instances (time steps) sampled at 10 Hz, about 21 hours in flight time spread across 6 flights. 
# This notebook looks at Flight 1006 in more detail for testing, which has 108318 instances, or about 3 hours of flight. The held-out flight `line` subset of Flight 1006 has 8391 instances, or about 14 minutes of flight.

# To get an idea of the magnetometer data, we can call some utility functions for plotting.

# Note that these are filtered using the `ind` Boolean indices corresponding to the held-out flight `line`.



def field_check(s, *args):
    """
    Internal helper function with multiple signatures for field checking.
    
    Signatures:
    - field_check(s, t): Find data fields of type t in struct s
    - field_check(s, field): Check if field exists in struct s  
    - field_check(s, field, t): Check if field exists and is of type t
    """
    
    def _get_field_names(obj):
        """Get field names of an object, similar to Julia's fieldnames()"""
        if hasattr(obj, '__dict__'):
            return list(vars(obj).keys())
        elif hasattr(obj, '__slots__'):
            return list(obj.__slots__)
        else:
            # For other objects, get non-callable, non-private attributes
            return [attr for attr in dir(obj) 
                    if not callable(getattr(obj, attr, None)) and not attr.startswith('_')]
    
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, type):
            # field_check(s, t::Union{DataType,UnionAll})
            """
            Internal helper function to find data fields of a specified type in given struct.

            **Arguments:**
            - `s`: struct
            - `t`: type

            **Returns:**
            - `fields`: data fields of type `t` in struct `s`
            """
            t = arg
            fields = _get_field_names(s)
            # Find indices where field values are of type t, then return corresponding field names
            field_type_checks = [isinstance(getattr(s, f), t) for f in fields]
            indices = [i for i, check in enumerate(field_type_checks) if check]
            return [fields[i] for i in indices]
        else:
            # field_check(s, field::Symbol)
            """
            Internal helper function to check if a specified data field is in a given struct.

            **Arguments:**
            - `s`:     struct
            - `field`: data field

            **Returns:**
            - `AssertionError` if `field` is not in struct `s`
            """
            field = arg
            t = type(s)
            assert hasattr(s, field), f"{field} field not in {t} type"
    elif len(args) == 2:
        # field_check(s, field::Symbol, t::Union{DataType,UnionAll})
        """
        Internal helper function to check if a specified data field is in a given
        struct and of a given type.

        **Arguments:**
        - `s`:     struct
        - `field`: data field
        - `t`:     type

        **Returns:**
        - `AssertionError` if `field` is not in struct `s` or not type `t`
        """
        field, t = args
        field_check(s, field)
        assert isinstance(getattr(s, field), t), f"{field} is not {t} type"
    else:
        raise ValueError("Invalid number of arguments for field_check")



def plot_mag(xyz,
             ind=None,
             detrend_data: bool = False,
             use_mags: List[str] = None,
             vec_terms: List[str] = None,
             ylim: Tuple = (),
             dpi: int = 200,
             show_plot: bool = True,
             save_plot: bool = False,
             plot_png: str = "scalar_mags.png"):
    """
    Plot scalar or vector (fluxgate) magnetometer data from a given flight test.

    **Arguments:**
    - `xyz`:          `XYZ` flight data struct
    - `ind`:          (optional) selected data indices
    - `detrend_data`: (optional) if true, detrend plot data
    - `use_mags`:     (optional) scalar or vector (fluxgate) magnetometers to plot {`:all_mags`, `:comp_mags` or `:mag_1_c`, `:mag_1_uc`, `:flux_a`, etc.}
        - `:all_mags`  = all provided scalar magnetometer fields (e.g., `:mag_1_c`, `:mag_1_uc`, etc.)
        - `:comp_mags` = provided compensation(s) between `:mag_1_uc` & `:mag_1_c`, etc.
    - `vec_terms`:    (optional) vector magnetometer (fluxgate) terms to plot {`:all` or `:x`,`:y`,`:z`,`:t`}
    - `ylim`:         (optional) length-`2` plot `y` limits (`ymin`,`ymax`) [nT]
    - `dpi`:          (optional) dots per inch (image resolution)
    - `show_plot`:    (optional) if true, show `p1`
    - `save_plot`:    (optional) if true, save `p1` as `plot_png`
    - `plot_png`:     (optional) plot file name to save (`.png` extension optional)

    **Returns:**
    - `p1`: plot of scalar or vector (fluxgate) magnetometer data
    """
    num_mag_max = 6 # const value from julia Magnav
    
    # Set defaults
    if ind is None:
        ind = np.ones(xyz.traj.N, dtype=bool)
    if use_mags is None:
        use_mags = ['all_mags']
    if vec_terms is None:
        vec_terms = ['all']
    
    print('xyz.traj.tt', xyz.traj.tt, xyz.traj.tt.__class__)
    print('ind', ind, ind.__class__)
    print('masked values of tt', xyz.traj.tt[ind], xyz.traj.tt[ind].__class__)
    #[xyz.traj.tt[k] for (k,g) in enumerate(ind) if g>0]
    tt = (xyz.traj.tt[ind] - xyz.traj.tt[ind][0]) / 60 # tt = (xyz.traj.tt[ind] .- xyz.traj.tt[ind][1]) / 60
    xlab = "time [min]"

    # Get available fields
    fields = [attr for attr in dir(xyz) if not attr.startswith('_')]
    
    # Create magnetometer field lists
    list_c = [f"mag_{i}_c" for i in range(1, num_mag_max + 1)]
    list_uc = [f"mag_{i}_uc" for i in range(1, num_mag_max + 1)]
    
    # Find available magnetometer fields
    mags_c = [mag for mag in list_c if hasattr(xyz, mag)]
    mags_uc = [mag for mag in list_uc if hasattr(xyz, mag)]
    
    # Find indices where both compensated and uncompensated exist
    mags_c_ = []
    mags_uc_ = []
    for i, (c_field, uc_field) in enumerate(zip(list_c, list_uc)):
        if hasattr(xyz, c_field) and hasattr(xyz, uc_field):
            mags_c_.append(i)
            mags_uc_.append(i)
    
    mags_all = mags_c + mags_uc

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    if 'comp_mags' in use_mags:

        ylab = "magnetic field error [nT]"

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(ylim)

        for i in range(len(mags_c_)):
            val = (getattr(xyz, list_uc[mags_uc_[i]])[ind] -
                   getattr(xyz, list_c[mags_c_[i]])[ind])
            if detrend_data:
                val = detrend(val)
            ax.plot(tt, val, linewidth=2, label=f"mag_{i+1} comp")
            print(f"==== mag_{i+1} comp ====")
            print(f"avg comp = {np.round(np.mean(val), 3)} nT")
            print(f"std dev  = {np.round(np.std(val), 3)} nT")

    elif any(mag in field_check(xyz, MagV) for mag in use_mags):

        if 'all' in vec_terms:
            vec_terms = ['x', 'y', 'z', 't']

        ylab = "magnetic field [nT]"
        if detrend_data:
            ylab = f"detrended {ylab}"

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(ylim)

        available_vec_mags = field_check(xyz, MagV)
        for use_mag in [mag for mag in use_mags if mag in available_vec_mags]:

            flux = getattr(xyz, use_mag)
            if hasattr(flux, '__call__'):
                flux = flux(ind)
            else:
                # Assume it's already data, slice it
                flux_data = MagV()
                if hasattr(flux, 'x'):
                    flux_data.x = flux.x[ind] if hasattr(flux.x, '__getitem__') else flux.x
                if hasattr(flux, 'y'):
                    flux_data.y = flux.y[ind] if hasattr(flux.y, '__getitem__') else flux.y
                if hasattr(flux, 'z'):
                    flux_data.z = flux.z[ind] if hasattr(flux.z, '__getitem__') else flux.z
                if hasattr(flux, 't'):
                    flux_data.t = flux.t[ind] if hasattr(flux.t, '__getitem__') else flux.t
                flux = flux_data

            for vec_term in vec_terms:
                if hasattr(flux, vec_term):
                    val = getattr(flux, vec_term)
                    if detrend_data:
                        val = detrend(val)
                    ax.plot(tt, val, linewidth=2, label=f"{use_mag} {vec_term}")

    elif any(mag in ['all_mags'] + mags_all for mag in use_mags):

        if 'all_mags' in use_mags:
            use_mags = mags_all

        ylab = "magnetic field [nT]"
        if detrend_data:
            ylab = f"detrended {ylab}"

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(ylim)

        for mag in use_mags:
            if hasattr(xyz, mag):
                val = getattr(xyz, mag)[ind]
                if detrend_data:
                    val = detrend(val)
                ax.plot(tt, val, linewidth=2, label=f"{mag}")

    else:

        ylab = ""

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(ylim)

        for mag in use_mags:
            if hasattr(xyz, mag):
                val = getattr(xyz, mag)[ind]
                if detrend_data:
                    val = detrend(val)
                ax.plot(tt, val, linewidth=2, label=f"{mag}")

    # Add legend if there are labeled lines
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    if show_plot:
        plt.show()
    
    if save_plot:
        # Ensure .png extension
        if not plot_png.endswith('.png'):
            plot_png += '.png'
        plt.savefig(plot_png, dpi=dpi, bbox_inches='tight')

    return fig

# Plotting functions would be implemented separately
use_mags  = ['mag_1_uc', 'mag_4_uc', 'mag_5_uc']
plot_mag(xyz, ind=ind, use_mags=use_mags)  # equivalent Julia function
# plot_flux(xyz, ind=ind)  # equivalent Julia function
# plot_INS(xyz, ind=ind)  # equivalent Julia function