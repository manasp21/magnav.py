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

XYZ20 = MagNav.XYZ20
Traj = MagNav.Traj
INS = MagNav.INS


import magnavpy.analysis_util as au
import magnavpy.signal_util as su
from magnavpy.plot_functions import plot_mag
import magnavpy.compensation as compensation
# import magnavpy.compensation as compensation
import magnavpy.tolles_lawson as tolles_lawson

"""
imported julia code from ai
"""

import h5py
import logging
from dataclasses import dataclass

from pathlib import Path
import copy
from typing import Union, Tuple, List, Any, Optional, Dict, BinaryIO


#
# @dataclass
# class MagV:
#     """Magnetometer vector data"""
#     x: np.ndarray
#     y: np.ndarray
#     z: np.ndarray
#     t: np.ndarray
#
# @dataclass
# class Traj:
#     """Trajectory data structure"""
#     N: int
#     dt: float
#     tt: np.ndarray
#     lat: np.ndarray
#     lon: np.ndarray
#     utm_z: np.ndarray
#     vn: np.ndarray
#     ve: np.ndarray
#     vd: np.ndarray
#     fn: np.ndarray
#     fe: np.ndarray
#     fd: np.ndarray
#     Cnb: np.ndarray
#
# @dataclass
# class INS:
#     """INS data structure"""
#     N: int
#     dt: float
#     tt: np.ndarray
#     ins_lat: np.ndarray
#     ins_lon: np.ndarray
#     ins_alt: np.ndarray
#     ins_vn: np.ndarray
#     ins_ve: np.ndarray
#     ins_vd: np.ndarray
#     ins_fn: np.ndarray
#     ins_fe: np.ndarray
#     ins_fd: np.ndarray
#     ins_Cnb: np.ndarray
#     ins_P: np.ndarray
#
# @dataclass
# class XYZ20:
#     """XYZ20 data structure containing all flight data"""
#     info: Any
#     traj: Traj
#     ins: INS
#     flux_a: MagV
#     flux_b: MagV
#     flux_c: MagV
#     flux_d: MagV
#     flight: np.ndarray
#     line: np.ndarray
#     year: np.ndarray
#     doy: np.ndarray
#     utm_x: np.ndarray
#     utm_y: np.ndarray
#     utm_z: np.ndarray
#     msl: np.ndarray
#     baro: np.ndarray
#     diurnal: np.ndarray
#     igrf: np.ndarray
#     mag_1_c: np.ndarray
#     mag_1_lag: np.ndarray
#     mag_1_dc: np.ndarray
#     mag_1_igrf: np.ndarray
#     mag_1_uc: np.ndarray
#     mag_2_uc: np.ndarray
#     mag_3_uc: np.ndarray
#     mag_4_uc: np.ndarray
#     mag_5_uc: np.ndarray
#     mag_6_uc: np.ndarray
#     ogs_mag: np.ndarray
#     ogs_alt: np.ndarray
#     ins_wander: np.ndarray
#     ins_roll: np.ndarray
#     ins_pitch: np.ndarray
#     ins_yaw: np.ndarray
#     roll_rate: np.ndarray
#     pitch_rate: np.ndarray
#     yaw_rate: np.ndarray
#     ins_acc_x: np.ndarray
#     ins_acc_y: np.ndarray
#     ins_acc_z: np.ndarray
#     lgtl_acc: np.ndarray
#     ltrl_acc: np.ndarray
#     nrml_acc: np.ndarray
#     pitot_p: np.ndarray
#     static_p: np.ndarray
#     total_p: np.ndarray
#     cur_com_1: np.ndarray
#     cur_ac_hi: np.ndarray
#     cur_ac_lo: np.ndarray
#     cur_tank: np.ndarray
#     cur_flap: np.ndarray
#     cur_strb: np.ndarray
#     cur_srvo_o: np.ndarray
#     cur_srvo_m: np.ndarray
#     cur_srvo_i: np.ndarray
#     cur_heat: np.ndarray
#     cur_acpwr: np.ndarray
#     cur_outpwr: np.ndarray
#     cur_bat_1: np.ndarray
#     cur_bat_2: np.ndarray
#     vol_acpwr: np.ndarray
#     vol_outpwr: np.ndarray
#     vol_bat_1: np.ndarray
#     vol_bat_2: np.ndarray
#     vol_res_p: np.ndarray
#     vol_res_n: np.ndarray
#     vol_back_p: np.ndarray
#     vol_back_n: np.ndarray
#     vol_gyro_1: np.ndarray
#     vol_gyro_2: np.ndarray
#     vol_acc_p: np.ndarray
#     vol_acc_n: np.ndarray
#     vol_block: np.ndarray
#     vol_back: np.ndarray
#     vol_srvo: np.ndarray
#     vol_cabt: np.ndarray
#     vol_fan: np.ndarray
#     aux_1: np.ndarray
#     aux_2: np.ndarray
#     aux_3: np.ndarray

import magnavpy.common_types as common_types
MagV = common_types.MagV

def add_extension(filename: str, extension: str) -> str:
    """Add extension to filename if not already present"""
    if not filename.endswith(extension):
        return filename + extension
    return filename




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

#
# def get_XYZ20(xyz_h5: str, 
#               info: str = None,
#               tt_sort: bool = True,
#               silent: bool = False) -> XYZ20:
#     """
#     Load XYZ20 data from HDF5 file
#
#     Parameters:
#     xyz_h5: path to HDF5 file
#     info: information string (defaults to filename)
#     tt_sort: whether to sort by time
#     silent: suppress info messages
#     """
#
#     if info is None:
#         info = Path(xyz_h5).name
#         xyz_h5 = add_extension(xyz_h5, ".h5")
#     else:
#         xyz_h5 = info.loc[info['flight']==xyz_h5]['xyz_file'].values[0]
#     fields = "fields20"
#
#     if not silent:
#         logging.info(f"reading in XYZ20 data: {xyz_h5}")
#
#     # Open HDF5 file for reading
#     with h5py.File(xyz_h5, "r") as xyz:
#         # Find maximum length across all datasets
#         print('xyz of h5', xyz)
#         N = max([xyz[k].shape[0] for k in xyz.keys() if xyz[k].shape != ()]) #max([len(read(xyz, k)) for k in xyz.keys()])
#         d = {}
#
#         # Sort by time if requested
#         if tt_sort:
#             tt_data = read_check(xyz, 'tt', N, silent)
#             ind = np.argsort(tt_data)
#         else:
#             ind = np.arange(N)
#
#         # Read all fields
#         for field in xyz_fields(fields):
#             if field != 'ignore':
#                 d[field] = read_check(xyz, field, N, silent)[ind]
#
#         # Read info field
#         field = 'info'
#         info_data = read_check(xyz, field, info)
#         d[field] = info_data
#
#         # Read auxiliary fields
#         for field in ['aux_1', 'aux_2', 'aux_3']:
#             d[field] = read_check(xyz, field, N, True)
#
#     # Calculate time step
#     dt = round(d['tt'][1] - d['tt'][0], 9) if N > 1 else 0.1
#
#     # Convert degrees to radians for angular measurements
#     for field in ['lat', 'lon', 'ins_roll', 'ins_pitch', 'ins_yaw',
#                   'roll_rate', 'pitch_rate', 'yaw_rate']:
#         d[field] = deg2rad(d[field])
#
#     # Calculate IGRF difference for convenience
#     d['igrf'] = d['mag_1_dc'] - d['mag_1_igrf']
#
#     # Calculate trajectory velocities & specific forces from position
#     d['vn'] = fdm(d['utm_y']) / dt
#     d['ve'] = fdm(d['utm_x']) / dt
#     d['vd'] = -fdm(d['utm_z']) / dt
#     d['fn'] = fdm(d['vn']) / dt
#     d['fe'] = fdm(d['ve']) / dt
#     d['fd'] = fdm(d['vd']) / dt - g_earth
#
#     # Direction cosine matrix (body to navigation) from roll, pitch, yaw
#     d['Cnb'] = np.zeros((3, 3, N))  # unknown
#     d['ins_Cnb'] = euler2dcm(d['ins_roll'], d['ins_pitch'], d['ins_yaw'], 'body2nav')
#     d['ins_P'] = np.zeros((1, 1, N))  # unknown
#
#     # INS velocities in NED direction
#     d['ins_ve'] = -d['ins_vw']
#     d['ins_vd'] = -d['ins_vu']
#
#     # INS specific forces from measurements, rotated by wander angle (CW for NED)
#     ins_f = np.zeros((N, 3))
#     for i in range(N):
#         wander_dcm = euler2dcm(0, 0, -d['ins_wander'][i], 'body2nav')
#         acc_vector = np.array([d['ins_acc_x'][i], -d['ins_acc_y'][i], -d['ins_acc_z'][i]])
#         ins_f[i, :] = wander_dcm @ acc_vector
#
#     d['ins_fn'] = ins_f[:, 0]
#     d['ins_fe'] = ins_f[:, 1] 
#     d['ins_fd'] = ins_f[:, 2]
#
#     # Alternative INS specific forces from finite differences (commented out)
#     # d['ins_fn'] = fdm(-d['ins_vn']) / dt
#     # d['ins_fe'] = fdm(-d['ins_ve']) / dt
#     # d['ins_fd'] = fdm(-d['ins_vd']) / dt - g_earth
#
#     return XYZ20(
#         d['info'],
#         Traj(N, dt, d['tt'], d['lat'], d['lon'], d['utm_z'], d['vn'],
#              d['ve'], d['vd'], d['fn'], d['fe'], d['fd'], d['Cnb']),
#         INS(N, dt, d['tt'], d['ins_lat'], d['ins_lon'], d['ins_alt'],
#             d['ins_vn'], d['ins_ve'], d['ins_vd'], d['ins_fn'],
#             d['ins_fe'], d['ins_fd'], d['ins_Cnb'], d['ins_P']),
#         MagV(d['flux_a_x'], d['flux_a_y'], d['flux_a_z'], d['flux_a_t']),
#         MagV(d['flux_b_x'], d['flux_b_y'], d['flux_b_z'], d['flux_b_t']),
#         MagV(d['flux_c_x'], d['flux_c_y'], d['flux_c_z'], d['flux_c_t']),
#         MagV(d['flux_d_x'], d['flux_d_y'], d['flux_d_z'], d['flux_d_t']),
#         d['flight'], d['line'], d['year'], d['doy'],
#         d['utm_x'], d['utm_y'], d['utm_z'], d['msl'],
#         d['baro'], d['diurnal'], d['igrf'], d['mag_1_c'],
#         d['mag_1_lag'], d['mag_1_dc'], d['mag_1_igrf'], d['mag_1_uc'],
#         d['mag_2_uc'], d['mag_3_uc'], d['mag_4_uc'], d['mag_5_uc'],
#         d['mag_6_uc'], d['ogs_mag'], d['ogs_alt'], d['ins_wander'],
#         d['ins_roll'], d['ins_pitch'], d['ins_yaw'], d['roll_rate'],
#         d['pitch_rate'], d['yaw_rate'], d['ins_acc_x'], d['ins_acc_y'],
#         d['ins_acc_z'], d['lgtl_acc'], d['ltrl_acc'], d['nrml_acc'],
#         d['pitot_p'], d['static_p'], d['total_p'], d['cur_com_1'],
#         d['cur_ac_hi'], d['cur_ac_lo'], d['cur_tank'], d['cur_flap'],
#         d['cur_strb'], d['cur_srvo_o'], d['cur_srvo_m'], d['cur_srvo_i'],
#         d['cur_heat'], d['cur_acpwr'], d['cur_outpwr'], d['cur_bat_1'],
#         d['cur_bat_2'], d['vol_acpwr'], d['vol_outpwr'], d['vol_bat_1'],
#         d['vol_bat_2'], d['vol_res_p'], d['vol_res_n'], d['vol_back_p'],
#         d['vol_back_n'], d['vol_gyro_1'], d['vol_gyro_2'], d['vol_acc_p'],
#         d['vol_acc_n'], d['vol_block'], d['vol_back'], d['vol_srvo'],
#         d['vol_cabt'], d['vol_fan'], d['aux_1'], d['aux_2'],
#         d['aux_3']
#     )



"""
end imported code from ai
"""

seed(33)  # for reproducibility



"""
setup dataframes begin
"""

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


# def get_XYZ(flight: str, df_flight: pd.DataFrame,
#             tt_sort: bool = True,
#             reorient_vec: bool = False,
#             silent: bool = False):
#     """
#     Get XYZ data for a specific flight from a DataFrame.
#
#     Parameters:
#     -----------
#     flight : str
#         Flight identifier
#     df_flight : pandas.DataFrame
#         DataFrame containing flight data
#     tt_sort : bool, optional
#         Whether to sort by time (default: True)
#     reorient_vec : bool, optional
#         Whether to reorient vectors (default: False)
#     silent : bool, optional
#         Whether to suppress output (default: False)
#
#     Returns:
#     --------
#     xyz : object
#         XYZ data object
#     """
#
#     # Find first index where flight column matches the input flight
#     flight_symbols = df_flight['flight'].astype(str)
#     ind = None
#     for i, f in enumerate(flight_symbols):
#         if f == flight:
#             ind = i
#             break
#
#     if ind is None:
#         raise ValueError(f"Flight '{flight}' not found in DataFrame")
#
#     xyz_file = str(df_flight.iloc[ind]['xyz_file'])
#     xyz_type = str(df_flight.iloc[ind]['xyz_type'])
#
#     # Call appropriate get_XYZ function based on xyz_type
#     if xyz_type == 'XYZ0':
#         xyz = get_XYZ0(xyz_file, tt_sort=tt_sort, silent=silent)
#     elif xyz_type == 'XYZ1':
#         xyz = get_XYZ1(xyz_file, tt_sort=tt_sort, silent=silent)
#     elif xyz_type == 'XYZ20':
#         xyz = get_XYZ20(xyz_file, tt_sort=tt_sort, silent=silent)
#     elif xyz_type == 'XYZ21':
#         xyz = get_XYZ21(xyz_file, tt_sort=tt_sort, silent=silent)
#     else:
#         raise ValueError(f"{xyz_type} xyz_type not defined")
#
#     # Optionally reorient vectors
#     if reorient_vec:
#         xyz_reorient_vec_(xyz)
#
#     return xyz



flight = 'Flt1006'  # select flight, full list in df_flight

from magnavpy.create_xyz import get_XYZ
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








TL_i = 6  # select first calibration box of 1006.04
# from MagNav.analysis_util import *
# TL_ind = au.get_ind_segs(xyz, lines=[df_cal.t_start[TL_i], df_cal.t_end[TL_i]])
# TL_ind = au.get_segments(xyz, lines=[df_cal.t_start[TL_i], df_cal.t_end[TL_i]])
TL_ind_other = au.get_ind_xyz(xyz, tt_lim=[df_cal.t_start[TL_i], df_cal.t_end[TL_i]])  # equivalent Julia function
TL_ind = TL_ind_other

# Here `df_all` is filtered into `df_options` to ensure that the selected flight line(s) for testing correspond with the selected flight (`:Flt1006`).

df_options = df_all[df_all.flight == flight]
print(df_options)

# For testing, we use Boolean indices (mask) corresponding to flight line 1006.08 in `df_options`.

line = 1006.08  # select flight line (row) from df_options
ind = au.get_ind_xyz_line_df(xyz, line, df_options)  # get Boolean indices

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


# Plotting functions would be implemented separately
use_mags  = ['mag_1_uc', 'mag_4_uc', 'mag_5_uc']
show_plot = True
save_plot = False

p1 = plot_mag(xyz, ind=ind, use_mags=use_mags)  # equivalent Julia function
p2 = plot_mag(xyz, ind=ind, show_plot=show_plot, save_plot=save_plot, # plot scalar magnetometers
              use_mags     = use_mags,
              detrend_data = True)
p3 = plot_mag(xyz, ind=ind, show_plot=show_plot, save_plot=save_plot, # plot vector magnetometer (fluxgate)
              use_mags     = ['flux_d'], # try changing to :flux_a, :flux_b, :flux_c
              detrend_data = True)

lpf     = compensation.get_bpf(pass1=0.0,pass2=0.2, fs=10.0) # get low-pass filter
lpf_sig = -compensation.bpf_data(xyz.cur_strb[ind], bpf=lpf) # apply low-pass filter, sign switched for easier comparison


## Tolles Lawson Callibration
lambda_val       = 0.025   # ridge parameter for ridge regression
use_vec = 'flux_d' # selected vector (flux) magnetometer
terms_A = ['permanent', 'induced', 'eddy'] # Tolles-Lawson terms to use
flux    = getattr(xyz, use_vec) # load Flux D data

TL_d_4  = tolles_lawson.create_TL_coef(flux, xyz.mag_4_uc, TL_ind, # create Tolles-Lawson
                         terms=terms_A, lambda_val=lambda_val)       # coefficients with Flux D & Mag 4

A = tolles_lawson.create_TL_A_modified(flux.x[ind], flux.y[ind], flux.z[ind])     # Tolles-Lawson `A` matrix for Flux D
### should return 8391 X 18 matrix .. , now after using the _modified function, it is returning
# returning 108318 X 18 matrix

mag_1_sgl = xyz.mag_1_c[ind]  # professionally compensated tail stinger, Mag 1
mag_4_uc  = xyz.mag_4_uc[ind] # uncompensated Mag 4

mag_4_c   = mag_4_uc - au.detrend(A@TL_d_4, mean_only=True) # compensated Mag 4 # use @ instead of * for multiplying matrices



##### Begin training NN
features = ['mag_4_uc', 'TL_A_flux_d', 'lpf_cur_com_1', 'lpf_cur_strb', 'lpf_cur_outpwr', 'lpf_cur_ac_lo']
y_type      = 'd'
use_mag     = 'mag_4_uc'
sub_diurnal = True
sub_igrf    = True
norm_type_x = 'standardize'
norm_type_y = 'standardize'

model_type  = 'm2b'
η_adam      = 0.001
epoch_adam  = 100
hidden      = [8]

comp_params = compensation.NNCompParams(features_setup = features,
                           model_type     = model_type,
                           y_type         = y_type,
                           use_mag        = use_mag,
                           use_vec        = use_vec,
                           terms_A        = terms_A,
                           sub_diurnal    = sub_diurnal,
                           sub_igrf       = sub_igrf,
                           norm_type_x    = norm_type_x,
                           norm_type_y    = norm_type_y,
                           TL_coef        = TL_d_4,
                           eta_adam         = η_adam,
                           epoch_adam     = epoch_adam,
                           hidden         = hidden)

(comp_params,y_train,y_train_hat,err_train,feats) = compensation.comp_train_df(comp_params,lines_train,df_all,df_flight,df_map)
# (_,y_test_hat,_) = compensation.comp_test(comp_params,[line],df_all,df_flight,df_map)
