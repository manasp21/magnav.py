import pytest
import numpy as np
import scipy.io
import os

# Assuming MagNavPy classes and functions are in these locations based on user prompt
from MagNavPy.src.magnav import Traj, INS, MapS, EKF_RT, FILTres, MapCache
from MagNavPy.src.ekf import crlb, ekf
# model_functions might contain map_interpolate and map_params.
# If these are methods of MapS or another class, imports would change.
from MagNavPy.src.model_functions import map_interpolate, map_params

# Helper for loading .mat files and extracting the specific variable
def load_mat_variable(base_path, file_name, variable_name):
    return scipy.io.loadmat(os.path.join(base_path, file_name))[variable_name]

# Base path for test data, relative to this test file's location
# This file is in MagNavPy/tests/
# Data is in MagNav.jl/test/test_data/
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "MagNav.jl", "test", "test_data")

# Load data (mirroring Julia script)
ekf_data_content  = load_mat_variable(TEST_DATA_PATH, "test_data_ekf.mat", "ekf_data")
ins_data_content  = load_mat_variable(TEST_DATA_PATH, "test_data_ins.mat", "ins_data")
map_data_content  = load_mat_variable(TEST_DATA_PATH, "test_data_map.mat", "map_data")
params_content    = load_mat_variable(TEST_DATA_PATH, "test_data_params.mat", "params")
traj_data_content = load_mat_variable(TEST_DATA_PATH, "test_data_traj.mat", "traj")

# Extract data from loaded content
# From ekf_data
P0 = ekf_data_content["P0"]
Qd = ekf_data_content["Qd"]
R_val  = ekf_data_content["R"] # Expecting this to be a scalar or 1x1 array for magnetometer

# From ins_data
ins_lat  = np.deg2rad(ins_data_content["lat"].ravel())
ins_lon  = np.deg2rad(ins_data_content["lon"].ravel())
ins_alt  = ins_data_content["alt"].ravel()
ins_vn   = ins_data_content["vn"].ravel()
ins_ve   = ins_data_content["ve"].ravel()
ins_vd   = ins_data_content["vd"].ravel()
ins_fn   = ins_data_content["fn"].ravel()
ins_fe   = ins_data_content["fe"].ravel()
ins_fd   = ins_data_content["fd"].ravel()
ins_Cnb  = ins_data_content["Cnb"] # Assuming shape (3, 3, N) from Julia Cnb[:,:,t]

# From map_data
map_info = "Map" # Hardcoded as in Julia script; or map_data_content["map_info"] if exists
map_map  = map_data_content["map"]
map_xx   = np.deg2rad(map_data_content["xx"].ravel())
map_yy   = np.deg2rad(map_data_content["yy"].ravel())
map_alt_data  = map_data_content["alt"] # Renamed to avoid conflict with alt from traj_data
# map_params from MagNav.jl returns (map_map,map_mask,map_edges,map_origin,map_res). We need map_mask.
# Assuming Python map_params has a similar return or can provide the mask.
# For now, assuming it returns a tuple where the mask is an element.
# Julia: map_mask = MagNav.map_params(map_map,map_xx,map_yy)[2] (index 2 is 3rd element)
# This implies map_params returns multiple values and mask is the second one (if 0-indexed) or third (if 1-indexed).
# Let's assume map_params returns (some_val, mask_val, ...) or similar.
# This part might need adjustment based on the actual Python map_params signature.
# For now, assuming it returns a structure or tuple from which mask can be extracted.
# If map_params returns (map_object_with_mask_attribute), access would differ.
# Given Julia's direct indexing `[2]`, it's likely a tuple.
# Let's assume map_params returns (processed_map, mask, other_things...)
_, map_mask, _, _, _ = map_params(map_map, map_xx, map_yy) # Adjust if return differs

# From params (these are expected to be scalars)
dt       = params_content["dt"].item()
baro_tau = params_content["baro_tau"].item()
acc_tau  = params_content["acc_tau"].item()
gyro_tau = params_content["gyro_tau"].item()
fogm_tau = params_content["meas_tau"].item() # Julia uses "meas_tau"

# From traj_data
tt       = traj_data_content["tt"].ravel()
lat      = np.deg2rad(traj_data_content["lat"].ravel())
lon      = np.deg2rad(traj_data_content["lon"].ravel())
alt      = traj_data_content["alt"].ravel()
vn       = traj_data_content["vn"].ravel()
ve       = traj_data_content["ve"].ravel()
vd       = traj_data_content["vd"].ravel()
fn       = traj_data_content["fn"].ravel()
fe       = traj_data_content["fe"].ravel()
fd       = traj_data_content["fd"].ravel()
Cnb      = traj_data_content["Cnb"] # Assuming shape (3, 3, N)
mag_1_c  = traj_data_content["mag_1_c"].ravel()
N        = len(lat)

# Create Traj and INS objects
traj = Traj(N, dt, tt, lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb)
ins  = INS(N, dt, tt, ins_lat, ins_lon, ins_alt, ins_vn, ins_ve, ins_vd,
           ins_fn, ins_fe, ins_fd, ins_Cnb, np.zeros((3,3,N))) # Assuming last arg dCnb_CG

# Create MapS and MapCache objects
mapS = MapS(map_info, map_map, map_xx, map_yy, map_alt_data, map_mask)
map_cache = MapCache(maps=[mapS]) # Assuming MapCache class and constructor

# Interpolate map
# Python equivalent of: (itp_mapS,der_mapS) = map_interpolate(mapS,:linear;return_vert_deriv=true)
itp_mapS, der_mapS = map_interpolate(mapS, method='linear', return_vert_deriv=True)

# CRLB calculation
crlb_P_py = crlb(lat, lon, alt, vn, ve, vd, fn, fe, fd, Cnb, dt, itp_mapS,
                 P0=P0, Qd=Qd, R=R_val,
                 baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                 fogm_tau=fogm_tau, core=False)

# EKF calculation
filt_res_py = ekf(ins_lat, ins_lon, ins_alt, ins_vn, ins_ve, ins_vd,
                  ins_fn, ins_fe, ins_fd, ins_Cnb, mag_1_c, dt, itp_mapS,
                  P0=P0, Qd=Qd, R=R_val,
                  baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                  fogm_tau=fogm_tau, core=False)

# Expected values from MAT file for comparison
ekf_data_crlb_P = ekf_data_content["crlb_P"]
ekf_data_x_out  = ekf_data_content["x_out"]
ekf_data_P_out  = ekf_data_content["P_out"]

@pytest.fixture(scope="module")
def common_data():
    """Fixture to provide common data to multiple tests."""
    # This can be expanded if more data needs to be shared or setup is complex
    assert N == 100, f"Expected N=100 for shape tests based on Julia, but got N={N}"
    return {
        "crlb_P_py": crlb_P_py,
        "filt_res_py": filt_res_py,
        "ekf_data_crlb_P": ekf_data_crlb_P,
        "ekf_data_x_out": ekf_data_x_out,
        "ekf_data_P_out": ekf_data_P_out,
        "traj": traj,
        "ins": ins,
        "itp_mapS": itp_mapS,
        "map_cache": map_cache,
        "der_mapS": der_mapS,
        "map_alt_data": map_alt_data,
        "mag_1_c": mag_1_c,
        "P0": P0, "Qd": Qd, "R_val": R_val,
        "baro_tau": baro_tau, "acc_tau": acc_tau, "gyro_tau": gyro_tau, "fogm_tau": fogm_tau,
        "dt": dt, "N": N, "tt": tt,
        "ins_lat": ins_lat, "ins_lon": ins_lon, "ins_alt": ins_alt,
        "ins_vn": ins_vn, "ins_ve": ins_ve, "ins_vd": ins_vd,
        "ins_fn": ins_fn, "ins_fe": ins_fe, "ins_fd": ins_fd, "ins_Cnb": ins_Cnb,
        "ekf_rt_nx": EKF_RT(P=P0, Qd=Qd, R=R_val, baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau, fogm_tau=fogm_tau, core=False, ny=1).nx, # ny=1 for scalar mag
        "ekf_rt_ny": 1, # scalar mag
    }

def test_crlb_calculations(common_data):
    """Tests for CRLB functionality."""
    np.testing.assert_allclose(common_data["crlb_P_py"][:,:,0], common_data["ekf_data_crlb_P"][:,:,0], atol=1e-6)
    np.testing.assert_allclose(common_data["crlb_P_py"][:,:,-1], common_data["ekf_data_crlb_P"][:,:,-1], atol=1e-6)

    # Test crlb with Traj object and map objects, assuming Python crlb supports this
    # This requires crlb to be flexible with its first two arguments (traj data, map data)
    crlb_traj_itp = crlb(common_data["traj"], common_data["itp_mapS"], P0=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
                         baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"], gyro_tau=common_data["gyro_tau"],
                         fogm_tau=common_data["fogm_tau"], core=False)
    assert crlb_traj_itp.shape == (18, 18, common_data["N"])

    crlb_traj_cache = crlb(common_data["traj"], common_data["map_cache"], P0=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
                           baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"], gyro_tau=common_data["gyro_tau"],
                           fogm_tau=common_data["fogm_tau"], core=False)
    assert crlb_traj_cache.shape == (18, 18, common_data["N"])

def test_ekf_calculations(common_data):
    """Tests for EKF functionality."""
    filt_res_py = common_data["filt_res_py"]
    ekf_data_x_out = common_data["ekf_data_x_out"]
    ekf_data_P_out = common_data["ekf_data_P_out"]

    np.testing.assert_allclose(filt_res_py.x[:,0], ekf_data_x_out[:,0], atol=1e-6)
    np.testing.assert_allclose(filt_res_py.x[:,-1], ekf_data_x_out[:,-1], atol=1e-3)
    np.testing.assert_allclose(filt_res_py.P[:,:,0], ekf_data_P_out[:,:,0], atol=1e-6)
    np.testing.assert_allclose(filt_res_py.P[:,:,-1], ekf_data_P_out[:,:,-1], atol=1e-3)

    # Test EKF with INS object and other variations, assuming Python ekf supports this
    res1 = ekf(common_data["ins"], common_data["mag_1_c"], common_data["itp_mapS"],
               P0=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
               baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"],
               gyro_tau=common_data["gyro_tau"], fogm_tau=common_data["fogm_tau"], core=False)
    assert isinstance(res1, FILTres)

    res2 = ekf(common_data["ins"], common_data["mag_1_c"], common_data["map_cache"],
               P0=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
               baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"],
               gyro_tau=common_data["gyro_tau"], fogm_tau=common_data["fogm_tau"], core=True) # core=True here
    assert isinstance(res2, FILTres)

    # Test with R as a tuple, e.g., (1.0, 10.0)
    R_tuple_test = (1.0, 10.0) # Based on Julia test R=(1,10)
    res3 = ekf(common_data["ins"], common_data["mag_1_c"], common_data["itp_mapS"],
               P0=common_data["P0"], Qd=common_data["Qd"], R=R_tuple_test, # Using tuple R
               baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"],
               gyro_tau=common_data["gyro_tau"], fogm_tau=common_data["fogm_tau"], core=False)
    assert isinstance(res3, FILTres)

    res4 = ekf(common_data["ins"], common_data["mag_1_c"], common_data["itp_mapS"],
               der_mapS=common_data["der_mapS"], map_alt=common_data["map_alt_data"], # Optional map args
               P0=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
               baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"],
               gyro_tau=common_data["gyro_tau"], fogm_tau=common_data["fogm_tau"], core=False)
    assert isinstance(res4, FILTres)

def test_ekf_rt_functionality(common_data):
    """Tests for EKF_RT (Real-Time EKF) functionality."""
    # Create EKF_RT instance
    # ny=1 for scalar magnetometer measurement
    ekf_rt_obj = EKF_RT(P=common_data["P0"], Qd=common_data["Qd"], R=common_data["R_val"],
                        baro_tau=common_data["baro_tau"], acc_tau=common_data["acc_tau"],
                        gyro_tau=common_data["gyro_tau"], fogm_tau=common_data["fogm_tau"],
                        core=False, ny=common_data["ekf_rt_ny"])

    N_val = common_data["N"]
    x_rt = np.zeros((common_data["ekf_rt_nx"], N_val))
    P_rt = np.zeros((common_data["ekf_rt_nx"], common_data["ekf_rt_nx"], N_val))
    r_rt = np.zeros((common_data["ekf_rt_ny"], N_val))

    # Loop for real-time simulation
    # Assuming EKF_RT instance is callable (defines __call__) or has a compatible method
    for t in range(N_val):
        current_ins_Cnb_t = common_data["ins_Cnb"][:,:,t] # Assuming ins_Cnb is (3,3,N)
        current_mag_1_c_t = common_data["mag_1_c"][t]

        filt_res_step = ekf_rt_obj(common_data["ins_lat"][t], common_data["ins_lon"][t], common_data["ins_alt"][t],
                                   common_data["ins_vn"][t], common_data["ins_ve"][t], common_data["ins_vd"][t],
                                   common_data["ins_fn"][t], common_data["ins_fe"][t], common_data["ins_fd"][t],
                                   current_ins_Cnb_t, current_mag_1_c_t, common_data["tt"][t],
                                   common_data["itp_mapS"], dt=common_data["dt"])
        
        x_rt[:,t]   = filt_res_step.x.ravel() # Ensure x is 1D for assignment
        P_rt[:,:,t] = filt_res_step.P
        r_rt[:,t]   = filt_res_step.r.ravel() # Ensure r is 1D

    # Compare EKF_RT results with batch EKF results (filt_res_py from common_data)
    filt_res_batch = common_data["filt_res_py"]
    # Using a slightly looser tolerance for RT vs batch if minor numerical differences are expected
    # Or tight if they should be identical. Julia used '≈' which is isapprox default.
    atol_rt_vs_batch = 1e-9 # Assuming they should be very close
    np.testing.assert_allclose(x_rt[:,0], filt_res_batch.x[:,0], atol=atol_rt_vs_batch)
    np.testing.assert_allclose(x_rt[:,-1], filt_res_batch.x[:,-1], atol=atol_rt_vs_batch)
    np.testing.assert_allclose(P_rt[:,:,0], filt_res_batch.P[:,:,0], atol=atol_rt_vs_batch)
    np.testing.assert_allclose(P_rt[:,:,-1], filt_res_batch.P[:,:,-1], atol=atol_rt_vs_batch)
    np.testing.assert_allclose(r_rt[:,0], filt_res_batch.r[:,0], atol=atol_rt_vs_batch)
    np.testing.assert_allclose(r_rt[:,-1], filt_res_batch.r[:,-1], atol=atol_rt_vs_batch)

    # Further EKF_RT tests (type checks with different inputs for the last time step)
    last_idx = N_val - 1
    last_ins_Cnb = common_data["ins_Cnb"][:,:,last_idx]
    last_mag_1_c = common_data["mag_1_c"][last_idx]

    # Test 1: EKF_RT with itp_mapS for last step
    res_rt_test1 = ekf_rt_obj(common_data["ins_lat"][last_idx], common_data["ins_lon"][last_idx], common_data["ins_alt"][last_idx],
                              common_data["ins_vn"][last_idx], common_data["ins_ve"][last_idx], common_data["ins_vd"][last_idx],
                              common_data["ins_fn"][last_idx], common_data["ins_fe"][last_idx], common_data["ins_fd"][last_idx],
                              last_ins_Cnb, last_mag_1_c, common_data["tt"][last_idx],
                              common_data["itp_mapS"], dt=common_data["dt"])
    assert isinstance(res_rt_test1, FILTres)

    # Test 2: EKF_RT with map_cache for last step
    res_rt_test2 = ekf_rt_obj(common_data["ins_lat"][last_idx], common_data["ins_lon"][last_idx], common_data["ins_alt"][last_idx],
                              common_data["ins_vn"][last_idx], common_data["ins_ve"][last_idx], common_data["ins_vd"][last_idx],
                              common_data["ins_fn"][last_idx], common_data["ins_fe"][last_idx], common_data["ins_fd"][last_idx],
                              last_ins_Cnb, last_mag_1_c, common_data["tt"][last_idx],
                              common_data["map_cache"], dt=common_data["dt"]) # Using map_cache
    assert isinstance(res_rt_test2, FILTres)
    
    # Test 3: EKF_RT with der_mapS and map_alt for last step
    res_rt_test3 = ekf_rt_obj(common_data["ins_lat"][last_idx], common_data["ins_lon"][last_idx], common_data["ins_alt"][last_idx],
                              common_data["ins_vn"][last_idx], common_data["ins_ve"][last_idx], common_data["ins_vd"][last_idx],
                              common_data["ins_fn"][last_idx], common_data["ins_fe"][last_idx], common_data["ins_fd"][last_idx],
                              last_ins_Cnb, last_mag_1_c, common_data["tt"][last_idx],
                              common_data["itp_mapS"], dt=common_data["dt"],
                              der_mapS=common_data["der_mapS"], map_alt=common_data["map_alt_data"]) # Optional args
    assert isinstance(res_rt_test3, FILTres)