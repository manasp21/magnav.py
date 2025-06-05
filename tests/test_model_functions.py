import os
import pytest
import numpy as np
import scipy.io
from scipy.linalg import expm

# Functions from magnavpy.model_functions
from magnavpy.model_functions import (
    create_model,
    get_Phi,
    get_H,
    get_h_basic,
    fogm,
    get_pinson
)
from magnavpy.map_utils import map_interpolate, upward_fft
from magnavpy.magnav import MapS

# Set random seed for reproducibility
np.random.seed(2)

# Helper function to safely extract scalar values
def get_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        else:
            return value[0]
    return value

# Helper function to load .mat files
def load_mat_file(filename):
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "..", "..", "MagNav.jl", "test", "test_data", filename)
    return scipy.io.loadmat(data_path, squeeze_me=True)

# Load test data
ekf_data_dict = load_mat_file("test_data_ekf.mat")["ekf_data"]
grid_data_dict = load_mat_file("test_data_grid.mat")["grid_data"]
map_data_dict = load_mat_file("test_data_map.mat")["map_data"]
params_dict = load_mat_file("test_data_params.mat")["params"]
pinson_data_dict = load_mat_file("test_data_pinson.mat")["pinson_data"]
traj_data_dict = load_mat_file("test_data_traj.mat")["traj"]

# Prepare data
xn = grid_data_dict["xn"].flatten()
xl = grid_data_dict["xl"].flatten()
x_state = grid_data_dict["x"].flatten()

map_info = "Map"
map_map = map_data_dict["map"]
map_xx = np.deg2rad(np.array(map_data_dict["xx"].tolist()).flatten().astype(float))
map_yy = np.deg2rad(np.array(map_data_dict["yy"].tolist()).flatten().astype(float))
map_alt = map_data_dict["alt"]
map_mask = np.ones_like(map_map, dtype=bool)

# Convert parameters to scalars using helper function
dt = get_scalar(params_dict["dt"])
init_pos_sigma = get_scalar(params_dict["init_pos_sigma"])
init_alt_sigma = get_scalar(params_dict["init_alt_sigma"])
init_vel_sigma = get_scalar(params_dict["init_vel_sigma"])
meas_var = get_scalar(params_dict["meas_R"])
VRW_sigma = np.sqrt(get_scalar(params_dict["VRW_var"]))
ARW_sigma = np.sqrt(get_scalar(params_dict["ARW_var"]))
baro_sigma = get_scalar(params_dict["baro_std"])
ha_sigma = get_scalar(params_dict["ha_sigma"])
a_hat_sigma = get_scalar(params_dict["a_hat_sigma"])
acc_sigma = get_scalar(params_dict["acc_sigma"])
gyro_sigma = get_scalar(params_dict["gyro_sigma"])
fogm_sigma_param = get_scalar(params_dict["meas_sigma"])
baro_tau = get_scalar(params_dict["baro_tau"])
acc_tau = get_scalar(params_dict["acc_tau"])
gyro_tau = get_scalar(params_dict["gyro_tau"])
fogm_tau_param = get_scalar(params_dict["meas_tau"])

# Extract trajectory point data using helper function
lat = np.deg2rad(get_scalar(traj_data_dict["lat"]))
lon = np.deg2rad(get_scalar(traj_data_dict["lon"]))
alt = get_scalar(traj_data_dict["alt"])
vn = get_scalar(traj_data_dict["vn"])
ve = get_scalar(traj_data_dict["ve"])
vd = get_scalar(traj_data_dict["vd"])
fn = get_scalar(traj_data_dict["fn"])
fe = get_scalar(traj_data_dict["fe"])
fd = get_scalar(traj_data_dict["fd"])

# Robustly load Cnb
raw_cnb_data = traj_data_dict["Cnb"]
actual_cnb_array = raw_cnb_data
if raw_cnb_data.ndim == 0 and hasattr(raw_cnb_data, 'item') and isinstance(raw_cnb_data.item(), np.ndarray):
    actual_cnb_array = raw_cnb_data.item()

if actual_cnb_array.ndim == 3:
    Cnb = actual_cnb_array[:,:,0]
elif actual_cnb_array.ndim == 2 and actual_cnb_array.shape == (3,3):
    Cnb = actual_cnb_array
else:
    raise ValueError(f"Unexpected shape for Cnb: {actual_cnb_array.shape}")

fogm_data_expected = np.array([
    -0.00573724460026025,
    0.09446430193246710,
    0.03368954850407684,
    0.06685250250778804,
    0.04552990438603038
])

# Create MapS object
mapS_obj = MapS(map_info, map_map, map_xx, map_yy, map_alt, map_mask)

# Interpolate map
print(f"DEBUG_ERROR_6_TEST: map_interpolate.__module__: {map_interpolate.__module__}")
print(f"DEBUG_ERROR_6_TEST: map_interpolate.__qualname__: {map_interpolate.__qualname__}")
itp_mapS, der_mapS = map_interpolate(mapS_obj, 'linear', return_vert_deriv=True)

map_up_alt = np.array([mapS_obj.alt, mapS_obj.alt + 5.0])
mapS_upward_fft_result = upward_fft(mapS_obj, map_up_alt)
itp_mapS3D, der_mapS3D = map_interpolate(mapS_upward_fft_result, 'linear', return_vert_deriv=True)

# Call create_model
P0_py, Qd_py, R_py = create_model(dt, lat,
                               init_pos_sigma=init_pos_sigma,
                               init_alt_sigma=init_alt_sigma,
                               init_vel_sigma=init_vel_sigma,
                               meas_var=meas_var,
                               VRW_sigma=VRW_sigma,
                               ARW_sigma=ARW_sigma,
                               baro_sigma=baro_sigma,
                               ha_sigma=ha_sigma,
                               a_hat_sigma=a_hat_sigma,
                               acc_sigma=acc_sigma,
                               gyro_sigma=gyro_sigma,
                               fogm_sigma=fogm_sigma_param,
                               baro_tau=baro_tau,
                               acc_tau=acc_tau,
                               gyro_tau=gyro_tau,
                               fogm_tau=fogm_tau_param)

# --- Test Functions ---
RTOL = 1e-6
ATOL = 1e-6

def test_create_model():
    # Extract MATLAB arrays from object arrays
    def extract_matlab_array(data):
        if isinstance(data, np.ndarray) and data.dtype == object:
            return data.item()
        return data
        
    P0_mat = extract_matlab_array(ekf_data_dict["P0"])
    Qd_mat = extract_matlab_array(ekf_data_dict["Qd"])
    R_mat = extract_matlab_array(ekf_data_dict["R"])
    
    # Compare results
    np.testing.assert_allclose(P0_py, P0_mat, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Qd_py, Qd_mat, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(R_py, R_mat, rtol=RTOL, atol=ATOL)

# ... rest of test functions ...