import os
import pytest
import numpy as np
import scipy.io
from scipy.linalg import expm # Included as it's common in state-space, though not directly used by name here

# Functions from MagNavPy.src.model_functions
from MagNavPy.src.model_functions import (
    create_model,
    get_Phi,
    get_H,
    get_h_basic, # Assuming MagNav.get_h maps to this
    fogm,
    get_pinson,
    map_interpolate,
    upward_fft,
    map_grad,
    map_params
)

# Data structures from MagNavPy.src.magnav
from MagNavPy.src.magnav import MapS

# Set random seed for reproducibility (affects np.random functions)
np.random.seed(2)

# Helper function to load .mat files
def load_mat_file(filename):
    # Construct the correct path relative to this test file
    # This test file is in MagNavPy/tests/
    # Data is in MagNav.jl/test/test_data/
    base_path = os.path.dirname(__file__)
    # Path adjustment: from MagNavPy/tests/test_model_functions.py to MagNav.jl/test/test_data/
    data_path = os.path.join(base_path, "..", "..", "MagNav.jl", "test", "test_data", filename)
    return scipy.io.loadmat(data_path, squeeze_me=True) # squeeze_me=True helps with scalar/vector dimensions

# Load test data
# The keys like "ekf_data" are struct names within the .mat files in the Julia code
ekf_data_dict = load_mat_file("test_data_ekf.mat")["ekf_data"]
grid_data_dict = load_mat_file("test_data_grid.mat")["grid_data"]
map_data_dict = load_mat_file("test_data_map.mat")["map_data"]
params_dict = load_mat_file("test_data_params.mat")["params"]
pinson_data_dict = load_mat_file("test_data_pinson.mat")["pinson_data"]
traj_data_dict = load_mat_file("test_data_traj.mat")["traj"]


# Prepare data similar to Julia script
# For scalars, .item() can be used if necessary, but squeeze_me=True should handle most.
xn = grid_data_dict["xn"].flatten()
xl = grid_data_dict["xl"].flatten()
x_state = grid_data_dict["x"].flatten() # Renamed from x to avoid conflict if x is used as loop var

map_info = "Map" # map_data_dict.get("map_info", "Map") # Assuming "Map" if not in .mat
map_map = map_data_dict["map"]
map_xx = np.deg2rad(map_data_dict["xx"].flatten())
map_yy = np.deg2rad(map_data_dict["yy"].flatten())
map_alt = map_data_dict["alt"] # squeeze_me=True should make this scalar if it is
# map_mask = MagNav.map_params(map_map,map_xx,map_yy)[2] (Julia 1-indexed tuple access)
# Assuming map_params returns a tuple and the second element (index 1 in Python) is the mask.
map_mask = map_params(map_map, map_xx, map_yy)[1]

dt = params_dict["dt"]
init_pos_sigma = params_dict["init_pos_sigma"]
init_alt_sigma = params_dict["init_alt_sigma"]
init_vel_sigma = params_dict["init_vel_sigma"]
init_att_sigma = params_dict["init_att_sigma"]
meas_var = params_dict["meas_R"]
VRW_sigma = np.sqrt(params_dict["VRW_var"])
ARW_sigma = np.sqrt(params_dict["ARW_var"])
baro_sigma = params_dict["baro_std"]
ha_sigma = params_dict["ha_sigma"]
a_hat_sigma = params_dict["a_hat_sigma"]
acc_sigma = params_dict["acc_sigma"]
gyro_sigma = params_dict["gyro_sigma"]
fogm_sigma_param = params_dict["meas_sigma"] # Renamed to avoid conflict
baro_tau = params_dict["baro_tau"]
acc_tau = params_dict["acc_tau"]
gyro_tau = params_dict["gyro_tau"]
fogm_tau_param = params_dict["meas_tau"] # Renamed to avoid conflict

# Extract trajectory point data
lat = np.deg2rad(traj_data_dict["lat"].flatten()[0])
lon = np.deg2rad(traj_data_dict["lon"].flatten()[0])
alt = traj_data_dict["alt"].flatten()[0]
vn = traj_data_dict["vn"].flatten()[0]
ve = traj_data_dict["ve"].flatten()[0]
vd = traj_data_dict["vd"].flatten()[0]
fn = traj_data_dict["fn"].flatten()[0]
fe = traj_data_dict["fe"].flatten()[0]
fd = traj_data_dict["fd"].flatten()[0]
Cnb = traj_data_dict["Cnb"][:,:,0] # Julia Cnb[:,:,1] -> Python Cnb[:,:,0]

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
itp_mapS, der_mapS = map_interpolate(mapS_obj, 'linear', return_vert_deriv=True)

map_up_alt = np.array([mapS_obj.alt, mapS_obj.alt + 5.0]) # Ensure float for calculations
mapS_upward_fft_result = upward_fft(mapS_obj, map_up_alt)
itp_mapS3D, der_mapS3D = map_interpolate(mapS_upward_fft_result, 'linear', return_vert_deriv=True)

# Call create_model
P0_py, Qd_py, R_py = create_model(dt, lat,
                               init_pos_sigma=init_pos_sigma,
                               init_alt_sigma=init_alt_sigma,
                               init_vel_sigma=init_vel_sigma,
                               init_att_sigma=init_att_sigma,
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
# Define a common tolerance
RTOL = 1e-6
ATOL = 1e-6

def test_create_model():
    np.testing.assert_allclose(P0_py, ekf_data_dict["P0"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Qd_py, ekf_data_dict["Qd"], rtol=RTOL, atol=ATOL)
    # R is scalar measurement variance
    expected_R = ekf_data_dict["R"]
    assert R_py == pytest.approx(expected_R, rel=RTOL, abs=ATOL)

def test_get_pinson():
    p17_py = get_pinson(17, lat, vn, ve, vd, fn, fe, fd, Cnb,
                        baro_tau=baro_tau,
                        acc_tau=acc_tau,
                        gyro_tau=gyro_tau,
                        fogm_tau=fogm_tau_param,
                        vec_states=False,
                        fogm_state=False)
    np.testing.assert_allclose(p17_py, pinson_data_dict["P17"], rtol=RTOL, atol=ATOL)

    p18_py = get_pinson(17 + 1, lat, vn, ve, vd, fn, fe, fd, Cnb,
                        baro_tau=baro_tau,
                        acc_tau=acc_tau,
                        gyro_tau=gyro_tau,
                        fogm_tau=fogm_tau_param,
                        vec_states=False,
                        fogm_state=True)
    np.testing.assert_allclose(p18_py, pinson_data_dict["P18"], rtol=RTOL, atol=ATOL)

    p_vec_states_py = get_pinson(17 + 3, lat, vn, ve, vd, fn, fe, fd, Cnb,
                                 baro_tau=baro_tau, # Pass other taus for consistency if defaults differ
                                 acc_tau=acc_tau,
                                 gyro_tau=gyro_tau,
                                 fogm_tau=fogm_tau_param,
                                 vec_states=True,
                                 fogm_state=False)
    assert isinstance(p_vec_states_py, np.ndarray)
    assert p_vec_states_py.ndim == 2

def test_get_Phi():
    phi_py = get_Phi(18, lat, vn, ve, vd, fn, fe, fd, Cnb,
                     baro_tau, acc_tau, gyro_tau, fogm_tau_param, dt)
    np.testing.assert_allclose(phi_py, pinson_data_dict["Phi"], rtol=RTOL, atol=ATOL)

def test_get_H():
    H_py = get_H(itp_mapS, x_state, lat, lon, alt, core=False)
    np.testing.assert_allclose(H_py, grid_data_dict["H"], rtol=RTOL, atol=ATOL)

    H_3D_py = get_H(itp_mapS3D, x_state, lat, lon, alt) # Assuming default core matches Julia
    assert isinstance(H_3D_py, np.ndarray)
    # Julia's `isa Vector` implies 1D array.
    assert H_3D_py.ndim == 1 or (H_3D_py.ndim == 2 and (H_3D_py.shape[0] == 1 or H_3D_py.shape[1] == 1))


def test_get_h():
    # Test 1: Basic h value
    # Julia: MagNav.get_h(itp_mapS,x,lat,lon,alt;core=false)[1] ≈ grid_data["h"]
    # Assumes get_h_basic returns scalar h, or (h, dh_dalt) if der_map is provided.
    # Here, der_map is not provided to get_h_basic.
    h_val_1 = get_h_basic(itp_mapS, x_state, lat, lon, alt, core=False)
    # If get_h_basic returns a tuple even without der_map, take the first element.
    # However, it's more likely to return a scalar directly in this case.
    if isinstance(h_val_1, tuple): h_val_1 = h_val_1[0]
    np.testing.assert_allclose(h_val_1, grid_data_dict["h"], rtol=RTOL, atol=ATOL)

    # Test 2: h value for RBPF state
    # Julia: MagNav.get_h(itp_mapS,[xn;0;xl[2:end]],lat,lon,alt;core=false)[1] ≈ grid_data["hRBPF"]
    state_rbpf = np.concatenate((xn, np.array([0.0]), xl[1:])) # xl[1:] in Python is Julia's xl[2:end]
    h_val_2 = get_h_basic(itp_mapS, state_rbpf, lat, lon, alt, core=False)
    if isinstance(h_val_2, tuple): h_val_2 = h_val_2[0]
    np.testing.assert_allclose(h_val_2, grid_data_dict["hRBPF"], rtol=RTOL, atol=ATOL)

    # Test 3: h value with 3D map interpolation
    # Julia: MagNav.get_h(itp_mapS3D,x,lat,lon,alt) isa Vector
    # Assuming default core for get_h_basic matches Julia's get_h default.
    h_val_3 = get_h_basic(itp_mapS3D, x_state, lat, lon, alt)
    if isinstance(h_val_3, tuple): h_val_3 = h_val_3[0] # Primary value
    assert isinstance(h_val_3, np.ndarray)
    assert h_val_3.ndim == 1 # Expecting a 1D vector

    # Test 4: h value with derivative map, core=false
    # Julia: MagNav.get_h(itp_mapS,der_mapS,x,lat,lon,alt,map_alt;core=false) isa Vector
    # Assumes get_h_basic signature: (itp, x, lat, lon, alt, core, map_alt_ref, der_map)
    # The Julia test `isa Vector` on the direct output implies the function returns a single vector.
    h_val_4 = get_h_basic(itp_mapS, x_state, lat, lon, alt, core=False, map_alt_ref=map_alt, der_map=der_mapS)
    assert isinstance(h_val_4, np.ndarray), "Expected ndarray for h_val_4"
    assert h_val_4.ndim == 1, "Expected 1D vector for h_val_4"

    # Test 5: h value with derivative map, core=true
    h_val_5 = get_h_basic(itp_mapS, x_state, lat, lon, alt, core=True, map_alt_ref=map_alt, der_map=der_mapS)
    assert isinstance(h_val_5, np.ndarray), "Expected ndarray for h_val_5"
    assert h_val_5.ndim == 1, "Expected 1D vector for h_val_5"

    # Test 6: h value with 3D derivative map, core=false
    h_val_6 = get_h_basic(itp_mapS3D, x_state, lat, lon, alt, core=False, map_alt_ref=map_alt, der_map=der_mapS3D)
    assert isinstance(h_val_6, np.ndarray), "Expected ndarray for h_val_6"
    assert h_val_6.ndim == 1, "Expected 1D vector for h_val_6"

    # Test 7: h value with 3D derivative map, core=true
    h_val_7 = get_h_basic(itp_mapS3D, x_state, lat, lon, alt, core=True, map_alt_ref=map_alt, der_map=der_mapS3D)
    assert isinstance(h_val_7, np.ndarray), "Expected ndarray for h_val_7"
    assert h_val_7.ndim == 1, "Expected 1D vector for h_val_7"


def test_map_grad():
    # Julia: MagNav.map_grad(itp_mapS,lat,lon,alt)[1:2] ≈ reverse(vec(grid_data["grad"]))
    # map_grad in Python assumed to return (dM_dlat, dM_dlon, dM_dalt)
    grad_py_tuple = map_grad(itp_mapS, lat, lon, alt)
    grad_py_lat_lon = np.array([grad_py_tuple[0], grad_py_tuple[1]])

    # grid_data["grad"] is likely [dM_dlon; dM_dlat] (2x1) from MATLAB via Julia.
    # vec(grid_data["grad"]) -> [dM_dlon, dM_dlat]
    # reverse(...) -> [dM_dlat, dM_dlon]
    expected_grad_jl_order = grid_data_dict["grad"].flatten()
    expected_grad_for_comparison = expected_grad_jl_order[::-1] # Reverse to [dM_dlat, dM_dlon]
    np.testing.assert_allclose(grad_py_lat_lon, expected_grad_for_comparison, rtol=RTOL, atol=ATOL)

    # Julia: MagNav.map_grad(itp_mapS3D,lat,lon,alt) isa Vector
    # This implies the Python map_grad might return a 1D NumPy array in this case.
    grad_3D_py_output = map_grad(itp_mapS3D, lat, lon, alt)
    assert isinstance(grad_3D_py_output, np.ndarray)
    assert grad_3D_py_output.ndim == 1
    assert grad_3D_py_output.size == 3 # Expect 3 components for 3D gradient

def test_fogm():
    # Julia: fogm_data ≈ MagNav.fogm(fogm_sigma,fogm_tau,dt,length(fogm_data))
    # Global np.random.seed(2) should make np.random.randn() deterministic if used in fogm.
    len_fogm_data = len(fogm_data_expected)
    generated_fogm_data = fogm(fogm_sigma_param, fogm_tau_param, dt, len_fogm_data)
    # If fogm itself needs a seed for perfect reproducibility with Julia's randn:
    # generated_fogm_data = fogm(fogm_sigma_param, fogm_tau_param, dt, len_fogm_data, seed=2)
    np.testing.assert_allclose(generated_fogm_data, fogm_data_expected, rtol=RTOL, atol=ATOL)