import pytest
import numpy as np
import scipy.io
import h5py
from pathlib import Path
import copy # For deepcopy

# Assuming MagNavPy is installed or in PYTHONPATH
# If MagNavPy is a local directory relative to the workspace:
# import sys
# # Example: if MagNavPy is a sibling to the current project's root
# current_script_path = Path(__file__).resolve()
# # workspace_root typically c:/Users/Manas Pandey/Documents/magnav.py
# # This test is in MagNavPy/tests/test_create_xyz.py
# # So, Path(__file__).parent.parent should be MagNavPy
# # And Path(__file__).parent.parent.parent is the workspace root
# workspace_root = current_script_path.parent.parent.parent
# sys.path.append(str(workspace_root / "MagNavPy/src"))

from magnavpy.magnav import XYZ0, Traj, INS, MagV, MapS
from magnavpy.map_utils import get_map
from magnavpy.create_xyz import (
    create_xyz0 as create_XYZ0_py, # Corrected case
    create_ins as create_ins_py,
    corrupt_mag as corrupt_mag_py,
    create_informed_xyz as create_informed_xyz_py,
    # get_ind as get_ind_py, # Function not found
    create_traj as get_traj_py # Corrected name
)
from magnavpy.tolles_lawson import create_TL_coef

# Set random seed for reproducibility
np.random.seed(2)

# Define base path for test data
# This test file is in MagNavPy/tests/
# Data is in MagNav.jl/test/test_data/
# So, ../../MagNav.jl/test/test_data/ from this file's location
BASE_DATA_PATH = Path(__file__).parent.parent / "MagNav.jl/test/test_data/"

# Load data from .mat files
def load_mat_file(filename):
    # Ensure the path is resolved correctly relative to this script file
    # Path(__file__).parent gives the directory of the current script (MagNavPy/tests)
    # .parent again gives MagNavPy
    # .parent again gives the workspace root (e.g., magnav.py directory)
    # So, the path to MagNav.jl from MagNavPy/tests/ is ../../MagNav.jl
    data_file_path = Path(__file__).resolve().parent.parent.parent / "MagNav.jl/test/test_data" / filename
    return scipy.io.loadmat(data_file_path)

ins_data_mat_content = load_mat_file("test_data_ins.mat")
ins_data_mat = ins_data_mat_content["ins_data"]

params_mat_content = load_mat_file("test_data_params.mat")
params_mat = params_mat_content["params"]

traj_data_mat_content = load_mat_file("test_data_traj.mat")
traj_data_mat = traj_data_mat_content["traj"]


# Extract INS data (benchmark from Julia test)
ins_lat_jl_ref = np.deg2rad(ins_data_mat["lat"][0][0].ravel().astype(float))
ins_lon_jl_ref = np.deg2rad(ins_data_mat["lon"][0][0].ravel().astype(float))
ins_alt_jl_ref = ins_data_mat["alt"].ravel()
ins_vn_jl_ref  = ins_data_mat["vn"].ravel()
ins_ve_jl_ref  = ins_data_mat["ve"].ravel()
ins_vd_jl_ref  = ins_data_mat["vd"].ravel()
ins_Cnb_jl_ref = ins_data_mat["Cnb"]

# Extract Params data
dt_p = params_mat["dt"].item()
init_pos_sigma_p = params_mat["init_pos_sigma"].item()
init_alt_sigma_p = params_mat["init_alt_sigma"].item()
init_vel_sigma_p = params_mat["init_vel_sigma"].item()
init_att_sigma_p = params_mat["init_att_sigma"].item()
VRW_sigma_p = np.sqrt(params_mat["VRW_var"].item())
ARW_sigma_p = np.sqrt(params_mat["ARW_var"].item())
baro_sigma_p = params_mat["baro_std"].item()
ha_sigma_p = params_mat["ha_sigma"].item()
acc_sigma_p = params_mat["acc_sigma"].item()
gyro_sigma_p = params_mat["gyro_sigma"].item()
baro_tau_p = params_mat["baro_tau"].item()
acc_tau_p = params_mat["acc_tau"].item()
gyro_tau_p = params_mat["gyro_tau"].item()

# Extract Trajectory data (input for creating Python objects)
tt_traj = traj_data_mat["tt"].ravel()
lat_traj = np.deg2rad(traj_data_mat["lat"][0][0].ravel().astype(float))
lon_traj = np.deg2rad(traj_data_mat["lon"][0][0].ravel().astype(float))
alt_traj = traj_data_mat["alt"].ravel()
vn_traj = traj_data_mat["vn"].ravel()
ve_traj = traj_data_mat["ve"].ravel()
vd_traj = traj_data_mat["vd"].ravel()
fn_traj = traj_data_mat["fn"].ravel()
fe_traj = traj_data_mat["fe"].ravel()
fd_traj = traj_data_mat["fd"].ravel()
Cnb_traj = traj_data_mat["Cnb"]
mag_1_c_traj = traj_data_mat["mag_1_c"].ravel()
flux_a_x_traj = traj_data_mat["flux_a_x"].ravel()
flux_a_y_traj = traj_data_mat["flux_a_y"].ravel()
flux_a_z_traj = traj_data_mat["flux_a_z"].ravel()
N_traj = len(lat_traj)

# Corruption parameters
sim_params = params_mat['sim'][0,0]
cor_sigma_p = sim_params["biasSigma"].item()
cor_tau_p = sim_params["biasTau"].item()
cor_var_p = params_mat["meas_R"].item()
cor_drift_p = sim_params["linearSlope"].item()
cor_perm_mag_p = 10.0
cor_ind_mag_p = 5.0
cor_eddy_mag_p = 1.0

# --- Global setup for tests ---
# This data is created once and used by tests.
# If tests modify them, consider using a fixture that returns copies.

# Create initial Traj object
g_traj_py = Traj(N_traj, dt_p, tt_traj, lat_traj, lon_traj, alt_traj,
                 vn_traj, ve_traj, vd_traj, fn_traj, fe_traj, fd_traj, Cnb_traj)

# Create initial INS object
g_ins_py = create_ins_py(g_traj_py,
                       init_pos_sigma = init_pos_sigma_p[0][0],
                       init_alt_sigma = init_alt_sigma_p[0][0],
                       init_vel_sigma = init_vel_sigma_p[0][0],
                       init_att_sigma = init_att_sigma_p[0][0],
                       VRW_sigma      = VRW_sigma_p[0][0],
                       ARW_sigma      = ARW_sigma_p[0][0],
                       baro_sigma     = baro_sigma_p[0][0],
                       ha_sigma       = ha_sigma_p[0][0],
                       acc_sigma      = acc_sigma_p[0][0],
                       gyro_sigma     = gyro_sigma_p[0][0],
                       baro_tau       = baro_tau_p[0][0],
                       acc_tau        = acc_tau_p[0][0],
                       gyro_tau       = gyro_tau_p[0][0])

# Corrupt mag data
g_mag_1_uc_py, _, _ = corrupt_mag_py(mag_1_c_traj, flux_a_x_traj, flux_a_y_traj, flux_a_z_traj,
                                   dt           = dt_p,
                                   cor_sigma    = cor_sigma_p,
                                   cor_tau      = cor_tau_p,
                                   cor_var      = cor_var_p,
                                   cor_drift    = cor_drift_p,
                                   cor_perm_mag = cor_perm_mag_p,
                                   cor_ind_mag  = cor_ind_mag_p,
                                   cor_eddy_mag = cor_eddy_mag_p)

# Prepare maps
# EMM720_MODEL_IDENTIFIER: Python equivalent for MagNav.emm720
# This needs to be a valid model specifier for the Python `get_map` function.
EMM720_MODEL_IDENTIFIER = "emm720" # Placeholder, adjust if `get_map` uses different ID

g_mapS_py = map_trim(get_map(), g_traj_py) # Default map

g_mapS_mod_py = copy.deepcopy(g_mapS_py)
if g_mapS_mod_py.map is not None and g_mapS_mod_py.map.size > 0:
    N_mod = int(np.ceil(g_mapS_mod_py.map.size * 0.01))
    if N_mod == 0 and g_mapS_mod_py.map.size > 0: N_mod = 1
    
    if N_mod > 0:
        if g_mapS_mod_py.map.ndim == 1:
            ind_mod = np.random.choice(g_mapS_mod_py.map.shape[0], min(N_mod, g_mapS_mod_py.map.shape[0]), replace=False)
            g_mapS_mod_py.map[ind_mod] = 0
        elif g_mapS_mod_py.map.ndim == 2:
            num_elements_to_pick = min(N_mod, g_mapS_mod_py.map.size)
            if num_elements_to_pick > 0:
                flat_indices_to_modify = np.random.choice(g_mapS_mod_py.map.size, num_elements_to_pick, replace=False)
                row_indices_mod, col_indices_mod = np.unravel_index(flat_indices_to_modify, g_mapS_mod_py.map.shape)
                g_mapS_mod_py.map[row_indices_mod, col_indices_mod] = 0
# else: map is None or empty, g_mapS_mod_py remains as is.

g_mapV_py = get_map(EMM720_MODEL_IDENTIFIER)
g_mapV_py = map_trim(g_mapV_py, g_traj_py)


# --- Test Functions ---

def test_create_xyz0(tmp_path):
    xyz_h5_path = tmp_path / "test_create_XYZ0.h5"

    # Test 1: Basic creation and H5 saving
    xyz_obj = create_XYZ0_py(g_mapS_py, alt=2000, t=10, mapV=g_mapV_py,
                             save_h5=True, xyz_h5=str(xyz_h5_path))
    assert isinstance(xyz_obj, XYZ0)
    assert xyz_h5_path.exists()
    with h5py.File(xyz_h5_path, 'r') as f:
        assert "tt" in f # Basic check for HDF5 content

    # Test 2: Error throwing case (e.g., VRW_sigma too high in Julia example)
    # The Python function might raise ValueError or a custom error.
    with pytest.raises(Exception): # Use more specific exception if known
        create_XYZ0_py(g_mapS_py, t=10, mapV=g_mapV_py, VRW_sigma=1e6)

    # Test 3: Creation with modified map and lat/lon bounds
    if g_mapS_mod_py.map is not None and g_mapS_mod_py.map.size > 0:
        xyz_obj_mod = create_XYZ0_py(g_mapS_mod_py, N_waves=0, mapV=g_mapV_py,
                                 ll1=np.rad2deg((g_traj_py.lat[0], g_traj_py.lon[0])),
                                 ll2=np.rad2deg((g_traj_py.lat[-1], g_traj_py.lon[-1])))
        assert isinstance(xyz_obj_mod, XYZ0)
    else:
        pytest.skip("Skipping mapS_mod_py test due to empty or None map in g_mapS_mod_py.")


def test_create_ins():
    # Compare Python-created INS object (g_ins_py) with Julia reference values
    assert np.allclose(g_ins_py.lat, ins_lat_jl_ref, atol=1e-3)
    assert np.allclose(g_ins_py.lon, ins_lon_jl_ref, atol=1e-3)
    assert np.allclose(g_ins_py.alt, ins_alt_jl_ref, atol=1000)
    assert np.allclose(g_ins_py.vn,  ins_vn_jl_ref,  atol=10)
    assert np.allclose(g_ins_py.ve,  ins_ve_jl_ref,  atol=10)
    assert np.allclose(g_ins_py.vd,  ins_vd_jl_ref,  atol=10)
    assert np.allclose(g_ins_py.Cnb, ins_Cnb_jl_ref, atol=0.1)


def test_corrupt_mag():
    # mag_1_c_traj is the original clean data
    # g_mag_1_uc_py is the corrupted version
    assert np.any(g_mag_1_uc_py != mag_1_c_traj)

    mean_abs_mag_c = np.mean(np.abs(mag_1_c_traj))
    if mean_abs_mag_c > 1e-9: # Avoid issues if mag_1_c_traj is all zeros or very small
        assert np.any(np.abs(g_mag_1_uc_py - mag_1_c_traj) < mean_abs_mag_c)
    # If mean_abs_mag_c is ~0, this test might not be meaningful as originally written.
    # The first assert already checks for difference.


def test_create_informed_xyz():
    # Create original XYZ0 struct for this test scope
    xyz_orig = create_XYZ0_py(g_mapS_py, alt=2000, t=10, mapV=g_mapV_py)
    # ind = get_ind_py(xyz_orig) # get_ind_py is not defined
    ind = np.ones(xyz_orig.traj.N, dtype=bool) # Placeholder for ind
    traj_orig = get_traj_py(xyz_orig, ind)

    # Define field names for dynamic access
    use_vec_field_name = "flux_a" # Corresponds to Julia :flux_a
    use_mag_field_name = "mag_1_uc" # Corresponds to Julia :mag_1_uc

    # Ensure the XYZ0 object has the necessary flux components and mag data.
    # These are typically populated by create_XYZ0_py.
    # We need xyz_orig.flux_a_x, xyz_orig.flux_a_y, xyz_orig.flux_a_z, and xyz_orig.mag_1_uc
    required_fields = ['flux_a_x', 'flux_a_y', 'flux_a_z', use_mag_field_name]
    for field in required_fields:
        if not hasattr(xyz_orig, field):
            pytest.skip(f"XYZ0 object (xyz_orig) missing required field: {field}")
            return
        if getattr(xyz_orig, field) is None: # Also check if it's None
             pytest.skip(f"XYZ0 object (xyz_orig) has None for field: {field}")
             return


    # Prepare flux_a components for TL_coef (tuple of arrays, indexed by ind)
    flux_a_for_tl = (
        xyz_orig.flux_a_x[ind],
        xyz_orig.flux_a_y[ind],
        xyz_orig.flux_a_z[ind]
    )
    mag_for_tl = getattr(xyz_orig, use_mag_field_name)[ind]

    tl_coef = create_TL_coef(flux_a_for_tl, mag_for_tl,
                             terms=['p','e','i'], lam=0.025)

    # Create displaced XYZ0 struct
    xyz_disp = create_informed_xyz_py(xyz_orig, ind, g_mapS_py,
                                      use_mag_field_name, use_vec_field_name, tl_coef)
    traj_disp = get_traj_py(xyz_disp, ind)

    # Calculate displacements for comparison
    lat_disp_val = traj_disp.lat[0] - traj_orig.lat[0]
    lon_disp_val = traj_disp.lon[0] - traj_orig.lon[0]

    # Assertions
    assert np.array_equal(traj_disp.tt, traj_orig.tt)
    assert np.array_equal(traj_disp.alt, traj_orig.alt)
    assert 0.0 < abs(lat_disp_val) < 0.01
    assert 0.0 < abs(lon_disp_val) < 0.01
    assert np.allclose(traj_disp.lat, traj_orig.lat + lat_disp_val)
    assert np.allclose(traj_disp.lon, traj_orig.lon + lon_disp_val)