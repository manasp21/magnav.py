import os
import numpy as np
import pytest
from scipy.io import loadmat

# Assumptions:
# 1. MagNavPy is in PYTHONPATH or installed.
# 2. The module structure is as guessed (e.g., magnavpy.magnav, magnavpy.mpf).
#    Adjust imports if the actual structure differs.
# 3. Python versions of functions (mpf, helpers) and classes (INS, MapS) have
#    signatures and behavior compatible with this translation.
# 4. Data shapes (e.g., Cnb as (3,3,N)) are handled consistently by Python code.

try:
    from magnavpy.magnav import INS, MapS, MapCache
    # Attempt to import get_years if it exists in a known location
    # from magnavpy.utils import get_years # Example path
except ImportError:
    # Fallback or raise error if essential components are missing
    # For now, define INS, MapS, Map_Cache as placeholders if not found,
    # to allow the rest of the script structure to be shown.
    # In a real scenario, these imports must succeed.
    class INS: pass
    class MapS: pass
    class Map_Cache: pass
    print("Warning: magnavpy.magnav components not found, using placeholders.")

try:
    from magnavpy.mpf import mpf, sys_resample, part_cov, filter_exit
except ImportError:
    def mpf(*args, **kwargs): pass
    def sys_resample(*args, **kwargs): pass
    def part_cov(*args, **kwargs): pass
    def filter_exit(*args, **kwargs): pass
    print("Warning: magnavpy.mpf components not found, using placeholders.")

try:
    from magnavpy.map_utils import map_params, map_interpolate
except ImportError:
    def map_params(*args, **kwargs): return None, np.array([]) # return dummy mask
    def map_interpolate(*args, **kwargs): pass
    print("Warning: magnavpy.map_functions components not found, using placeholders.")


# Helper function for get_years (decimal year calculation)
# This should ideally be part of MagNavPy's utilities.
def get_years(year, day_of_year):
    """
    Converts year and day_of_year to a decimal year.
    Example: 2020, day 185 -> 2020 + (185-1)/days_in_year
    """
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    return year + (day_of_year - 1) / days_in_year

# Determine base path for data files
# Test script is in MagNavPy/tests/
# Data is in MagNav.jl/test/test_data/
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "MagNav.jl", "test", "test_data")

# Load data
ekf_data_path = os.path.join(TEST_DATA_DIR, "test_data_ekf.mat")
ekf_data_mat = loadmat(ekf_data_path)
# Handle cases where 'ekf_data' might be a struct-like object in the .mat file
ekf_data = ekf_data_mat['ekf_data'] if 'ekf_data' in ekf_data_mat else ekf_data_mat

ins_data_path = os.path.join(TEST_DATA_DIR, "test_data_ins.mat")
ins_data_mat = loadmat(ins_data_path)
ins_data = ins_data_mat['ins_data'] if 'ins_data' in ins_data_mat else ins_data_mat

map_data_path = os.path.join(TEST_DATA_DIR, "test_data_map.mat")
map_data_mat = loadmat(map_data_path)
map_data = map_data_mat['map_data'] if 'map_data' in map_data_mat else map_data_mat

params_path = os.path.join(TEST_DATA_DIR, "test_data_params.mat")
params_mat = loadmat(params_path)
params = params_mat['params'] if 'params' in params_mat else params_mat

traj_data_path = os.path.join(TEST_DATA_DIR, "test_data_traj.mat")
traj_data_mat = loadmat(traj_data_path)
# The Julia code uses 'traj', but .mat files might have different top-level keys
# Check common patterns for struct-like .mat files
traj_data = traj_data_mat['traj'] if 'traj' in traj_data_mat else traj_data_mat


# Extract variables from loaded data
# Ensure to handle scalar extraction from numpy arrays (e.g., val[0,0])
# Convert to float to avoid object dtypes from loadmat
P0 = ekf_data['P0'].squeeze().astype(float)
Qd = ekf_data['Qd'].squeeze().astype(float)
R_julia = ekf_data['R'].squeeze()
R = float(R_julia.item()) if R_julia.size == 1 else R_julia.astype(float) # Ensure R is scalar float or float array

ins_lat = np.deg2rad(ins_data['lat'][0][0].flatten().astype(float))
ins_lon = np.deg2rad(ins_data['lon'][0][0].flatten().astype(float))
ins_alt = ins_data['alt'].flatten().astype(float)
ins_vn = ins_data['vn'].flatten().astype(float)
ins_ve = ins_data['ve'].flatten().astype(float)
ins_vd = ins_data['vd'].flatten().astype(float)
ins_fn = ins_data['fn'].flatten().astype(float)
ins_fe = ins_data['fe'].flatten().astype(float)
ins_fd = ins_data['fd'].flatten().astype(float)
# For Cnb, ensure it's a numeric type. If it's an object array of arrays, this needs more care.
# Assuming Cnb from .mat is already a numeric array (e.g., double)
ins_Cnb = ins_data['Cnb'].astype(float)  # Shape (3, 3, N) in Julia/MAT
N = len(ins_lat)

map_info_val = map_data['map_info'] if 'map_info' in map_data.dtype.fields else "Map" # Access directly if key exists
map_info = str(map_info_val.item()) if isinstance(map_info_val, np.ndarray) and map_info_val.size == 1 else str(map_info_val)

map_map = map_data['map']
map_xx = np.deg2rad(np.array(map_data['xx'].tolist()).flatten().astype(float))
map_yy = np.deg2rad(np.array(map_data['yy'].tolist()).flatten().astype(float))
map_alt_julia = map_data['alt'].squeeze()
map_alt = float(map_alt_julia.item()) if map_alt_julia.size == 1 else map_alt_julia.astype(float)

# MagNav.map_params(map_map,map_xx,map_yy)[2] -> Python map_params(...)[1]
# Assuming map_params returns (some_other_info, mask)
_, map_mask = map_params(map_map, map_xx, map_yy)

dt = float(params['dt'].squeeze())
num_part = int(params['num_particles'].squeeze())
thresh = float(params['resampleThresh'].squeeze())
baro_tau = float(params['baro_tau'].squeeze())
acc_tau = float(params['acc_tau'].squeeze())
gyro_tau = float(params['gyro_tau'].squeeze())
fogm_tau = float(params['meas_tau'].squeeze()) # 'meas_tau' in Julia
date = get_years(2020, 185)
core = False # Default, or load if 'core' in params: bool(params['core'].squeeze())

tt = traj_data['tt'].flatten().astype(float)
mag_1_c = traj_data['mag_1_c'].flatten().astype(float)

# Create INS object
# The Julia INS constructor: INS(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb,ins_ωb=zeros(3,3,N))
# Python INS constructor signature is assumed. This is a critical assumption.
# ins_Cnb from Julia is (3,3,N). Python might expect (N,3,3).
# ins_wb (gyro biases/cov) from Julia is (3,3,N).
# For now, pass them as is, assuming Python INS handles these shapes.
ins_wb_zeros = np.zeros((3, 3, N)) # Matches Julia zeros(3,3,N)
ins_obj = INS(N, dt, tt, ins_lat, ins_lon, ins_alt, ins_vn, ins_ve, ins_vd,
              ins_fn, ins_fe, ins_fd, ins_Cnb, ins_wb_zeros)

# Create MapS object
map_obj = MapS(map_info, map_map, map_xx, map_yy, map_alt, map_mask)
map_cache_obj = Map_Cache(maps=[map_obj])
itp_mapS = map_interpolate(map_obj, method='linear') # :linear -> 'linear'

# Helper to create a sliced INS object for testing
def get_sliced_ins(sl_range, original_dt, original_tt,
                   original_lat, original_lon, original_alt,
                   original_vn, original_ve, original_vd,
                   original_fn, original_fe, original_fd,
                   original_Cnb, original_wb):
    N_sliced = len(original_tt[sl_range])
    return INS(N_sliced, original_dt, original_tt[sl_range],
               original_lat[sl_range], original_lon[sl_range], original_alt[sl_range],
               original_vn[sl_range], original_ve[sl_range], original_vd[sl_range],
               original_fn[sl_range], original_fe[sl_range], original_fd[sl_range],
               original_Cnb[:,:,sl_range], original_wb[:,:,sl_range])


# Set random seed for reproducibility
np.random.seed(2)
filt_res_1 = mpf(ins_lat, ins_lon, ins_alt, ins_vn, ins_ve, ins_vd,
                 ins_fn, ins_fe, ins_fd, ins_Cnb, mag_1_c, dt, itp_mapS,
                 P0=P0, Qd=Qd, R=R,
                 num_part=num_part, thresh=thresh,
                 baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                 fogm_tau=fogm_tau, date=date, core=core)

np.random.seed(2)
filt_res_2 = mpf(ins_obj, mag_1_c, itp_mapS,
                 P0=P0, Qd=Qd, R=R,
                 num_part=num_part, thresh=thresh,
                 baro_tau=baro_tau, acc_tau=acc_tau, gyro_tau=gyro_tau,
                 fogm_tau=fogm_tau, date=date, core=core)

# Common parameters for some specific mpf tests below, taken from global setup
common_mpf_params = {
    "P0": P0, "Qd": Qd, "R": R,
    "baro_tau": baro_tau, "acc_tau": acc_tau, "gyro_tau": gyro_tau,
    "fogm_tau": fogm_tau, "date": date, "core": core
}

@pytest.fixture(scope="module")
def mpf_results():
    return filt_res_1, filt_res_2

def test_mpf_main_comparisons(mpf_results):
    """Tests comparing two mpf outputs (raw arrays vs INS object)."""
    res1, res2 = mpf_results
    # Assuming .x is (state_dim, num_timesteps)
    np.testing.assert_allclose(res1.x[:, 0], res2.x[:, 0], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(res1.x[:, -1], res2.x[:, -1], atol=1e-7, rtol=1e-6)

    # Assuming .P is (state_dim, state_dim, num_timesteps)
    np.testing.assert_allclose(res1.P[:, :, 0], res2.P[:, :, 0], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(res1.P[:, :, -1], res2.P[:, :, -1], atol=1e-7, rtol=1e-6)

def test_mpf_convergence_flag():
    """Tests the convergence flag '.c' of the mpf filter output."""
    # Test 1: mpf with raw arrays, specific num_part and thresh
    res_c1 = mpf(ins_lat, ins_lon, ins_alt, ins_vn, ins_ve, ins_vd,
                 ins_fn, ins_fe, ins_fd, ins_Cnb, mag_1_c, dt, itp_mapS,
                 num_part=100, thresh=0.1, **common_mpf_params)
    assert res_c1.c is True

    # Test 2: mpf with INS object, specific num_part
    res_c2 = mpf(ins_obj, mag_1_c, itp_mapS, num_part=100,
                 thresh=thresh, # Using global thresh here
                 **common_mpf_params)
    assert res_c2.c is True

    # Test 3: mpf with sliced INS object and map_cache
    sl = slice(0, 10)
    ins_sliced_obj = get_sliced_ins(
        sl, dt, tt, ins_lat, ins_lon, ins_alt,
        ins_vn, ins_ve, ins_vd, ins_fn, ins_fe, ins_fd,
        ins_Cnb, ins_wb_zeros
    )
    res_c3 = mpf(ins_sliced_obj, mag_1_c[sl], map_cache_obj, num_part=100,
                 thresh=thresh, # Using global thresh
                 **common_mpf_params)
    assert res_c3.c is True

    # Test 4: mpf with zero magnetometer readings
    zero_mag = np.zeros_like(mag_1_c)
    res_c4 = mpf(ins_obj, zero_mag, itp_mapS, num_part=100,
                 thresh=thresh, # Using global thresh
                 **common_mpf_params)
    assert res_c4.c is False

def test_mpf_helper_sys_resample():
    """Tests the sys_resample helper function."""
    # Test 1: Weights [0,1,0] -> should pick middle particle (index 1) exclusively
    # Expected output: array of [1,1,1] (0-indexed)
    np.testing.assert_allclose(sys_resample(np.array([0.0, 1.0, 0.0])),
                               np.array([1, 1, 1]), atol=1e-7)

    # Test 2: Weights [0.3,0.3,0.3]
    # Julia test: res[end] ≈ 3 (1-based index)
    # Python (0-based): If output is [0,1,2], res[-1] is 2.
    # This assumes a deterministic behavior for uniform weights.
    # The exact output can vary based on the random start of systematic resampling.
    # For this test to be robust, sys_resample might need a fixed seed or be deterministic.
    # For now, assuming a typical output like [0,1,2] for weights [1/3,1/3,1/3]
    res_sr = sys_resample(np.array([0.3, 0.3, 0.3]))
    assert res_sr[-1] == pytest.approx(2) # Expecting 0-indexed last particle to be index 2

def test_mpf_helper_part_cov():
    """Tests the part_cov helper function. These tests are highly
    dependent on the specific (and somewhat unusual) behavior
    implied by the Julia tests."""

    # Test 1: part_cov([0,1,0], zeros(3,3), ones(3)) ≈ ones(3,3)
    # Assumed interpretation: if weights select one particle whose "mean" (or value)
    # is ones(3), and its "added covariance" is zeros(3,3), the result is outer(mean,mean).
    weights1 = np.array([0.0, 1.0, 0.0])
    cov_term1 = np.zeros((3, 3))
    mean_val1 = np.ones(3)
    expected1 = np.ones((3, 3)) # np.outer(mean_val1, mean_val1)
    np.testing.assert_allclose(part_cov(weights1, cov_term1, mean_val1),
                               expected1, atol=1e-7)

    # Test 2: part_cov([0,1,0], zeros(3,3), ones(3), ones(3,3)) ≈ 2*ones(3,3)
    # Assumed: fourth arg is an additional covariance matrix to add to result of previous logic.
    # So, expected1 + ones(3,3) = 2*ones(3,3)
    added_cov2 = np.ones((3,3))
    expected2 = 2 * np.ones((3, 3))
    np.testing.assert_allclose(part_cov(weights1, cov_term1, mean_val1, added_cov2),
                               expected2, atol=1e-7)

def test_mpf_helper_filter_exit():
    """Tests the filter_exit helper function. These tests are also
    highly dependent on specific behavior implied by Julia tests."""

    # Test 1: filter_exit([0][:,:], [0][:,:], 0, true) ≈ zeros(2,2,1)
    # Julia [0][:,:] -> np.array([[0.0]])
    # Assumes filter_exit returns a covariance structure, and for these inputs,
    # it's a 2x2x1 zero matrix. State dimension seems to become 2.
    x_out1 = np.array([[0.0]])
    P_out1 = np.array([[0.0]])
    N1 = 0
    coreS1 = True
    expected_P1 = np.zeros((2, 2, 1))
    # This test implies filter_exit might only return P, or its first return is P.
    # Or the test is specifically on a P-like attribute of the return.
    # For now, assume it directly returns the P-like structure.
    np.testing.assert_allclose(filter_exit(x_out1, P_out1, N1, coreS1),
                               expected_P1, atol=1e-7)

    # Test 2: filter_exit([1][:,:], [1][:,:], 1, false)[:,:,1] (Julia) ≈ [1 0; 0 1]
    # Python: result[:,:,0]
    x_out2 = np.array([[1.0]])
    P_out2 = np.array([[1.0]])
    N2 = 1
    coreS2 = False
    expected_P_slice2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    result_fe2 = filter_exit(x_out2, P_out2, N2, coreS2)
    np.testing.assert_allclose(result_fe2[:, :, 0], # Python 0-indexed slice
                               expected_P_slice2, atol=1e-7)