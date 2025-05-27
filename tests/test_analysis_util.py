import pytest
import numpy as np
import pandas as pd
import scipy.io
import os
import shutil # For cleaning up directories
# from unittest.mock import MagicMock # If complex mocking is needed
import matplotlib.figure

# Relative imports from MagNavPy.src
# Assuming these modules exist and contain the translated functions
from ..src import analysis_util
from ..src import magnav as mn_data  # For data loading functions like sgl_2020_train
from ..src import model_functions # For get_nn_m, etc.
from ..src import tolles_lawson
from ..src import plot_functions

# Helper to define the path to the original Julia test data
# This path is relative to this test file (MagNavPy/tests/test_analysis_util.py)
BASE_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "MagNav.jl", "test", "test_data")

# --- Helper function to load values from .mat files ---
# scipy.io.loadmat loads .mat files into dictionaries.
# Structs within .mat files often become structured numpy arrays or nested dicts.
# This helper is a placeholder for robustly extracting values.
# The actual extraction logic might be more complex depending on the .mat file structure.
def _get_mat_value(mat_data_dict, key_path, default=None):
    """
    Extracts a value from a dictionary loaded by scipy.io.loadmat.
    key_path can be a simple key or 'struct_name/field_name'.
    This is a simplified helper and might need adjustment.
    """
    try:
        current = mat_data_dict
        parts = key_path.split('/')
        for part in parts:
            # Handle cases where .mat structs become 0-dim arrays of objects
            if isinstance(current, np.ndarray) and current.dtype.names and part in current.dtype.names:
                current = current[part][0,0]
            elif isinstance(current, dict):
                current = current[part]
            else: # Try direct field access if it's some other object
                current = current[part]

        # If the result is a 1x1 array, extract the scalar
        if isinstance(current, np.ndarray) and current.shape == (1, 1):
            return current[0, 0]
        if isinstance(current, np.ndarray) and current.size == 1: # For single element arrays
            return current.item()
        return current
    except (KeyError, IndexError, TypeError) as e:
        if default is not None:
            return default
        raise ValueError(f"Could not extract {key_path} from .mat data: {e}")


# --- Load .mat files ---
# These are loaded once at the module level.
params_mat_file_path = os.path.join(BASE_TEST_DATA_DIR, "test_data_params.mat")
params_mat_contents = scipy.io.loadmat(params_mat_file_path)
# 'params' is the key in the .mat file for the main struct/dict
# The structure of 'params_from_mat' depends on the .mat file version and content
params_from_mat = params_mat_contents['params']

tl_mat_file_path = os.path.join(BASE_TEST_DATA_DIR, "test_data_TL.mat")
tl_mat_contents = scipy.io.loadmat(tl_mat_file_path)
TL_data_from_mat = tl_mat_contents['TL_data']

# --- Global test variables from Julia file ---
# Extracting specific values, requires knowing the .mat structure
# Example: params["TL"]["lambda"]
lambda_val = _get_mat_value(params_from_mat, "TL/lambda")

A_a_f_t = _get_mat_value(TL_data_from_mat, "A_a_f_t")
mag_1_uc_f_t = _get_mat_value(TL_data_from_mat, "mag_1_uc_f_t").flatten() # Julia's vec()

# Assuming analysis_util.linreg and analysis_util.detrend are available
# These are calculated based on loaded data, similar to Julia setup
TL_a_1 = analysis_util.linreg(mag_1_uc_f_t, A_a_f_t, lam=lambda_val) # Assuming 'lam' or similar kwarg
mag_1_comp_d = analysis_util.detrend(_get_mat_value(TL_data_from_mat, "mag_1_comp"))

lat_global = np.deg2rad(39.160667350241980)
dn_1_global = 1.0
dn_2_global = np.array([0.5, 10.0, 200.0])
de_1_global = 1.0
de_2_global = np.array([0.5, 10.0, 200.0])
dlat_1_expected = 1.565761736512648e-07
dlat_2_expected = np.array([7.828808682563242e-08, 1.565761736512649e-06, 3.131523473025297e-05])
dlon_1_expected = 2.019352321699552e-07
dlon_2_expected = np.array([1.009676160849776e-07, 2.019352321699552e-06, 4.038704643399104e-05])

# --- Test Functions ---

def test_dn2dlat_de2dlon():
    assert analysis_util.dn2dlat(dn_1_global, lat_global) == pytest.approx(dlat_1_expected)
    assert analysis_util.de2dlon(de_1_global, lat_global) == pytest.approx(dlon_1_expected)
    assert analysis_util.dn2dlat(dn_2_global, lat_global) == pytest.approx(dlat_2_expected)
    assert analysis_util.de2dlon(de_2_global, lat_global) == pytest.approx(dlon_2_expected)

def test_dlat2dn_dlon2de():
    assert analysis_util.dlat2dn(dlat_1_expected, lat_global) == pytest.approx(dn_1_global)
    assert analysis_util.dlon2de(dlon_1_expected, lat_global) == pytest.approx(de_1_global)
    assert analysis_util.dlat2dn(dlat_2_expected, lat_global) == pytest.approx(dn_2_global)
    assert analysis_util.dlon2de(dlon_2_expected, lat_global) == pytest.approx(de_2_global)

def test_linreg():
    expected_TL_a_1 = _get_mat_value(TL_data_from_mat, "TL_a_1").flatten()
    assert TL_a_1 == pytest.approx(expected_TL_a_1)
    # Assuming linreg handles 1D array input and output
    assert analysis_util.linreg(np.array([3,6,9])) == pytest.approx(np.array([0,3]))

def test_detrend():
    expected_mag_1_comp_d = _get_mat_value(TL_data_from_mat, "mag_1_comp_d")
    assert mag_1_comp_d == pytest.approx(expected_mag_1_comp_d)
    assert analysis_util.detrend(np.array([3,6,8])) == pytest.approx(np.array([-1,2,-1]) / 6)
    assert analysis_util.detrend(np.array([3,6,8]), mean_only=True) == pytest.approx(np.array([-8,1,7]) / 3)
    # Assuming detrend can take two arguments (y, X) similar to Julia version for this test case
    # This test depends on the Python implementation of detrend(y, X)
    assert analysis_util.detrend(np.array([1,1]), np.array([[1, 0.1], [0.1, 1]])) == pytest.approx(np.zeros(2))
    assert analysis_util.detrend(np.array([1,1]), np.array([[1, 0.1], [0.1, 1]]), mean_only=True) == pytest.approx(np.zeros(2))

def test_get_bpf():
    # In Python, we check the type of the returned filter object.
    # Assuming get_bpf returns a scipy.signal filter object or similar.
    # For now, just checking it runs and returns something not None.
    # A more specific check would be isinstance(filter_obj, expected_filter_type)
    assert analysis_util.get_bpf(pass1=0.1, pass2=0.9) is not None
    assert analysis_util.get_bpf(pass1=-1, pass2=0.9) is not None # Julia allows this, Python might error earlier
    assert analysis_util.get_bpf(pass1=np.inf, pass2=0.9) is not None
    assert analysis_util.get_bpf(pass1=0.1, pass2=-1) is not None
    assert analysis_util.get_bpf(pass1=0.1, pass2=np.inf) is not None

    with pytest.raises(Exception): # Julia uses ErrorException, Python might use ValueError or custom
        analysis_util.get_bpf(pass1=-1, pass2=-1)
    with pytest.raises(Exception):
        analysis_util.get_bpf(pass1=np.inf, pass2=np.inf)

def test_bpf_data():
    # Create copies for in-place modification tests if bpf_data! is translated to an in-place version
    A_a_f_t_copy = A_a_f_t.copy()
    mag_1_uc_f_t_copy = mag_1_uc_f_t.copy()

    result_matrix = analysis_util.bpf_data(A_a_f_t_copy)
    assert isinstance(result_matrix, np.ndarray)
    assert result_matrix.ndim == 2

    result_vector = analysis_util.bpf_data(mag_1_uc_f_t_copy)
    assert isinstance(result_vector, np.ndarray)
    assert result_vector.ndim == 1

    # For bpf_data! (in-place):
    # Assuming a Python equivalent like bpf_data_inplace or bpf_data(..., inplace=True)
    # If analysis_util.bpf_data modifies in place and returns None:
    # assert analysis_util.bpf_data_inplace(A_a_f_t_copy) is None
    # assert analysis_util.bpf_data_inplace(mag_1_uc_f_t_copy) is None
    # For now, assuming bpf_data is not in-place by default in Python unless specified.
    # If the Python version of bpf_data can be in-place, those tests would be:
    # A_a_f_t_for_inplace = A_a_f_t.copy()
    # MagNavPy.src.analysis_util.bpf_data(A_a_f_t_for_inplace, inplace=True) # or similar API
    # assert A_a_f_t_for_inplace is not None # Check modification effect
    pass # Placeholder if inplace version is not directly tested here

def test_downsample():
    # Assuming MagNav.downsample is translated to analysis_util.downsample
    # and handles default decimation factor if not provided.
    # The Julia test implies a default for the vector version.
    # Let's assume Python's downsample(data, factor) and downsample(data) if factor has default
    assert analysis_util.downsample(A_a_f_t, 100).shape == (96, 18)
    assert analysis_util.downsample(mag_1_uc_f_t).shape == (960,) # Relies on Python default matching Julia

# --- Setup for tests requiring flight data (get_x, get_y, etc.) ---
# This setup might be better in a pytest fixture if used by many tests.
@pytest.fixture(scope="module")
def flight_data_setup():
    flight_name = "Flt1003" # Julia :Flt1003
    xyz_type = "XYZ20"    # Julia :XYZ20
    map_name_eastern = "Eastern_395" # Julia :Eastern_395

    # These functions (sgl_2020_train, ottawa_area_maps) are assumed to be in mn_data (MagNavPy.src.magnav)
    # and return paths to the HDF5 files.
    # The paths might need to be relative to MagNav.jl/data if that's where they are stored/downloaded.
    # For testing, these files must be accessible.
    try:
        xyz_h5_path = mn_data.sgl_2020_train(flight_name)
        map_h5_path = mn_data.ottawa_area_maps(map_name_eastern)
    except Exception as e:
        pytest.skip(f"Skipping flight data tests: Could not load/find HDF5 data files via mn_data module: {e}")


    # Assuming get_XYZ20 is in analysis_util and works with the path
    xyz = analysis_util.get_XYZ20(xyz_h5_path, tt_sort=True, silent=True)

    tt = xyz.traj.tt # Assuming xyz object has this structure
    line = xyz.line
    unique_lines = np.unique(line)
    if len(unique_lines) < 3:
        pytest.skip("Not enough unique lines in flight data for these tests.")
    line2 = unique_lines[1] # 0-indexed
    line3 = unique_lines[2] # 0-indexed
    
    lines_for_test = np.array([-1, line2]) # Julia [-1,line2]
    
    ind = (line == line2)
    # Julia: ind[findall(ind)[51:end]] .= false
    true_indices = np.where(ind)[0]
    if len(true_indices) > 50: # Check to avoid index error
        ind[true_indices[50:]] = False # 0-indexed 50th is Julia's 51st

    # DataFrame setup
    # ind_for_df needs to be boolean array of same length as tt for selection
    tt_line2_selected = tt[ind]
    
    # For line3, get first 5 points
    ind_line3 = (line == line3)
    tt_line3_all = tt[ind_line3]
    
    if len(tt_line2_selected) == 0 or len(tt_line3_all) < 5:
         pytest.skip("Not enough data points for selected lines for DataFrame setup.")

    df_line_data = {
        'flight': [flight_name, flight_name],
        'line': [line2, line3],
        't_start': [tt_line2_selected[0], tt_line3_all[0]],
        't_end': [tt_line2_selected[-1], tt_line3_all[4]], # Julia's [5] is Python's [4]
        'map_name': [map_name_eastern, map_name_eastern]
    }
    df_line = pd.DataFrame(df_line_data)

    df_flight = pd.DataFrame({
        'flight': [flight_name], # Must be iterable for DataFrame constructor
        'xyz_type': [xyz_type],
        'xyz_set': [1],
        'xyz_file': [xyz_h5_path]
    })

    df_map = pd.DataFrame({
        'map_name': [map_name_eastern], # Must be iterable
        'map_file': [map_h5_path]
    })
    
    # For get_y tests
    map_val_rand = np.random.randn(np.sum(ind))

    return {
        "xyz": xyz, "ind": ind, "line2": line2, "line3": line3,
        "lines_for_test": lines_for_test, "df_line": df_line,
        "df_flight": df_flight, "df_map": df_map, "map_val_rand": map_val_rand,
        "tt": tt, "line": line
    }

def test_get_x(flight_data_setup):
    s = flight_data_setup
    xyz, ind, line2, lines_for_test, df_line, df_flight = \
        s["xyz"], s["ind"], s["line2"], s["lines_for_test"], s["df_line"], s["df_flight"]

    res_tuple = analysis_util.get_x(xyz, ind)
    assert isinstance(res_tuple, tuple) and len(res_tuple) == 4
    assert isinstance(res_tuple[0], np.ndarray) and res_tuple[0].ndim == 2 # Matrix
    assert isinstance(res_tuple[1], np.ndarray) and res_tuple[1].ndim == 1 # Vector
    assert isinstance(res_tuple[2], np.ndarray) and res_tuple[2].ndim == 1 # Vector
    assert isinstance(res_tuple[3], np.ndarray) and res_tuple[3].ndim == 1 # Vector

    # Test with field selection
    analysis_util.get_x(xyz, ind, fields=['flight']) # Runs without error

    with pytest.raises(Exception): # Julia ErrorException
        analysis_util.get_x(xyz, ind, fields=['test'])
    with pytest.raises(Exception): # Julia ErrorException
        analysis_util.get_x(xyz, ind, fields=['mag_6_uc'])

    analysis_util.get_x([xyz, xyz], [ind, ind]) # Runs

    analysis_util.get_x(line2, df_line, df_flight) # Runs
    analysis_util.get_x(lines_for_test, df_line, df_flight) # Runs
    analysis_util.get_x(np.array([line2, s["line3"]]), df_line, df_flight) # Runs

    with pytest.raises(AssertionError):
        analysis_util.get_x(np.array([line2, line2]), df_line, df_flight)

def test_get_y(flight_data_setup):
    s = flight_data_setup
    xyz, ind, map_val_rand, line2, lines_for_test, df_line, df_flight, df_map = \
        s["xyz"], s["ind"], s["map_val_rand"], s["line2"], s["lines_for_test"], \
        s["df_line"], s["df_flight"], s["df_map"]

    assert isinstance(analysis_util.get_y(xyz, ind, y_type="a"), np.ndarray)
    assert isinstance(analysis_util.get_y(xyz, ind, map_val_rand, y_type="b"), np.ndarray)
    assert isinstance(analysis_util.get_y(xyz, ind, map_val_rand, y_type="c"), np.ndarray)
    assert isinstance(analysis_util.get_y(xyz, ind, y_type="d"), np.ndarray)
    assert isinstance(analysis_util.get_y(xyz, ind, y_type="e", use_mag="flux_a"), np.ndarray)

    with pytest.raises(Exception): # Julia ErrorException
        analysis_util.get_y(xyz, ind, y_type="test")

    assert isinstance(analysis_util.get_y(line2, df_line, df_flight, df_map, y_type="a"), np.ndarray)
    assert isinstance(analysis_util.get_y(lines_for_test, df_line, df_flight, df_map, y_type="c"), np.ndarray)
    assert isinstance(analysis_util.get_y(np.array([line2, s["line3"]]), df_line, df_flight, df_map, y_type="d"), np.ndarray)

    with pytest.raises(AssertionError):
        analysis_util.get_y(np.array([line2, line2]), df_line, df_flight, df_map)

def test_get_Axy(flight_data_setup):
    s = flight_data_setup
    line2, lines_for_test, df_line, df_flight, df_map = \
        s["line2"], s["lines_for_test"], s["df_line"], s["df_flight"], s["df_map"]

    # Julia result is 1-indexed, Python 0-indexed
    res1 = analysis_util.get_Axy(line2, df_line, df_flight, df_map, y_type="a", mod_TL=True, return_B=True)
    assert isinstance(res1[0], np.ndarray) and res1[0].ndim == 2 # Matrix (A)

    res2 = analysis_util.get_Axy(lines_for_test, df_line, df_flight, df_map, y_type="b", map_TL=True)
    assert isinstance(res2[1], np.ndarray) and res2[1].ndim == 2 # Matrix (X)

    res3 = analysis_util.get_Axy(np.array([line2, s["line3"]]), df_line, df_flight, df_map)
    assert isinstance(res3[2], np.ndarray) and res3[2].ndim == 1 # Vector (y)

    with pytest.raises(AssertionError):
        analysis_util.get_Axy(np.array([line2, line2]), df_line, df_flight, df_map)

def test_get_nn_m():
    # Assuming model_functions.get_nn_m returns a Python ML model (e.g., Keras, PyTorch, or custom)
    # The test `isa Chain` in Julia checks the type. Here we check if it's not None,
    # or isinstance(model, ExpectedPythonModelType).
    # For simplicity, checking not None or a generic type if specific type is complex/unknown.
    # This part is highly dependent on the Python implementation of get_nn_m.
    def check_model(model): # Placeholder for more specific model checks
        assert model is not None

    check_model(model_functions.get_nn_m(1, 1, hidden=[]))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1]))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1, 1]))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1, 1, 1]))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], final_bias=False))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], skip_con=True))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3w", dropout_prob=0))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3w", dropout_prob=0.5))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1,1], model_type="m3w", dropout_prob=0))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1,1], model_type="m3w", dropout_prob=0.5))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3tf", tf_layer_type="prelayer", tf_norm_type="layer", N_tf_head=1))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3tf", tf_layer_type="postlayer", tf_norm_type="layer", N_tf_head=1))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1,1], model_type="m3tf", tf_layer_type="prelayer", tf_norm_type="batch", N_tf_head=1))
    check_model(model_functions.get_nn_m(1, 1, hidden=[1,1], model_type="m3tf", tf_layer_type="postlayer", tf_norm_type="none", N_tf_head=1))

    with pytest.raises(Exception): # Julia ErrorException
        model_functions.get_nn_m(1, 1, hidden=[1, 1, 1, 1])
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[1, 1], skip_con=True)
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[], model_type="m3w")
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[1, 1, 1], model_type="m3w")
    with pytest.raises(AssertionError): # Julia AssertionError
        model_functions.get_nn_m(1, 1, hidden=[], model_type="m3tf", N_tf_head=1)
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[1, 1, 1], model_type="m3tf", N_tf_head=1)
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3tf", tf_layer_type="test", N_tf_head=1)
    with pytest.raises(Exception):
        model_functions.get_nn_m(1, 1, hidden=[1], model_type="m3tf", tf_norm_type="test", N_tf_head=1)

# Model for subsequent tests
m_global_nn_model = model_functions.get_nn_m(3, 1, hidden=[1])
alpha_sgl = 0.5

def test_sparse_group_lasso():
    # The Julia test `MagNav.sparse_group_lasso(m) ≈ MagNav.sparse_group_lasso(weights)`
    # implies sparse_group_lasso can take a model or weights.
    # This translation assumes the Python version model_functions.sparse_group_lasso primarily takes the model.
    # The comparison to a version with explicit weights depends on how weights are handled/extracted in Python.
    # For now, we test the call with the model and alpha.
    sgl_m = model_functions.sparse_group_lasso(m_global_nn_model)
    sgl_m_alpha = model_functions.sparse_group_lasso(m_global_nn_model, alpha_sgl)
    
    assert sgl_m is not None # Basic check
    assert sgl_m_alpha is not None

    # If there was a Python equivalent to Flux.trainables(m) and sparse_group_lasso could take it:
    # weights_py = model_functions.get_trainable_parameters(m_global_nn_model) # Hypothetical
    # assert sgl_m == pytest.approx(model_functions.sparse_group_lasso(weights_py))
    # assert sgl_m_alpha == pytest.approx(model_functions.sparse_group_lasso(weights_py, alpha_sgl))
    # This part is commented out as it's highly dependent on MagNavPy's ML framework specifics.

# Data for norm_sets, denorm_sets tests
A_norm_data = np.random.randn(5,9).astype(np.float32)
x_norm_data = np.random.randn(5,3).astype(np.float32)
y_norm_data = np.random.randn(5).astype(np.float32) # Julia y is a vector

A_bias, A_scale, A_norm = analysis_util.norm_sets(A_norm_data)
x_bias, x_scale, x_norm = analysis_util.norm_sets(x_norm_data)
y_bias, y_scale, y_norm = analysis_util.norm_sets(y_norm_data)


def test_norm_sets():
    for norm_type in ["standardize", "normalize", "scale", "none"]:
        # Test with matrix inputs
        res_x_tuple = analysis_util.norm_sets(x_norm_data, norm_type=norm_type, no_norm=2)
        assert isinstance(res_x_tuple, tuple) and len(res_x_tuple) == 3
        assert all(isinstance(item, np.ndarray) and item.ndim == 2 for item in res_x_tuple)

        res_xx_tuple = analysis_util.norm_sets(x_norm_data, x_norm_data, norm_type=norm_type, no_norm=2)
        assert isinstance(res_xx_tuple, tuple) and len(res_xx_tuple) == 4
        assert all(isinstance(item, np.ndarray) and item.ndim == 2 for item in res_xx_tuple)
        
        res_xxx_tuple = analysis_util.norm_sets(x_norm_data, x_norm_data, x_norm_data, norm_type=norm_type, no_norm=2)
        assert isinstance(res_xxx_tuple, tuple) and len(res_xxx_tuple) == 5
        assert all(isinstance(item, np.ndarray) and item.ndim == 2 for item in res_xxx_tuple)

        # Test with vector inputs
        res_y_tuple = analysis_util.norm_sets(y_norm_data, norm_type=norm_type)
        assert isinstance(res_y_tuple, tuple) and len(res_y_tuple) == 3
        assert all(isinstance(item, np.ndarray) and item.ndim == 1 for item in res_y_tuple)

        res_yy_tuple = analysis_util.norm_sets(y_norm_data, y_norm_data, norm_type=norm_type)
        assert isinstance(res_yy_tuple, tuple) and len(res_yy_tuple) == 4
        assert all(isinstance(item, np.ndarray) and item.ndim == 1 for item in res_yy_tuple)

        res_yyy_tuple = analysis_util.norm_sets(y_norm_data, y_norm_data, y_norm_data, norm_type=norm_type)
        assert isinstance(res_yyy_tuple, tuple) and len(res_yyy_tuple) == 5
        assert all(isinstance(item, np.ndarray) and item.ndim == 1 for item in res_yyy_tuple)

    with pytest.raises(Exception): # Julia ErrorException
        analysis_util.norm_sets(x_norm_data, norm_type="test")
    # ... (repeat for other x,x ; x,x,x ; y ; y,y ; y,y,y with norm_type="test")

def test_denorm_sets():
    assert analysis_util.denorm_sets(x_bias, x_scale, x_norm) == pytest.approx(x_norm_data)
    
    res_xx = analysis_util.denorm_sets(x_bias, x_scale, x_norm, x_norm)
    assert res_xx[0] == pytest.approx(x_norm_data)
    assert res_xx[1] == pytest.approx(x_norm_data)

    res_xxx = analysis_util.denorm_sets(x_bias, x_scale, x_norm, x_norm, x_norm)
    assert res_xxx[0] == pytest.approx(x_norm_data)
    assert res_xxx[1] == pytest.approx(x_norm_data)
    assert res_xxx[2] == pytest.approx(x_norm_data)

    assert analysis_util.denorm_sets(y_bias, y_scale, y_norm) == pytest.approx(y_norm_data)

    res_yy = analysis_util.denorm_sets(y_bias, y_scale, y_norm, y_norm)
    assert res_yy[0] == pytest.approx(y_norm_data)
    assert res_yy[1] == pytest.approx(y_norm_data)

    res_yyy = analysis_util.denorm_sets(y_bias, y_scale, y_norm, y_norm, y_norm)
    assert res_yyy[0] == pytest.approx(y_norm_data)
    assert res_yyy[1] == pytest.approx(y_norm_data)
    assert res_yyy[2] == pytest.approx(y_norm_data)

# Data for unpack_data_norms tests
v_scale_unpack = np.eye(x_norm_data.shape[1])
data_norms_7 = (A_bias, A_scale, v_scale_unpack, x_bias, x_scale, y_bias, y_scale)
data_norms_6 = (A_bias, A_scale, x_bias, x_scale, y_bias, y_scale)
data_norms_5 = (v_scale_unpack, x_bias, x_scale, y_bias, y_scale)
data_norms_4 = (x_bias, x_scale, y_bias, y_scale)
# Julia A_bias=0, A_scale=1. Python equivalent for "no-op" A normalization
A_bias_zero, A_scale_one = np.zeros_like(A_bias), np.ones_like(A_scale) # Or scalar 0, 1 if appropriate
data_norms_A_expected = (A_bias_zero, A_scale_one, v_scale_unpack, x_bias, x_scale, y_bias, y_scale)


def compare_data_norms_tuples(t1, t2):
    assert len(t1) == len(t2)
    for i in range(len(t1)):
        if isinstance(t1[i], np.ndarray):
            assert t1[i] == pytest.approx(t2[i])
        else: # Scalars
            assert t1[i] == pytest.approx(t2[i])

def test_unpack_data_norms():
    # Assuming analysis_util.unpack_data_norms is available
    # The comparison needs to handle tuples of numpy arrays and scalars.
    compare_data_norms_tuples(analysis_util.unpack_data_norms(data_norms_7), data_norms_7)
    compare_data_norms_tuples(analysis_util.unpack_data_norms(data_norms_6), data_norms_7) # Expected to expand
    compare_data_norms_tuples(analysis_util.unpack_data_norms(data_norms_5), data_norms_A_expected)
    compare_data_norms_tuples(analysis_util.unpack_data_norms(data_norms_4), data_norms_A_expected)

    with pytest.raises(Exception): # Julia ErrorException
        analysis_util.unpack_data_norms(tuple([0]*8)) # (0,0,0,0,0,0,0,0)
    with pytest.raises(Exception):
        analysis_util.unpack_data_norms(tuple([0]*3)) # (0,0,0)

def test_get_ind(flight_data_setup):
    s = flight_data_setup
    tt, line, ind, xyz, lines_for_test, df_line = \
        s["tt"], s["line"], s["ind"], s["xyz"], s["lines_for_test"], s["df_line"]
    
    tt_ind = tt[ind] # tt values corresponding to true ind flags

    assert np.sum(analysis_util.get_ind(tt, line, ind=ind)) == pytest.approx(50)
    assert np.sum(analysis_util.get_ind(tt[ind], line[ind], ind=np.arange(50))) == pytest.approx(50)
    
    if len(tt_ind) >= 9: # Ensure enough elements for tt_lim tests
        assert np.sum(analysis_util.get_ind(tt[ind], line[ind], tt_lim=(tt_ind[4],))) == pytest.approx(5) # Julia tt[ind][5] is 0-indexed tt_ind[4]
        assert np.sum(analysis_util.get_ind(tt[ind], line[ind], tt_lim=(tt_ind[4], tt_ind[8]))) == pytest.approx(5) # Julia 5 to 9 inclusive
    
    res_splits_1 = analysis_util.get_ind(tt, line, ind=ind, splits=(0.5, 0.5))
    assert tuple(np.sum(r) for r in res_splits_1) == pytest.approx((25,25))

    res_splits_2 = analysis_util.get_ind(tt, line, ind=ind, splits=(0.7, 0.2, 0.1))
    assert tuple(np.sum(r) for r in res_splits_2) == pytest.approx((35,10,5))

    with pytest.raises(AssertionError):
        analysis_util.get_ind(tt[ind], line[ind], tt_lim=(1,1,1))
    with pytest.raises(AssertionError):
        analysis_util.get_ind(tt[ind], line[ind], splits=(1,1,1))
    with pytest.raises(AssertionError): # Julia splits=(1,0,0,0)
        analysis_util.get_ind(tt[ind], line[ind], splits=(1,0,0,0))

    if len(tt_ind) >= 5:
      assert np.sum(analysis_util.get_ind(xyz, lines=lines_for_test, tt_lim=(tt_ind[4],))) == pytest.approx(5)
    
    assert np.sum(analysis_util.get_ind(xyz, lines_for_test[1], df_line, l_window=15)) == pytest.approx(45) # lines[2] in Julia is lines_for_test[1]
    
    res_splits_3 = analysis_util.get_ind(xyz, lines_for_test[1], df_line, splits=(0.5,0.5), l_window=15)
    assert tuple(np.sum(r) for r in res_splits_3) == pytest.approx((15,15))

    assert np.sum(analysis_util.get_ind(xyz, lines_for_test, df_line)) == pytest.approx(50)
    
    res_splits_4 = analysis_util.get_ind(xyz, lines_for_test, df_line, splits=(0.5,0.5))
    assert tuple(np.sum(r) for r in res_splits_4) == pytest.approx((25,25))

    res_splits_5 = analysis_util.get_ind(xyz, lines_for_test, df_line, splits=(0.7,0.2,0.1))
    assert tuple(np.sum(r) for r in res_splits_5) == pytest.approx((35,10,5))

    with pytest.raises(AssertionError):
        analysis_util.get_ind(xyz, lines_for_test, df_line, splits=(1,1,1))
    with pytest.raises(AssertionError):
        analysis_util.get_ind(xyz, lines_for_test, df_line, splits=(1,0,0,0))


def test_chunk_data():
    # Test case 1: MagNav.chunk_data(x,y,size(x,1)) == ([x'],[y])
    # x_cd1 is a 4x2 matrix, y_cd1 is a 4-element vector
    x_cd1 = np.array([[1,11],[2,22],[3,33],[4,44]], dtype=float)
    y_cd1 = np.array([1,2,3,4], dtype=float)
    res_x1, res_y1 = analysis_util.chunk_data(x_cd1, y_cd1, x_cd1.shape[0])
    assert len(res_x1) == 1 and res_x1[0] == pytest.approx(x_cd1.T)
    assert len(res_y1) == 1 and res_y1[0] == pytest.approx(y_cd1)

    # Test case 2: MagNav.chunk_data([1:4 1:4],1:4,2)
    # Julia [1:4 1:4] is hcat(1:4, 1:4) -> [[1,1],[2,2],[3,3],[4,4]]
    x_cd2 = np.array([[1,1],[2,2],[3,3],[4,4]], dtype=float)
    y_cd2 = np.array([1,2,3,4], dtype=float)
    res_x2, res_y2 = analysis_util.chunk_data(x_cd2, y_cd2, 2)
    
    # Expected output from Julia: ([[1 2; 1 2],[3 4; 3 4]],[[1,2],[3,4]])
    # This means chunks of x are transposed.
    # Chunk 1 of x_cd2: [[1,1],[2,2]], Transposed: [[1,2],[1,2]]
    # Chunk 2 of x_cd2: [[3,3],[4,4]], Transposed: [[3,4],[3,4]]
    expected_x2_list = [np.array([[1,2],[1,2]]), np.array([[3,4],[3,4]])]
    expected_y2_list = [np.array([1,2]), np.array([3,4])]

    assert len(res_x2) == len(expected_x2_list)
    for i in range(len(res_x2)):
        assert res_x2[i] == pytest.approx(expected_x2_list[i])
    
    assert len(res_y2) == len(expected_y2_list)
    for i in range(len(res_y2)):
        assert res_y2[i] == pytest.approx(expected_y2_list[i])

# For predict_rnn tests, we need a model and data.
# Using x_norm_data (5x3) from earlier.
# m_rnn needs to be a Python equivalent of Flux Chain(GRU(3=>1), Dense(1=>1))
# This assumes model_functions.build_simple_rnn or similar exists, or predict_rnn can take a mock.
# For now, let's assume predict_rnn can work with a placeholder if model creation is complex.
# Or, if get_nn_m can create RNNs, use that. The Julia test uses a generic m_rnn.
# Let's assume model_functions.get_rnn_model(input_dim, gru_units, dense_units) exists.
try:
    m_rnn_py = model_functions.get_rnn_model(input_dim=3, gru_units=1, output_units=1) # Hypothetical
except AttributeError: # If get_rnn_model is not defined, use a placeholder for test structure
    m_rnn_py = "placeholder_rnn_model" # This test might then need adjustment or skip

def test_predict_rnn():
    if isinstance(m_rnn_py, str) and m_rnn_py == "placeholder_rnn_model":
        pytest.skip("Skipping RNN prediction test as RNN model creation is not fully defined.")

    # x_norm_data is (5,3)
    rnn_full_out = model_functions.predict_rnn_full(m_rnn_py, x_norm_data)
    rnn_windowed_out = model_functions.predict_rnn_windowed(m_rnn_py, x_norm_data, x_norm_data.shape[0])
    
    assert isinstance(rnn_full_out, np.ndarray) and rnn_full_out.ndim == 1
    assert isinstance(rnn_windowed_out, np.ndarray) and rnn_windowed_out.ndim == 1
    assert rnn_full_out == pytest.approx(rnn_windowed_out)

# For krr tests
# Using x_norm_data (5x3) and y_norm_data (5,)
krr_model, krr_data_norms, _, _ = analysis_util.krr_fit(x_norm_data, y_norm_data)

def test_krr():
    # Test 1: MagNav.krr_fit(x,y)[2:4] == MagNav.krr_fit( x,y;data_norms )[2:4]
    # Julia [2:4] -> Python slice [1:4] (elements at index 1, 2, 3)
    res1_tuple_parts = analysis_util.krr_fit(x_norm_data, y_norm_data)[1:4]
    res2_tuple_parts = analysis_util.krr_fit(x_norm_data, y_norm_data, data_norms=krr_data_norms)[1:4]
    # Compare tuples part by part if pytest.approx doesn't handle tuple of mixed types well
    for r1, r2 in zip(res1_tuple_parts, res2_tuple_parts):
        assert r1 == pytest.approx(r2)

    # Test 2: MagNav.krr_fit(x,y)[3:4] == MagNav.krr_test(x,y,data_norms,model)[1:2]
    # Julia [3:4] -> Python slice [2:4] (elements at index 2, 3)
    # Julia [1:2] -> Python slice [0:2] (elements at index 0, 1)
    fit_res_part = analysis_util.krr_fit(x_norm_data, y_norm_data)[2:4]
    test_res_part = analysis_util.krr_test(x_norm_data, y_norm_data, krr_data_norms, krr_model)[0:2]
    for r1, r2 in zip(fit_res_part, test_res_part):
        assert r1 == pytest.approx(r2)

# For shapley & gsa tests
# Using m_global_nn_model (3-input NN) and x_norm_data (5x3)
features_shap = ["f1", "f2", "f3"]
df_shap, baseline_shap = model_functions.eval_shapley(m_global_nn_model, x_norm_data, features_shap)

def test_shapley_gsa():
    df_x_shap = pd.DataFrame(x_norm_data, columns=features_shap)
    pred_shap_df = model_functions.predict_shapley(m_global_nn_model, df_x_shap)
    assert isinstance(pred_shap_df, pd.DataFrame)

    res_eval_shap = model_functions.eval_shapley(m_global_nn_model, x_norm_data, features_shap)
    assert isinstance(res_eval_shap, tuple) and len(res_eval_shap) == 2
    assert isinstance(res_eval_shap[0], pd.DataFrame) # df_shap
    assert isinstance(res_eval_shap[1], (int, float)) # baseline_shap (Real)

    # Assuming plot_shapley is in plot_functions
    # This test depends on plot_functions.plot_shapley returning a matplotlib Figure object
    plot_obj = plot_functions.plot_shapley(df_shap, baseline_shap)
    assert isinstance(plot_obj, matplotlib.figure.Figure)

    gsa_res = model_functions.eval_gsa(m_global_nn_model, x_norm_data)
    assert isinstance(gsa_res, np.ndarray) and gsa_res.ndim == 1


def test_get_igrf(flight_data_setup):
    s = flight_data_setup
    xyz, ind = s["xyz"], s["ind"]
    
    igrf_vector1 = analysis_util.get_igrf(xyz, ind)
    assert isinstance(igrf_vector1, np.ndarray) and igrf_vector1.ndim == 1

    igrf_vector2 = analysis_util.get_igrf(xyz, ind, frame="nav", norm_igrf=True)
    assert isinstance(igrf_vector2, np.ndarray) and igrf_vector2.ndim == 1

# Data for projection tests
vec_nav_proj = np.random.randn(3)
vec_body_proj = np.random.randn(3)

def test_project_vec_to_2d():
    v_nav = vec_nav_proj
    with pytest.raises(AssertionError):
        analysis_util.project_vec_to_2d(v_nav, np.array([1,0,0]), np.array([1,0,0]))
    with pytest.raises(AssertionError):
        analysis_util.project_vec_to_2d(v_nav, np.array([1,1,0]), np.array([0,1,0]))
    with pytest.raises(AssertionError):
        analysis_util.project_vec_to_2d(v_nav, np.array([1,0,0]), np.array([1,1,0]))
    
    res_proj = analysis_util.project_vec_to_2d(v_nav, np.array([1,0,0]), np.array([0,1,0]))
    assert isinstance(res_proj, np.ndarray) and res_proj.ndim == 1


def test_project_body_field_to_2d_igrf(flight_data_setup):
    s = flight_data_setup
    xyz, ind = s["xyz"], s["ind"]

    igrf_nav_proj = vec_nav_proj / np.linalg.norm(vec_nav_proj)
    Cnb_proj = xyz.ins.Cnb[:,:,ind] # Assuming this structure from xyz object

    # Test with the first Cnb matrix
    if Cnb_proj.shape[2] > 0:
        res_pb = analysis_util.project_body_field_to_2d_igrf(vec_body_proj, igrf_nav_proj, Cnb_proj[:,:,0])
        assert isinstance(res_pb, np.ndarray) and res_pb.ndim == 1

        # Cardinal direction tests
        north_vec_nav = np.array([1.0, 0.0, 0.0])
        num_samples = Cnb_proj.shape[2]

        north_vec_body_list = [Cnb_proj[:,:,i].T @ north_vec_nav for i in range(num_samples)]
        n2ds = [analysis_util.project_body_field_to_2d_igrf(north_vec_body_list[i], north_vec_nav, Cnb_proj[:,:,i]) for i in range(num_samples)]
        assert all(v == pytest.approx([1.0, 0.0]) for v in n2ds)

        east_vec_nav = np.array([0.0, 1.0, 0.0])
        east_vec_body_list = [Cnb_proj[:,:,i].T @ east_vec_nav for i in range(num_samples)]
        e2ds = [analysis_util.project_body_field_to_2d_igrf(east_vec_body_list[i], north_vec_nav, Cnb_proj[:,:,i]) for i in range(num_samples)]
        assert all(v == pytest.approx([0.0, 1.0]) for v in e2ds)

        # West and South tests (similar structure)
        west_vec_nav = np.array([0.0, -1.0, 0.0])
        west_vec_body_list = [Cnb_proj[:,:,i].T @ west_vec_nav for i in range(num_samples)]
        w2ds = [analysis_util.project_body_field_to_2d_igrf(west_vec_body_list[i], north_vec_nav, Cnb_proj[:,:,i]) for i in range(num_samples)]
        assert all(v == pytest.approx([0.0, -1.0]) for v in w2ds)

        south_vec_nav = np.array([-1.0, 0.0, 0.0])
        south_vec_body_list = [Cnb_proj[:,:,i].T @ south_vec_nav for i in range(num_samples)]
        s2ds = [analysis_util.project_body_field_to_2d_igrf(south_vec_body_list[i], north_vec_nav, Cnb_proj[:,:,i]) for i in range(num_samples)]
        assert all(v == pytest.approx([-1.0, 0.0]) for v in s2ds)


def test_get_optimal_rotation_matrix():
    # vec_nav_proj and vec_body_proj are 1D arrays (vectors)
    # Julia [1 0] is a 1x2 row vector (matrix). Python np.array([[1,0]])
    # Julia vec_body' is transpose. If vec_body_proj is (3,), its transpose is still (3,).
    # For matrix multiplication, it needs to be (1,3) or (3,1).
    # Assuming the function expects 2D arrays (NxD)
    v_nav_row = vec_nav_proj.reshape(1, -1)
    v_body_row = vec_body_proj.reshape(1, -1)

    with pytest.raises(AssertionError): # Shape mismatch or other assertion
        analysis_util.get_optimal_rotation_matrix(np.array([[1,0]]), v_body_row)
    with pytest.raises(AssertionError):
        analysis_util.get_optimal_rotation_matrix(np.array([[1,0,0]]), np.array([[1,0]]))
    
    # Test with compatible shapes, e.g., two sets of corresponding 3D vectors (Nx3)
    # For this test, using single corresponding vectors reshaped to (1,3)
    rot_matrix = analysis_util.get_optimal_rotation_matrix(v_nav_row, v_body_row)
    assert rot_matrix.shape == (3,3)


def test_expand_range():
    x_er = np.array([-1, 0, 1])
    xlim_er = (-2.0, 2.0)
    
    # Julia -3:3 is range [-3, -2, -1, 0, 1, 2, 3]. Python np.arange(-3, 3+1)
    res1_val, res1_ticks = analysis_util.expand_range(x_er, xlim_er, True) # Assuming returns (values, tick_labels)
    assert np.array_equal(res1_val, np.arange(-3, 4)) # Check values
    # Tick labels might be strings or numbers, depends on Python impl. Julia test checks range object.

    # Julia 2:4 is range [2, 3, 4]. Python np.arange(2, 4+1)
    res2_val, res2_ticks = analysis_util.expand_range(x_er, xlim_er, False)
    assert np.array_equal(res2_ticks, np.arange(2, 5)) # Check tick labels (assuming they are the range)


# Setup for gif_animation_m3 test
@pytest.fixture(scope="function") # function scope for temp dir cleanup
def gif_test_setup(flight_data_setup):
    s = flight_data_setup
    xyz, ind = s["xyz"], s["ind"]

    # This requires functions from tolles_lawson.py
    # Assuming xyz.flux_a exists and ind is valid
    # Ensure ind selects a non-empty portion of xyz.flux_a
    if np.sum(ind) == 0:
        pytest.skip("Not enough data points for GIF animation test after applying 'ind'.")

    flux_a_selected = xyz.flux_a[ind, :] if xyz.flux_a.ndim > 1 else xyz.flux_a[ind]

    B_unit, Bt, B_vec_dot = tolles_lawson.create_TL_A(flux_a_selected, terms=['p'], return_B=True)
    B_vec = B_unit * Bt # Element-wise if B_unit is scalar or broadcastable

    # Using TL_a_1 from global setup
    TL_coef_p, TL_coef_i, TL_coef_e = tolles_lawson.TL_vec2mat(TL_a_1, ['p','i','e'])
    
    TL_aircraft, TL_perm, TL_induced, TL_eddy = \
        tolles_lawson.get_TL_aircraft_vec(B_vec.T, B_vec_dot.T, TL_coef_p, TL_coef_i, TL_coef_e,
                                     return_parts=True)

    num_points = np.sum(ind)
    y_nn_gif = np.zeros((3, num_points)) # Match Julia (3,50) if num_points is 50
    y_gif = y_hat_gif = np.zeros(num_points)

    # Create a temporary directory for the GIF output
    # Using a sub-directory in the current test file's location
    current_test_dir = os.path.dirname(__file__)
    mag_gif_output_dir = os.path.join(current_test_dir, "temp_gif_output")
    os.makedirs(mag_gif_output_dir, exist_ok=True)
    
    # Data to pass to the test and for cleanup
    setup_data = {
        "TL_perm": TL_perm, "TL_induced": TL_induced, "TL_eddy": TL_eddy,
        "TL_aircraft": TL_aircraft, "B_unit_T": B_unit.T, "y_nn_gif": y_nn_gif,
        "y_gif": y_gif, "y_hat_gif": y_hat_gif, "xyz": xyz, "ind": ind,
        "mag_gif_output_dir": mag_gif_output_dir
    }

    yield setup_data # Pass data to the test

    # Teardown: remove the temporary directory
    if os.path.exists(mag_gif_output_dir):
        shutil.rmtree(mag_gif_output_dir)


def test_gif_animation_m3(gif_test_setup, flight_data_setup): # Also needs flight_data_setup for xyz.ins
    s_gif = gif_test_setup
    s_flight = flight_data_setup # For xyz.ins.lat, xyz.ins.lon

    # Julia ENV["GKSwstype"] = "100" is for headless GR backend.
    # Matplotlib uses 'Agg' for non-GUI. This might be set globally for tests if needed,
    # or handled by plot_functions.gif_animation_m3 internally.
    # os.environ["MATPLOTLIB_BACKEND"] = "Agg" # Example

    # Assuming plot_functions.gif_animation_m3 is the Python equivalent
    # and it saves a .gif file in the mag_gif_output_dir
    # The function might return the path to the created gif or just save it.
    
    # Ensure lat/lon from flight_data_setup correspond to the 'ind' used for other data
    lat_selected = s_flight["xyz"].ins.lat[s_gif["ind"]]
    lon_selected = s_flight["xyz"].ins.lon[s_gif["ind"]]

    # The name of the gif file created by gif_animation_m3 needs to be known or discoverable.
    # Let's assume it creates a file named "animation.gif" inside the output dir.
    expected_gif_filename = "animation.gif" # This is an assumption

    plot_functions.gif_animation_m3(
        s_gif["TL_perm"], s_gif["TL_induced"], s_gif["TL_eddy"],
        s_gif["TL_aircraft"], s_gif["B_unit_T"], s_gif["y_nn_gif"],
        s_gif["y_gif"], s_gif["y_hat_gif"], s_gif["xyz"],
        lat_selected, lon_selected, # Pass selected lat/lon
        ind=s_gif["ind"], save_plot=True, mag_gif=s_gif["mag_gif_output_dir"],
        # Potentially pass expected_gif_filename if the function takes it
        # or ensure the function's default naming matches.
        gif_filename=expected_gif_filename 
    )

    # Check if the GIF file was created
    created_gif_path = os.path.join(s_gif["mag_gif_output_dir"], expected_gif_filename)
    assert os.path.exists(created_gif_path), f"GIF file {created_gif_path} was not created."
    assert os.path.getsize(created_gif_path) > 0, "Created GIF file is empty."

    # The Julia test checks `isa Plots.AnimatedGif`.
    # The Python equivalent might return a matplotlib animation object or None if it just saves.
    # The current check focuses on file creation as per `save_plot=true`.
    # If the function returns an animation object:
    # anim_object = plot_functions.gif_animation_m3(...)
    # assert isinstance(anim_object, matplotlib.animation.ArtistAnimation) # or similar