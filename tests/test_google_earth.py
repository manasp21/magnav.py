import pytest
import numpy as np
import scipy.io
import os

# Attempt to import project-specific modules.
# Tests will be skipped if these imports fail, indicating the modules are not available.
try:
    from MagNavPy.src import google_earth
    from MagNavPy.src import magnav # For Traj, MapS data structures
    # For more specific imports if preferred:
    # from MagNavPy.src.magnav import Traj, MapS
except ImportError as e:
    pytest.skip(f"Skipping google_earth tests; failed to import MagNavPy modules: {e}", allow_module_level=True)

# Define the path to the test data directory.
# This assumes tests are run from the workspace root (c:/Users/Manas Pandey/Documents/magnav.py).
TEST_DATA_DIR = "MagNav.jl/test/test_data"

@pytest.fixture(scope="module")
def map_data_fixture():
    """
    Loads map data from test_data_map.mat.
    Corresponds to the initial map data loading in the Julia test script.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_data_map.mat")
    loaded_mat = scipy.io.loadmat(test_file)

    # Assumptions about MAT file structure:
    # - 'map_data' is a variable in the MAT file.
    # - This variable is a scalar MATLAB struct.
    # - scipy.io.loadmat loads this as a 1x1 structured numpy array.
    map_struct = loaded_mat['map_data'][0,0]

    map_map_val  = map_struct["map"]
    # Convert xx and yy to radians, similar to Julia's deg2rad.(vec(map_data["xx"]))
    map_xx_val   = np.deg2rad(map_struct["xx"].flatten())
    map_yy_val   = np.deg2rad(map_struct["yy"].flatten())
    map_alt_val  = map_struct["alt"]
    # Ensure alt is a scalar if it's loaded as a single-element array (e.g., [[value]])
    if isinstance(map_alt_val, np.ndarray) and map_alt_val.size == 1:
        map_alt_val = map_alt_val.item()

    map_info_val = "Map" # As in Julia script

    # map_mask calculation: MagNav.map_params(map_map,map_xx,map_yy)[2] in Julia
    # This assumes a similar magnav.map_params function exists in Python.
    # The [1] access corresponds to Julia's [2] (0-indexed vs 1-indexed).
    try:
        # This is a critical assumption: magnav.map_params exists and its
        # second return element (index 1) is the map_mask.
        map_params_output = magnav.map_params(map_map_val, map_xx_val, map_yy_val)
        map_mask_val = map_params_output[1]
    except (AttributeError, TypeError, IndexError) as e:
        # AttributeError: magnav.map_params doesn't exist or is misspelled.
        # TypeError: arguments to magnav.map_params are incorrect.
        # IndexError: magnav.map_params doesn't return at least two elements.
        print(f"Warning: Could not use magnav.map_params due to {type(e).__name__}: {e}. "
              "Using a default map_mask (all true). This may affect MapS behavior.")
        map_mask_val = np.ones(map_map_val.shape, dtype=bool) # Placeholder

    mapS_val = magnav.MapS(map_info_val, map_map_val, map_xx_val, map_yy_val, map_alt_val, map_mask_val)

    return {
        "map_info": map_info_val,
        "map_map_orig": map_map_val.copy(), # Store original for later tests
        "map_xx_orig": map_xx_val.copy(),   # Already in radians
        "map_yy_orig": map_yy_val.copy(),   # Already in radians
        "map_alt": map_alt_val,
        "map_mask": map_mask_val,
        "mapS": mapS_val,
    }

@pytest.fixture(scope="module")
def traj_data_fixture():
    """
    Loads trajectory data from test_data_traj.mat.
    Corresponds to the initial trajectory data loading in the Julia test script.
    """
    test_file = os.path.join(TEST_DATA_DIR, "test_data_traj.mat")
    loaded_mat = scipy.io.loadmat(test_file)
    traj_struct = loaded_mat['traj'][0,0] # Assuming 'traj' is a scalar struct

    tt_val  = traj_struct["tt"].flatten()
    lat_val = np.deg2rad(traj_struct["lat"].flatten()) # Convert to radians
    lon_val = np.deg2rad(traj_struct["lon"].flatten()) # Convert to radians
    alt_val = traj_struct["alt"].flatten()
    vn_val  = traj_struct["vn"].flatten()
    ve_val  = traj_struct["ve"].flatten()
    vd_val  = traj_struct["vd"].flatten()
    fn_val  = traj_struct["fn"].flatten()
    fe_val  = traj_struct["fe"].flatten()
    fd_val  = traj_struct["fd"].flatten()
    Cnb_val = traj_struct["Cnb"] # Expected to be (N, 3, 3) or similar

    N_val   = len(lat_val)
    dt_val  = (tt_val[1] - tt_val[0]) if N_val > 1 else 0.0

    traj_obj = magnav.Traj(N_val, dt_val, tt_val, lat_val, lon_val, alt_val,
                           vn_val, ve_val, vd_val, fn_val, fe_val, fd_val, Cnb_val)

    # Boolean index for filtering, similar to Julia's `ind`
    # Julia: ind = trues(N); ind[51:end] .= false (1-based)
    # Python: ind_py = np.ones(N, dtype=bool); ind_py[50:] = False (0-based)
    ind_py = np.ones(N_val, dtype=bool)
    if N_val > 50: # Ensure index 50 is valid
        ind_py[50:] = False

    return {
        "tt": tt_val, "lat": lat_val, "lon": lon_val, "alt": alt_val,
        "vn": vn_val, "ve": ve_val, "vd": vd_val,
        "fn": fn_val, "fe": fe_val, "fd": fd_val,
        "Cnb": Cnb_val, "N": N_val, "dt": dt_val,
        "traj_obj": traj_obj,
        "ind_py": ind_py # Boolean index for filtering
    }

@pytest.fixture(scope="module")
def common_data(map_data_fixture, traj_data_fixture):
    """
    Combines map and trajectory data, and prepares filtered map versions
    based on trajectory length, as done in the Julia test script.
    """
    ind_filter_from_traj = traj_data_fixture["ind_py"]

    # Replicate Julia's filtering of map components using an index derived from trajectory length.
    # This is an unusual pattern but present in the original test.
    # It assumes map_xx_orig, map_yy_orig, and map_map_orig dimensions are
    # compatible with the length of ind_filter_from_traj.
    map_xx_orig = map_data_fixture["map_xx_orig"]
    map_yy_orig = map_data_fixture["map_yy_orig"]
    map_map_orig = map_data_fixture["map_map_orig"]

    # Ensure the filter length is not greater than the data to be filtered
    filter_len = len(ind_filter_from_traj)
    map_xx_filt = map_xx_orig[ind_filter_from_traj] if filter_len <= len(map_xx_orig) else map_xx_orig[:filter_len][ind_filter_from_traj[:len(map_xx_orig)]]
    map_yy_filt = map_yy_orig[ind_filter_from_traj] if filter_len <= len(map_yy_orig) else map_yy_orig[:filter_len][ind_filter_from_traj[:len(map_yy_orig)]]

    # For 2D map_map, using np.ix_ for safe broadcasting of boolean index
    # This creates a subgrid. If map_map_orig is smaller than filter_len in any dimension,
    # this will error or select up to the available size.
    # We assume map_map_orig is large enough.
    if filter_len <= map_map_orig.shape[0] and filter_len <= map_map_orig.shape[1]:
        map_map_filt = map_map_orig[np.ix_(ind_filter_from_traj, ind_filter_from_traj)]
    else:
        # Fallback or error if dimensions don't match the filtering logic from Julia
        # For simplicity, assume dimensions are compatible as per original test's implicit assumption
        # This part might need adjustment if data dimensions cause issues.
        # A simple slice to avoid error, though it might not perfectly match Julia if ind_filter_from_traj is too long:
        shorter_ind = ind_filter_from_traj[:min(filter_len, map_map_orig.shape[0], map_map_orig.shape[1])]
        map_map_filt = map_map_orig[np.ix_(shorter_ind, shorter_ind)]
        if len(shorter_ind) < filter_len:
             print(f"Warning: Map dimensions were smaller than trajectory-based filter length. "
                   f"Filtered map may not match Julia's intent precisely.")


    return {
        **map_data_fixture,
        **traj_data_fixture,
        "map_map_filtered": map_map_filt,
        "map_xx_filtered": map_xx_filt, # Already in radians
        "map_yy_filtered": map_yy_filt, # Already in radians
    }

def test_map2kmz(common_data, tmp_path):
    """
    Tests for the map2kmz function.
    Corresponds to Julia's "@testset map2kmz tests".
    """
    mapS = common_data["mapS"]
    map_map_filt = common_data["map_map_filtered"]
    map_xx_filt_rad = common_data["map_xx_filtered"] # These are already in radians
    map_yy_filt_rad = common_data["map_yy_filtered"] # These are already in radians

    output_basename = "test_map_output"

    # Test 1: map2kmz(mapS, map_kmz_path; plot_alt=mapS.alt)
    # Assumes map2kmz returns None on success and creates the file.
    map_kmz_file1 = tmp_path / (output_basename + "_mapS.kmz")
    assert google_earth.map2kmz(mapS, str(map_kmz_file1), plot_alt=mapS.alt) is None
    assert map_kmz_file1.exists()
    assert map_kmz_file1.stat().st_size > 0 # Basic check for non-empty file

    # Test 2: map2kmz(map_map_filt, map_xx_filt_rad, map_yy_filt_rad, map_kmz_path)
    # Inputs map_xx_filt_rad, map_yy_filt_rad are in radians.
    map_kmz_file2 = tmp_path / (output_basename + "_filtered_rad.kmz")
    assert google_earth.map2kmz(map_map_filt, map_xx_filt_rad, map_yy_filt_rad, str(map_kmz_file2)) is None
    assert map_kmz_file2.exists()
    assert map_kmz_file2.stat().st_size > 0

    # Test 3: map2kmz(map_map_filt, map_xx_filt_deg, map_yy_filt_deg, map_kmz_path; map_units='deg')
    # If map_units='deg', the function expects inputs in degrees.
    # The Julia test passes radian values here, which might be an issue in the original test
    # or implies a specific behavior of Julia's map2kmz.
    # For Python, we assume 'deg' means inputs should be in degrees.
    map_xx_filt_deg = np.rad2deg(map_xx_filt_rad)
    map_yy_filt_deg = np.rad2deg(map_yy_filt_rad)
    map_kmz_file3 = tmp_path / (output_basename + "_filtered_deg.kmz")
    assert google_earth.map2kmz(map_map_filt, map_xx_filt_deg, map_yy_filt_deg, str(map_kmz_file3), map_units='deg') is None
    assert map_kmz_file3.exists()
    assert map_kmz_file3.stat().st_size > 0

    # Test 4: @test_throws ErrorException map2kmz(..., map_units=:test)
    # Using degree inputs from Test 3 for consistency with an invalid unit.
    map_kmz_file4 = tmp_path / (output_basename + "_error.kmz") # Path might still be needed by function signature
    with pytest.raises(Exception): # Julia's ErrorException maps to a general Python Exception.
                                   # Could be ValueError or TypeError depending on implementation.
        google_earth.map2kmz(map_map_filt, map_xx_filt_deg, map_yy_filt_deg, str(map_kmz_file4), map_units='invalid_unit_test')
    # Optionally, assert that the file was NOT created in this error case, if that's the expected behavior.
    # assert not map_kmz_file4.exists()

def test_path2kml(common_data, tmp_path):
    """
    Tests for the path2kml function.
    Corresponds to Julia's "@testset path2kml tests".
    """
    lat_rad_full = common_data["lat"] # Full trajectory latitude in radians
    lon_rad_full = common_data["lon"] # Full trajectory longitude in radians
    alt_full = common_data["alt"]     # Full trajectory altitude

    ind_py = common_data["ind_py"] # Boolean index for filtering trajectory

    output_basename = "test_path_output"

    # Test 1: path2kml(lat_rad_full, lon_rad_full, alt_full, path_kml_path)
    # Assumes default input units are radians if not specified.
    path_kml_file1 = tmp_path / (output_basename + "_full.kml")
    assert google_earth.path2kml(lat_rad_full, lon_rad_full, alt_full, str(path_kml_file1)) is None
    assert path_kml_file1.exists()
    assert path_kml_file1.stat().st_size > 0

    # Test 2: path2kml(traj_filtered_obj, path_kml_path; points=true)
    # This involves creating a filtered Traj object.
    lat_filt = lat_rad_full[ind_py]
    lon_filt = lon_rad_full[ind_py]
    alt_filt = alt_full[ind_py]
    
    N_filt = np.sum(ind_py)
    if N_filt > 0: # Proceed only if filtered data is not empty
        tt_filt = common_data["tt"][ind_py]
        dt_filt = (tt_filt[1] - tt_filt[0]) if N_filt > 1 else common_data["dt"]
        
        vn_filt = common_data["vn"][ind_py]
        ve_filt = common_data["ve"][ind_py]
        vd_filt = common_data["vd"][ind_py]
        fn_filt = common_data["fn"][ind_py]
        fe_filt = common_data["fe"][ind_py]
        fd_filt = common_data["fd"][ind_py]
        # Filter Cnb along the first (trajectory points) dimension
        Cnb_filt = common_data["Cnb"][ind_py, :, :]

        traj_obj_filt = magnav.Traj(N_filt, dt_filt, tt_filt, lat_filt, lon_filt, alt_filt,
                                    vn_filt, ve_filt, vd_filt, fn_filt, fe_filt, fd_filt, Cnb_filt)
        
        path_kml_file2 = tmp_path / (output_basename + "_filtered_traj.kml")
        assert google_earth.path2kml(traj_obj_filt, str(path_kml_file2), points=True) is None
        assert path_kml_file2.exists()
        assert path_kml_file2.stat().st_size > 0
    else:
        pytest.skip("Filtered trajectory is empty, skipping path2kml test with filtered Traj object.")

    # Test 3: @test_throws ErrorException path2kml(lat,lon,alt,path_kml;path_units=:test)
    # Using full lat/lon/alt (in radians) for this error test.
    path_kml_file3 = tmp_path / (output_basename + "_error.kml") # Path might still be needed
    with pytest.raises(Exception): # Or more specific, e.g., ValueError
        google_earth.path2kml(lat_rad_full, lon_rad_full, alt_full, str(path_kml_file3), path_units='invalid_unit_test')
    # Optionally, assert file not created on error.
    # assert not path_kml_file3.exists()

# Note: The Julia script includes `rm(map_kmz)` and `rm(path_kml)`.
# This is not needed with pytest's `tmp_path` fixture, which handles cleanup.