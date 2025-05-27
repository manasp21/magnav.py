import pytest
import numpy as np
import scipy.io
import torch
import os

# MagNavPy imports
# Assuming MagNavPy is installed or in PYTHONPATH, and these modules/functions exist
from MagNavPy.src.magnav import (
    get_map, MapCache, map_interpolate, get_XYZ0, Traj, INS, FILTres, CRLBout, INSout, FILTout
)
from MagNavPy.src.nekf import nekf_train, nekf
from MagNavPy.src.compensation import create_TL_A
from MagNavPy.src.ekf import run_filt # Assuming run_filt is generic

# Helper to define base data directory relative to this test file
# This test file is in MagNavPy/tests/
# Data is in MagNav.jl/test/test_data/
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_JULIA_TEST_DATA_DIR = os.path.join(_BASE_DIR, "..", "..", "MagNav.jl", "test", "test_data")

@pytest.fixture(scope="module") # Scope to module to run setup once
def nekf_test_data():
    """Loads and preprocesses data required for NEKF tests."""
    map_file = os.path.join(_JULIA_TEST_DATA_DIR, "test_data_map.mat")
    traj_file = os.path.join(_JULIA_TEST_DATA_DIR, "test_data_traj.mat")

    # Load map data
    # Assumes get_map can handle the file path and symbol-like string for data key
    mapS = get_map(map_file, 'map_data')
    map_cache = MapCache(maps=[mapS]) # Assumes MapCache constructor
    itp_mapS = map_interpolate(mapS)  # Assumes map_interpolate works on MapS object

    # Load trajectory data
    # Assumes get_XYZ0 signature: get_XYZ0(filepath, data_key_1, data_key_2, silent=Boolean)
    xyz_data = get_XYZ0(traj_file, 'traj', 'none', silent=True)

    traj = xyz_data.traj
    ins = xyz_data.ins
    flux_a = xyz_data.flux_a
    mag_1_uc = xyz_data.mag_1_uc
    mag_1_c = xyz_data.mag_1_c

    # Create index
    # Assumes traj object has an 'N' attribute for number of points, like in Julia
    num_points = traj.N
    ind = np.ones(num_points, dtype=bool)
    # Julia's 51:end becomes Python's 50: (0-indexed start)
    ind[50:] = False

    # Prepare 'x' for NEKF training (input features)
    # Corresponds to Julia: x = [xyz.mag_1_uc create_TL_A(flux_a;terms=[:p])][ind,:]
    mag_1_uc_col = mag_1_uc
    if mag_1_uc_col.ndim == 1:
        mag_1_uc_col = mag_1_uc_col.reshape(-1, 1)

    # Assumes create_TL_A takes flux_a and a list of terms like ['p']
    tl_a_matrix = create_TL_A(flux_a, terms=['p'])
    if tl_a_matrix.ndim == 1: # Ensure 2D for hstack
        tl_a_matrix = tl_a_matrix.reshape(-1, 1)
    
    if mag_1_uc_col.shape[0] != tl_a_matrix.shape[0]:
        raise ValueError(
            f"Row mismatch for hstack: mag_1_uc_col ({mag_1_uc_col.shape[0]}) vs "
            f"tl_a_matrix ({tl_a_matrix.shape[0]})"
        )

    x_full = np.hstack((mag_1_uc_col, tl_a_matrix))
    x = x_full[ind, :] # Select rows based on ind

    # NEKF training
    # This matches the Julia test which runs a minimal training.
    # Assumes nekf_train takes xyz_data object, ind array, full mag_1_c, map, and sliced x.
    m, data_norms = nekf_train(xyz_data, ind, mag_1_c, itp_mapS, x,
                               epoch_adam=1, l_window=10)
    # m is expected to be a torch.nn.Module
    # data_norms is expected to be a tuple (v_scale, x_bias, x_scale)

    v_scale, x_bias, x_scale = data_norms
    # x is already sliced by ind, so x_nn will also be for the indexed subset
    x_nn = ((x - x_bias) / x_scale) * v_scale

    # For subsetting INS/Traj objects for nekf and run_filt calls:
    # This assumes INS and Traj Python objects have a `get_subset(ind)` method.
    # If not, these lines would need adjustment based on actual API,
    # or nekf/run_filt would need to accept full objects + index, or pre-sliced arrays.
    if not hasattr(ins, 'get_subset') or not callable(getattr(ins, 'get_subset')):
        raise NotImplementedError("INS class must have a get_subset(ind) method for this test.")
    if not hasattr(traj, 'get_subset') or not callable(getattr(traj, 'get_subset')):
        raise NotImplementedError("Traj class must have a get_subset(ind) method for this test.")

    ins_subset = ins.get_subset(ind)
    traj_subset = traj.get_subset(ind)

    return {
        "ins_subset": ins_subset,
        "traj_subset": traj_subset,
        "mag_1_c_indexed": mag_1_c[ind], # nekf/run_filt expect indexed mag data
        "itp_mapS": itp_mapS,
        "map_cache": map_cache,
        "x_nn": x_nn, # This is x_nn for the 'ind' subset
        "m": m,       # Trained model
        "data_norms": data_norms, # For the first assertion
        # For run_filt, which might need full traj/ins if it subsets internally,
        # but Julia test implies passing subsetted objects.
        # The current setup passes subsetted objects.
    }

def test_nekf_operations(nekf_test_data):
    """
    Tests NEKF training output types, NEKF step with different map types,
    and overall filter output using run_filt.
    Corresponds to @testset "nekf tests" in the Julia original.
    """
    # Unpack fixture data relevant for assertions
    ins_subset = nekf_test_data["ins_subset"]
    traj_subset = nekf_test_data["traj_subset"]
    mag_1_c_indexed = nekf_test_data["mag_1_c_indexed"]
    itp_mapS = nekf_test_data["itp_mapS"]
    map_cache = nekf_test_data["map_cache"]
    x_nn = nekf_test_data["x_nn"]
    m = nekf_test_data["m"]
    data_norms = nekf_test_data["data_norms"]

    # Test 1: nekf_train output types (model 'm' and 'data_norms')
    # Julia: @test nekf_train(...) isa Tuple{Chain,Tuple}
    # The training is done in the fixture. We assert on its results.
    assert isinstance(m, torch.nn.Module), \
        "NEKF model 'm' should be a torch.nn.Module (analogous to Flux.Chain)"
    assert isinstance(data_norms, tuple), \
        "data_norms from nekf_train should be a tuple"
    assert len(data_norms) == 3, \
        "data_norms tuple should contain v_scale, x_bias, x_scale (3 elements)"

    # Test 2: nekf function with interpolated map
    # Julia: @test nekf(ins(ind),xyz.mag_1_c[ind],itp_mapS ,x_nn,m) isa MagNav.FILTres
    filt_res_itp = nekf(ins_subset, mag_1_c_indexed, itp_mapS, x_nn, m)
    assert isinstance(filt_res_itp, FILTres), \
        "nekf output with itp_mapS should be of type FILTres"

    # Test 3: nekf function with map cache
    # Julia: @test nekf(ins(ind),xyz.mag_1_c[ind],map_cache,x_nn,m) isa MagNav.FILTres
    filt_res_cache = nekf(ins_subset, mag_1_c_indexed, map_cache, x_nn, m)
    assert isinstance(filt_res_cache, FILTres), \
        "nekf output with map_cache should be of type FILTres"

    # Test 4: run_filt with 'nekf' type
    # Julia: @test run_filt(traj(ind),ins(ind),xyz.mag_1_c[ind],itp_mapS,:nekf;
    #                       x_nn=x_nn,m=m) isa Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    # Assumes run_filt takes 'nekf' as a string type and **kwargs for model-specific params
    crlb_out, ins_out, filt_out = run_filt(traj_subset, ins_subset, mag_1_c_indexed,
                                           itp_mapS, "nekf", x_nn=x_nn, m=m)
    assert isinstance(crlb_out, CRLBout), "run_filt: crlb_out type mismatch"
    assert isinstance(ins_out, INSout), "run_filt: ins_out type mismatch"
    assert isinstance(filt_out, FILTout), "run_filt: filt_out type mismatch"

# General comments and assumptions for this translation:
# 1. Python Equivalents: Assumes `MagNavPy.src.*` modules provide Python versions of
#    `get_map`, `MapCache`, `map_interpolate`, `get_XYZ0`, `create_TL_A`, `nekf_train`,
#    `nekf`, and `run_filt` with analogous functionality to their Julia counterparts.
#    Data types like `MapS`, `Traj`, `INS`, `FILTres`, `CRLBout`, `INSout`, `FILTout`
#    are also assumed to be defined in `MagNavPy.src.magnav`.
# 2. Data Loading: `get_map` and `get_XYZ0` are expected to handle `.mat` file inputs
#    and string arguments (e.g., 'map_data', 'traj', 'none') that correspond to Julia's symbols.
# 3. Traj/INS Subsetting: `Traj` and `INS` Python objects are critically assumed to have a
#    `get_subset(ind)` method. This method should return a new instance of the respective
#    class containing data sliced by the boolean index `ind`, mimicking Julia's
#    `traj(ind)` or `ins(ind)` behavior. The test fixture raises a NotImplementedError
#    if these methods are missing.
# 4. NN Model: `nekf_train` is expected to return a `torch.nn.Module` instance as the
#    trained model, which is analogous to `Flux.Chain` in Julia.
# 5. `run_filt`: The generic `run_filt` function is assumed to accept "nekf" as a string
#    identifier for the filter type and use `x_nn` and `m` passed as keyword arguments
#    when this type is specified.
# 6. Data Paths: Test data file paths are constructed relative to the location of this
#    test script, assuming a standard project layout where `MagNav.jl` and `MagNavPy`
#    are sibling directories.
# 7. Indexing: Julia's 1-based indexing and `end` keyword are translated to Python's
#    0-based indexing (e.g., `ind[51:end]` becomes `ind[50:]`).
# 8. Attribute Access: Assumes Python `Traj` object has an `.N` attribute for the
#    number of data points, similar to the Julia version.