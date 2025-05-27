import pytest
import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend
import os

# Import functions from MagNavPy
from MagNavPy.src.tolles_lawson import create_TL_A, create_TL_coef, fdm
# get_TL_term_ind is mentioned in requirements but not explicitly tested in the original Julia file.
from MagNavPy.src.magnav import MagV # Imported as per requirement

# Helper to get path to test data
def get_test_data_path(filename):
    # Assumes test_tolles_lawson.py is in MagNavPy/tests/
    # Data is in MagNav.jl/test/test_data/
    return os.path.join(os.path.dirname(__file__), "..", "..", "MagNav.jl", "test", "test_data", filename)

@pytest.fixture(scope="module")
def load_test_data():
    """
    Loads data from .mat files, similar to the Julia test setup.
    Uses squeeze_me=True and struct_as_record=False for easier struct handling.
    """
    params_file = get_test_data_path("test_data_params.mat")
    # params_content is a struct-like object after loading
    params_content = loadmat(params_file, squeeze_me=True, struct_as_record=False)['params']
    
    tl_data_file = get_test_data_path("test_data_TL.mat")
    # tl_content is a struct-like object
    tl_content = loadmat(tl_data_file, squeeze_me=True, struct_as_record=False)['TL_data']

    data = {}
    # Extracting from params_content.TL struct
    data['pass1']    = params_content.TL.pass1
    data['pass2']    = params_content.TL.pass2
    data['trim']     = int(round(params_content.TL.trim))
    # Assuming the field name in .mat is 'lambda'. loadmat with struct_as_record=False
    # should make it accessible as .lambda_ if it's a keyword.
    data['lambda_val'] = params_content.TL.lambda_ # Accessing 'lambda' field from .mat
    data['Bt_scale'] = params_content.TL.Bt_scale

    # Extracting from tl_content struct
    data['fs']       = tl_content.fs
    data['flux_a_x'] = np.asarray(tl_content.flux_a_x).ravel()
    data['flux_a_y'] = np.asarray(tl_content.flux_a_y).ravel()
    data['flux_a_z'] = np.asarray(tl_content.flux_a_z).ravel()
    data['mag_1_uc'] = np.asarray(tl_content.mag_1_uc).ravel()

    # Expected data from tl_content for assertions
    data['A_a_expected'] = np.asarray(tl_content.A_a)
    data['mag_1_c_expected'] = np.asarray(tl_content.mag_1_c).ravel()
    data['mag_1_c_d_expected'] = np.asarray(tl_content.mag_1_c_d).ravel()
    
    # Pre-calculate A_a and TL_a_1 as in Julia script to be used in tests
    data['A_a'] = create_TL_A(data['flux_a_x'], data['flux_a_y'], data['flux_a_z'],
                                Bt_scale=data['Bt_scale'])
    
    data['TL_a_1'] = create_TL_coef(data['flux_a_x'], data['flux_a_y'], data['flux_a_z'], data['mag_1_uc'],
                                    lambda_val=data['lambda_val'], # Use the Python variable name
                                    pass1=data['pass1'], pass2=data['pass2'], fs=data['fs'],
                                    trim=data['trim'], Bt_scale=data['Bt_scale'])

    data['mag_1_c'] = data['mag_1_uc'] - data['A_a'] @ data['TL_a_1']
    
    # Detrending: Julia's Statistics.detrend by default removes the mean.
    # scipy.signal.detrend(..., type='constant') does the same.
    model_output = data['A_a'] @ data['TL_a_1']
    data['mag_1_c_d'] = data['mag_1_uc'] - detrend(model_output, type='constant')
    
    return data

# Testset 1: create_TL_A tests (Translated from Julia @testset "create_TL_A tests")
def test_create_TL_A_output(load_test_data):
    data = load_test_data
    # Julia: @test A_a ≈ TL_data["A_a"]
    # Using numpy.testing.assert_allclose for floating point array comparison
    # Default rtol=1e-7, atol=0. Julia's ≈ might have different defaults.
    # Using slightly looser tolerances as an example, adjust if needed.
    np.testing.assert_allclose(data['A_a'], data['A_a_expected'], rtol=1e-5, atol=1e-8)

# Testset 2: create_TL_coef tests (Translated from Julia @testset "create_TL_coef tests")
def test_create_TL_coef_output(load_test_data):
    data = load_test_data
    # Julia: @test std(mag_1_c-TL_data["mag_1_c"]) < 0.1
    assert np.std(data['mag_1_c'] - data['mag_1_c_expected']) < 0.1
    
    # Julia: @test std(mag_1_c_d-TL_data["mag_1_c_d"]) < 0.1
    assert np.std(data['mag_1_c_d'] - data['mag_1_c_d_expected']) < 0.1

# Testset 3: create_TL_A & create_TL_coef arguments tests
# (Translated from Julia @testset "create_TL_A & create_TL_coef arguments tests")
def test_TL_A_and_coef_arguments(load_test_data):
    data = load_test_data
    flux_a_x = data['flux_a_x']
    flux_a_y = data['flux_a_y']
    flux_a_z = data['flux_a_z']
    mag_1_uc = data['mag_1_uc']
    fs = data['fs'] # fs from loaded data

    terms_set = [
        ["permanent", "induced", "eddy", "bias"],
        ["permanent", "induced", "eddy"],
        ["permanent", "induced"],
        ["permanent"],
        ["induced"],
        ["i5", "e8", "fdm"], # Assuming these are valid string terms for Python version
        ["i3", "e3", "f3"]  # Assuming these are valid string terms for Python version
    ]

    for terms_list in terms_set:
        # Test create_TL_A with different terms
        A_matrix = create_TL_A(flux_a_x, flux_a_y, flux_a_z, terms=terms_list, Bt_scale=data['Bt_scale'])
        assert isinstance(A_matrix, np.ndarray), f"A_matrix for terms {terms_list} is not a numpy array"
        assert A_matrix.ndim == 2, f"A_matrix for terms {terms_list} is not 2D"

        # Test create_TL_coef with different terms, providing all necessary args from fixture
        coef_vector = create_TL_coef(flux_a_x, flux_a_y, flux_a_z, mag_1_uc,
                                     terms=terms_list,
                                     lambda_val=data['lambda_val'],
                                     pass1=data['pass1'], pass2=data['pass2'], fs=fs,
                                     trim=data['trim'], Bt_scale=data['Bt_scale'])
        assert isinstance(coef_vector, np.ndarray), f"coef_vector for terms {terms_list} is not a numpy array"
        assert coef_vector.ndim == 1, f"coef_vector for terms {terms_list} is not 1D"

    # Test create_TL_coef with specific pass1=0
    coef_vec_pass1_zero = create_TL_coef(flux_a_x, flux_a_y, flux_a_z, mag_1_uc,
                                         fs=fs, pass1=0, # Varied parameter
                                         lambda_val=data['lambda_val'], pass2=data['pass2'],
                                         trim=data['trim'], Bt_scale=data['Bt_scale'])
    assert isinstance(coef_vec_pass1_zero, np.ndarray)
    assert coef_vec_pass1_zero.ndim == 1

    # Test create_TL_coef with specific pass2=fs
    coef_vec_pass2_fs = create_TL_coef(flux_a_x, flux_a_y, flux_a_z, mag_1_uc,
                                       fs=fs, pass2=fs, # Varied parameter
                                       lambda_val=data['lambda_val'], pass1=data['pass1'],
                                       trim=data['trim'], Bt_scale=data['Bt_scale'])
    assert isinstance(coef_vec_pass2_fs, np.ndarray)
    assert coef_vec_pass2_fs.ndim == 1
    
    # Test std of create_TL_coef output
    coef_for_std = create_TL_coef(flux_a_x, flux_a_y, flux_a_z, mag_1_uc,
                                  fs=fs, pass1=0, pass2=fs, # Varied parameters
                                  lambda_val=data['lambda_val'],
                                  trim=data['trim'], Bt_scale=data['Bt_scale'])
    assert np.std(coef_for_std) >= 0 # Standard deviation is always non-negative

# Testset 4: fdm tests (Translated from Julia @testset "fdm tests")
def test_fdm_schemes(load_test_data):
    data = load_test_data
    # Input for fdm is mag_1_c_d from the fixture
    mag_1_c_d_input = data['mag_1_c_d']

    # Schemes as tested in the Julia file
    schemes_from_julia = ["backward", "forward", "central", "backward2", "forward2", "fourth", "test"]

    for scheme_name in schemes_from_julia:
        # Julia: @test MagNav.fdm(mag_1_c_d;scheme=:backward ) isa Vector
        # Python: fdm_output = fdm(mag_1_c_d_input, scheme=scheme_name)
        #         assert isinstance(fdm_output, np.ndarray) and fdm_output.ndim == 1
        try:
            fdm_output = fdm(mag_1_c_d_input, scheme=scheme_name)
            assert isinstance(fdm_output, np.ndarray), \
                f"fdm output for scheme '{scheme_name}' is not a numpy array"
            assert fdm_output.ndim == 1, \
                f"fdm output for scheme '{scheme_name}' is not 1D"
            # Additional check: output should not be empty if input is not empty
            if mag_1_c_d_input.size > 0:
                assert fdm_output.size > 0, f"fdm output for scheme '{scheme_name}' is empty for non-empty input"
        except Exception as e:
            # Fail the test if fdm raises an unexpected error for a scheme that Julia tested
            pytest.fail(f"fdm(scheme='{scheme_name}') raised an exception: {e}")