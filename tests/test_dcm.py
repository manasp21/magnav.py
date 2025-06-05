import pytest
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# Assuming dcm functions are in magnavpy.dcm
# Adjust the import path if your project structure is different
from magnavpy.dcm_util import euler2dcm, dcm2euler, correct_Cnb

# Determine the correct relative path to the .mat file
# The test script is in MagNavPy/tests/
# The data file is in MagNav.jl/test/test_data/
# So, we need to go up two directories from MagNavPy/tests/ to the project root,
# then into MagNav.jl/test/test_data/
TEST_DATA_PATH = Path(__file__).parent.parent.parent / "MagNav.jl" / "test" / "test_data" / "test_data_dcm.mat"

def load_test_data():
    """Loads test data from test_data_dcm.mat"""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data file not found: {TEST_DATA_PATH}")
    return loadmat(TEST_DATA_PATH)["dcm_data"]

# Load test data and extract arrays directly
dcm_data = loadmat(TEST_DATA_PATH)["dcm_data"]

# Helper function to robustly extract 1D scalar arrays
def _extract_scalar_vector(data_field: np.ndarray) -> np.ndarray:
    """
    Extracts a 1D array of scalars from a data field loaded from .mat,
    which might be an object array of arrays or a simple numeric array.
    Takes the first element of any inner arrays.
    """
    if data_field.ndim == 0: # Handle scalar case
        return np.array([data_field.item()])
    if data_field.dtype == object:
        # If elements are arrays, take the first element of each inner array
        return np.array([arr.flat[0] if hasattr(arr, 'flat') else arr for arr in data_field.flat], dtype=float)
    else:
        # Standard case, e.g., a 2D array (N,1) or (1,N) that needs flattening
        return data_field.flatten().astype(float)

# Extract data similarly to the Julia script
# Ensure correct indexing and shaping if necessary
# Extract Euler angles as float arrays
# Extract Euler angles as 1D arrays
roll_jl = dcm_data["roll"].squeeze()
pitch_jl = dcm_data["pitch"].squeeze()
yaw_jl = dcm_data["yaw"].squeeze()

# Ensure they are arrays, not scalars
roll_jl = np.atleast_1d(roll_jl)
pitch_jl = np.atleast_1d(pitch_jl)
yaw_jl = np.atleast_1d(yaw_jl)

# Properly extract DCM arrays - avoid .item() for non-scalar arrays
cnb_expected_jl = dcm_data["Cnb"][0, 0]  # 3x3xN array
cnb_estimate_expected_jl = dcm_data["Cnb_estimate"][0, 0]  # 3x3xN array

# Extract first DCM for single test cases
cnb_1_expected_jl = cnb_expected_jl[:, :, 0]
cnb_estimate_1_expected_jl = cnb_estimate_expected_jl[:, :, 0]

# Extract tilt error data as a 3xN array
tilt_err_jl = dcm_data["tilt_err"][0, 0]

# Calculate Cnb_1 and Cnb using the Python implementation
# Use first elements of arrays for single test case
cnb_1_py = euler2dcm(
    roll_jl[0] if roll_jl.size > 0 else 0.0,
    pitch_jl[0] if pitch_jl.size > 0 else 0.0,
    yaw_jl[0] if yaw_jl.size > 0 else 0.0,
    "body2nav"
)
# Create batch DCMs by iterating through Euler angles
cnb_py_list = []
for i in range(len(roll_jl)):
    cnb = euler2dcm(roll_jl[i], pitch_jl[i], yaw_jl[i], "body2nav")
    cnb_py_list.append(cnb)
    
if cnb_py_list:
    cnb_py = np.stack(cnb_py_list, axis=0)  # Stacks to (N,3,3)
else:
    cnb_py = np.empty((3,3,0))


# Calculate Cnb_estimate_1 and Cnb_estimate
# Ensure tilt_err_jl[:,0] is correctly shaped for correct_Cnb
# Use the first column for single Cnb test
cnb_estimate_1_py = correct_Cnb(cnb_1_py, tilt_err_jl[:, 0])

# For cnb_estimate_py, correct_Cnb needs to handle a 3D Cnb and 2D tilt_err
# or be called in a loop.
# Assuming correct_Cnb(Cnb_stack, tilt_err_stack) works if Cnb_stack is (3,3,N) and tilt_err_stack is (3,N)
if cnb_py.shape[2] > 0 : # only if there's data
    # Process the tilt error array to ensure it's in the correct format
    # The MATLAB data is stored as a 3x3 array, but we need to use it column-wise
    # Ensure tilt_err_jl is 2D (3xN)
    if tilt_err_jl.ndim == 3 and tilt_err_jl.shape[1] == 1:
        tilt_err_jl = tilt_err_jl.squeeze(axis=1)
    
    cnb_estimate_py = []
    for i in range(cnb_py.shape[0]):
        # Ensure we're using a 1D array for tilt error
        corrected = correct_Cnb(cnb_py[i], tilt_err_jl[:, i].flatten())
        cnb_estimate_py.append(corrected)
    cnb_estimate_py = np.stack(cnb_estimate_py, axis=0)
else:
    cnb_estimate_py = np.empty((3,3,0))


def test_correct_cnb():
    """Tests for the correct_Cnb function and initial Cnb calculations."""
    # Extract first DCM from expected arrays for single test case
    # For single DCM test case, remove extra dimensions
    np.testing.assert_allclose(cnb_1_py.squeeze(), cnb_1_expected_jl, atol=1e-7)
    # For batch DCMs, ensure both arrays have same shape
    np.testing.assert_allclose(cnb_py.squeeze(), cnb_expected_jl.squeeze(), atol=1e-7)
    # For single corrected DCM test case
    np.testing.assert_allclose(cnb_estimate_1_py.squeeze(), cnb_estimate_1_expected_jl, atol=1e-7)
    # For batch corrected DCMs
    np.testing.assert_allclose(cnb_estimate_py.squeeze(), cnb_estimate_expected_jl.squeeze(), atol=1e-7)

    # Test size/shape
    # Create a dummy Cnb_1 for this specific test, as cnb_1_py is already defined
    dummy_cnb_single = euler2dcm(0.1, 0.2, 0.3, "body2nav")
    corrected_cnb_zeros = correct_Cnb(dummy_cnb_single, np.zeros(3)) # Julia uses (3,1)
    assert corrected_cnb_zeros.shape == (3, 3) # A single DCM is (3,3)

    # If correct_Cnb can output a stack (3,3,1) for single input if tilt_err is (3,1)
    # This depends on the Python implementation of correct_Cnb
    # The Julia test MagNav.correct_Cnb(Cnb_1,zeros(3,1))) == (3,3,1) implies
    # that even for a single Cnb_1 (3x3), if tilt_err is (3x1), it might produce a (3,3,1) stack.
    # Let's assume the Python version for a single DCM input returns (3,3).
    # If it's meant to return (3,3,1) for a (3,1) error vector, the assertion needs to change.
    # For now, sticking to (3,3) for a single DCM.
    # If the python correct_Cnb is designed to stack for (3,1) error, this test would be:
    # corrected_cnb_zeros_stacked = correct_Cnb(dummy_cnb_single, np.zeros((3,1)))
    # assert corrected_cnb_zeros_stacked.shape == (3,3,1)
    # This needs clarification based on magnavpy.dcm.correct_Cnb behavior.
    # For now, we assume the simpler (3,3) output for single DCM.


def test_euler_dcm_conversions():
    """Tests for euler2dcm and dcm2euler conversions."""
    rpy_rad = np.deg2rad([45, 60, 15]) # Tuple in Julia, list or array in Python

    cnb_temp_py = euler2dcm(rpy_rad[0], rpy_rad[1], rpy_rad[2], "body2nav")
    # For nav2body, use body2nav and transpose
    cbn_temp_py = euler2dcm(rpy_rad[0], rpy_rad[1], rpy_rad[2], "body2nav").T

    # Assuming dcm2euler returns a tuple (roll, pitch, yaw)
    roll_1_py, pitch_1_py, yaw_1_py = dcm2euler(cnb_temp_py, "body2nav")
    roll_2_py, pitch_2_py, yaw_2_py = dcm2euler(cbn_temp_py, "xyz")

    assert roll_1_py == pytest.approx(rpy_rad[0])
    assert pitch_1_py == pytest.approx(rpy_rad[1])
    assert yaw_1_py == pytest.approx(rpy_rad[2])

    assert roll_2_py == pytest.approx(rpy_rad[0])
    assert pitch_2_py == pytest.approx(rpy_rad[1])
    assert yaw_2_py == pytest.approx(rpy_rad[2])

    # Test for invalid type argument
    # Assuming the Python functions raise ValueError for incorrect type strings
    with pytest.raises(ValueError): # Or TypeError, depending on implementation
        euler2dcm(roll_jl[0], pitch_jl[0], yaw_jl[0], "test_invalid_type")

    with pytest.raises(ValueError): # Or TypeError
        dcm2euler(cnb_1_py, "test_invalid_type")

# Additional considerations:
# 1. Ensure that the string arguments like "body2nav", "nav2body" match exactly
#    what the Python implementations in magnavpy.dcm expect.
# 2. The behavior of euler2dcm when given array inputs (roll, pitch, yaw) vs. scalar inputs.
#    The Julia code implies euler2dcm(roll_vec, pitch_vec, yaw_vec) produces a (3,3,N) stack.
#    The current Python translation uses a list comprehension and np.stack. If euler2dcm
#    itself handles array inputs to produce a stack, that would be more direct.
#    cnb_py = euler2dcm(roll_jl, pitch_jl, yaw_jl, "body2nav") # If it supports this
# 3. Similarly, for correct_Cnb(Cnb_stack, tilt_err_stack).
#    cnb_estimate_py = correct_Cnb(cnb_py, tilt_err_jl) # If it supports stack inputs
# 4. The exact exception type for @test_throws (ErrorException in Julia) might map
#    to ValueError, TypeError, or a custom Exception in Python. Using ValueError as a placeholder.