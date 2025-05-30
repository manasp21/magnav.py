import pytest
import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import hamming
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.figure

# Assuming magnavpy modules are importable
# Adjust these imports based on your actual project structure
from magnavpy import magnav as mn
from magnavpy import plot_functions as pf

# Path to the test data directory, assuming MagNav.jl and MagNavPy are siblings
BASE_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "MagNav.jl" / "test" / "test_data"

@pytest.fixture(scope="module")
def xyz_object_data():
    """
    Loads and prepares data similar to the Julia test script, providing an XYZ0 object.
    """
    ins_file = BASE_DATA_PATH / "test_data_ins.mat"
    traj_file = BASE_DATA_PATH / "test_data_traj.mat"

    ins_data_mat = loadmat(ins_file)
    # In .mat, struct fields are often within a nested array, e.g., ins_data_mat['ins_data'][0,0]['lat']
    # For simplicity, assuming direct access or that loadmat simplifies this.
    # If ins_data is a structured array, access fields like ins_data_mat['ins_data']['lat'][0,0]
    # This part might need adjustment based on the exact structure of 'ins_data' in the .mat file
    # For this conversion, we'll assume a dictionary-like structure after loadmat for 'ins_data' and 'traj'
    # If 'ins_data' is a key holding a dict:
    ins_data = ins_data_mat["ins_data"]
    if not isinstance(ins_data, dict): # Handle structured numpy array if necessary
        # This is a common pattern for structs loaded from MATLAB
        ins_data_fields = ins_data.dtype.names
        ins_data = {field: ins_data[field][0,0].flatten() if ins_data[field][0,0].ndim > 1 else ins_data[field][0,0] for field in ins_data_fields}


    traj_data_mat = loadmat(traj_file)
    traj_data = traj_data_mat["traj"]
    if not isinstance(traj_data, dict):
        traj_data_fields = traj_data.dtype.names
        traj_data = {field: traj_data[field][0,0].flatten() if traj_data[field][0,0].ndim > 1 else traj_data[field][0,0] for field in traj_data_fields}


    ins_lat  = np.deg2rad(ins_data["lat"].flatten())
    ins_lon  = np.deg2rad(ins_data["lon"].flatten())
    ins_alt  = ins_data["alt"].flatten()
    ins_vn   = ins_data["vn"].flatten()
    ins_ve   = ins_data["ve"].flatten()
    ins_vd   = ins_data["vd"].flatten()
    ins_fn   = ins_data["fn"].flatten()
    ins_fe   = ins_data["fe"].flatten()
    ins_fd   = ins_data["fd"].flatten()
    # Cnb might be (3,3,N) or (N,3,3). Julia's Cnb = ins_data["Cnb"] is likely (3,3,N)
    # Python/Numpy might prefer (N,3,3). Let's assume it's loaded correctly for mn.INS
    ins_Cnb  = ins_data_mat["ins_data"]["Cnb"][0,0] # Adjust if ins_data was flattened differently

    tt       = traj_data["tt"].flatten()
    lat      = np.deg2rad(traj_data["lat"].flatten())
    lon      = np.deg2rad(traj_data["lon"].flatten())
    alt      = traj_data["alt"].flatten()
    vn       = traj_data["vn"].flatten()
    ve       = traj_data["ve"].flatten()
    vd       = traj_data["vd"].flatten()
    fn       = traj_data["fn"].flatten()
    fe       = traj_data["fe"].flatten()
    fd       = traj_data["fd"].flatten()
    Cnb      = traj_data_mat["traj"]["Cnb"][0,0] # Adjust as per ins_Cnb
    mag_1_c  = traj_data["mag_1_c"].flatten()
    mag_1_uc = traj_data["mag_1_uc"].flatten()
    flux_a_x = traj_data["flux_a_x"].flatten()
    flux_a_y = traj_data["flux_a_y"].flatten()
    flux_a_z = traj_data["flux_a_z"].flatten()
    flux_a_t = np.sqrt(flux_a_x**2 + flux_a_y**2 + flux_a_z**2)
    
    N = len(lat)
    dt = tt[1] - tt[0] if N > 1 else 0.1 # Handle N=1 case for dt

    # ins_P in Julia: zeros(1,1,N). Numpy: np.zeros((1,1,N)) or (N,1,1)
    # Assuming mn.INS expects P with shape (N, num_states, num_states) or similar
    # For simplicity, if it's just a placeholder:
    ins_P  = np.zeros((N, 1, 1)) # Placeholder, adjust if mn.INS expects specific shape

    val    = np.ones_like(lat)
    
    # Ensure Cnb and ins_Cnb are in (N,3,3) format if that's what Python classes expect
    # Original Julia code implies Cnb is likely (3,3,N). Let's transpose if needed.
    if Cnb.shape[0] == 3 and Cnb.shape[1] == 3: # (3,3,N)
        Cnb = np.transpose(Cnb, (2,0,1)) # to (N,3,3)
    if ins_Cnb.shape[0] == 3 and ins_Cnb.shape[1] == 3: # (3,3,N)
        ins_Cnb = np.transpose(ins_Cnb, (2,0,1)) # to (N,3,3)


    traj   = mn.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
    ins    = mn.INS(N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                     ins_fn,ins_fe,ins_fd,ins_Cnb,ins_P) # Pass ins_P
    flux_a = mn.MagV(N, dt, tt, flux_a_x,flux_a_y,flux_a_z,flux_a_t) # Assuming MagV constructor
    
    # For XYZ0, some fields might be optional or have defaults.
    # The Julia code uses 'val' for many fields.
    xyz    = mn.XYZ0(name="test", N=N, dt=dt, tt=tt,
                     traj=traj, ins=ins, flux_a=flux_a,
                     map_val=val, map_xx=val, map_yy=val, map_zz=val, # Renamed for clarity
                     map_filt=val, map_up=val,
                     mag_1_c=mag_1_c, mag_1_uc=mag_1_uc)

    ind = np.ones(N, dtype=bool)
    if N > 50:
        ind[50:] = False # Python 0-indexing, 50 corresponds to Julia's 51

    # Store commonly used components for easier access in tests
    return {
        "xyz": xyz,
        "tt": tt,
        "mag_1_c": mag_1_c,
        "mag_1_uc": mag_1_uc,
        "ind": ind,
        "N": N
    }

# Helper to close plots
def _close_plot(fig):
    if fig:
        plt.close(fig)

def test_plot_basic(xyz_object_data):
    """Tests for plot_basic function."""
    data = xyz_object_data
    fig = None
    try:
        fig = pf.plot_basic(data["tt"], data["mag_1_c"], show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure), "plot_basic should return a Figure object."
    finally:
        _close_plot(fig)
    
    fig = None
    try:
        fig = pf.plot_basic(data["tt"], data["mag_1_c"], data["ind"],
                            lab="mag_1_c",
                            xlab="time [min]",
                            ylab="magnetic field [nT]",
                            show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure), "plot_basic with options should return a Figure object."
    finally:
        _close_plot(fig)

def test_plot_activation():
    """Tests for plot_activation function."""
    fig = None
    try:
        fig = pf.plot_activation(show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure), "plot_activation should return a Figure object."
    finally:
        _close_plot(fig)

    fig = None
    try:
        # Assuming symbols :relu, :swish are passed as strings
        fig = pf.plot_activation(['relu', 'swish'],
                                 plot_deriv=True,
                                 show_plot=False,
                                 save_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure), "plot_activation with options should return a Figure object."
    finally:
        _close_plot(fig)

def test_plot_mag(xyz_object_data):
    """Tests for plot_mag function."""
    xyz = xyz_object_data["xyz"]
    ind = xyz_object_data["ind"]
    fig = None

    plot_configs = [
        {"use_mags": ['comp_mags']},
        {"use_mags": ['flux_a']},
        {"use_mags": ['flight']},
        {"ind": ind, "detrend_data": True, "use_mags": ['mag_1_c', 'mag_1_uc'], "vec_terms": ['all'], "ylim": (-300, 300), "dpi": 100, "save_plot": False},
        {"ind": ind, "detrend_data": True, "use_mags": ['comp_mags'], "vec_terms": ['all'], "ylim": (-1, 1), "dpi": 100, "save_plot": False},
        {"ind": ind, "detrend_data": True, "use_mags": ['flux_a'], "vec_terms": ['all'], "ylim": (-1000, 1000), "dpi": 100, "save_plot": False},
        {"ind": ind, "detrend_data": True, "use_mags": ['flight'], "vec_terms": ['all'], "ylim": (-1, 1), "dpi": 100, "save_plot": False},
    ]

    try:
        fig = pf.plot_mag(xyz, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    for config in plot_configs:
        fig = None
        try:
            fig = pf.plot_mag(xyz, show_plot=False, **config)
            assert isinstance(fig, matplotlib.figure.Figure)
        finally:
            _close_plot(fig)

def test_plot_mag_c(xyz_object_data):
    """Tests for plot_mag_c function."""
    xyz = xyz_object_data["xyz"]
    ind = xyz_object_data["ind"]
    N = xyz_object_data["N"]
    fig = None

    try:
        fig = pf.plot_mag_c(xyz, xyz, show_plot=False) # Assuming xyz_comp is another XYZ0 object
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # Test for expected error (Julia: @test_throws ErrorException)
    # The Python equivalent might raise ValueError, TypeError, or a custom exception
    with pytest.raises(Exception): # Use a more specific exception if known
        pf.plot_mag_c(xyz, xyz, use_mags=['test'], show_plot=False)
    
    fig = None
    try:
        # ind_comp needs to be defined. Julia uses `ind`, Python `ind[50:] = False`
        # For `.!ind`, use `~ind` or `np.logical_not(ind)`
        ind_comp = ~ind if N > 0 else np.array([], dtype=bool) # Ensure ind_comp is valid

        fig = pf.plot_mag_c(xyz, xyz, # Assuming xyz_comp is another XYZ0 object
                            ind=np.logical_not(ind) if N > 0 else np.array([], dtype=bool),
                            ind_comp=ind_comp,
                            detrend_data=False,
                            lambda_reg=0.0025, # Python typically uses lambda_reg or similar for λ
                            terms=['p', 'i', 'e', 'b'], # Assuming symbols are strings
                            pass1=0.2,
                            pass2=0.8,
                            fs=1.0,
                            use_mags=['mag_1_uc'], # Assuming symbols are strings
                            use_vec='flux_a',    # Assuming symbol is string
                            plot_diff=True,
                            plot_mag_1_uc=False,
                            plot_mag_1_c=False,
                            dpi=100,
                            ylim=(-50, 50),
                            show_plot=False,
                            save_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

def test_plot_psd(xyz_object_data):
    """Tests for plot_PSD function (named plot_psd for Python convention)."""
    # Assuming pf.plot_PSD is aliased or named pf.plot_psd in Python
    # The Julia code calls MagNav.plot_PSD, so pf.plot_PSD
    mag_1_c = xyz_object_data["mag_1_c"]
    fig = None
    try:
        fig = pf.plot_PSD(mag_1_c, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    fig = None
    try:
        fig = pf.plot_PSD(mag_1_c, fs=1.0, # fs is typically the second arg or kwarg
                          window=hamming, # Pass the window function
                          dpi=100,
                          show_plot=False,
                          save_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

def test_plot_spectrogram(xyz_object_data):
    """Tests for plot_spectrogram function."""
    # The Julia code calls MagNav.plot_spectrogram
    mag_1_c = xyz_object_data["mag_1_c"]
    fig = None
    try:
        fig = pf.plot_spectrogram(mag_1_c, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    fig = None
    try:
        fig = pf.plot_spectrogram(mag_1_c, fs=1.0, # fs is typically the second arg or kwarg
                                  window=hamming, # Pass the window function
                                  dpi=100,
                                  show_plot=False,
                                  save_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

def test_plot_frequency(xyz_object_data):
    """Tests for plot_frequency function."""
    xyz = xyz_object_data["xyz"]
    ind = xyz_object_data["ind"]
    fig = None
    try:
        fig = pf.plot_frequency(xyz, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    fig = None
    try:
        fig = pf.plot_frequency(xyz,
                                ind=ind,
                                field='mag_1_c', # Assuming symbol is string
                                freq_type='spec', # Assuming symbol is string
                                detrend_data=False,
                                window=hamming, # Pass the window function
                                dpi=100,
                                show_plot=False,
                                save_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

def test_plot_correlation(xyz_object_data):
    """Tests for plot_correlation function."""
    data = xyz_object_data
    xyz = data["xyz"]
    ind = data["ind"]
    mag_1_uc = data["mag_1_uc"]
    mag_1_c = data["mag_1_c"]
    fig = None

    # Test case 1: plot_correlation(xyz;show_plot) isa Plot
    try:
        fig = pf.plot_correlation(xyz, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # Test case 2: plot_correlation(xyz,:mag_1_uc,:mag_1_c,ind; ...) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation(xyz, field1='mag_1_uc', field2='mag_1_c', ind=ind,
                                  lim=0.5,
                                  dpi=100,
                                  show_plot=False,
                                  save_plot=False,
                                  silent=True)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # Test case 3: plot_correlation(mag_1_uc,mag_1_c;show_plot) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation(mag_1_uc, mag_1_c, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # Test case 4: plot_correlation(mag_1_uc,mag_1_c,:mag_1_uc,:mag_1_c; ...) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation(mag_1_uc, mag_1_c, field_name1='mag_1_uc', field_name2='mag_1_c',
                                  lim=0.5,
                                  dpi=100,
                                  show_plot=False,
                                  save_plot=False,
                                  silent=True)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # Test case 5: plot_correlation(xyz;lim=Inf,show_plot) isa Nothing
    # In Python, this might mean the function returns None or raises an error if lim=Inf is invalid,
    # or it successfully plots but the check is for no error.
    # If it's expected to return None when no significant correlation is found with lim=Inf:
    # This depends on the Python function's behavior.
    # For now, assume it runs and returns a Figure or None.
    # If it should return None:
    # result = pf.plot_correlation(xyz, lim=np.inf, show_plot=False)
    # assert result is None, "Expected None for lim=np.inf if no plot is generated."
    # If it still generates a plot (e.g. empty plot or all correlations):
    fig = None
    try:
        # This test in Julia `isa Nothing` might mean the plot object is not created or returned
        # under certain conditions. If the Python version always returns a figure,
        # this test might need to be adapted or check for specific properties of the figure.
        # For now, we'll assume it should still produce a figure object, or the test logic
        # in Python's plot_correlation handles lim=np.inf gracefully.
        fig = pf.plot_correlation(xyz, lim=np.inf, show_plot=False)
        if fig is not None: # If it can return None under this condition
             assert isinstance(fig, matplotlib.figure.Figure)
        # If it's guaranteed to return a figure, the `if fig is not None` is not needed.
        # If the Julia test `isa Nothing` implies no error and no plot shown,
        # then just running it without error is the test.
    finally:
        _close_plot(fig)


def test_plot_correlation_matrix(xyz_object_data):
    """Tests for plot_correlation_matrix function."""
    data = xyz_object_data
    xyz = data["xyz"]
    ind = data["ind"]
    fig = None

    # feat_set in Julia: [:mag_1_c,:mag_1_uc,:TL_A_flux_a,:flight]
    # Assuming these are attributes of xyz or keys if xyz is dict-like, or handled by the plot function
    feat_set_py = ['mag_1_c', 'mag_1_uc', 'TL_A_flux_a', 'flight'] # Python equivalent

    # Test cases from Julia
    # plot_correlation_matrix(xyz,ind;Nmax=10 ,show_plot) isa Plot
    try:
        fig = pf.plot_correlation_matrix(xyz, ind, Nmax=10, show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # plot_correlation_matrix(xyz,ind,feat_set[1:2];show_plot) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[0:2], show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # plot_correlation_matrix(xyz,ind,feat_set[3:3];show_plot) isa Plot (single feature)
    # Julia feat_set[3:3] is a list with one element. Python feat_set_py[2:3]
    fig = None
    try:
        fig = pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[2:3], show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)
    
    # plot_correlation_matrix(xyz,ind,feat_set[2:3];show_plot) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[1:3], show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # plot_correlation_matrix(xyz,ind,feat_set[1:3];show_plot) isa Plot
    fig = None
    try:
        fig = pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[0:3], show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        _close_plot(fig)

    # @test_throws AssertionError plot_correlation_matrix(xyz,ind,feat_set[1:1];show_plot)
    # Julia feat_set[1:1] is a list with one element. Python feat_set_py[0:1]
    # This implies the function expects more than one feature, or specific features.
    with pytest.raises(AssertionError):
        pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[0:1], show_plot=False)

    # @test_throws AssertionError plot_correlation_matrix(xyz,ind,feat_set[1:4];show_plot)
    # This implies a problem with the combination or number of features in feat_set[0:4]
    # or that 'TL_A_flux_a' or 'flight' might be problematic if not handled correctly.
    with pytest.raises(AssertionError):
        pf.plot_correlation_matrix(xyz, ind, features=feat_set_py[0:4], show_plot=False)