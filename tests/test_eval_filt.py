"""
import pytest
import numpy as np
import scipy.io
import os
from pathlib import Path

# Assuming MagNavPy structure:
# MagNavPy/src/magnav.py contains Traj, INS, MapS, CRLBout, INSout, FILTout, FILTres, Map_Cache
# MagNavPy/src/eval_filt.py contains run_filt, eval_results, eval_crlb, eval_ins, eval_filt,
#                                   map_params, map_interpolate, upward_fft, chisq_pdf, chisq_cdf, chisq_q
# MagNavPy/src/plot_functions.py contains plotting functions like plot_filt, plot_mag_map, etc.
#                                      and potentially add_extension, points_ellipse, conf_ellipse_inplace,
#                                      conf_ellipse, units_ellipse, gif_ellipse

from magnavpy import magnav
from magnavpy import eval_filt
from magnavpy import plot_functions # Assuming plot functions are here
# from magnavpy import map_functions # Or map_params, map_interpolate, upward_fft might be here

# It's good practice for tests involving plotting to use a non-interactive backend.
# import matplotlib
# matplotlib.use('Agg') # Uncomment if GUI windows pop up during tests or if running in headless CI

# --- Data Loading ---
# Paths are relative to the workspace root c:/Users/Manas Pandey/Documents/magnav.py
# Pytest is often run from the project root (e.g., MagNavPy or the workspace root).
# If run from workspace root:
BASE_DIR = Path("MagNav.jl/test/test_data")

# If tests are run from MagNavPy directory, adjust path:
# BASE_DIR = Path("../MagNav.jl/test/test_data")

# Or, to be more robust if test file location is fixed relative to data:
# SCRIPT_DIR = Path(__file__).resolve().parent
# BASE_DIR = SCRIPT_DIR.parent.parent / "MagNav.jl" / "test" / "test_data"


def load_mat_file(filename):
    return scipy.io.loadmat(BASE_DIR / filename)

ins_data_mat = load_mat_file("test_data_ins.mat")
ins_data = ins_data_mat["ins_data"] # This is likely a structured array or dict

map_data_mat = load_mat_file("test_data_map.mat")
map_data = map_data_mat["map_data"]

traj_data_mat = load_mat_file("test_data_traj.mat")
traj_data = traj_data_mat["traj"] # This is likely a structured array or dict

# Helper to extract data from loaded .mat structures, similar to Julia's `vec(data["field"])`
def get_vec(data_struct, field_name):
    # scipy.io.loadmat might return 2D arrays (N,1) or (1,N)
    # .ravel() ensures it's 1D
    if isinstance(data_struct, dict): # If it's a dict of arrays
        return data_struct[field_name].ravel()
    # If it's a NumPy structured array (e.g. data_struct[0,0]['field_name'])
    # This part might need adjustment based on actual .mat structure
    try:
        return data_struct[field_name].ravel()
    except (TypeError, IndexError, KeyError): # Fallback for complex structured arrays
         # Common pattern for scalar structs from loadmat
        if hasattr(data_struct, "item") and isinstance(data_struct.item(), dict):
             return data_struct.item()[field_name].ravel()
        elif data_struct.dtype.fields and field_name in data_struct.dtype.fields:
            return data_struct[field_name].ravel()
        else: # Default if direct access works (e.g. if already a simple dict)
            raise ValueError(f"Could not extract {field_name} from mat struct")


ins_lat  = np.deg2rad(get_vec(ins_data, "lat"))
ins_lon  = np.deg2rad(get_vec(ins_data, "lon"))
ins_alt  = get_vec(ins_data, "alt")
ins_vn   = get_vec(ins_data, "vn")
ins_ve   = get_vec(ins_data, "ve")
ins_vd   = get_vec(ins_data, "vd")
ins_fn   = get_vec(ins_data, "fn")
ins_fe   = get_vec(ins_data, "fe")
ins_fd   = get_vec(ins_data, "fd")
# Cnb is likely (3,3,N)
ins_Cnb  = ins_data["Cnb"] if isinstance(ins_data, dict) else ins_data[0,0]["Cnb"]


map_info = "Map" # String as in Julia
map_map  = map_data["map"] if isinstance(map_data, dict) else map_data[0,0]["map"]
map_xx   = np.deg2rad(get_vec(map_data, "xx"))
map_yy   = np.deg2rad(get_vec(map_data, "yy"))
map_alt  = (map_data["alt"] if isinstance(map_data, dict) else map_data[0,0]["alt"]).item() # Assuming scalar alt

# map_params returns a tuple, Julia's [2] is Python's [1] (0-indexed)
# Assuming eval_filt.map_params exists and has similar signature
map_params_result = eval_filt.map_params(map_map, map_xx, map_yy)
map_mask = map_params_result[1]

tt       = get_vec(traj_data, "tt")
lat      = np.deg2rad(get_vec(traj_data, "lat"))
lon      = np.deg2rad(get_vec(traj_data, "lon"))
alt      = get_vec(traj_data, "alt")
vn       = get_vec(traj_data, "vn")
ve       = get_vec(traj_data, "ve")
vd       = get_vec(traj_data, "vd")
fn       = get_vec(traj_data, "fn")
fe       = get_vec(traj_data, "fe")
fd       = get_vec(traj_data, "fd")
Cnb      = traj_data["Cnb"] if isinstance(traj_data, dict) else traj_data[0,0]["Cnb"]
mag_1_c  = get_vec(traj_data, "mag_1_c")
mag_1_uc = get_vec(traj_data, "mag_1_uc")
flux_a_x = get_vec(traj_data, "flux_a_x")
flux_a_y = get_vec(traj_data, "flux_a_y")
flux_a_z = get_vec(traj_data, "flux_a_z")
flux_a_t = np.sqrt(flux_a_x**2 + flux_a_y**2 + flux_a_z**2)
N        = len(lat)
dt       = tt[1] - tt[0]

# Instantiate custom types
# Ensure these classes are defined in magnav.py and accept these arguments
traj = magnav.Traj(N,dt,tt,lat,lon,alt,vn,ve,vd,fn,fe,fd,Cnb)
ins  = magnav.INS(N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                  ins_fn,ins_fe,ins_fd,ins_Cnb,np.zeros((3,3,N)))
ins2 = magnav.INS(N,dt,tt,ins_lat,ins_lon,ins_alt,ins_vn,ins_ve,ins_vd,
                  ins_fn,ins_fe,ins_fd,ins_Cnb,np.ones((3,3,N)))

mapS       = magnav.MapS(map_info,map_map,map_xx,map_yy,map_alt,map_mask)
map_cache  = magnav.Map_Cache(maps=[mapS]) # Assuming Map_Cache takes a list of maps

# Assuming eval_filt.map_interpolate and eval_filt.upward_fft exist
# ':linear' becomes 'linear' string
itp_mapS   = eval_filt.map_interpolate(mapS, 'linear')
itp_mapS3D = eval_filt.map_interpolate(
    eval_filt.upward_fft(mapS, [mapS.alt, mapS.alt + 5000]), # Increased altitude diff to match typical usage
    'linear'
)

# Assuming itp_mapS is a callable that can take arrays lat, lon
map_val    = itp_mapS(lat, lon)

# --- Test Sets ---

def test_run_filt():
    # Test return types of run_filt
    # Tuple[MagNav.CRLBout,MagNav.INSout,MagNav.FILTout]
    # -> tuple of (magnav.CRLBout, magnav.INSout, magnav.FILTout)
    result1 = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'ekf',
                                 extract=True, run_crlb=True)
    assert isinstance(result1, tuple) and len(result1) == 3
    assert isinstance(result1[0], magnav.CRLBout)
    assert isinstance(result1[1], magnav.INSout)
    assert isinstance(result1[2], magnav.FILTout)

    # MagNav.FILTout -> magnav.FILTout
    result2 = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'ekf',
                                 extract=True, run_crlb=False)
    assert isinstance(result2, magnav.FILTout)

    # Tuple{MagNav.FILTres,Array} -> tuple of (magnav.FILTres, np.ndarray)
    result3 = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'mpf',
                                 extract=False, run_crlb=True)
    assert isinstance(result3, tuple) and len(result3) == 2
    assert isinstance(result3[0], magnav.FILTres)
    assert isinstance(result3[1], np.ndarray) # Assuming 'Array' maps to numpy array

    # MagNav.FILTres -> magnav.FILTres
    result4 = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'mpf',
                                 extract=False, run_crlb=False)
    assert isinstance(result4, magnav.FILTres)

    # @test_throws ErrorException -> pytest.raises(Exception) or specific error
    with pytest.raises(Exception): # Julia's ErrorException is generic
        eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'test_invalid_filter_type')

    # Test running multiple filters, expecting None (or some summary object)
    # Julia code expects Nothing, which is None in Python
    result5 = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, ['ekf', 'mpf'])
    assert result5 is None


# This part depends on the output of a specific run_filt call
# For simplicity, re-run it here or use a fixture if it's slow
filt_res_tuple = eval_filt.run_filt(traj, ins, mag_1_c, itp_mapS, 'ekf', extract=False, run_crlb=True)
filt_res_ekf = filt_res_tuple[0] # This is FILTres
crlb_P_ekf = filt_res_tuple[1]   # This is Array (numpy.ndarray)

def test_eval_results():
    # Tuple{MagNav.CRLBout,MagNav.INSout,MagNav.FILTout}
    eval_out = eval_filt.eval_results(traj, ins, filt_res_ekf, crlb_P_ekf)
    assert isinstance(eval_out, tuple) and len(eval_out) == 3
    assert isinstance(eval_out[0], magnav.CRLBout)
    assert isinstance(eval_out[1], magnav.INSout)
    assert isinstance(eval_out[2], magnav.FILTout)

    # MagNav.CRLBout
    crlb_out_direct = eval_filt.eval_crlb(traj, crlb_P_ekf)
    assert isinstance(crlb_out_direct, magnav.CRLBout)

    # MagNav.INSout
    ins_out_direct = eval_filt.eval_ins(traj, ins2) # Using ins2 as in Julia
    assert isinstance(ins_out_direct, magnav.INSout)

    # MagNav.FILTout
    filt_out_direct = eval_filt.eval_filt(traj, ins2, filt_res_ekf) # Using ins2
    assert isinstance(filt_out_direct, magnav.FILTout)

# For subsequent tests, get the fully evaluated results
(crlb_out, ins_out, filt_out) = eval_filt.eval_results(traj, ins, filt_res_ekf, crlb_P_ekf)

# Global for plotting tests, if needed, or pass as arguments
# p1_fig, p1_ax = plt.subplots() # If using matplotlib for inplace plots

show_plot_flag = False # As in Julia code

def test_plot_filt():
    # For plotting functions, we mainly test if they run without error
    # and return the expected type (e.g., a plot object or None for inplace)
    # Assuming plot_functions.plot_filt_inplace for the '!' version
    # If plot_filt! modifies a matplotlib Axes object:
    # fig, ax = plt.subplots()
    # assert plot_functions.plot_filt_inplace(ax, traj, ins, filt_out, show_plot=show_plot_flag) is None
    # For now, skipping direct test of inplace modification if API is unknown.

    # NTuple{5,Plots.Plot} -> tuple of 5 plot objects
    plot_tuple_1 = plot_functions.plot_filt(traj, ins, filt_out, plot_vel=True, show_plot=show_plot_flag)
    assert isinstance(plot_tuple_1, tuple) and len(plot_tuple_1) == 5
    # Can add more specific checks if plot objects are known (e.g. matplotlib Figure/Axes)

    # NTuple{3,Plots.Plot} -> tuple of 3 plot objects
    plot_tuple_2 = plot_functions.plot_filt(traj, ins, filt_out, plot_vel=False, show_plot=show_plot_flag)
    assert isinstance(plot_tuple_2, tuple) and len(plot_tuple_2) == 3

    # NTuple{4,Plots.Plot} -> tuple of 4 plot objects
    plot_tuple_err = plot_functions.plot_filt_err(traj, filt_out, crlb_out, plot_vel=True, show_plot=show_plot_flag)
    assert isinstance(plot_tuple_err, tuple) and len(plot_tuple_err) == 4


def test_plot_mag_map():
    # Plots.Plot -> some plot object
    plot_obj_1 = plot_functions.plot_mag_map(traj, mag_1_c, itp_mapS, order='magmap', show_plot=show_plot_flag)
    assert plot_obj_1 is not None # Check it returns something

    plot_obj_2 = plot_functions.plot_mag_map(traj, mag_1_c, map_cache, order='magmap', show_plot=show_plot_flag)
    assert plot_obj_2 is not None

    plot_obj_3 = plot_functions.plot_mag_map(traj, mag_1_c, itp_mapS3D, order='mapmag', show_plot=show_plot_flag)
    assert plot_obj_3 is not None

    with pytest.raises(Exception): # Or more specific error if known
        plot_functions.plot_mag_map(traj, mag_1_c, itp_mapS, order='test_invalid_order', show_plot=show_plot_flag)

    plot_err_1 = plot_functions.plot_mag_map_err(traj, mag_1_c, itp_mapS, show_plot=show_plot_flag)
    assert plot_err_1 is not None

    plot_err_2 = plot_functions.plot_mag_map_err(traj, mag_1_c, map_cache, show_plot=show_plot_flag)
    assert plot_err_2 is not None

    plot_err_3 = plot_functions.plot_mag_map_err(traj, mag_1_c, itp_mapS3D, show_plot=show_plot_flag)
    assert plot_err_3 is not None


def test_plot_autocor():
    autocorr_plot = plot_functions.plot_autocor(mag_1_c - map_val, dt, 1, show_plot=show_plot_flag)
    assert autocorr_plot is not None


def test_chisq():
    # ≈ maps to pytest.approx for scalars
    assert eval_filt.chisq_pdf(0) == pytest.approx(0)
    assert eval_filt.chisq_cdf(0) == pytest.approx(0)
    assert eval_filt.chisq_q(0)   == pytest.approx(0) # Assuming chisq_q is similar

    assert 0.4 < eval_filt.chisq_pdf(0.5) < 0.6
    assert 0.4 < eval_filt.chisq_cdf(0.5) < 0.6
    assert 0.4 < eval_filt.chisq_q(0.5) < 0.6


# Prepare P for ellipse tests
P_ellipse = crlb_P_ekf[0:2, 0:2, :] # Slicing in Python is exclusive for end index

# fig_ellipse, ax_ellipse = plt.subplots() # For inplace ellipse plots

# Path for saving GIF, relative to this test file's directory
# SCRIPT_DIR = Path(__file__).resolve().parent
# ellipse_gif_base_path = SCRIPT_DIR / "conf_ellipse"

# Or relative to workspace root, if that's where output should go
# For consistency with Julia's @__DIR__, let's assume output in test dir
ellipse_gif_base_path = Path("MagNavPy/tests/conf_ellipse") # Output in tests dir

def test_ellipse():
    # Julia: ENV["GKSwstype"] = "100" (for headless GR backend)
    # Python/Matplotlib: matplotlib.use('Agg') can be set at top of file if needed.

    # NTuple{2,Vector} -> tuple of 2 numpy arrays
    points = plot_functions.points_ellipse(P_ellipse[:, :, 0])
    assert isinstance(points, tuple) and len(points) == 2
    assert isinstance(points[0], np.ndarray)
    assert isinstance(points[1], np.ndarray)

    # Inplace modification, returns None
    # assert plot_functions.conf_ellipse_inplace(ax_ellipse, P_ellipse[:, :, 0]) is None
    # assert plot_functions.conf_ellipse_inplace(ax_ellipse, P_ellipse[:, :, 0], plot_eigax=True) is None
    # Skipping inplace tests for now if matplotlib Axes object (ax_ellipse) setup is complex here

    # Returns a plot object
    conf_plot = plot_functions.conf_ellipse(P_ellipse[:, :, 0])
    assert conf_plot is not None # Check it returns a plot object

    # Array -> np.ndarray
    assert isinstance(plot_functions.units_ellipse(P_ellipse, conf_units='deg'), np.ndarray)
    assert isinstance(plot_functions.units_ellipse(P_ellipse, conf_units='rad'), np.ndarray)

    # Assuming filt_res_ekf (FILTres) and filt_out (FILTout) are valid inputs
    assert isinstance(plot_functions.units_ellipse(filt_res_ekf, filt_out, conf_units='ft'), np.ndarray)
    assert isinstance(plot_functions.units_ellipse(filt_res_ekf, filt_out, conf_units='m'), np.ndarray)

    with pytest.raises(Exception): # Or specific error
        plot_functions.units_ellipse(filt_res_ekf, filt_out, conf_units='test_invalid_unit')

    # GIF generation tests
    # Ensure the directory for the GIF exists if the function doesn't create it
    ellipse_gif_base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Assuming add_extension is part of plot_functions or handled within gif_ellipse
    # ellipse_gif_full_path_str = str(ellipse_gif_base_path) + ".gif"
    # Using a fixed name for the test, as Julia code does.
    ellipse_gif_filename = "conf_ellipse.gif" 
    # Place it in the same directory as the test script for simplicity, or adjust path
    gif_output_path = Path(__file__).resolve().parent / ellipse_gif_filename


    # Plots.AnimatedGif -> type of animation object or path to file
    # Assuming gif_ellipse returns the path to the saved GIF or an animation object
    gif_result1 = plot_functions.gif_ellipse(P_ellipse, save_plot=True, filename=str(gif_output_path))
    assert gif_result1 is not None # Or check if file exists: assert gif_output_path.is_file()
    if gif_output_path.is_file(): os.remove(gif_output_path)


    gif_result2 = plot_functions.gif_ellipse(filt_res_ekf, filt_out, save_plot=True, filename=str(gif_output_path))
    assert gif_result2 is not None
    if gif_output_path.is_file(): os.remove(gif_output_path)

    gif_result3 = plot_functions.gif_ellipse(filt_res_ekf, filt_out, mapS, save_plot=True, filename=str(gif_output_path))
    assert gif_result3 is not None
    if gif_output_path.is_file(): os.remove(gif_output_path)

    # Final cleanup of the GIF file if created by the last call and not removed
    # The Julia code does:
    # ellipse_gif = MagNav.add_extension(ellipse_gif,".gif")
    # rm(ellipse_gif)
    # This implies the filename used by gif_ellipse might be predictable.
    # If gif_ellipse always creates "conf_ellipse.gif" in the specified path:
    # final_gif_path = ellipse_gif_base_path.with_suffix(".gif") # pathlib way
    # if final_gif_path.is_file():
    #     os.remove(final_gif_path)
    # The test above already removes gif_output_path after each creation.
    # This final cleanup might be redundant if tests handle their own files.
    # If the Julia code's rm() is for a file named literally "conf_ellipse.gif" in @__DIR__
    # then the Python equivalent would be:
    # final_cleanup_path = Path(__file__).resolve().parent / "conf_ellipse.gif"
    # if final_cleanup_path.is_file():
    #    os.remove(final_cleanup_path)
    # This is handled by individual test cleanups above.
"""