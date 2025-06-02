# MagNavPy/src/plot_functions.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend as scipy_detrend, welch, spectrogram
# Note: firwin, filtfilt were in thought process but not directly used in final Julia code shown
from scipy.stats import pearsonr, linregress
from scipy.special import expit as sigmoid # for σ
import math

# Assuming these will be available from other modules in MagNavPy
# It's crucial these imports are correct relative to the project structure.
# For example: from .magnav import XYZ, MagV (if MagV is a class/type)
# from .analysis_util import get_x
# from .tolles_lawson import create_TL_A, create_TL_coef

# Constants (assuming based on Julia code context)
NUM_MAG_MAX = 9 # Assumption, based on mag_9_uc in Julia comments/usage

# Helper for derivatives
def derivative(func, x, dx=1e-6):
    """Computes numerical derivative."""
    return (func(x + dx) - func(x - dx)) / (2 * dx)

# Activation functions
def relu(x):
    """Rectified Linear Unit."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU."""
    return np.where(x > 0, 1.0, 0.0) # Ensure float output

# sigmoid is imported as scipy.special.expit

def sigmoid_derivative(x):
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1 - s)

def swish(x):
    """Swish activation function."""
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivative of Swish."""
    sx = sigmoid(x)
    return sx + x * sx * (1 - sx) # f'(x) = f(x) + sigmoid(x)*(1-f(x)) where f(x) = x*sigmoid(x) -> simpler: sigmoid(x) + x * sigmoid_derivative(x)

def tanh(x):
    """Hyperbolic tangent."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh."""
    return 1 - np.tanh(x)**2

# Helper for detrend to match Julia's detrend options
def detrend_mean_only(data):
    """Subtracts the mean from the data."""
    return data - np.mean(data)

def detrend_linear(data):
    """Subtracts a linear fit from the data (default for scipy.signal.detrend)."""
    return scipy_detrend(data, type='linear')


def plot_basic(tt, y, ind=None, lab="", xlab="time [min]", ylab="",
               show_plot=True, save_plot=False, plot_png="data_vs_time.png"):
    """
    Plot data vs time.

    Args:
        tt (np.ndarray): length-N time vector [s]
        y (np.ndarray): length-N data vector
        ind (np.ndarray, optional): boolean array for selected data indices. Defaults to all true.
        lab (str, optional): data (legend) label. Defaults to "".
        xlab (str, optional): x-axis label. Defaults to "time [min]".
        ylab (str, optional): y-axis label. Defaults to "".
        show_plot (bool, optional): if true, show plot. Defaults to True.
        save_plot (bool, optional): if true, save plot as plot_png. Defaults to False.
        plot_png (str, optional): plot file name to save. Defaults to "data_vs_time.png".

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    if ind is None:
        ind = np.ones_like(tt, dtype=bool)

    fig, ax = plt.subplots()
    # Ensure tt[ind] is not empty before accessing tt[ind][0]
    if np.any(ind) and len(tt[ind]) > 0:
        time_axis = (tt[ind] - tt[ind][0]) / 60
        data_axis = y[ind]
        ax.plot(time_axis, data_axis, label=lab)
    else:
        # Handle empty selection: plot nothing or a message
        ax.text(0.5, 0.5, "No data selected for plotting", ha='center', va='center', transform=ax.transAxes)


    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if lab:
        ax.legend()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): # Allow other extensions
            plot_png += ".png"
        plt.savefig(plot_png)
        print(f"Plot saved to {plot_png}")
    if show_plot:
        plt.show()

    return fig

def plot_activation(activation_func_names=None, plot_deriv=False, show_plot=True,
                    save_plot=False, plot_png="act_func.png"):
    """
    Plot activation function(s) or their derivative(s).

    Args:
        activation_func_names (list, optional): activation function names (strings: 'relu', 'sigmoid', 'swish', 'tanh').
                                              Defaults to ['relu', 'sigmoid', 'swish', 'tanh'].
        plot_deriv (bool, optional): if true, plot derivatives. Defaults to False.
        show_plot (bool, optional): if true, show plot. Defaults to True.
        save_plot (bool, optional): if true, save plot. Defaults to False.
        plot_png (str, optional): plot file name to save. Defaults to "act_func.png".

    Returns:
        matplotlib.figure.Figure: The plot figure.
    """
    if activation_func_names is None:
        activation_func_names = ['relu', 'sigmoid', 'swish', 'tanh']

    dpi = 500 if save_plot else plt.rcParams['figure.dpi'] # Use default dpi if not saving high-res
    x = np.linspace(-6, 4, 1000)

    func_map = {
        'relu': (relu, relu_derivative),
        'sigmoid': (sigmoid, sigmoid_derivative),
        'swish': (swish, swish_derivative),
        'tanh': (tanh, tanh_derivative),
    }

    label_map = {
        'relu': "ReLU", 'sigmoid': "Sigmoid", 'swish': "Swish", 'tanh': "tanh"
    }
    linestyle_map = { # From Julia code
        'relu': 'solid', 'sigmoid': 'solid', 'swish': 'dashed', 'tanh': 'dotted'
    }

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlabel("z")
    ax.set_xlim(-6, 4)
    # Julia's margin=2*mm. Matplotlib handles margins via tight_layout or subplots_adjust.
    # plt.tight_layout() is often sufficient.

    target_funcs_data = []
    if not plot_deriv:
        ax.set_ylabel("f(z)")
        ax.set_ylim(-2, 4)
        for name in activation_func_names:
            if name in func_map:
                target_funcs_data.append((x, func_map[name][0](x), label_map[name], linestyle_map[name]))
    else:
        ax.set_ylabel("f'(z)")
        ax.set_ylim(-0.5, 1.5)
        for name in activation_func_names:
            if name in func_map:
                target_funcs_data.append((x, func_map[name][1](x), label_map[name], linestyle_map[name]))

    for x_vals, y_vals, label, ls in target_funcs_data:
        ax.plot(x_vals, y_vals, label=label, linewidth=2, linestyle=ls)

    ax.legend(loc='upper left')
    plt.tight_layout()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")):
            plot_png += ".png"
        plt.savefig(plot_png)
        print(f"Plot saved to {plot_png}")
    if show_plot:
        plt.show()

    return fig

# Placeholder for MagV type if it's a specific class.
# If it's just a convention for objects with .x, .y, .z, .t, then hasattr checks are fine.
# class MagVPlaceholder:
#     def __init__(self, x, y, z, t): self.x, self.y, self.z, self.t = x, y, z, t

def plot_mag(xyz, ind=None, detrend_data=False, use_mags=None,
             vec_terms=None, ylim=None, dpi=None, show_plot=True,
             save_plot=False, plot_png="scalar_mags.png"):
    """
    Plot scalar or vector (fluxgate) magnetometer data.
    xyz: An object expected to have attributes like xyz.traj.tt, xyz.mag_1_c,
    and for vector mags, attributes like xyz.flux_a which itself has .x, .y, .z, .t.
    """
    if ind is None: ind = np.ones_like(xyz.traj.tt, dtype=bool)
    if use_mags is None: use_mags = ['all_mags']
    if vec_terms is None: vec_terms = ['all']
    if dpi is None: dpi = plt.rcParams['figure.dpi']


    if not np.any(ind) or len(xyz.traj.tt[ind]) == 0:
        fig, ax = plt.subplots(dpi=dpi)
        ax.text(0.5, 0.5, "No data selected for plotting", ha='center', va='center', transform=ax.transAxes)
        if show_plot: plt.show()
        return fig
        
    tt_plot = (xyz.traj.tt[ind] - xyz.traj.tt[ind][0]) / 60
    xlab = "time [min]"

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlabel(xlab)

    list_c_names = [f"mag_{i}_c" for i in range(1, NUM_MAG_MAX + 1)]
    list_uc_names = [f"mag_{i}_uc" for i in range(1, NUM_MAG_MAX + 1)]

    available_mags_c = [name for name in list_c_names if hasattr(xyz, name)]
    available_mags_uc = [name for name in list_uc_names if hasattr(xyz, name)]
    
    # Paired mags (both _c and _uc exist for the same number)
    paired_mag_indices = []
    for i in range(1, NUM_MAG_MAX + 1):
        if f"mag_{i}_c" in available_mags_c and f"mag_{i}_uc" in available_mags_uc:
            paired_mag_indices.append(i)

    all_available_scalar_mags = available_mags_c + available_mags_uc
    plot_title = "Magnetometer Data" # Default title

    if 'comp_mags' in use_mags:
        ylab = "magnetic field error [nT]"
        plot_title = "Magnetometer Compensation Error"
        for i in paired_mag_indices:
            uc_data = getattr(xyz, f"mag_{i}_uc")[ind]
            c_data = getattr(xyz, f"mag_{i}_c")[ind]
            val = uc_data - c_data
            if detrend_data: val = detrend_linear(val)
            ax.plot(tt_plot, val, label=f"mag_{i} comp", linewidth=2)
            print(f"==== mag_{i} comp ====")
            print(f"avg comp = {np.mean(val):.3f} nT, std dev = {np.std(val):.3f} nT")

    # Check for vector magnetometers. This is a heuristic.
    # Assumes vector mag fields are explicitly named in use_mags (e.g., 'flux_a')
    # and these fields on xyz are objects with .x, .y, .z, .t attributes.
    elif any(hasattr(xyz, mag_name) and all(hasattr(getattr(xyz, mag_name), term) for term in ['x','y','z','t'])
             for mag_name in use_mags if isinstance(mag_name, str)):
        
        actual_vec_plot_terms = ['x', 'y', 'z', 't'] if 'all' in vec_terms else \
                                [term for term in vec_terms if term in ['x', 'y', 'z', 't']]
        ylab = "magnetic field [nT]"
        if detrend_data: ylab = f"detrended {ylab}"
        plot_title = "Vector Magnetometer Data"

        for mag_name_str in use_mags:
            if hasattr(xyz, mag_name_str):
                vector_mag_obj = getattr(xyz, mag_name_str)
                for term_to_plot in actual_vec_plot_terms:
                    if hasattr(vector_mag_obj, term_to_plot):
                        val_full_series = getattr(vector_mag_obj, term_to_plot)
                        val = val_full_series[ind] # Assuming components are indexable like main data
                        if detrend_data: val = detrend_linear(val)
                        ax.plot(tt_plot, val, label=f"{mag_name_str} {term_to_plot}", linewidth=2)
                    else:
                        print(f"Warning: Term '{term_to_plot}' not found in vector mag '{mag_name_str}'.")
            # else: # This mag_name_str might not be a vector mag, could be scalar handled below
            #    pass

    elif 'all_mags' in use_mags or any(m in all_available_scalar_mags for m in use_mags):
        mags_to_plot_final = all_available_scalar_mags if 'all_mags' in use_mags else \
                             [m for m in use_mags if m in all_available_scalar_mags]
        ylab = "magnetic field [nT]"
        if detrend_data: ylab = f"detrended {ylab}"
        plot_title = "Scalar Magnetometer Data"
        for mag_name_str in mags_to_plot_final:
            val = getattr(xyz, mag_name_str)[ind]
            if detrend_data: val = detrend_linear(val)
            ax.plot(tt_plot, val, label=mag_name_str, linewidth=2)
    else: # Fallback for specific scalar mags not caught by 'all_mags' logic
        ylab = "magnetic field [nT]"
        if detrend_data: ylab = f"detrended {ylab}"
        mags_to_plot_specific = [m for m in use_mags if hasattr(xyz,m) and isinstance(getattr(xyz,m), np.ndarray)]
        if mags_to_plot_specific:
            plot_title = "Selected Scalar Magnetometer Data"
            for mag_name_str in mags_to_plot_specific:
                val = getattr(xyz, mag_name_str)[ind]
                if detrend_data: val = detrend_linear(val)
                ax.plot(tt_plot, val, label=mag_name_str, linewidth=2)
        else:
            print(f"Warning: No plottable magnetometers found for use_mags: {use_mags}")


    ax.set_ylabel(ylab)
    ax.set_title(plot_title)
    if ylim: ax.set_ylim(ylim)
    if any(ax.get_legend_handles_labels()[1]): ax.legend()
    plt.tight_layout()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
        plt.savefig(plot_png, dpi=dpi)
        print(f"Plot saved to {plot_png}")
    if show_plot: plt.show()
    return fig


def plot_mag_c(xyz, xyz_comp, ind=None, ind_comp=None, detrend_data=True,
               lambda_ridge=0.025, terms=None, pass1=0.1, pass2=0.9, fs=10.0,
               use_mags=None, use_vec='flux_a', plot_diff=False,
               plot_mag_1_uc=True, plot_mag_1_c_provided=True, # Renamed from plot_mag_1_c
               ylim=None, dpi=None, show_plot=True, save_plot=False,
               plot_png="scalar_mags_comp.png"):
    """
    Plot compensated magnetometer(s) data.
    WARNING: This function relies on 'create_TL_A' and 'create_TL_coef'
    from '.tolles_lawson' module, which are not defined here.
    It will not work correctly without them.
    """
    print("WARNING: plot_mag_c is partially implemented and depends on external functions " + \
          "(create_TL_A, create_TL_coef) from other modules (e.g., .tolles_lawson).")

    if ind is None: ind = np.ones(xyz.traj.N, dtype=bool) # Assuming xyz.traj.N
    if ind_comp is None: ind_comp = np.ones(xyz_comp.traj.N, dtype=bool)
    if terms is None: terms = ['permanent', 'induced', 'eddy']
    if use_mags is None: use_mags = ['all_mags']
    if dpi is None: dpi = plt.rcParams['figure.dpi']

    if not (np.any(ind) and np.any(ind_comp)):
        fig, ax = plt.subplots(dpi=dpi)
        ax.text(0.5,0.5, "No data for compensation plot.", ha='center', va='center')
        if show_plot: plt.show()
        return fig

    # Check for essential external functions (placeholders)
    try:
        from .tolles_lawson import create_TL_A, create_TL_coef
    except ImportError:
        print("ERROR: plot_mag_c cannot proceed without 'create_TL_A' and 'create_TL_coef' from .tolles_lawson.")
        fig, ax = plt.subplots(dpi=dpi)
        ax.text(0.5,0.5, "Missing dependencies for plot_mag_c", ha='center', va='center', color='red')
        if show_plot: plt.show()
        return fig

    if not hasattr(xyz, use_vec) or not hasattr(xyz_comp, use_vec):
        print(f"ERROR: Vector magnetometer '{use_vec}' not found in xyz or xyz_comp.")
        # Handle error appropriately
        return plt.figure() # Return empty figure

    A_matrix_full_flight = create_TL_A(getattr(xyz, use_vec), terms=terms) # This is for the flight to be compensated
    A_matrix_selected = A_matrix_full_flight[ind, :]


    tt_plot = (xyz.traj.tt[ind] - xyz.traj.tt[ind][0]) / 60
    
    if not hasattr(xyz, 'mag_1_c') or not hasattr(xyz, 'mag_1_uc'):
        print("Warning: xyz.mag_1_c or xyz.mag_1_uc not found for baseline comparison.")
        # Fallback or error
        return plt.figure()

    mag_1_c_provided_data = xyz.mag_1_c[ind]
    mag_1_uc_data = xyz.mag_1_uc[ind]

    if detrend_data:
        mag_1_c_provided_data = detrend_linear(mag_1_c_provided_data)
        mag_1_uc_data = detrend_linear(mag_1_uc_data)

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("magnetic field [nT]")
    plot_title = "Compensated Scalar Magnetometer Data"

    if plot_mag_1_uc and not plot_diff:
        ax.plot(tt_plot, mag_1_uc_data, label="mag_1_uc (input)", color='cyan', linewidth=1.5, linestyle=':')

    available_uc_mags_in_xyz = [f"mag_{i}_uc" for i in range(1, NUM_MAG_MAX + 1) if hasattr(xyz, f"mag_{i}_uc")]
    
    mags_to_process = []
    if 'all_mags' in use_mags:
        mags_to_process = available_uc_mags_in_xyz
    else:
        mags_to_process = [m for m in use_mags if m in available_uc_mags_in_xyz]

    color_map = { # From Julia
        'mag_1_uc': 'red', 'mag_2_uc': 'purple', 'mag_3_uc': 'green', 'mag_4_uc': 'black',
        'mag_5_uc': 'orange', 'mag_6_uc': 'gray', 'mag_7_uc': 'violet', 'mag_8_uc': 'brown',
        'mag_9_uc': 'pink',
    }
    default_color = 'grey' # Fallback color

    for mag_name_uc_to_process in mags_to_process:
        if not hasattr(xyz_comp, mag_name_uc_to_process):
            print(f"Warning: {mag_name_uc_to_process} not found in xyz_comp. Skipping compensation for this mag.")
            continue
        
        mag_uc_for_comp_coeffs = getattr(xyz_comp, mag_name_uc_to_process)[ind_comp]
        vec_for_comp_coeffs = getattr(xyz_comp, use_vec) # This needs to be the MagV object / arrays for ind_comp

        # The create_TL_coef expects the vector mag data for ind_comp, not the whole thing
        # This part needs careful handling of how vec_for_comp_coeffs is structured and indexed
        # Assuming create_TL_coef can handle the MagV object and ind_comp correctly, or expects pre-indexed arrays
        # For now, passing the MagV object and ind_comp separately.
        # This might need adjustment based on actual create_TL_A/coef implementation.
        # A common pattern is: vec_data_for_coeffs = {axis: getattr(vec_for_comp_coeffs, axis)[ind_comp] for axis in ['x','y','z','t']}
        
        tl_coeffs = create_TL_coef(vec_for_comp_coeffs, # Or pre-indexed vector data for ind_comp
                                   mag_uc_for_comp_coeffs,
                                   ind_comp, # This might be redundant if vec_for_comp_coeffs is pre-indexed
                                   lambda_ridge=lambda_ridge, terms=terms,
                                   pass1=pass1, pass2=pass2, fs=fs)

        current_mag_uc_flight_data = getattr(xyz, mag_name_uc_to_process)[ind]
        
        # Compensation is applied: B_comp = B_uncomp - A * coeffs
        # Julia: mag_uc - detrend(A*TL_coef;mean_only=true)
        # The detrend(mean_only=true) on A*coeffs is important.
        compensated_noise_estimate = A_matrix_selected @ tl_coeffs
        compensated_noise_estimate_detrended = detrend_mean_only(compensated_noise_estimate)
        
        mag_c_calculated = current_mag_uc_flight_data - compensated_noise_estimate_detrended
        
        # For reporting mean_diff, Julia uses: mean(mag_c - xyz.mag_1_c[ind])
        # This is before the final detrend_data step on mag_c_calculated for plotting
        mean_diff_to_provided_mag1c = np.mean(mag_c_calculated - xyz.mag_1_c[ind])


        if detrend_data: # Final detrend for plotting
            mag_c_calculated = detrend_linear(mag_c_calculated)
            # current_mag_uc_flight_data_detrended = detrend_linear(current_mag_uc_flight_data) # if needed

        label_base = mag_name_uc_to_process[:-3] + "_c (calc)" # e.g. mag_1_c (calc)
        plot_val = mag_c_calculated
        
        if plot_diff:
            label_base = f"Δ {label_base} vs provided mag_1_c"
            plot_val = mag_c_calculated - mag_1_c_provided_data # Diff w.r.t provided (and possibly detrended) mag_1_c

        color = color_map.get(mag_name_uc_to_process, default_color)
        # Julia plots [1:end-1]. If this is due to filter artifacts, it might be needed.
        # For now, plotting the whole series. If data length changes, adjust.
        ax.plot(tt_plot, plot_val, label=label_base, color=color, linewidth=1.5)

        if plot_diff:
            print(f"=== {label_base} ===")
            print(f"avg diff (calc_c vs provided_mag_1_c, before final detrend) = {mean_diff_to_provided_mag1c:.3f} nT")
            print(f"std dev of plotted diff = {np.std(plot_val):.3f} nT")

    if plot_mag_1_c_provided and not plot_diff:
        # Julia plots [1:end-1] here too.
        ax.plot(tt_plot, mag_1_c_provided_data, label="mag_1_c (provided)", color='blue', linestyle='--', linewidth=2)

    if ylim: ax.set_ylim(ylim)
    if any(ax.get_legend_handles_labels()[1]): ax.legend(fontsize='small')
    ax.set_title(plot_title)
    plt.tight_layout()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
        plt.savefig(plot_png, dpi=dpi)
        print(f"Plot saved to {plot_png}")
    if show_plot: plt.show()
    return fig


def plot_psd(x, fs=10, window='hamming', nperseg=None, dpi=None, show_plot=True,
             save_plot=False, plot_png="PSD.png"):
    """Plots Power Spectral Density using scipy.signal.welch."""
    if dpi is None: dpi = plt.rcParams['figure.dpi']
    if nperseg is None: nperseg = min(256, len(x)) if len(x) > 0 else 256
    if len(x) == 0:
        print("Warning: Empty data for PSD plot.")
        return plt.figure()

    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg, scaling='density')
    Pxx_db = 10 * np.log10(Pxx)

    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(freqs, Pxx_db)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power/Frequency [dB/Hz]")
    ax.set_title("Power Spectral Density (Welch)")
    plt.tight_layout()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
        plt.savefig(plot_png, dpi=dpi); print(f"Plot saved to {plot_png}")
    if show_plot: plt.show()
    return fig

def plot_spectrogram_mpl(x, fs=10, window='hamming', nperseg=None, dpi=None,
                         show_plot=True, save_plot=False, plot_png="spectrogram.png"):
    """Plots Spectrogram using scipy.signal.spectrogram and pcolormesh."""
    if dpi is None: dpi = plt.rcParams['figure.dpi']
    if nperseg is None: nperseg = min(256, len(x)) if len(x) > 0 else 256
    if len(x) == 0:
        print("Warning: Empty data for Spectrogram plot.")
        return plt.figure()

    f, t, Sxx = spectrogram(x, fs=fs, window=window, nperseg=nperseg, scaling='density')
    Sxx_db = 10 * np.log10(Sxx)

    fig, ax = plt.subplots(dpi=dpi)
    mesh = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='Intensity [dB]')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Spectrogram")
    plt.tight_layout()

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
        plt.savefig(plot_png, dpi=dpi); print(f"Plot saved to {plot_png}")
    if show_plot: plt.show()
    return fig

def plot_frequency(xyz, ind=None, field='mag_1_uc', freq_type='PSD',
                   detrend_data=True, window='hamming', nperseg=None, dpi=None,
                   show_plot=True, save_plot=False, plot_png=None):
    """Plots frequency domain data (PSD or Spectrogram) for a field in xyz."""
    if ind is None: ind = np.ones_like(xyz.traj.tt, dtype=bool)
    if not hasattr(xyz, field):
        raise ValueError(f"Field '{field}' not found in xyz data.")
    
    x_data = getattr(xyz, field)[ind]
    if not np.any(ind) or len(x_data) == 0 :
        print(f"Warning: No data for field '{field}' after indexing. Skipping frequency plot.")
        return plt.figure()


    if detrend_data: x_data = detrend_linear(x_data)
    fs = 1.0 / xyz.traj.dt # Assumes xyz.traj.dt is sample period in seconds

    if plot_png is None: plot_png = f"{field}_{freq_type.lower()}.png"
    
    fig_out = None
    if freq_type.upper() == 'PSD':
        fig_out = plot_psd(x_data, fs=fs, window=window, nperseg=nperseg, dpi=dpi,
                           show_plot=show_plot, save_plot=save_plot, plot_png=plot_png)
    elif freq_type.upper() in ['SPECTROGRAM', 'SPEC']:
        fig_out = plot_spectrogram_mpl(x_data, fs=fs, window=window, nperseg=nperseg, dpi=dpi,
                                       show_plot=show_plot, save_plot=save_plot, plot_png=plot_png)
    else:
        raise ValueError(f"Unknown freq_type: {freq_type}. Choose 'PSD' or 'spectrogram'.")
    return fig_out


def plot_correlation(x_data, y_data, xfeature_name='feature_1', yfeature_name='feature_2',
                     limit=0, dpi=None, show_plot=True, save_plot=False, plot_png=None,
                     silent=True):
    """Plots correlation between two data series."""
    if dpi is None: dpi = plt.rcParams['figure.dpi']
    x_data, y_data = np.array(x_data), np.array(y_data)

    if len(x_data) < 2 or len(y_data) < 2 or len(x_data) != len(y_data):
        if not silent: print(f"Correlation plot for {yfeature_name} vs {xfeature_name} skipped: insufficient/mismatched data.")
        return None

    corr_coeff, _ = pearsonr(x_data, y_data)
    slope, intercept, _, _, _ = linregress(x_data, y_data)
    title = f"{yfeature_name} vs {xfeature_name}"
    if not silent: print(f"{title} | Pearson corr: {corr_coeff:.5f}, slope: {slope:.5f}")

    fig_out = None
    if abs(corr_coeff) > limit:
        fig_out, ax = plt.subplots(dpi=dpi)
        ax.scatter(x_data, y_data, label=f"Data (Corr: {corr_coeff:.2f})", color='black', s=4, alpha=0.6)
        line_x = np.array([np.min(x_data), np.max(x_data)])
        ax.plot(line_x, slope * line_x + intercept, color='red', label=f"LinReg: y={slope:.2f}x+{intercept:.2f}")
        ax.set_xlabel(str(xfeature_name)); ax.set_ylabel(str(yfeature_name))
        ax.set_title(title); ax.legend(); plt.tight_layout()

        if plot_png is None: plot_png = f"{str(xfeature_name)}-{str(yfeature_name)}_corr.png"
        if save_plot:
            if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
            plt.savefig(plot_png, dpi=dpi); print(f"Plot saved to {plot_png}")
        if show_plot: plt.show()
    elif not silent:
        print(f"Correlation ({abs(corr_coeff):.3f}) for {title} below limit ({limit}). Plot skipped.")
    return fig_out

def plot_correlation_xyz(xyz, xfeature='mag_1_c', yfeature='mag_1_uc', ind=None,
                         limit=0, dpi=None, show_plot=True, save_plot=False,
                         plot_png=None, silent=True):
    """Plots correlation between two features from an XYZ object."""
    if ind is None: ind = np.ones_like(xyz.traj.tt, dtype=bool)
    if not (hasattr(xyz, xfeature) and hasattr(xyz, yfeature)):
        raise ValueError(f"Features '{xfeature}' or '{yfeature}' not found in xyz data.")
    
    x_d = getattr(xyz, xfeature)[ind]
    y_d = getattr(xyz, yfeature)[ind]
    if plot_png is None: plot_png = f"{xfeature}-{yfeature}_corr.png"

    return plot_correlation(x_d, y_d, xfeature_name=xfeature, yfeature_name=yfeature,
                            limit=limit, dpi=dpi, show_plot=show_plot, save_plot=save_plot,
                            plot_png=plot_png, silent=silent)

def downsample_data(data_array, max_points):
    """Helper to downsample data if it exceeds max_points, taking every Nth point."""
    if len(data_array) > max_points:
        step = math.ceil(len(data_array) / max_points) # Ensure step is at least 1
        return data_array[::step]
    return data_array

def plot_correlation_matrix(data_matrix, features, dpi=None, Nmax=1000,
                            show_plot=True, save_plot=False,
                            plot_png="correlation_matrix.png"):
    """
    Plots a scatter plot matrix for features.
    data_matrix: (N_samples, N_features) numpy array.
    features: List of N_features string names.
    """
    if dpi is None: dpi = plt.rcParams['figure.dpi']
    Nf = len(features)
    if not (2 <= Nf <= 5): # As per Julia original constraint
        raise ValueError(f"Number of features must be 2-5, got {Nf}")
    if data_matrix.shape[1] != Nf:
        raise ValueError(f"Data matrix columns ({data_matrix.shape[1]}) != num features ({Nf})")

    fig, axes = plt.subplots(Nf, Nf, figsize=(Nf * 2.2, Nf * 2.2), dpi=dpi, squeeze=False)

    for i in range(Nf): # y-axis of scatter (row in subplot grid)
        for j in range(Nf): # x-axis of scatter (col in subplot grid)
            ax = axes[i, j]
            if i == j: # Diagonal
                diag_data = downsample_data(data_matrix[:, i], Nmax)
                ax.hist(diag_data, bins='auto', color='grey', edgecolor='k', density=True)
                ax.set_title(features[i], fontsize=9, y=1.0, pad=-13) # Title inside plot
                ax.set_yticks([])
                if i < Nf - 1: ax.set_xticks([]) # Hide x-ticks except for bottom row
                else: ax.tick_params(axis='x', labelsize=7) # Small ticks for bottom hist
            elif i > j: # Lower triangle: Scatter plots
                x_scatter = downsample_data(data_matrix[:, j], Nmax)
                y_scatter = downsample_data(data_matrix[:, i], Nmax)
                ax.scatter(x_scatter, y_scatter, s=2, c='k', alpha=0.5)
                
                # Add correlation coefficient text
                if len(data_matrix[:,j]) >=2 and len(data_matrix[:,i]) >=2 : # Pearsonr needs at least 2 points
                    corr, _ = pearsonr(data_matrix[:, j], data_matrix[:, i])
                    ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes,
                            fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.7))
            else: # Upper triangle: Off
                ax.axis('off')

            # Axis labels for outer plots
            if j == 0 and i > j : ax.set_ylabel(features[i], fontsize=8) # Leftmost column y-labels
            if i == Nf - 1 and j < i : ax.set_xlabel(features[j], fontsize=8) # Bottom row x-labels
            
            # Tick labels
            if i > j : # Only for scatter plots in lower triangle
                ax.tick_params(axis='both', labelsize=7)
                if j > 0 : ax.set_yticklabels([]) # No y-tick labels for inner scatters
                if i < Nf -1 : ax.set_xticklabels([]) # No x-tick labels for inner scatters


    fig.subplots_adjust(hspace=0.15, wspace=0.15) # Reduce spacing
    # plt.tight_layout(pad=0.5) # Alternative spacing adjustment

    if save_plot:
        if not plot_png.lower().endswith((".png", ".pdf", ".svg")): plot_png += ".png"
        plt.savefig(plot_png, dpi=dpi); print(f"Plot saved to {plot_png}")
    if show_plot: plt.show()
    return fig


def plot_correlation_matrix_xyz(xyz, ind=None, features_setup=None,
                                terms=None, sub_diurnal=False, sub_igrf=False,
                                bpf_mag=False, dpi=None, Nmax=1000,
                                show_plot=True, save_plot=False,
                                plot_png="correlation_matrix_xyz.png"):
    """
    Plots correlation matrix for features derived from XYZ data.
    WARNING: Relies on 'get_x' from '.analysis_util' (not defined here).
    """
    print("WARNING: plot_correlation_matrix_xyz relies on 'get_x' from '.analysis_util' (or similar).")
    if ind is None: ind = np.ones_like(xyz.traj.tt, dtype=bool)
    if features_setup is None: features_setup = ['mag_1_uc', 'TL_A_flux_a'] # Julia default
    if terms is None: terms = ['permanent']

    try:
        from .analysis_util import get_x # Attempt to import
    except ImportError:
        print("ERROR: Cannot import 'get_x' from .analysis_util. plot_correlation_matrix_xyz will fail.")
        fig_err, ax_err = plt.subplots(dpi=dpi if dpi else plt.rcParams['figure.dpi'])
        ax_err.text(0.5, 0.5, "Missing 'get_x' dependency", ha='center', va='center', color='red', fontsize=12)
        if show_plot: plt.show()
        return fig_err

    # Call get_x to obtain data_matrix and final_feature_names
    # The exact signature and return order of get_x must match.
    # Julia: (x,_,features,_) = get_x(...) -> Python: data_matrix, _, final_feature_names, _
    data_matrix, _, final_feature_names, _ = get_x(
        xyz, ind, features_setup,
        terms=terms, sub_diurnal=sub_diurnal,
        sub_igrf=sub_igrf, bpf_mag=bpf_mag
    )

    return plot_correlation_matrix(data_matrix, final_feature_names,
                                   dpi=dpi, Nmax=Nmax, show_plot=show_plot,
                                   save_plot=save_plot, plot_png=plot_png)

# Example of how XYZ and MagV might be structured for type hinting or internal use,
# if not imported. These are NOT full implementations.
# class TrajectoryPlaceholder:
#   def __init__(self, tt, N, dt): self.tt, self.N, self.dt = tt, N, dt
# class XYZPlaceholder:
#   def __init__(self, tt, dt, mag_1_uc_data, mag_1_c_data=None, flux_a_data=None):
#       self.traj = TrajectoryPlaceholder(tt, len(tt), dt)
#       self.mag_1_uc = np.array(mag_1_uc_data)
#       if mag_1_c_data is not None: self.mag_1_c = np.array(mag_1_c_data)
#       if flux_a_data is not None: # flux_a_data could be a dict {'x':[], 'y':[], ..}
#           self.flux_a = type('FluxAPlaceholder', (), flux_a_data)() # Simple object from dict
# Add other fields as needed...

if __name__ == '__main__':
    # Example Usage (requires dummy data or actual MagNavPy objects)
    print("plot_functions.py loaded. Contains plotting utilities for MagNavPy.")
    print("To run examples, define dummy XYZ data or integrate with MagNavPy modules.")

    # Dummy data for basic plot
    # tt_dummy = np.linspace(0, 600, 6000) # 10 minutes of data at 10Hz
    # y_dummy = np.sin(tt_dummy / 60) + np.random.randn(len(tt_dummy)) * 0.1
    # plot_basic(tt_dummy, y_dummy, lab="Dummy Sine Wave", show_plot=True)

    # plot_activation(show_plot=True)
    # plot_activation(activation_func_names=['relu', 'tanh'], plot_deriv=True, show_plot=True)
    
    # Further examples would require creating XYZ-like objects and potentially
    # stubbing/mocking the dependent analysis_util and tolles_lawson functions.
    pass