import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Union, Optional, Tuple

def linreg_matrix(y: np.ndarray, x_matrix: np.ndarray, lambda_ridge: float = 0) -> np.ndarray:
    """
    Linear regression with data matrix. Solves (X'X + lambda*I)beta = X'y.

    Args:
        y: length-N observed data vector (1D array).
        x_matrix: N x Nf input data matrix (Nf is number of features).
        lambda_ridge: (optional) ridge parameter.

    Returns:
        coef: linear regression coefficients (Nf x 1 array).
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    term1 = x_matrix.T @ x_matrix
    if lambda_ridge > 0:
        term1 += lambda_ridge * np.eye(x_matrix.shape[1])
    term2 = x_matrix.T @ y
    
    try:
        coef = np.linalg.solve(term1, term2)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        coef = np.linalg.pinv(term1) @ term2
    return coef.flatten() # Return as 1D array

def get_bpf_sos(pass1: float = 0.1, pass2: float = 0.9, fs: float = 10.0, pole: int = 4) -> np.ndarray:
    """
    Create Butterworth bandpass (or low-pass or high-pass) filter coefficients (SOS format).
    Set pass1 = -1 for low-pass filter or pass2 = -1 for high-pass filter.

    Args:
        pass1: (optional) first passband frequency [Hz].
        pass2: (optional) second passband frequency [Hz].
        fs: (optional) sampling frequency [Hz].
        pole: (optional) number of poles for Butterworth filter.
              Order = pole for bandpass, order = pole for low/high pass in SciPy's butter.

    Returns:
        sos: Second-order sections representation of the filter.
    """
    nyquist = fs / 2.0
    order = pole # For scipy.signal.butter, order is the number of poles.

    if (pass1 > 0 and pass1 < nyquist) and (pass2 > 0 and pass2 < nyquist):
        if pass1 >= pass2:
            raise ValueError(f"For bandpass filter, pass1 ({pass1}) must be less than pass2 ({pass2}).")
        # Wn for bandpass is [lowcut, highcut] in terms of fs for recent scipy versions if fs is passed
        # otherwise, it's normalized to Nyquist. Let's stick to normalized for clarity with older scipy.
        Wn_low = pass1 / nyquist
        Wn_high = pass2 / nyquist
        sos = sp_signal.butter(order, [Wn_low, Wn_high], btype='bandpass', analog=False, output='sos')
    elif (pass1 <= 0 or pass1 >= nyquist) and (pass2 > 0 and pass2 < nyquist):
        Wn = pass2 / nyquist
        sos = sp_signal.butter(order, Wn, btype='lowpass', analog=False, output='sos')
    elif (pass1 > 0 and pass1 < nyquist) and (pass2 <= 0 or pass2 >= nyquist):
        Wn = pass1 / nyquist
        sos = sp_signal.butter(order, Wn, btype='highpass', analog=False, output='sos')
    else:
        raise ValueError(f"Passband frequencies pass1={pass1}, pass2={pass2} are invalid for fs={fs}.")
    
    return sos

def bpf_data(data: Union[np.ndarray, pd.DataFrame], sos: Optional[np.ndarray] = None, **kwargs_for_get_bpf) -> np.ndarray:
    """
    Bandpass (or low-pass or high-pass) filter data (vector or columns of matrix/DataFrame)
    using a zero-phase filter (sosfiltfilt).

    Args:
        data: Data vector (1D array) or matrix (2D array) or DataFrame.
              If matrix/DataFrame, filtering is applied column-wise.
        sos: (optional) Precomputed filter SOS coefficients. If None, `get_bpf_sos` is called with `kwargs_for_get_bpf`.
        **kwargs_for_get_bpf: Arguments for `get_bpf_sos` if `sos` is not provided.
                               Example: fs=10.0, pass1=0.1, pass2=0.9, pole=4

    Returns:
        data_f: Filtered data (np.ndarray).
    """
    if sos is None:
        sos = get_bpf_sos(**kwargs_for_get_bpf)

    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy(dtype=float)
    elif isinstance(data, np.ndarray):
        data_np = data.astype(float)
    else:
        raise TypeError("Input data must be a NumPy array or Pandas DataFrame.")

    if data_np.ndim == 1:
        if len(data_np) == 0 or np.std(data_np) <= np.finfo(data_np.dtype).eps: # Handle empty or constant
             return data_np.copy()
        # Check if data length is sufficient for sosfiltfilt
        if len(data_np) <= 3 * (sos.shape[0] * 2): # sosfiltfilt padding requirement
            # Fallback or warning for short data. For now, return as is or apply sosfilt.
            # For simplicity, returning as is if too short for filtfilt.
            # Consider sp_signal.sosfilt for short sequences if needed.
            return data_np.copy() 
        return sp_signal.sosfiltfilt(sos, data_np)
    elif data_np.ndim == 2:
        data_f = np.empty_like(data_np)
        for i in range(data_np.shape[1]):
            col = data_np[:, i]
            if len(col) == 0 or np.std(col) <= np.finfo(col.dtype).eps: # Handle empty or constant
                data_f[:, i] = col
                continue
            if len(col) <= 3 * (sos.shape[0] * 2):
                data_f[:, i] = col # Return as is if too short
                continue
            data_f[:, i] = sp_signal.sosfiltfilt(sos, col)
        return data_f
    else:
        raise ValueError("Input data must be 1D or 2D.")

def bpf_data_inplace(data: Union[np.ndarray, pd.DataFrame], sos: Optional[np.ndarray] = None, **kwargs_for_get_bpf) -> None:
    """
    In-place version of bpf_data. Modifies the input `data`.

    Args:
        data: Data vector (1D array) or matrix (2D array) or DataFrame to be filtered in-place.
        sos: (optional) Precomputed filter SOS coefficients.
        **kwargs_for_get_bpf: Arguments for `get_bpf_sos` if `sos` is not provided.
    """
    filtered_data = bpf_data(data, sos, **kwargs_for_get_bpf)
    if isinstance(data, pd.DataFrame):
        # Check if columns match before assignment
        if data.shape == filtered_data.shape:
            data[:] = filtered_data
        else: # Handle cases where bpf_data might return original due to length
            for i, col_name in enumerate(data.columns):
                if i < filtered_data.shape[1]:
                    data[col_name] = filtered_data[:, i]
    elif isinstance(data, np.ndarray):
        if data.shape == filtered_data.shape:
            data[:] = filtered_data
        # else: data was returned as is by bpf_data, no need to reassign
    else:
        raise TypeError("Input data for in-place filtering must be a NumPy array or Pandas DataFrame.")

def detrend_data(y_data: np.ndarray, x_data: Optional[np.ndarray] = None, 
                 lambda_ridge: float = 0, mean_only: bool = False) -> np.ndarray:
    """
    Detrend signal (remove mean and optionally slope).

    Args:
        y_data: Length-N observed data vector.
        x_data: (optional) Length-N independent variable vector or N x Nf matrix. 
                If None, defaults to np.arange(len(y_data)).
        lambda_ridge: (optional) Ridge parameter for linear regression.
        mean_only: (optional) If True, only remove mean (not slope).

    Returns:
        y_detrended: Detrended data vector.
    """
    y_data = np.asarray(y_data, dtype=float)
    if y_data.ndim != 1:
        raise ValueError("y_data must be a 1D array.")
    if len(y_data) == 0:
        return y_data.copy()

    if mean_only:
        return y_data - np.mean(y_data)
    else:
        if x_data is None:
            x_fit = np.arange(len(y_data), dtype=float)
        else:
            x_fit = np.asarray(x_data, dtype=float)

        if x_fit.ndim == 1:
            if len(x_fit) != len(y_data):
                raise ValueError("If x_data is 1D, it must have the same length as y_data.")
            x_matrix = np.vstack([np.ones(len(y_data)), x_fit]).T
        elif x_fit.ndim == 2:
            if x_fit.shape[0] != len(y_data):
                raise ValueError("If x_data is 2D, its first dimension must match y_data length.")
            x_matrix = np.hstack([np.ones((len(y_data), 1)), x_fit])
        else:
            raise ValueError("x_data must be 1D or 2D.")
            
        coef = linreg_matrix(y_data, x_matrix, lambda_ridge=lambda_ridge)
        y_trend = x_matrix @ coef
        return y_data - y_trend

def sos_freq_response(sos: np.ndarray, worN: Optional[Union[int, np.ndarray]] = None, fs: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency response of a filter given in SOS format.
    Wrapper for scipy.signal.sosfreqz.

    Args:
        sos: Array of second-order filter sections.
        worN: If a number, then compute at worN evenly spaced frequencies around the unit circle.
              If an array, compute the response at frequencies given in worN.
              Frequencies are in radians/sample.
        fs: The sampling frequency of the system.

    Returns:
        w: The frequencies at which h was computed, in the same units as fs. By default, w is normalized to the range [0, pi) (radians/sample).
        h: The frequency response, as complex numbers.
    """
    return sp_signal.sosfreqz(sos, worN=worN, fs=fs)

def filt_data_forward(data: np.ndarray, sos: np.ndarray, zi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter data forward using cascaded second-order sections.
    Wrapper for scipy.signal.sosfilt. This is a causal filter.

    Args:
        data: Input data array.
        sos: Array of second-order filter sections.
        zi: (optional) Initial conditions for the cascaded filter delays.

    Returns:
        filtered_data: The filtered output of the same shape as `data`.
        zf: (optional) If `zi` is None, this is not returned. Otherwise, `zf` contains the final filter delay values.
    """
    if zi is None:
        return sp_signal.sosfilt(sos, data), None # Or handle return based on zi presence
    else:
        return sp_signal.sosfilt(sos, data, zi=zi)


def periodogram_psd(x: np.ndarray, fs: float = 1.0, window: str = 'hann', 
                    nfft: Optional[int] = None, detrend: Union[str, bool] = 'constant', 
                    return_onesided: bool = True, scaling: str = 'density') -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using a periodogram.
    Wrapper for scipy.signal.periodogram.

    Args:
        x: Time series of measurement values.
        fs: Sampling frequency of the `x` time series.
        window: Desired window to use.
        nfft: Length of the FFT used.
        detrend: Specifies how to detrend each segment.
        return_onesided: If True, return a one-sided spectrum for real data.
        scaling: Selects between computing the power spectral density ('density')
                 or the power spectrum ('spectrum').

    Returns:
        f: Array of sample frequencies.
        Pxx: Power spectral density or power spectrum of x.
    """
    return sp_signal.periodogram(x, fs=fs, window=window, nfft=nfft, detrend=detrend, 
                                 return_onesided=return_onesided, scaling=scaling)

def welch_psd(x: np.ndarray, fs: float = 1.0, window: str = 'hann', 
              nperseg: Optional[int] = None, noverlap: Optional[int] = None, 
              nfft: Optional[int] = None, detrend: Union[str, bool] = 'constant', 
              return_onesided: bool = True, scaling: str = 'density', 
              average: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch’s method.
    Wrapper for scipy.signal.welch.

    Args:
        x: Time series of measurement values.
        fs: Sampling frequency of the `x` time series.
        window: Desired window to use.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        nfft: Length of the FFT used, if a zero padded FFT is desired.
        detrend: Specifies how to detrend each segment.
        return_onesided: If True, return a one-sided spectrum for real data.
        scaling: Selects between computing the power spectral density ('density')
                 or the power spectrum ('spectrum').
        average: Method to average periodograms ('mean' or 'median').

    Returns:
        f: Array of sample frequencies.
        Pxx: Power spectral density or power spectrum of x.
    """
    return sp_signal.welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, 
                           nfft=nfft, detrend=detrend, return_onesided=return_onesided, 
                           scaling=scaling, average=average)

def smooth_data(data: np.ndarray, window_len: int = 11, window: str = 'flat') -> np.ndarray:
    """
    Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.

    Args:
        data: The input signal (1D numpy array).
        window_len: The dimension of the smoothing window; should be an odd integer.
        window: The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
                'flat' window will produce a moving average smoothing.

    Returns:
        The smoothed signal.
    
    Reference: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    if data.ndim != 1:
        raise ValueError("smooth_data only accepts 1 dimension arrays.")
    if data.size < window_len:
        # raise ValueError("Input vector needs to be bigger than window size.")
        return data # Return original data if too short to smooth
    if window_len < 3:
        return data

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # Pad signal at the beginning and end to avoid edge effects
    s = np.r_[data[window_len - 1:0:-1], data, data[-2:-window_len - 1:-1]]
    
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    y = np.convolve(w / w.sum(), s, mode='valid')
    
    # Trim to original data length
    # The 'valid' mode convolution already handles part of this.
    # We need to select the central part of y that corresponds to original data.
    # Length of y is len(s) - window_len + 1 = (len(data) + 2*(window_len-1)) - window_len + 1
    # = len(data) + window_len - 1
    # We want a slice of length len(data)
    start_index = (len(y) - len(data)) // 2
    return y[start_index : start_index + len(data)]


def spectral_amplitude(data: np.ndarray, fs: float = 1.0, nfft: Optional[int] = None,
                       return_onesided: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectral amplitude of a signal.

    Args:
        data: Input time-series data (1D numpy array).
        fs: Sampling frequency of the data.
        nfft: (optional) Number of FFT points. If None, uses length of data.
        return_onesided: (optional) If True, returns a one-sided spectrum (for real signals).

    Returns:
        freqs: Array of frequencies.
        amp: Array of spectral amplitudes (magnitude of FFT components).
             Note: This is the raw magnitude. For a calibrated amplitude spectrum,
             further scaling (e.g., by N or 2/N) might be needed depending on convention.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    
    N = len(data)
    if N == 0:
        return np.array([]), np.array([])

    if nfft is None:
        nfft = N
    
    Y = np.fft.fft(data, n=nfft)
    
    if return_onesided:
        # Number of unique points for one-sided spectrum
        num_unique_pts = nfft // 2 + 1
        freqs = np.fft.fftfreq(nfft, d=1/fs)[:num_unique_pts]
        amp = np.abs(Y)[:num_unique_pts]
    else:
        freqs = np.fft.fftfreq(nfft, d=1/fs)
        amp = np.abs(Y)
        # Optionally fftshift if a centered spectrum is desired for two-sided
        # freqs = np.fft.fftshift(freqs)
        # amp = np.fft.fftshift(amp)
        
    return freqs, amp