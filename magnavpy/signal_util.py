import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Union, Optional

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
        pole: (optional) number of poles for Butterworth filter (order = pole for bandpass, pole/2 for low/high).

    Returns:
        sos: Second-order sections representation of the filter.
    """
    nyquist = fs / 2.0
    order = pole

    if (pass1 > 0 and pass1 < nyquist) and (pass2 > 0 and pass2 < nyquist):
        if pass1 >= pass2:
            raise ValueError(f"For bandpass filter, pass1 ({pass1}) must be less than pass2 ({pass2}).")
        Wn = [pass1 / nyquist, pass2 / nyquist]
        btype = 'bandpass'
        actual_order = order
    elif (pass1 <= 0 or pass1 >= nyquist) and (pass2 > 0 and pass2 < nyquist):
        Wn = pass2 / nyquist
        btype = 'lowpass'
        actual_order = order
    elif (pass1 > 0 and pass1 < nyquist) and (pass2 <= 0 or pass2 >= nyquist):
        Wn = pass1 / nyquist
        btype = 'highpass'
        actual_order = order
    else:
        raise ValueError(f"Passband frequencies pass1={pass1}, pass2={pass2} are invalid for fs={fs}.")

    sos = sp_signal.butter(actual_order, Wn, btype=btype, analog=False, output='sos', fs=fs)
    return sos

def bpf_data(data: Union[np.ndarray, pd.DataFrame], sos: Optional[np.ndarray] = None, **kwargs_for_get_bpf) -> np.ndarray:
    """
    Bandpass (or low-pass or high-pass) filter data (vector or columns of matrix/DataFrame).

    Args:
        data: Data vector (1D array) or matrix (2D array) or DataFrame.
              If matrix/DataFrame, filtering is applied column-wise.
        sos: (optional) Precomputed filter SOS coefficients. If None, `get_bpf_sos` is called with `kwargs_for_get_bpf`.
        **kwargs_for_get_bpf: Arguments for `get_bpf_sos` if `sos` is not provided.

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
        if np.std(data_np) <= np.finfo(data_np.dtype).eps:
            return data_np.copy()
        return sp_signal.sosfiltfilt(sos, data_np)
    elif data_np.ndim == 2:
        data_f = np.empty_like(data_np)
        for i in range(data_np.shape[1]):
            col = data_np[:, i]
            if np.std(col) <= np.finfo(col.dtype).eps: # Check if column is constant
                data_f[:, i] = col
            else:
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
        data[:] = filtered_data
    elif isinstance(data, np.ndarray):
        data[:] = filtered_data
    else:
        raise TypeError("Input data for in-place filtering must be a NumPy array or Pandas DataFrame.")