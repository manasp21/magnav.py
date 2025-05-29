# MagNavPy/src/tolles_lawson.py
# -*- coding: utf-8 -*-
"""
Tolles-Lawson aeromagnetic compensation algorithm, translated from MagNav.jl.
"""

import numpy as np
from typing import List, Tuple, Union, Sequence, Literal, Optional, cast
import logging

# Assuming these are in analysis_util.py or similar as per user instructions
from .analysis_util import get_bpf, bpf_data, linreg
from .magnav import MagV

logger = logging.getLogger(__name__)

# Define valid Tolles-Lawson term strings based on Julia symbols
TL_TermType = Literal[
    "permanent", "p", "permanent3", "p3",
    "induced", "i", "induced6", "i6", "induced5", "i5", "induced3", "i3",
    "eddy", "e", "eddy9", "e9", "eddy8", "e8", "eddy3", "e3",
    "fdm", "f", "fdm3", "f3",
    "bias", "b"
]

DEFAULT_TL_TERMS: Tuple[TL_TermType, ...] = ("permanent", "induced", "eddy")
VALID_FDM_SCHEMES = Literal["backward", "forward", "central", "central2",
                            "backward2", "forward2", "fourth", "central4"]

def fdm(x: np.ndarray, scheme: VALID_FDM_SCHEMES = "central") -> np.ndarray:
    """
    Finite difference method (FDM) applied to x.

    Args:
        x: data vector
        scheme: (optional) finite difference method scheme used
            - "backward":  1st derivative 1st-order backward difference
            - "forward":   1st derivative 1st-order forward  difference
            - "central":   1st derivative 2nd-order central  difference (also "central2")
            - "backward2": 1st derivative 2nd-order backward difference
            - "forward2":  1st derivative 2nd-order forward  difference
            - "fourth":    4th derivative central difference (also "central4")

    Returns:
        Vector of finite differences (length of x)
    """
    N = len(x)
    if N == 0:
        return np.array([], dtype=x.dtype)
    
    x_float = x.astype(float, copy=False) # Ensure float for division

    # Default to zeros, fill if scheme matches and N is sufficient
    dif = np.zeros_like(x_float)

    if scheme == "backward":
        if N > 1:
            dif[0] = x_float[1] - x_float[0] # Julia: dif_1
            if N > 2: # Julia: dif_mid = (x[2:end-1] - x[1:end-2])
                dif[1:N-1] = x_float[1:N-1] - x_float[0:N-2]
            if N > 0: # Ensure index is valid for single element assignment
                 dif[N-1] = x_float[N-1] - x_float[N-2] # Julia: dif_end
        # If N=1, returns zeros_like, matching Julia's else branch
    elif scheme == "forward":
        if N > 1:
            dif[0] = x_float[1] - x_float[0] # Julia: dif_1
            if N > 2: # Julia: dif_mid = (x[3:end] - x[2:end-1])
                dif[1:N-1] = x_float[2:N] - x_float[1:N-1]
            if N > 0:
                dif[N-1] = x_float[N-1] - x_float[N-2] # Julia: dif_end
    elif scheme in ["central", "central2"]:
        if N > 2: # Julia condition N > 2
            dif[0] = x_float[1] - x_float[0] # Julia: dif_1
            # Julia: dif_mid = (x[3:end] - x[1:end-2]) ./ 2
            dif[1:N-1] = (x_float[2:N] - x_float[0:N-2]) / 2.0
            dif[N-1] = x_float[N-1] - x_float[N-2] # Julia: dif_end
        elif N > 0 : # Handle N=1, N=2 for central (approximate as forward/backward or specific)
             # Julia returns zero(x) if N <= 2 for central. So this is fine.
             pass # Returns zeros_like
    elif scheme == "backward2":
        if N > 3: # Julia condition N > 3
            # dif_1   = x[2:3]     - x[1:2]
            dif[0:2] = x_float[1:3] - x_float[0:2]
            # dif_mid = (3*x[3:end-1] - 4*x[2:end-2] + x[1:end-3]  ) ./ 2
            dif[2:N-1] = (3*x_float[2:N-1] - 4*x_float[1:N-2] + x_float[0:N-3]) / 2.0
            # dif_end = (3*x[end]     - 4*x[end-1]   + x[end-2]    ) ./ 2
            dif[N-1] = (3*x_float[N-1] - 4*x_float[N-2] + x_float[N-3]) / 2.0
    elif scheme == "forward2":
        if N > 3: # Julia condition N > 3
            # dif_1   = (-x[3]        + 4*x[2]       - 3*x[1]      ) ./ 2
            dif[0] = (-x_float[2] + 4*x_float[1] - 3*x_float[0]) / 2.0
            # dif_mid = (-x[4:end]    + 4*x[3:end-1] - 3*x[2:end-2]) ./ 2
            dif[1:N-2] = (-x_float[3:N] + 4*x_float[2:N-1] - 3*x_float[1:N-2]) / 2.0
            # dif_end = x[end-1:end] - x[end-2:end-1]
            dif[N-2:N] = x_float[N-2:N] - x_float[N-3:N-1]
    elif scheme in ["fourth", "central4"]:
        if N > 4: # Julia condition N > 4
            # dif_1   = zeros(eltype(x),2)
            dif[0:2] = 0.0
            # dif_mid = ( ... ) ./ 16
            dif[2:N-2] = (  x_float[0:N-4] -
                           4*x_float[1:N-3] +
                           6*x_float[2:N-2] -
                           4*x_float[3:N-1] +
                             x_float[4:N  ] ) / 16.0
            # dif_end = zeros(eltype(x),2)
            dif[N-2:N] = 0.0
    # else: dif remains zeros_like, matching Julia's final else branch.
    return dif

def _create_TL_A_components(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, *,
                            Bt_actual: Optional[np.ndarray],
                            terms: Sequence[str],
                            Bt_scale: float,
                            return_B: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Helper function to create Tolles-Lawson A matrix from components."""
    
    if Bt_actual is None:
        Bt_calc = np.sqrt(Bx**2 + By**2 + Bz**2)
    else:
        Bt_calc = Bt_actual
    
    if isinstance(terms, str):
        terms_set = {terms}
    else:
        terms_set = set(terms)

    Bx_hat = Bx / Bt_calc
    By_hat = By / Bt_calc
    Bz_hat = Bz / Bt_calc

    Bx_dot = fdm(Bx)
    By_dot = fdm(By)
    Bz_dot = fdm(Bz)

    Bx_hat_Bx = Bx_hat * Bx / Bt_scale
    Bx_hat_By = Bx_hat * By / Bt_scale
    Bx_hat_Bz = Bx_hat * Bz / Bt_scale
    By_hat_By = By_hat * By / Bt_scale
    By_hat_Bz = By_hat * Bz / Bt_scale
    Bz_hat_Bz = Bz_hat * Bz / Bt_scale

    Bx_hat_Bx_dot = Bx_hat * Bx_dot / Bt_scale
    Bx_hat_By_dot = Bx_hat * By_dot / Bt_scale
    Bx_hat_Bz_dot = Bx_hat * Bz_dot / Bt_scale
    By_hat_Bx_dot = By_hat * Bx_dot / Bt_scale
    By_hat_By_dot = By_hat * By_dot / Bt_scale
    By_hat_Bz_dot = By_hat * Bz_dot / Bt_scale
    Bz_hat_Bx_dot = Bz_hat * Bx_dot / Bt_scale
    Bz_hat_By_dot = Bz_hat * By_dot / Bt_scale
    Bz_hat_Bz_dot = Bz_hat * Bz_dot / Bt_scale
    
    columns: List[np.ndarray] = []

    if any(s in terms_set for s in ("permanent", "p", "permanent3", "p3")):
        columns.extend([Bx_hat, By_hat, Bz_hat])

    if any(s in terms_set for s in ("induced", "i", "induced6", "i6")):
        columns.extend([Bx_hat_Bx, Bx_hat_By, Bx_hat_Bz, By_hat_By, By_hat_Bz, Bz_hat_Bz])
    if any(s in terms_set for s in ("induced5", "i5")): # Separate if, as in Julia
        columns.extend([Bx_hat_Bx, Bx_hat_By, Bx_hat_Bz, By_hat_By, By_hat_Bz])
    if any(s in terms_set for s in ("induced3", "i3")): # Separate if
        columns.extend([Bx_hat_Bx, By_hat_By, Bz_hat_Bz])

    if any(s in terms_set for s in ("eddy", "e", "eddy9", "e9")):
        columns.extend([Bx_hat_Bx_dot, Bx_hat_By_dot, Bx_hat_Bz_dot,
                        By_hat_Bx_dot, By_hat_By_dot, By_hat_Bz_dot,
                        Bz_hat_Bx_dot, Bz_hat_By_dot, Bz_hat_Bz_dot])
    if any(s in terms_set for s in ("eddy8", "e8")): # Separate if
         columns.extend([Bx_hat_Bx_dot, Bx_hat_By_dot, Bx_hat_Bz_dot,
                         By_hat_Bx_dot, By_hat_By_dot, By_hat_Bz_dot,
                         Bz_hat_Bx_dot, Bz_hat_By_dot])
    if any(s in terms_set for s in ("eddy3", "e3")): # Separate if
        columns.extend([Bx_hat_Bx_dot, By_hat_By_dot, Bz_hat_Bz_dot])

    if any(s in terms_set for s in ("fdm", "f", "fdm3", "f3")):
        columns.extend([Bx_dot, By_dot, Bz_dot])

    if any(s in terms_set for s in ("bias", "b")):
        columns.append(np.ones(len(Bt_calc), dtype=Bt_calc.dtype))

    if not columns:
        raise ValueError(f"Terms {terms} are invalid or result in no columns for the A matrix.")

    A = np.column_stack(columns)

    if return_B:
        B_dot_matrix = np.column_stack([Bx_dot, By_dot, Bz_dot])
        return A, Bt_calc, B_dot_matrix
    else:
        return A

def create_TL_A(flux_or_Bx: Union[MagV, np.ndarray],
                By: Optional[np.ndarray] = None,
                Bz: Optional[np.ndarray] = None,
                ind: Optional[np.ndarray] = None,
                *, 
                Bt: Optional[np.ndarray] = None,
                terms: Sequence[TL_TermType] = DEFAULT_TL_TERMS,
                Bt_scale: float = 50000.0,
                return_B: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create Tolles-Lawson `A` matrix. See _create_TL_A_components for details.
    Handles MagV input or Bx,By,Bz components.
    """
    _Bx_comp: np.ndarray
    _By_comp: np.ndarray
    _Bz_comp: np.ndarray
    _Bt_for_helper: Optional[np.ndarray] = Bt 

    if isinstance(flux_or_Bx, MagV):
        flux = cast(MagV, flux_or_Bx)
        _ind_actual = ind if ind is not None else np.ones(len(flux.x), dtype=bool)
        
        _Bx_comp, _By_comp, _Bz_comp = flux.x[_ind_actual], flux.y[_ind_actual], flux.z[_ind_actual]
        
        if Bt is not None:
            if len(Bt) != len(flux.x[_ind_actual]):
                 _Bt_for_helper = Bt[_ind_actual]
    elif isinstance(flux_or_Bx, np.ndarray):
        if By is None or Bz is None:
            raise ValueError("If first argument is Bx array, By and Bz must also be provided.")
        _Bx_comp, _By_comp, _Bz_comp = flux_or_Bx, By, Bz
    else:
        raise TypeError("First argument must be a MagV object or a NumPy array for Bx.")

    current_terms_str = cast(Sequence[str], terms)
    return _create_TL_A_components(_Bx_comp, _By_comp, _Bz_comp,
                                   Bt_actual=_Bt_for_helper,
                                   terms=current_terms_str,
                                   Bt_scale=Bt_scale,
                                   return_B=return_B)

def _create_TL_coef_components(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, B_scalar: np.ndarray, *,
                               Bt_actual: Optional[np.ndarray],
                               lambda_val: float,
                               terms: Sequence[str],
                               pass1: float,
                               pass2: float,
                               fs: float,
                               pole: int,
                               trim: int,
                               Bt_scale: float,
                               return_var: bool) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Helper function to create Tolles-Lawson coefficients from components."""
    perform_filter = False
    bpf_coeffs = None 
    if ((pass1 > 0) and (pass1 < fs / 2.0)) or \
       ((pass2 > 0) and (pass2 < fs / 2.0)):
        perform_filter = True
        bpf_coeffs = get_bpf(pass1=pass1, pass2=pass2, fs=fs, pole=pole)
    else:
        logger.info("Not filtering (or trimming) Tolles-Lawson data as pass frequencies are out of range.")

    A_unfiltered = _create_TL_A_components(Bx, By, Bz,
                                Bt_actual=Bt_actual,
                                terms=terms,
                                Bt_scale=Bt_scale,
                                return_B=False)
    A_unfiltered = cast(np.ndarray, A_unfiltered)
    
    A_to_use = A_unfiltered.copy()
    B_to_use = B_scalar.copy()

    if perform_filter and bpf_coeffs is not None:
        A_filt = bpf_data(A_to_use, bpf=bpf_coeffs) # Assuming bpf_data handles 2D A
        B_filt = bpf_data(B_to_use, bpf=bpf_coeffs) 
        
        if trim > 0:
            if len(A_filt) > 2 * trim and len(B_filt) > 2 * trim :
                A_to_use = A_filt[trim:-trim, :]
                B_to_use = B_filt[trim:-trim]
            else:
                logger.warning(f"Cannot trim {trim} samples, data too short. Using untrimmed filtered data.")
                A_to_use = A_filt
                B_to_use = B_filt
        else: # trim is 0
            A_to_use = A_filt
            B_to_use = B_filt
    
    coef = linreg(B_to_use, A_to_use, lambda_val=lambda_val).flatten()

    if return_var:
        B_comp_error = B_to_use - (A_to_use @ coef)
        B_var_val = np.var(B_comp_error)
        logger.info(f"TL fit error variance: {B_var_val}")
        return coef, B_var_val
    else:
        return coef

def create_TL_coef(flux_or_Bx: Union[MagV, np.ndarray],
                   arg2: np.ndarray, 
                   arg3: Optional[np.ndarray] = None,
                   B_scalar_for_bx_case: Optional[np.ndarray] = None,
                   *, 
                   Bt: Optional[np.ndarray] = None,
                   lambda_val: float = 0.0,
                   terms: Sequence[TL_TermType] = DEFAULT_TL_TERMS,
                   pass1: float = 0.1,
                   pass2: float = 0.9,
                   fs: float = 10.0,
                   pole: int = 4,
                   trim: int = 20,
                   Bt_scale: float = 50000.0,
                   return_var: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Create Tolles-Lawson coefficients. See _create_TL_coef_components for details.
    Handles MagV input or Bx,By,Bz components.

    Args:
        flux_or_Bx: MagV object or Bx (x-component) as np.ndarray.
        arg2: If flux_or_Bx is MagV, this is B_scalar (scalar measurements).
              If flux_or_Bx is Bx (array), this is By (y-component).
        arg3: If flux_or_Bx is MagV, this is `ind` (selected indices, optional).
              If flux_or_Bx is Bx (array), this is Bz (z-component).
        B_scalar_for_bx_case: Scalar measurements [nT]. Required if using Bx,By,Bz components.
        (Keyword arguments follow)
    """
    _Bx_comp: np.ndarray
    _By_comp: np.ndarray
    _Bz_comp: np.ndarray
    _B_scalar_comp: np.ndarray
    _Bt_for_helper: Optional[np.ndarray] = Bt
    current_terms_str = cast(Sequence[str], terms)

    if isinstance(flux_or_Bx, MagV):
        flux = cast(MagV, flux_or_Bx)
        _B_scalar_comp = arg2 
        _ind_actual = arg3 if (arg3 is not None and arg3.dtype == bool) else np.ones(len(flux.x), dtype=bool)
        
        _Bx_comp, _By_comp, _Bz_comp = flux.x[_ind_actual], flux.y[_ind_actual], flux.z[_ind_actual]
        _B_scalar_comp = _B_scalar_comp[_ind_actual]

        if Bt is not None:
            if len(Bt) != len(flux.x[_ind_actual]):
                _Bt_for_helper = Bt[_ind_actual]
    elif isinstance(flux_or_Bx, np.ndarray):
        _Bx_comp = flux_or_Bx
        _By_comp = arg2 
        if arg3 is None: raise ValueError("Bz (arg3) must be provided if flux_or_Bx is Bx array.")
        _Bz_comp = arg3 
        if B_scalar_for_bx_case is None:
            raise ValueError("B_scalar_for_bx_case must be provided if flux_or_Bx is Bx array.")
        _B_scalar_comp = B_scalar_for_bx_case
    else:
        raise TypeError("First argument must be a MagV object or a NumPy array for Bx.")

    return _create_TL_coef_components(_Bx_comp, _By_comp, _Bz_comp, _B_scalar_comp,
                                      Bt_actual=_Bt_for_helper,
                                      lambda_val=lambda_val,
                                      terms=current_terms_str,
                                      pass1=pass1, pass2=pass2, fs=fs,
                                      pole=pole, trim=trim, Bt_scale=Bt_scale,
                                      return_var=return_var)

def get_TL_term_ind(term_to_find: str,
                    current_terms: Sequence[str]) -> np.ndarray:
    """
    Finds indices corresponding to `term_to_find` in a TL coefficient vector
    generated using `current_terms`.
    """
    terms_list = list(current_terms) if not isinstance(current_terms, str) else [current_terms]

    x_dummy = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) # Min length for all fdm schemes
    bt_dummy = np.sqrt(3 * x_dummy**2) # Dummy Bt

    A_total_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                          Bt_actual=bt_dummy, terms=terms_list,
                                          Bt_scale=50000.0, return_B=False)
    N_total_cols = cast(np.ndarray, A_total_obj).shape[1]
    
    N_cols_for_term_to_find = 0
    try:
        A_for_term_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                                 Bt_actual=bt_dummy, terms=[term_to_find],
                                                 Bt_scale=50000.0, return_B=False)
        N_cols_for_term_to_find = cast(np.ndarray, A_for_term_obj).shape[1]
    except ValueError: # term_to_find alone might be invalid
        pass # N_cols_for_term_to_find remains 0

    offset_cols = 0
    # Iterate through terms_list to find the first occurrence of term_to_find or its category
    # This logic needs to correctly identify the block of columns.
    # The Julia code `findfirst(term .== terms)` is simpler if `term` is a main category
    # and `terms` is a list of main categories.
    # This Python version assumes term_to_find is a string that would activate a block in _create_TL_A_components.
    
    # Simplified: find the first term in terms_list that *activates* term_to_find's block
    # This is still complex. The original Julia `get_TL_term_ind` is likely used with
    # `term` being one of the main categories like `:permanent` and `terms` being the
    # list of main categories used for the full A matrix.
    
    # Replicating the Julia structure more directly:
    # Assume term_to_find is a primary category.
    # Find its position in current_terms (if current_terms contains primary categories).
    
    # This function is hard to translate perfectly without knowing its exact usage context
    # regarding aliases vs primary terms in `term_to_find` and `current_terms`.
    # The provided Julia code: `i_term  = findfirst(term .== terms)`
    # This implies `term` (e.g. `:permanent`) must be literally in `terms`.
    
    idx_in_list = -1
    try:
        idx_in_list = terms_list.index(term_to_find) # term_to_find must be literally in terms_list
    except ValueError:
         # If term_to_find is not literally in terms_list, the Julia logic would fail.
         # Return empty or raise error, as per Julia's implicit behavior or assert.
        logger.warning(f"Term '{term_to_find}' not directly in current_terms: {terms_list}. "
                       "get_TL_term_ind might not work as expected.")
        return np.zeros(N_total_cols, dtype=bool)


    terms_before_it = terms_list[:idx_in_list]
    if terms_before_it:
        A_before_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                               Bt_actual=bt_dummy, terms=terms_before_it,
                                               Bt_scale=50000.0, return_B=False)
        offset_cols = cast(np.ndarray, A_before_obj).shape[1]
    
    # N_cols_for_term_to_find is how many columns `term_to_find` *itself* generates.
    # This was calculated earlier.

    bool_indices = np.zeros(N_total_cols, dtype=bool)
    if N_cols_for_term_to_find > 0:
        start_idx = offset_cols
        end_idx = offset_cols + N_cols_for_term_to_find
        if end_idx <= N_total_cols: # Ensure indices are within bounds
            bool_indices[start_idx:end_idx] = True
        else:
            logger.warning(f"Calculated indices for '{term_to_find}' [{start_idx}:{end_idx}] "
                           f"exceed total columns {N_total_cols}.")
            # Partial fill if possible
            valid_end_idx = min(end_idx, N_total_cols)
            if start_idx < valid_end_idx:
                 bool_indices[start_idx:valid_end_idx] = True


    return bool_indices