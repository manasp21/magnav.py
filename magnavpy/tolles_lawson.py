# MagNavPy/src/tolles_lawson.py
# -*- coding: utf-8 -*-
"""
Tolles-Lawson aeromagnetic compensation algorithm, translated from MagNav.jl.
"""

import numpy as np
from typing import List, Tuple, Union, Sequence, Literal, Optional, cast
import logging

# Assuming these are in analysis_util.py or similar as per user instructions
from magnavpy.signal_util import get_bpf_sos, bpf_data, linreg_matrix
from magnavpy.common_types import MagV

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
            # dif[N-1] needs to be handled carefully if N=1 or N=2
            if N > 1: # Ensure index is valid for single element assignment
                 dif[N-1] = x_float[N-1] - x_float[N-2] # Julia: dif_end
        # If N=1, returns zeros_like, matching Julia's else branch
    elif scheme == "forward":
        if N > 1:
            dif[0] = x_float[1] - x_float[0] # Julia: dif_1
            if N > 2: # Julia: dif_mid = (x[3:end] - x[2:end-1])
                dif[1:N-1] = x_float[2:N] - x_float[1:N-1]
            if N > 1:
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
                             x_float[4:N  ] ) / 16.0 # Julia uses 16, not dx^4. Assuming dt=1.
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
    
    # Handle potential zero Bt_calc values to avoid division by zero
    # Replace zeros with a small epsilon or handle as appropriate for the physics
    # For now, we assume Bt_calc will not contain zeros where division occurs.
    # A robust solution might involve np.where(Bt_calc == 0, epsilon, Bt_calc)
    # or specific handling if Bx, By, Bz are all zero.
    # If Bt_calc can be zero, Bx_hat etc. become NaN or Inf.
    # The original Julia code doesn't explicitly handle Bt == 0.
    # We'll add a small epsilon to prevent division by zero if Bt_calc is zero.
    epsilon = np.finfo(float).eps
    Bt_calc_safe = np.where(Bt_calc == 0, epsilon, Bt_calc)

    if isinstance(terms, str):
        terms_set = {terms}
    else:
        terms_set = set(terms)

    Bx_hat = Bx / Bt_calc_safe
    By_hat = By / Bt_calc_safe
    Bz_hat = Bz / Bt_calc_safe

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
        columns.append(np.ones(len(Bt_calc_safe), dtype=Bt_calc_safe.dtype))

    if not columns:
        raise ValueError(f"Terms {terms} are invalid or result in no columns for the A matrix.")

    A = np.column_stack(columns)

    if return_B:
        B_dot_matrix = np.column_stack([Bx_dot, By_dot, Bz_dot])
        return A, Bt_calc, B_dot_matrix # Return original Bt_calc, not Bt_calc_safe
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

    Args:
        flux_or_Bx: MagV object or Bx (x-component) as np.ndarray.
        By: If flux_or_Bx is Bx array, this is By (y-component).
        Bz: If flux_or_Bx is Bx array, this is Bz (z-component).
        ind: If flux_or_Bx is MagV, this is `ind` (selected indices, optional boolean array).
        Bt: (optional) Magnitude of vector magnetometer measurements or scalar
            magnetometer measurements for modified Tolles-Lawson [nT].
        terms: (optional) Tolles-Lawson terms to use.
        Bt_scale: (optional) Scaling factor for induced & eddy current terms [nT].
        return_B: (optional) If true, also return `Bt_calc` & `B_dot_matrix`.

    Returns:
        Tolles-Lawson `A` matrix, and optionally `Bt_calc` and `B_dot_matrix`.
    """
    _Bx_comp: np.ndarray
    _By_comp: np.ndarray
    _Bz_comp: np.ndarray
    _Bt_for_helper: Optional[np.ndarray] = Bt 
    _ind_actual: Optional[np.ndarray] = ind

    if isinstance(flux_or_Bx, MagV):
        flux = cast(MagV, flux_or_Bx)
        if _ind_actual is not None:
            if not (_ind_actual.dtype == bool and len(_ind_actual) == len(flux.x)):
                raise ValueError("`ind` must be a boolean array of the same length as MagV data.")
        else:
            _ind_actual = np.ones(len(flux.x), dtype=bool)
        
        _Bx_comp, _By_comp, _Bz_comp = flux.x[_ind_actual], flux.y[_ind_actual], flux.z[_ind_actual]
        
        if _Bt_for_helper is not None:
            if len(_Bt_for_helper) != len(flux.x): # Original length before indexing
                raise ValueError("Provided `Bt` must have the same original length as MagV data.")
            _Bt_for_helper = _Bt_for_helper[_ind_actual]

    elif isinstance(flux_or_Bx, np.ndarray):
        if By is None or Bz is None:
            raise ValueError("If first argument is Bx array, By and Bz must also be provided.")
        _Bx_comp, _By_comp, _Bz_comp = flux_or_Bx, By, Bz
        if _ind_actual is not None:
            logger.warning("`ind` argument is ignored when Bx, By, Bz arrays are provided directly.")
        # If Bt is provided, it should match the length of Bx, By, Bz
        if _Bt_for_helper is not None and len(_Bt_for_helper) != len(_Bx_comp):
            raise ValueError("Provided `Bt` must have the same length as Bx, By, Bz arrays.")
    else:
        raise TypeError("First argument must be a MagV object or a NumPy array for Bx.")

    current_terms_str = cast(Sequence[str], terms) # TL_TermType is a subset of str
    return _create_TL_A_components(_Bx_comp, _By_comp, _Bz_comp,
                                   Bt_actual=_Bt_for_helper,
                                   terms=current_terms_str,
                                   Bt_scale=Bt_scale,
                                   return_B=return_B)

def create_TL_A_modified_2(flux, ind,
                     Bt       = None,
                     terms    = None,
                     Bt_scale = 50000,
                     return_B = False):
    return create_TL_A_modified_1(flux.x[ind],flux.y[ind],flux.z[ind], Bt=Bt,terms=terms,Bt_scale=Bt_scale,return_B=return_B)

def create_TL_A_modified_1(Bx, By, Bz, 
                Bt=None,
                terms=None,
                Bt_scale=50000,
                return_B=False):
    
    # Default parameter handling
    if Bt is None:
        Bt = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    if terms is None:
        terms = ['permanent', 'induced', 'eddy']
    
    # Ensure terms is a list
    if not isinstance(terms, list):
        terms = [terms]

    Bx_hat = Bx / Bt
    By_hat = By / Bt
    Bz_hat = Bz / Bt

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

    # note: original (slightly incorrect) eddy current terms
    # Bx_hat_Bx_dot = Bx_hat * fdm(Bx_hat) * Bt / Bt_scale
    # Bx_hat_By_dot = Bx_hat * fdm(By_hat) * Bt / Bt_scale
    # Bx_hat_Bz_dot = Bx_hat * fdm(Bz_hat) * Bt / Bt_scale
    # By_hat_Bx_dot = By_hat * fdm(Bx_hat) * Bt / Bt_scale
    # By_hat_By_dot = By_hat * fdm(By_hat) * Bt / Bt_scale
    # By_hat_Bz_dot = By_hat * fdm(Bz_hat) * Bt / Bt_scale
    # Bz_hat_Bx_dot = Bz_hat * fdm(Bx_hat) * Bt / Bt_scale
    # Bz_hat_By_dot = Bz_hat * fdm(By_hat) * Bt / Bt_scale
    # Bz_hat_Bz_dot = Bz_hat * fdm(Bz_hat) * Bt / Bt_scale

    A = np.empty((len(Bt), 0), dtype=Bt.dtype)
    # print("A matrix=", A)
    
    # add (3) permanent field terms - all
    if any(term in terms for term in ['permanent', 'p', 'permanent3', 'p3']):
        A = np.column_stack([A, Bx_hat, By_hat, Bz_hat]) if A.size > 0 else np.column_stack([Bx_hat, By_hat, Bz_hat])

    # add (6) induced field terms - all
    if any(term in terms for term in ['induced', 'i', 'induced6', 'i6']):
        new_cols = np.column_stack([Bx_hat_Bx, Bx_hat_By, Bx_hat_Bz, By_hat_By, By_hat_Bz, Bz_hat_Bz])
        A = np.column_stack([A, new_cols]) if A.size > 0 else new_cols

    # add (5) induced field terms - all except Bz_hat_Bz
    if any(term in terms for term in ['induced5', 'i5']):
        new_cols = np.column_stack([Bx_hat_Bx, Bx_hat_By, Bx_hat_Bz, By_hat_By, By_hat_Bz])
        A = np.column_stack([A, new_cols]) if A.size > 0 else new_cols

    # add (3) induced field terms - Bx_hat_Bx, By_hat_By, Bz_hat_Bz
    if any(term in terms for term in ['induced3', 'i3']):
        new_cols = np.column_stack([Bx_hat_Bx, By_hat_By, Bz_hat_Bz])
        A = np.column_stack([A, new_cols]) if A.size > 0 else new_cols

    # add (9) eddy current terms - all
    if any(term in terms for term in ['eddy', 'e', 'eddy9', 'e9']):
        new_cols1 = np.column_stack([Bx_hat_Bx_dot, Bx_hat_By_dot, Bx_hat_Bz_dot])
        A = np.column_stack([A, new_cols1]) if A.size > 0 else new_cols1
        new_cols2 = np.column_stack([By_hat_Bx_dot, By_hat_By_dot, By_hat_Bz_dot])
        A = np.column_stack([A, new_cols2])
        new_cols3 = np.column_stack([Bz_hat_Bx_dot, Bz_hat_By_dot, Bz_hat_Bz_dot])
        A = np.column_stack([A, new_cols3])

    # add (8) eddy current terms - all except Bz_hat_Bz_dot
    if any(term in terms for term in ['eddy8', 'e8']):
        new_cols1 = np.column_stack([Bx_hat_Bx_dot, Bx_hat_By_dot, Bx_hat_Bz_dot])
        A = np.column_stack([A, new_cols1]) if A.size > 0 else new_cols1
        new_cols2 = np.column_stack([By_hat_Bx_dot, By_hat_By_dot, By_hat_Bz_dot])
        A = np.column_stack([A, new_cols2])
        new_cols3 = np.column_stack([Bz_hat_Bx_dot, Bz_hat_By_dot])
        A = np.column_stack([A, new_cols3])

    # add (3) eddy current terms - Bx_hat_Bx_dot, By_hat_By_dot, Bz_hat_Bz_dot
    if any(term in terms for term in ['eddy3', 'e3']):
        new_cols = np.column_stack([Bx_hat_Bx_dot, By_hat_By_dot, Bz_hat_Bz_dot])
        A = np.column_stack([A, new_cols]) if A.size > 0 else new_cols

    # add (3) derivative terms - Bx_dot, By_dot, Bz_dot
    if any(term in terms for term in ['fdm', 'f', 'fdm3', 'f3']):
        new_cols = np.column_stack([Bx_dot, By_dot, Bz_dot])
        A = np.column_stack([A, new_cols]) if A.size > 0 else new_cols

    # add (1) bias term
    if any(term in terms for term in ['bias', 'b']):
        bias_col = np.ones(len(Bt), dtype=Bt.dtype)
        A = np.column_stack([A, bias_col]) if A.size > 0 else bias_col.reshape(-1, 1)
    
    A = A[0]
    if np.all(A == 0) and A.size == 0:
        raise ValueError(f"{terms} terms are invalid")

    if return_B:
        B_dot = np.column_stack([Bx_dot, By_dot, Bz_dot])
        return (A, Bt, B_dot)
    else:
        return A
    
def create_TL_A_modified(Bx, By, Bz, add_induced=True, add_eddy=True, Bt_scale=50000):
    """
    Create Tolles-Lawson A matrix using vector magnetometer measurements.

    Arguments:
    - `Bx, By, Bz` : vector magnetometer measurements
    - `add_induced, add_eddy` : (optional) add induced and/or eddy terms to Tolles-Lawson A matrix.
    - `Bt_scale` : (optional) scaling factor for induced and eddy current terms

    Returns:
    - `A` : Tolles-Lawson A matrix
    """
    Bt = np.sqrt(Bx**2 + By**2 + Bz**2)
    s  = Bt / Bt_scale # scale
    cosX, cosY, cosZ = Bx/Bt, By/Bt, Bz/Bt
    cosX_dot = np.gradient(cosX)
    cosY_dot = np.gradient(cosY)
    cosZ_dot = np.gradient(cosZ)        

    # (3) permanent moment
    A = np.column_stack((cosX, cosY, cosZ))

    # (6) induced moment
    if add_induced:
        A_ind = np.column_stack((s*cosX*cosX,
                                 s*cosX*cosY,
                                 s*cosX*cosZ,
                                 s*cosY*cosY,
                                 s*cosY*cosZ,
                                 s*cosZ*cosZ))
        A = np.column_stack((A, A_ind))

    # (9) eddy current
    if add_eddy:        
        A_edd = np.column_stack((s*cosX*cosX_dot,
                                 s*cosX*cosY_dot,
                                 s*cosX*cosZ_dot,
                                 s*cosY*cosX_dot,
                                 s*cosY*cosY_dot,
                                 s*cosY*cosZ_dot,
                                 s*cosZ*cosX_dot,
                                 s*cosZ*cosY_dot,
                                 s*cosZ*cosZ_dot))
        A = np.column_stack((A, A_edd))

    return A
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
        bpf_coeffs = get_bpf_sos(pass1=pass1, pass2=pass2, fs=fs, pole=pole)
    else:
        logger.info("Not filtering (or trimming) Tolles-Lawson data as pass frequencies are out of range or zero.")

    A_unfiltered_obj = _create_TL_A_components(Bx, By, Bz,
                                Bt_actual=Bt_actual,
                                terms=terms,
                                Bt_scale=Bt_scale,
                                return_B=False) # We only need A matrix here
    A_unfiltered = cast(np.ndarray, A_unfiltered_obj)
    
    A_to_use = A_unfiltered.copy()
    B_to_use = B_scalar.copy()

    if perform_filter and bpf_coeffs is not None:
        # Ensure A_to_use is 2D for bpf_data if it expects that
        if A_to_use.ndim == 1: A_to_use = A_to_use[:, np.newaxis]
        A_filt = bpf_data(A_to_use, sos=bpf_coeffs, axis=0) # Apply filter along time axis (axis 0)
        B_filt = bpf_data(B_to_use, sos=bpf_coeffs) # Assuming B_to_use is 1D
        
        if trim > 0:
            if A_filt.shape[0] > 2 * trim and len(B_filt) > 2 * trim :
                A_to_use = A_filt[trim:-trim, :]
                B_to_use = B_filt[trim:-trim]
            else:
                logger.warning(f"Cannot trim {trim} samples, data too short ({A_filt.shape[0]} samples). Using untrimmed filtered data.")
                A_to_use = A_filt
                B_to_use = B_filt
        else: # trim is 0 or negative (no trimming)
            A_to_use = A_filt
            B_to_use = B_filt
    
    coef = linreg_matrix(B_to_use, A_to_use, lambda_ridge=lambda_val).flatten()

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
    Create Tolles-Lawson coefficients.

    Args:
        flux_or_Bx: MagV object or Bx (x-component) as np.ndarray.
        arg2: If flux_or_Bx is MagV, this is B_scalar (scalar measurements).
              If flux_or_Bx is Bx (array), this is By (y-component).
        arg3: If flux_or_Bx is MagV, this is `ind` (selected indices, optional boolean array).
              If flux_or_Bx is Bx (array), this is Bz (z-component).
        B_scalar_for_bx_case: Scalar measurements [nT]. Required if using Bx,By,Bz components.
        Bt: (optional) Magnitude of vector magnetometer measurements or scalar
            magnetometer measurements for modified Tolles-Lawson [nT].
        lambda_val: (optional) Ridge parameter for linear regression.
        terms: (optional) Tolles-Lawson terms to use.
        pass1: (optional) First passband frequency for filtering [Hz].
        pass2: (optional) Second passband frequency for filtering [Hz].
        fs: (optional) Sampling frequency [Hz].
        pole: (optional) Number of poles for Butterworth filter.
        trim: (optional) Number of elements to trim from each end after filtering.
        Bt_scale: (optional) Scaling factor for induced & eddy current terms [nT].
        return_var: (optional) If true, also return fit error variance.

    Returns:
        Tolles-Lawson coefficients, and optionally fit error variance.
    """
    _Bx_comp: np.ndarray
    _By_comp: np.ndarray
    _Bz_comp: np.ndarray
    _B_scalar_comp: np.ndarray
    _Bt_for_helper: Optional[np.ndarray] = Bt
    current_terms_str = cast(Sequence[str], terms)

    if isinstance(flux_or_Bx, MagV):
        flux = cast(MagV, flux_or_Bx)
        _B_scalar_comp_all = arg2 # This is B_scalar before indexing
        _ind_actual = arg3 if (arg3 is not None and arg3.dtype == bool and len(arg3) == len(flux.x)) else np.ones(len(flux.x), dtype=bool)
        
        _Bx_comp, _By_comp, _Bz_comp = flux.x[_ind_actual], flux.y[_ind_actual], flux.z[_ind_actual]
        
        if len(_B_scalar_comp_all) != len(flux.x):
            raise ValueError("`B_scalar` (arg2) must have the same length as MagV data before indexing.")
        _B_scalar_comp = _B_scalar_comp_all[_ind_actual]

        if _Bt_for_helper is not None:
            if len(_Bt_for_helper) != len(flux.x): # Original length
                 raise ValueError("Provided `Bt` must have the same original length as MagV data.")
            _Bt_for_helper = _Bt_for_helper[_ind_actual]
            
    elif isinstance(flux_or_Bx, np.ndarray):
        _Bx_comp = flux_or_Bx
        _By_comp = arg2 
        if arg3 is None: raise ValueError("Bz (arg3) must be provided if flux_or_Bx is Bx array.")
        _Bz_comp = arg3 
        if B_scalar_for_bx_case is None:
            raise ValueError("B_scalar_for_bx_case must be provided if flux_or_Bx is Bx array.")
        _B_scalar_comp = B_scalar_for_bx_case
        
        # Length checks for array inputs
        if not (len(_Bx_comp) == len(_By_comp) == len(_Bz_comp) == len(_B_scalar_comp)):
            raise ValueError("Bx, By, Bz, and B_scalar_for_bx_case must all have the same length.")
        if _Bt_for_helper is not None and len(_Bt_for_helper) != len(_Bx_comp):
            raise ValueError("Provided `Bt` must have the same length as Bx, By, Bz arrays.")
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
    Finds boolean indices corresponding to `term_to_find` in a Tolles-Lawson
    coefficient vector that would be generated using `current_terms`.

    Args:
        term_to_find: The specific Tolles-Lawson term (e.g., "permanent", "induced3")
                      whose column indices are sought.
        current_terms: A sequence of Tolles-Lawson terms that define the full
                       set of columns in the 'A' matrix (and thus the coefficient vector).

    Returns:
        A boolean numpy array indicating the positions of `term_to_find`'s
        coefficients within the full coefficient vector.
    """
    # Use a minimal dummy dataset that satisfies fdm's length requirements for all schemes
    # Length 5 is needed for 'fourth'/'central4' scheme in fdm.
    x_dummy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    # Create a dummy Bt that is non-zero to avoid division by zero issues.
    # Bt_dummy must have the same length as x_dummy.
    bt_dummy = np.sqrt(x_dummy**2 + x_dummy**2 + x_dummy**2) # Example: Bx=By=Bz=x_dummy

    # Get the total number of columns for all current_terms
    A_total_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                          Bt_actual=bt_dummy, terms=list(current_terms),
                                          Bt_scale=50000.0, return_B=False)
    N_total_cols = cast(np.ndarray, A_total_obj).shape[1]
    
    N_cols_for_term_to_find = 0
    try:
        # Get the number of columns generated by term_to_find alone
        A_for_term_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                                 Bt_actual=bt_dummy, terms=[term_to_find],
                                                 Bt_scale=50000.0, return_B=False)
        N_cols_for_term_to_find = cast(np.ndarray, A_for_term_obj).shape[1]
    except ValueError:
        # This can happen if term_to_find is not a valid term string on its own
        logger.warning(f"Term '{term_to_find}' might be invalid or not produce columns alone.")
        # N_cols_for_term_to_find remains 0, leading to empty indices.
        pass 

    offset_cols = 0
    # Find the first occurrence of term_to_find or a term that activates the same block
    # This requires careful handling of aliases (e.g., "p" for "permanent").
    # The most straightforward way is to iterate through current_terms and sum up
    # columns from terms *before* the one that matches/includes term_to_find.
    
    # Simplified approach: find the first term in current_terms that IS term_to_find
    # or an alias that would trigger the same block as term_to_find.
    # This is complex due to aliases. The Julia version `findfirst(term .== terms)`
    # implies `term` is a primary category and `terms` is a list of primary categories.
    # Python's `TL_TermType` includes aliases.
    
    # Iterate through current_terms to find the starting position
    found_term_block = False
    for i, term_in_list in enumerate(current_terms):
        # Check if term_in_list is term_to_find or an alias that activates the same block
        # This check needs to be robust to aliases.
        # For simplicity, we assume term_to_find is one of the primary categories
        # or a specific alias that uniquely identifies a block.
        
        # A practical way: if term_to_find is "permanent", it matches "p", "permanent3" etc.
        # This requires mapping term_to_find to its canonical block.
        # For now, let's assume term_to_find is one of the keys in the if-conditions
        # of _create_TL_A_components.
        
        # If term_in_list activates the block corresponding to term_to_find
        # This is tricky. Let's use the Julia approach: find term_to_find *literally*
        # in current_terms, or a primary term it maps to.
        
        # The most direct translation of Julia's `findfirst(term .== terms)`
        # is to find `term_to_find` literally in `current_terms`.
        
        temp_A_before_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                                    Bt_actual=bt_dummy, terms=[term_in_list],
                                                    Bt_scale=50000.0, return_B=False)
        cols_for_this_term_in_list = cast(np.ndarray, temp_A_before_obj).shape[1]

        # This logic is to find the *first* term in current_terms that term_to_find belongs to.
        # Example: if term_to_find="p" and current_terms=["permanent", "induced"]
        # "permanent" is the block "p" belongs to.
        # This requires a mapping from term_to_find to its canonical block.
        # For now, we assume term_to_find is a canonical term or a unique alias.
        
        # A simpler interpretation: find the offset by summing columns of terms *before*
        # the first term in `current_terms` that is `term_to_find` or an alias for the same block.
        # The current Python code tries to find `term_to_find` literally.
        
        is_matching_block = False
        if term_to_find in ("permanent", "p", "permanent3", "p3") and \
           term_in_list in ("permanent", "p", "permanent3", "p3"): is_matching_block = True
        elif term_to_find in ("induced", "i", "induced6", "i6") and \
             term_in_list in ("induced", "i", "induced6", "i6"): is_matching_block = True
        elif term_to_find in ("induced5", "i5") and term_in_list in ("induced5", "i5"): is_matching_block = True
        elif term_to_find in ("induced3", "i3") and term_in_list in ("induced3", "i3"): is_matching_block = True
        elif term_to_find in ("eddy", "e", "eddy9", "e9") and \
             term_in_list in ("eddy", "e", "eddy9", "e9"): is_matching_block = True
        elif term_to_find in ("eddy8", "e8") and term_in_list in ("eddy8", "e8"): is_matching_block = True
        elif term_to_find in ("eddy3", "e3") and term_in_list in ("eddy3", "e3"): is_matching_block = True
        elif term_to_find in ("fdm", "f", "fdm3", "f3") and \
             term_in_list in ("fdm", "f", "fdm3", "f3"): is_matching_block = True
        elif term_to_find in ("bias", "b") and term_in_list in ("bias", "b"): is_matching_block = True
        
        if is_matching_block:
            found_term_block = True
            # N_cols_for_term_to_find should be the number of columns this block (term_in_list) generates
            A_block_obj = _create_TL_A_components(x_dummy, x_dummy, x_dummy,
                                                  Bt_actual=bt_dummy, terms=[term_in_list],
                                                  Bt_scale=50000.0, return_B=False)
            N_cols_for_term_to_find = cast(np.ndarray, A_block_obj).shape[1]
            break # Found the start of the relevant block

        if not found_term_block:
            offset_cols += cols_for_this_term_in_list

    bool_indices = np.zeros(N_total_cols, dtype=bool)
    if found_term_block and N_cols_for_term_to_find > 0:
        start_idx = offset_cols
        end_idx = offset_cols + N_cols_for_term_to_find
        if end_idx <= N_total_cols:
            bool_indices[start_idx:end_idx] = True
        else:
            logger.warning(f"Calculated indices for '{term_to_find}' (as part of block '{term_in_list if found_term_block else ''}') "
                           f"[{start_idx}:{end_idx}] exceed total columns {N_total_cols}.")
            # Attempt partial fill if start_idx is valid
            valid_end_idx = min(end_idx, N_total_cols)
            if start_idx < valid_end_idx:
                 bool_indices[start_idx:valid_end_idx] = True
    elif not found_term_block:
        logger.warning(f"Term block for '{term_to_find}' not found in current_terms: {current_terms}. "
                       "get_TL_term_ind will return all False.")

    return bool_indices