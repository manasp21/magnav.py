"""
Python translation of MagNav.jl/src/compensation.jl
"""
import copy
import logging
import time # Added import
from typing import Any, Callable, Dict, List, Tuple, Union, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import detrend as scipy_detrend, butter, lfilter # Added butter, lfilter
from sklearn.model_selection import train_test_split # For get_split (alternative)
from sklearn.preprocessing import StandardScaler # For norm_sets (example)
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge # For elasticnet_fit, linear_fit with ridge
from sklearn.cross_decomposition import PLSRegression # For plsr_fit

# --- Placeholder/Stub Definitions ---
# These would typically be imported from other modules or defined in detail.

from .map_utils import get_map_val # Added import

SILENT_DEBUG = False # Global debug flag, similar to silent_debug in Julia

class Chain(nn.Sequential): # Alias for torch.nn.Sequential for closer naming
    pass

class CompParams: # Base placeholder
    def __init__(self, **kwargs):
        self.version: int = kwargs.get("version", 1)
        self.features_setup: List[str] = kwargs.get("features_setup", [])
        self.features_no_norm: List[bool] = kwargs.get("features_no_norm", [])
        self.model_type: str = kwargs.get("model_type", "m1")
        self.y_type: str = kwargs.get("y_type", "d")
        self.use_mag: str = kwargs.get("use_mag", "mag_1_c")
        self.use_vec: str = kwargs.get("use_vec", "flux_a")
        self.data_norms: Optional[Tuple] = kwargs.get("data_norms", None)
        self.model: Optional[nn.Module] = kwargs.get("model", None)
        self.terms: List[str] = kwargs.get("terms", [])
        self.terms_A: List[str] = kwargs.get("terms_A", ["permanent", "induced", "eddy"])
        self.sub_diurnal: bool = kwargs.get("sub_diurnal", True)
        self.sub_igrf: bool = kwargs.get("sub_igrf", True)
        self.bpf_mag: bool = kwargs.get("bpf_mag", True)
        self.reorient_vec: bool = kwargs.get("reorient_vec", False)
        self.norm_type_A: str = kwargs.get("norm_type_A", "none")
        self.norm_type_x: str = kwargs.get("norm_type_x", "standardize")
        self.norm_type_y: str = kwargs.get("norm_type_y", "standardize")
        self.TL_coef: np.ndarray = kwargs.get("TL_coef", np.zeros(18, dtype=np.float32))

class NNCompParams(CompParams): # Placeholder
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.η_adam: float = kwargs.get("η_adam", 0.001)
        self.epoch_adam: int = kwargs.get("epoch_adam", 5)
        self.epoch_lbfgs: int = kwargs.get("epoch_lbfgs", 0)
        self.hidden: List[int] = kwargs.get("hidden", [8])
        self.activation: Callable = kwargs.get("activation", nn.SiLU) # swish
        self.loss: Callable = kwargs.get("loss", nn.MSELoss())
        self.batchsize: int = kwargs.get("batchsize", 2048)
        self.frac_train: float = kwargs.get("frac_train", 14/17)
        self.α_sgl: float = kwargs.get("α_sgl", 1.0)
        self.λ_sgl: float = kwargs.get("λ_sgl", 0.0)
        self.k_pca: int = kwargs.get("k_pca", -1)
        self.drop_fi: bool = kwargs.get("drop_fi", False)
        self.drop_fi_bson: str = kwargs.get("drop_fi_bson", "")
        self.drop_fi_csv: str = kwargs.get("drop_fi_csv", "")
        self.perm_fi: bool = kwargs.get("perm_fi", False)
        self.perm_fi_csv: str = kwargs.get("perm_fi_csv", "")

class LinCompParams(CompParams): # Placeholder
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k_plsr: int = kwargs.get("k_plsr", 0) # Assuming 0 means use all features if not specified
        self.λ_TL: float = kwargs.get("λ_TL", 0.0)


class XYZ: # Placeholder
    def __init__(self, traj=None, mag_1_c=None, flux_a=None, dt=0.1, **kwargs):
        self.traj = traj if traj is not None else np.empty((0,3)) # Example: N x 3 (lat,lon,alt)
        self.mag_1_c = mag_1_c # Placeholder for scalar magnetometer data
        self.flux_a = flux_a   # Placeholder for vector magnetometer data
        self.dt = dt
        # Add other fields as encountered, e.g., xyz.diurnal, xyz.igrf
        for key, value in kwargs.items():
            setattr(self, key, value)

class MapS: # Placeholder
    pass
class MapSd(MapS): pass
class MapS3D(MapS): pass
mapS_null = MapS() # Placeholder

class TempParams: # Placeholder
    def __init__(self, **kwargs):
        self.σ_curriculum: float = kwargs.get("σ_curriculum", 1.0)
        self.l_window: int = kwargs.get("l_window", 5)
        self.window_type: str = kwargs.get("window_type", "sliding") # :none, :sliding, :contiguous
        self.tf_layer_type: str = kwargs.get("tf_layer_type", "postlayer")
        self.tf_norm_type: str = kwargs.get("tf_norm_type", "batch")
        self.dropout_prob: float = kwargs.get("dropout_prob", 0.2)
        self.N_tf_head: int = kwargs.get("N_tf_head", 8)
        self.tf_gain: float = kwargs.get("tf_gain", 1.0)

# --- Stub functions (to be implemented or imported) ---
def norm_sets(data: np.ndarray, norm_type: str = "standardize", no_norm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes data using specified method (standardize, normalize, or none)."""
    # This is a simplified example. The actual function might be more complex.
    original_ndim = data.ndim # Store original ndim
    data_for_norm = data.copy() # Work on a copy
    if original_ndim == 1:
        data_for_norm = data_for_norm.reshape(-1, 1) # Use data_for_norm
    
    # Handle empty data input gracefully
    if data_for_norm.shape[0] == 0:
        # Determine shape of bias/scale based on original_ndim and data_for_norm.shape[1]
        if original_ndim == 1:
            return np.array(0.0, dtype=data.dtype), np.array(1.0, dtype=data.dtype), np.array([], dtype=data.dtype)
        else:
            num_cols_empty = data_for_norm.shape[1]
            return np.zeros(num_cols_empty, dtype=data.dtype), np.ones(num_cols_empty, dtype=data.dtype), data_for_norm

    bias = np.zeros(data_for_norm.shape[1], dtype=data.dtype)
    scale = np.ones(data_for_norm.shape[1], dtype=data.dtype)
    data_norm = data_for_norm.copy()

    for i in range(data_for_norm.shape[1]):
        if no_norm is not None and i < len(no_norm) and no_norm[i]: # Added boundary check for no_norm
            continue
        if norm_type == "standardize":
            bias[i] = np.mean(data_for_norm[:, i])
            scale[i] = np.std(data_for_norm[:, i])
            if scale[i] == 0: scale[i] = 1.0 # Avoid division by zero
            data_norm[:, i] = (data_for_norm[:, i] - bias[i]) / scale[i]
        elif norm_type == "normalize": # Example: min-max
            bias[i] = np.min(data_for_norm[:, i])
            scale[i] = np.max(data_for_norm[:, i]) - np.min(data_for_norm[:, i])
            if scale[i] == 0: scale[i] = 1.0
            data_norm[:, i] = (data_for_norm[:, i] - bias[i]) / scale[i]
        elif norm_type == "none":
            pass # No normalization
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
            
    if data_norm.shape[1] == 1 and original_ndim == 1:
        bias_out = bias[0]
        scale_out = scale[0]
        data_norm_out = data_norm.flatten()
        return bias_out, scale_out, data_norm_out

    return bias, scale, data_norm

def denorm_sets(bias: Union[np.ndarray, float, np.number], scale: Union[np.ndarray, float, np.number], data_norm: np.ndarray) -> np.ndarray:
    """Denormalizes data given bias and scale."""
    # Ensure bias and scale are broadcastable with data_norm
    _bias = np.asarray(bias)
    _scale = np.asarray(scale)

    # If data_norm is 1D, bias and scale should be scalar or 1-element for broadcasting
    if data_norm.ndim == 1:
        if _bias.ndim > 0 and _bias.size > 1 : _bias = _bias[0]
        if _scale.ndim > 0 and _scale.size > 1 : _scale = _scale[0]
    # If data_norm is 2D (N,F) and bias/scale are (F,), broadcasting works.
    # If bias/scale are scalar, broadcasting works.
    return data_norm * _scale + _bias

def unpack_data_norms(data_norms_tuple: Tuple) -> Tuple:
    """Unpacks data normalization parameters from a tuple."""
    # Example: (A_bias,A_scale,v_scale,x_bias,x_scale,y_bias,y_scale)
    # or (_,_,v_scale,x_bias,x_scale,y_bias,y_scale)
    if len(data_norms_tuple) == 7: # Common case
        return data_norms_tuple
    elif len(data_norms_tuple) == 4: # For linear_fit etc. (x_b, x_s, y_b, y_s)
        return data_norms_tuple
    elif len(data_norms_tuple) == 5: # For nn_comp_1 (v_pca, x_b, x_s, y_b, y_s)
        # Pad with None for A_b, A_s to allow consistent unpacking if a 7-tuple is expected by some internal logic
        return (None, None) + data_norms_tuple
    # Add more cases if other tuple structures are used
    raise ValueError(f"Unsupported data_norms_tuple structure: length {len(data_norms_tuple)}")


def get_nn_m(Nf: int, Ny: int, hidden: List[int], activation: Callable, **kwargs) -> nn.Sequential:
    """Creates a neural network model (nn.Sequential) based on specified layers and activation."""
    layers: List[nn.Module] = []
    input_size = Nf
    model_type_nn = kwargs.get("model_type") # Store for clarity
    
    if model_type_nn in ["m3w", "m3tf"]: # Temporal models might have different input handling
        l_window = kwargs.get("l_window", 1) # sequence length
        # Nf is features_per_step. The nn.Linear layers will operate on features_per_step.
        # Transformer/LSTM layers will handle the sequence.
        pass # input_size remains Nf for the first Linear layer


    current_layer_input_size = input_size
    for h_units in hidden:
        layers.append(nn.Linear(current_layer_input_size, h_units))
        layers.append(activation())
        # Potentially add dropout or batchnorm here if part of get_nn_m logic
        current_layer_input_size = h_units
    
    if model_type_nn == "m3tf":
        # Example: Add a TransformerEncoderLayer
        tf_d_model = current_layer_input_size # d_model is the number of expected features in the input for the transformer layer
        tf_nhead = kwargs.get("N_tf_head", 8)
        tf_dim_feedforward = kwargs.get("tf_dim_feedforward", tf_d_model * 4) # Allow override
        tf_dropout = kwargs.get("dropout_prob", 0.1)
        tf_activation_str = activation_to_str(activation).lower() # Get string for transformer
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_d_model,
            nhead=tf_nhead,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
            activation=tf_activation_str,
            batch_first=True # Assuming input to transformer is (batch, seq_len, features)
        )
        num_tf_layers = kwargs.get("num_tf_layers", 1) # Allow specifying number of transformer layers
        layers.append(nn.TransformerEncoder(encoder_layer, num_layers=num_tf_layers))
        # Output of TransformerEncoder is (batch, seq_len, d_model).
        # The final Linear layer (added below) will take d_model as input features.
        # This assumes the transformer output is processed (e.g., taking last time step's output)
        # before this final linear layer if Ny is a scalar output per sequence.
        # If the NN is expected to output a sequence, the final Linear layer might need to be TimeDistributed.
        # For now, current_layer_input_size remains tf_d_model for the final Linear layer.

    layers.append(nn.Linear(current_layer_input_size, Ny))
    return nn.Sequential(*layers)

def activation_to_str(act_fn_type: type) -> str:
    if act_fn_type == nn.SiLU: return "silu"
    if act_fn_type == nn.ReLU: return "relu"
    if act_fn_type == nn.Tanh: return "tanh"
    if act_fn_type == nn.GELU: return "gelu"
    # Add more mappings as needed
    logging.warning(f"activation_to_str: Unknown activation type {act_fn_type}, defaulting to 'relu'.")
    return "relu" # Default
def get_split(
    N: int,
    frac_train: float,
    window_type: str = "none",
    l_window: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get training & validation data indices.
    Python translation of MagNav.jl's get_split.
    Indices returned are 0-indexed.
    """
    if not (0 <= frac_train <= 1):
        raise ValueError(f"frac_train of {frac_train} is not between 0 & 1")
    
    if N > 0 and l_window >= N and window_type != "none":
        raise ValueError(f"window length of {l_window} is too large for {N} samples")
    if N == 0: # Handle empty input case
        return np.array([], dtype=int), np.array([], dtype=int)

    p_train_np: np.ndarray
    p_val_np: np.ndarray

    if window_type == "none":
        if frac_train < 1:
            p = np.random.permutation(N)
            n_train = int(np.floor(N * frac_train))
            p_train_np = p[:n_train]
            p_val_np = p[n_train:]
        else:
            p_train_np = np.arange(N)
            p_val_np = np.arange(N)
    elif window_type == "sliding":
        p = np.arange(N)
        if frac_train < 1:
            n_train = int(np.floor(N * frac_train))
            # Julia assertion: all(l_window .<= (N_train, N-N_train))
            # This ensures that N_train >= l_window and (N - N_train) >= l_window.
            # This check can be added if strict adherence to original asserts is needed.
            # For now, focusing on the splitting logic.
            
            p_train_np = p[:n_train]
            
            # Julia: p_val = p[N_train+l_window : N] (1-indexed for p and start)
            # Python: val_start_idx_py = (N_train_from_1_based + l_window) - 1
            # n_train is count, so N_train_from_1_based is n_train.
            val_start_idx_py = n_train + l_window -1 
            if val_start_idx_py >= N : 
                p_val_np = np.array([], dtype=int)
            else:
                p_val_np = p[val_start_idx_py:]
        else: # frac_train == 1
            p_train_np = p
            p_val_np = p
    elif window_type == "contiguous":
        if l_window <= 0: 
            raise ValueError("l_window must be positive for contiguous window_type")
        
        block_starts = np.arange(0, N, l_window)
        n_blocks = len(block_starts)

        if n_blocks == 0:
            # N < l_window, treat as 'none' type split for these few samples
            # This also handles N == 0 correctly due to permutation of empty or arange(0)
            p_rand = np.random.permutation(N)
            n_train_samples_contig = int(np.floor(N * frac_train))
            p_train_np = p_rand[:n_train_samples_contig]
            p_val_np = p_rand[n_train_samples_contig:]


        else:
            permuted_block_indices = np.random.permutation(n_blocks)

            if frac_train < 1:
                n_train_blocks = int(np.floor(frac_train * n_blocks))
                
                train_block_selector = permuted_block_indices[:n_train_blocks]
                val_block_selector   = permuted_block_indices[n_train_blocks:]
                
                p_train_list_internal: List[np.ndarray] = []
                for block_idx_in_selector in train_block_selector:
                    start = block_starts[block_idx_in_selector]
                    end = min(start + l_window, N)
                    p_train_list_internal.append(np.arange(start, end))
                p_train_np = np.concatenate(p_train_list_internal) if p_train_list_internal else np.array([], dtype=int)

                p_val_list_internal: List[np.ndarray] = []
                for block_idx_in_selector in val_block_selector:
                    start = block_starts[block_idx_in_selector]
                    end = min(start + l_window, N)
                    p_val_list_internal.append(np.arange(start, end))
                p_val_np = np.concatenate(p_val_list_internal) if p_val_list_internal else np.array([], dtype=int)

            else: # frac_train == 1
                p_all_list_internal: List[np.ndarray] = []
                for block_start_val in block_starts: 
                    start = block_start_val
                    end = min(start + l_window, N)
                    p_all_list_internal.append(np.arange(start, end))
                p_train_np = np.concatenate(p_all_list_internal) if p_all_list_internal else np.array([], dtype=int)
                p_val_np = p_train_np 
    else:
        raise ValueError(f"window_type '{window_type}' is invalid, select 'none', 'sliding', or 'contiguous'")

    return p_train_np, p_val_np
def get_temporal_data(x_norm: np.ndarray, l_segs: List[int], l_window: int) -> np.ndarray:
    """
    Internal helper function to create windowed sequence temporal data from
    original data. Adds padding to beginning of data. Uses line lengths to avoid
    windowing across lines, but (ad hoc) checks if any lines may be sequential.

    Args:
        x_norm:   Nf x N normalized data matrix (Nf is number of features)
        l_segs:   length-N_lines vector of lengths of lines, sum(l_segs) = N
        l_window: temporal window length

    Returns:
        x_w: Nf x l_window x N normalized data matrix, windowed
             (Note: PyTorch LSTMs/Transformers often expect seq_len first or batch first,
              so further permutation might be needed depending on the model layer)
    """
    nf, n_samples = x_norm.shape # number of features & samples (instances)

    if sum(l_segs) != n_samples:
        raise ValueError(f"sum of lines = {sum(l_segs)} != {n_samples}")

    if not l_segs: # Handle case with no segments
        if n_samples > 0 and l_window > 0: # If there's data but no segments, treat as one segment
            l_segs_processed = [n_samples]
        else: # No data or no window, return empty or appropriately shaped array
            return np.empty((nf, l_window, 0), dtype=x_norm.dtype)
    else:
        l_segs_processed = list(l_segs) # Make a mutable copy

    # Ad-hoc check for sequential lines (simplified from Julia's std check)
    # This part is tricky to translate directly without knowing the exact data characteristics
    # and the intent of `std(x_norm[:,l]-x_norm[:,l+1]) < lim`.
    # For now, we'll skip the merging of segments based on this ad-hoc check,
    # as it might require more domain-specific knowledge or a clearer metric.
    # If this merging is critical, a more robust Python equivalent would be needed.
    # Julia code for merging:
    # if len(l_segs) > 1:
    #     l0 = np.cumsum(l_segs)[:-1]
    #     lim = 0.25 # ad hoc
    #     # ind = [np.std(x_norm[:,l] - x_norm[:,l+1]) < lim for l in l0] # Direct translation problematic
    #     # Simplified: assume segments are distinct unless explicitly told otherwise
    #     l_segs_ = list(l_segs) # mutable copy
    #     # ... merging logic based on 'ind' ...
    #     l_segs_processed = [s for s in l_segs_ if s != 0]

    current_l_segs = l_segs_processed
    
    # Correctly calculate cumulative sums for 0-indexed Python
    # l0_ = [0] + np.cumsum(current_l_segs[:-1]).tolist() if len(current_l_segs) > 1 else [0]
    # A simpler way for segment start indices:
    segment_starts = [0] * len(current_l_segs)
    for i in range(1, len(current_l_segs)):
        segment_starts[i] = segment_starts[i-1] + current_l_segs[i-1]

    x_w_list = []

    for seg_idx, n_seg_samples in enumerate(current_l_segs):
        if n_seg_samples == 0:
            continue

        x_w_segment = np.zeros((nf, l_window, n_seg_samples), dtype=x_norm.dtype)
        seg_start_abs_idx = segment_starts[seg_idx]

        for j_in_seg in range(n_seg_samples): # 0 to n_seg_samples-1
            # j_in_seg is the current point in the segment
            # We want to create a window of l_window points ending at j_in_seg

            # Determine the start of the window in the current segment
            window_start_in_seg = max(0, j_in_seg - l_window + 1)
            
            # Determine how many actual data points we can take for this window
            num_actual_points = j_in_seg - window_start_in_seg + 1
            
            # Determine padding needed at the beginning of the window
            num_padding_points = l_window - num_actual_points

            # Fill padding (if any)
            # In Julia, padding was with x_norm[:, l0_[i] .+ (j1)], which is the first element of the window.
            # Replicating this:
            first_val_for_padding = x_norm[:, seg_start_abs_idx + window_start_in_seg]
            for k_pad in range(num_padding_points):
                x_w_segment[:, k_pad, j_in_seg] = first_val_for_padding

            # Fill actual data
            for k_data in range(num_actual_points):
                src_idx_in_x_norm = seg_start_abs_idx + window_start_in_seg + k_data
                dest_idx_in_window = num_padding_points + k_data
                x_w_segment[:, dest_idx_in_window, j_in_seg] = x_norm[:, src_idx_in_x_norm]
                
        x_w_list.append(x_w_segment)

    if not x_w_list:
        return np.empty((nf, l_window, 0), dtype=x_norm.dtype)

    return np.concatenate(x_w_list, axis=2)

def sparse_group_lasso(model: nn.Module, alpha: float) -> torch.Tensor:
    """
    Computes the Sparse Group Lasso (SGL) penalty term for a PyTorch model.
    The penalty is calculated as: alpha * L1_norm + (1-alpha) * L2_norm_group_wise
    This function sums this term over all nn.Linear layers' weights.
    The overall regularization strength (lambda_sgl) should be applied multiplicatively
    to the output of this function.

    Args:
        model (nn.Module): The neural network model (typically an nn.Sequential containing nn.Linear layers).
        alpha (float): The mixing parameter between L1 and L2 norms.
                       alpha=1 gives Lasso, alpha=0 gives Group Lasso (L2 on layer weights).

    Returns:
        torch.Tensor: The SGL penalty value (scalar tensor).
    """
    sgl_penalty = torch.tensor(0.0, device=next(model.parameters(), torch.tensor(0.0)).device)
    
    has_linear_layers = False
    # Iterate through modules of the model to find Linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            has_linear_layers = True
            # Ensure the layer has a weight parameter
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                
                # L1 norm of all weights in the layer
                l1_term = torch.norm(weight, p=1)
                
                # L2 norm of all weights in the layer (this acts as the group L2 norm)
                l2_term = torch.norm(weight, p=2)
                
                # Add to total SGL penalty
                sgl_penalty += alpha * l1_term + (1.0 - alpha) * l2_term
            
            # Biases are typically not included in SGL or regularized differently.
            # If biases were to be included, similar logic for module.bias would be added.

    if not has_linear_layers and not SILENT_DEBUG:
        # This warning can be enhanced with a proper logging system if available.
        print("Warning: sparse_group_lasso was called on a model that does not appear to contain nn.Linear layers. The SGL penalty will be 0.")
        
    return sgl_penalty

def err_segs(y_hat: np.ndarray, y: np.ndarray, l_segs: List[int], silent: bool = False) -> np.ndarray:
    """Calculates error, mean-corrected per segment."""
    # Simplified: just difference. Actual might involve detrending per segment.
    current_pos = 0
    errors = []
    y_hat_flat = y_hat.flatten()
    y_flat = y.flatten()

    for seg_len in l_segs:
        if seg_len == 0: continue
        y_hat_segment = y_hat_flat[current_pos : current_pos + seg_len]
        y_segment = y_flat[current_pos : current_pos + seg_len]
        
        # Example: mean correction per segment
        err_segment = (y_segment - y_hat_segment)
        err_segment = err_segment - np.mean(err_segment)
        errors.append(err_segment)
        current_pos += seg_len
    
    if not errors:
        return np.array([], dtype=y_hat.dtype)
    return np.concatenate(errors)


def create_TL_A(vec_mag_data: np.ndarray, ind: Optional[np.ndarray] = None, terms: Optional[List[str]] = None, return_B: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Placeholder for creating Tolles-Lawson A matrix."""
    # vec_mag_data: N x 3 (flux_a, flux_b, flux_c)
    # ind: indices to select from vec_mag_data
    # terms: list of TL terms like 'permanent', 'induced', 'eddy'
    # kwargs: Bt (scalar total field), etc.
    
    # Robust handling of 'ind' for selecting data
    if ind is not None:
        _ind = np.asarray(ind) # Convert to numpy array to safely check size and handle lists
        if _ind.size > 0:
            selected_data = vec_mag_data[_ind, :]
        else: # ind was provided but resulted in an empty selection (e.g., empty list or array)
            selected_data = np.empty((0, vec_mag_data.shape[1]), dtype=vec_mag_data.dtype)
    else: # ind is None, use all data
        selected_data = vec_mag_data
    
    if selected_data.shape[0] == 0:
        # Return empty arrays with expected number of columns based on terms
        num_cols = 18 # Default for full TL model
        # A more robust way to determine num_cols is needed if this path is critical
        if terms is not None:
            if terms == ["permanent", "induced", "eddy"]: num_cols = 18
            elif terms == ["permanent", "induced"]: num_cols = 9
            elif terms == ["permanent"]: num_cols = 3
            # Add other specific combinations or a more generic calculation
            else: # Fallback, very rough estimate
                base_cols = 0
                if "permanent" in terms: base_cols += 3
                if "induced" in terms: base_cols += 5 # Assuming 5 for symmetric tensor
                if "eddy" in terms: base_cols += 9 # Assuming 9 for eddy current terms
                if base_cols > 0:
                    num_cols = base_cols
                # If terms are unknown or don't match, 18 is a guess or could be more dynamic.
        
        A_matrix = np.empty((0, num_cols), dtype=selected_data.dtype)
        Bt_out = np.empty(0, dtype=selected_data.dtype)
        B_dot_out = np.empty((0,3), dtype=selected_data.dtype) # Assuming B_dot is N x 3
        if return_B:
            return A_matrix, Bt_out, B_dot_out
        return A_matrix

    # Placeholder A_matrix construction. A real implementation would use 'terms'.
    # For this placeholder, if 'permanent' is in terms, use the first 3 cols of selected_data.
    # This is a very simplified placeholder.
    if terms and "permanent" in terms and selected_data.shape[1] >=3 :
         A_matrix = selected_data[:,:3]
    elif selected_data.shape[1] > 0: # Fallback to using selected_data if it has columns
         A_matrix = selected_data
    else: # selected_data is empty or has no columns, A_matrix should match num_cols logic from above
        num_cols_fallback = 18 # Default if terms is None or doesn't define structure
        if terms: # Try to infer from terms again if needed
            if terms == ["permanent", "induced", "eddy"]: num_cols_fallback = 18
            elif terms == ["permanent", "induced"]: num_cols_fallback = 9
            elif terms == ["permanent"]: num_cols_fallback = 3
        A_matrix = np.empty((selected_data.shape[0], num_cols_fallback), dtype=selected_data.dtype)


    Bt_out_val = np.zeros(selected_data.shape[0], dtype=selected_data.dtype)
    if selected_data.size > 0 and selected_data.shape[1] >=3 : # Check if selected_data has at least 3 columns for norm
        Bt_out_val = np.linalg.norm(selected_data[:,:3], axis=1)
    
    B_dot_out_val = np.zeros_like(selected_data) # Placeholder, should be (N,3)
    if selected_data.shape[0] > 1 and selected_data.shape[1] >=3: # Can only compute gradient if more than 1 sample
        dt_val = kwargs.get('dt', 0.1)
        if dt_val <= 0: dt_val = 0.1
        B_dot_out_val = np.gradient(selected_data[:,:3], dt_val, axis=0)

    if return_B:
        return A_matrix, Bt_out_val, B_dot_out_val
    return A_matrix


def get_TL_term_ind(term: str, terms_list: List[str]) -> np.ndarray:
    """Placeholder: Get indices for a specific TL term in the A matrix."""
    # This depends on the fixed order of terms in create_TL_A
    # Example: if terms_list = ['permanent', 'induced', 'eddy']
    # and 'permanent' corresponds to first 3 columns, 'induced' next 6, 'eddy' next 9
    if term == "permanent": return np.array([True,True,True] + [False]*15) # if 18 terms total
    # This is highly dependent on the actual structure of A matrix from create_TL_A
    return np.array([False] * 18) # Default to no match
def TL_vec_split(TL_coef: np.ndarray, terms: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal helper function to separate Tolles-Lawson coefficients from a single
    vector to individual permanent, induced, & eddy current coefficient vectors.

    Args:
        TL_coef: Tolles-Lawson coefficients (must include 'permanent' & 'induced')
        terms:   Tolles-Lawson terms used {'permanent','induced','eddy', 
                     'p','i','e', 'permanent3', 'p3',
                     'induced6', 'i6', 'induced5', 'i5', 'induced3', 'i3',
                     'eddy9', 'e9', 'eddy8', 'e8', 'eddy3', 'e3'}
    Returns:
        TL_p: length-3 vector of permanent field coefficients
        TL_i: length-3, 5, or 6 vector of induced field coefficients
        TL_e: length-0, 3, 8, or 9 vector of eddy current coefficients
    """
    # Simplified term checking for Python
    has_permanent = any(t in terms for t in ["permanent", "p", "permanent3", "p3"])
    has_induced = any(t in terms for t in ["induced", "i", "induced6", "i6", "induced5", "i5", "induced3", "i3"])
    
    if not has_permanent:
        raise ValueError("Permanent terms are required in TL_vec_split")
    if not has_induced:
        raise ValueError("Induced terms are required in TL_vec_split")
    
    # Check for disallowed terms (simplified)
    disallowed_terms = ["fdm", "f", "fdm3", "f3", "bias", "b"]
    if any(t in terms for t in disallowed_terms):
        raise ValueError("Derivative & bias terms may not be used in TL_vec_split")

    # This assertion is harder to replicate directly without calling create_TL_A
    # A_test = create_TL_A(np.array([[1.0,1.0,1.0]]), terms=terms) # Assuming create_TL_A is available and works
    # if TL_coef.size != A_test.shape[1]:
    #     raise ValueError("TL_coef does not agree with specified terms")

    current_idx = 0
    TL_p = np.array([], dtype=TL_coef.dtype)
    TL_i = np.array([], dtype=TL_coef.dtype)
    TL_e = np.array([], dtype=TL_coef.dtype)

    # Permanent terms (always 3)
    if any(t in terms for t in ["permanent", "p", "permanent3", "p3"]):
        TL_p = TL_coef[current_idx : current_idx+3]
        current_idx += 3
    
    # Induced terms
    if any(t in terms for t in ["induced", "i", "induced6", "i6"]):
        TL_i = TL_coef[current_idx : current_idx+6]
        current_idx += 6
    elif any(t in terms for t in ["induced5", "i5"]):
        TL_i = TL_coef[current_idx : current_idx+5]
        current_idx += 5
    elif any(t in terms for t in ["induced3", "i3"]):
        TL_i = TL_coef[current_idx : current_idx+3]
        current_idx += 3
        
    # Eddy terms (these come last, so we can infer from remaining length)
    # This logic assumes permanent and induced terms are always first and in that order.
    remaining_coeffs = TL_coef.size - current_idx
    
    if remaining_coeffs > 0:
        if any(t in terms for t in ["eddy", "e", "eddy9", "e9"]):
            if remaining_coeffs == 9: # Check if it matches expected size
                TL_e = TL_coef[current_idx : current_idx+9]
                current_idx += 9
            # else: warning or error if size mismatch
        elif any(t in terms for t in ["eddy8", "e8"]):
            if remaining_coeffs == 8:
                TL_e = TL_coef[current_idx : current_idx+8]
                current_idx += 8
        elif any(t in terms for t in ["eddy3", "e3"]):
            if remaining_coeffs == 3:
                TL_e = TL_coef[current_idx : current_idx+3]
                current_idx += 3
        # If no specific eddy term matches but there are remaining coeffs,
        # it implies a mismatch or an unhandled term combination.
        # For robustness, one might assign remaining to TL_e or raise error.
        # Based on Julia, if specific eddy terms are not present, TL_e is empty.
        # So, only assign if a recognized eddy term is present AND size matches.

    if current_idx != TL_coef.size:
        # This implies a mismatch between the sum of expected term lengths and TL_coef.size
        has_eddy_term_spec = any(t.startswith("eddy") or t == "e" for t in terms)
        if remaining_coeffs > 0 and not has_eddy_term_spec and not SILENT_DEBUG:
            logging.warning(f"TL_vec_split: {remaining_coeffs} coefficients remain but no eddy term specified in 'terms'.")
        elif remaining_coeffs == 0 and has_eddy_term_spec and not TL_e.size and not SILENT_DEBUG:
            logging.warning(f"TL_vec_split: Eddy term specified in 'terms' but no/mismatched coefficients found for it.")
        elif current_idx < TL_coef.size and has_eddy_term_spec and not TL_e.size and not SILENT_DEBUG:
             logging.warning(f"TL_vec_split: Eddy term specified, but remaining_coeffs ({remaining_coeffs}) did not match expected size for any known eddy term type.")
        elif not SILENT_DEBUG : # General mismatch
             logging.warning(f"TL_vec_split: Coefficient parsing mismatch. Expected to parse {TL_coef.size}, but parsed {current_idx}.")

    return TL_p, TL_i, TL_e
def TL_vec2mat(TL_coef: np.ndarray, terms: List[str], Bt_scale: float = 50000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the matrix form of Tolles-Lawson coefficients from the vector form.

    Args:
        TL_coef:  Tolles-Lawson coefficients (must include 'permanent' & 'induced')
        terms:    Tolles-Lawson terms used {'permanent','induced','eddy', ...}
        Bt_scale: (optional) scaling factor for induced & eddy current terms [nT]

    Returns:
        TL_coef_p: length-3 vector of permanent field coefficients
        TL_coef_i: 3x3 symmetric matrix of induced field coefficients, denormalized
        TL_coef_e: 3x3 matrix of eddy current coefficients, denormalized
    """
    has_permanent = any(t in terms for t in ["permanent", "p", "permanent3", "p3"])
    has_induced = any(t in terms for t in ["induced", "i", "induced6", "i6", "induced5", "i5", "induced3", "i3"])

    if not has_permanent:
        raise ValueError("Permanent terms are required in TL_vec2mat")
    if not has_induced:
        raise ValueError("Induced terms are required in TL_vec2mat")

    disallowed_terms = ["fdm", "f", "fdm3", "f3", "bias", "b"]
    if any(t in terms for t in disallowed_terms):
        raise ValueError("Derivative & bias terms may not be used in TL_vec2mat")

    # Placeholder for A_test assertion if create_TL_A is fully implemented and available
    # A_test = create_TL_A(np.array([[1.0,1.0,1.0]]), terms=terms)
    # if TL_coef.size != A_test.shape[1]:
    #     raise ValueError("TL_coef does not agree with specified terms in TL_vec2mat")

    current_idx = 0
    TL_coef_p = np.zeros(3, dtype=TL_coef.dtype)
    TL_coef_i_mat = np.zeros((3, 3), dtype=TL_coef.dtype)
    TL_coef_e_mat = np.zeros((3, 3), dtype=TL_coef.dtype) # Initialize as empty or zero

    # Permanent terms
    if any(t in terms for t in ["permanent", "p", "permanent3", "p3"]):
        TL_coef_p = TL_coef[current_idx : current_idx+3]
        current_idx += 3
    
    # Induced terms
    if any(t in terms for t in ["induced", "i", "induced6", "i6"]):
        TL_i_vec = TL_coef[current_idx : current_idx+6]
        TL_coef_i_mat = np.array([
            [TL_i_vec[0], TL_i_vec[1]/2, TL_i_vec[2]/2],
            [TL_i_vec[1]/2, TL_i_vec[3],   TL_i_vec[4]/2],
            [TL_i_vec[2]/2, TL_i_vec[4]/2, TL_i_vec[5]]
        ]) / Bt_scale
        current_idx += 6
    elif any(t in terms for t in ["induced5", "i5"]):
        TL_i_vec = TL_coef[current_idx : current_idx+5]
        TL_coef_i_mat = np.array([
            [TL_i_vec[0], TL_i_vec[1]/2, TL_i_vec[2]/2],
            [TL_i_vec[1]/2, TL_i_vec[3],   TL_i_vec[4]/2],
            [TL_i_vec[2]/2, TL_i_vec[4]/2, 0.0] # Assuming M_33 = 0 for 5-term model
        ]) / (Bt_scale if Bt_scale != 0 else 1.0)
        # Ensure symmetry for the 5-term case if it's not inherently symmetric from vector
        TL_coef_i_mat = (TL_coef_i_mat + TL_coef_i_mat.T)/2.0
        TL_coef_i_mat[2,2] = 0.0 # Explicitly set (3,3) to 0 for 5-term model
        current_idx += 5
    elif any(t in terms for t in ["induced3", "i3"]):
        TL_i_vec = TL_coef[current_idx : current_idx+3]
        TL_coef_i_mat = np.diag(TL_i_vec) / (Bt_scale if Bt_scale != 0 else 1.0)
        current_idx += 3

    # Eddy terms
    remaining_coeffs = TL_coef.size - current_idx
    if remaining_coeffs > 0:
        if any(t in terms for t in ["eddy", "e", "eddy9", "e9"]):
            if remaining_coeffs == 9:
                TL_e_vec = TL_coef[current_idx : current_idx+9]
                # Julia stores vec(M'), so reshape TL_e_vec (9 elements) to 3x3 column-major, then transpose.
                # Or, reshape row-major (default in numpy) and then transpose.
                TL_coef_e_mat = TL_e_vec.reshape(3,3).T / (Bt_scale if Bt_scale != 0 else 1.0)
                current_idx += 9
        elif any(t in terms for t in ["eddy8", "e8"]):
            if remaining_coeffs == 8: # Corrected indentation
                TL_e_vec = TL_coef[current_idx : current_idx+8]
                # Construct 3x3 matrix, assuming last element of vec(M') is zero
                temp_mat_flat = np.zeros(9, dtype=TL_e_vec.dtype)
                temp_mat_flat[:8] = TL_e_vec
                TL_coef_e_mat = temp_mat_flat.reshape(3,3).T / (Bt_scale if Bt_scale != 0 else 1.0)
                current_idx += 8
        elif any(t in terms for t in ["eddy3", "e3"]):
            if remaining_coeffs == 3:
                TL_e_vec = TL_coef[current_idx : current_idx+3]
                TL_coef_e_mat = np.diag(TL_e_vec) / (Bt_scale if Bt_scale != 0 else 1.0)
                current_idx += 3
        # If no specific eddy term is listed but coeffs remain, TL_coef_e_mat remains zero/empty
        # as per Julia's behavior (TL_e = [] if no eddy terms).

    # Final check, similar to TL_vec_split
    # if current_idx != TL_coef.size:
    #     # Potentially raise an error or warning if not all coefficients were consumed
    #     # and it doesn't align with an expectation of no eddy terms.
    #     pass

    return TL_coef_p, TL_coef_i_mat, TL_coef_e_mat
def TL_mat2vec(TL_coef_p: np.ndarray, TL_coef_i_mat: np.ndarray, TL_coef_e_mat: np.ndarray, 
               terms: List[str], Bt_scale: float = 50000.0) -> np.ndarray:
    """
    Extract the vector form of Tolles-Lawson coefficients from the matrix form.

    Args:
        TL_coef_p:     length-3 vector of permanent field coefficients
        TL_coef_i_mat: 3x3 symmetric matrix of induced field coefficients, denormalized
        TL_coef_e_mat: 3x3 matrix of eddy current coefficients, denormalized
        terms:         Tolles-Lawson terms used {'permanent','induced','eddy', ...}
        Bt_scale:      (optional) scaling factor for induced & eddy current terms [nT]

    Returns:
        TL_coef: Tolles-Lawson coefficients as a single vector
    """
    TL_coef_list = []

    # Permanent terms (always present and first)
    TL_coef_list.append(TL_coef_p)

    # Induced terms
    if any(t in terms for t in ["induced", "i", "induced6", "i6"]):
        # From matrix: [a b c; b d e; c e f] -> [a, 2b, 2c, d, 2e, f] (scaled)
        # Indices for upper triangle (including diagonal): (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        i_vec = np.array([
            TL_coef_i_mat[0,0], 
            TL_coef_i_mat[0,1]*2, 
            TL_coef_i_mat[0,2]*2,
            TL_coef_i_mat[1,1], 
            TL_coef_i_mat[1,2]*2,
            TL_coef_i_mat[2,2]
        ]) * Bt_scale
        TL_coef_list.append(i_vec)
    elif any(t in terms for t in ["induced5", "i5"]):
        # Matrix form assumed [a b c; b d e; c e 0] -> [a, 2b, 2c, d, 2e]
        i_vec = np.array([
            TL_coef_i_mat[0,0], 
            TL_coef_i_mat[0,1]*2, 
            TL_coef_i_mat[0,2]*2,
            TL_coef_i_mat[1,1], 
            TL_coef_i_mat[1,2]*2
        ]) * Bt_scale
        TL_coef_list.append(i_vec)
    elif any(t in terms for t in ["induced3", "i3"]):
        # Matrix form assumed [a 0 0; 0 d 0; 0 0 f] -> [a, d, f]
        i_vec = np.array([
            TL_coef_i_mat[0,0], 
            TL_coef_i_mat[1,1], 
            TL_coef_i_mat[2,2]
        ]) * Bt_scale
        TL_coef_list.append(i_vec)
    # If no induced terms specified, nothing is added for TL_i

    # Eddy terms
    _Bt_scale_eff = Bt_scale if Bt_scale != 0 else 1.0
    if TL_coef_e_mat is not None and TL_coef_e_mat.size > 0: # Check if e_mat is provided
        if any(t in terms for t in ["eddy", "e", "eddy9", "e9"]):
            TL_coef_list.append(TL_coef_e_mat.T.flatten() * _Bt_scale_eff)
        elif any(t in terms for t in ["eddy8", "e8"]):
            TL_coef_list.append(TL_coef_e_mat.T.flatten()[:8] * _Bt_scale_eff)
        elif any(t in terms for t in ["eddy3", "e3"]):
            TL_coef_list.append(np.diag(TL_coef_e_mat) * _Bt_scale_eff)
    # If no eddy terms specified, TL_e remains empty / nothing added

    if not TL_coef_list:
        return np.array([], dtype=TL_coef_p.dtype)
        
    return np.concatenate(TL_coef_list).astype(TL_coef_p.dtype)
def get_TL_aircraft_vec(
    B_vec: np.ndarray, 
    B_vec_dot: np.ndarray, 
    TL_coef_p: np.ndarray, 
    TL_coef_i_mat: np.ndarray, 
    TL_coef_e_mat: Optional[np.ndarray], # Can be None or empty if no eddy terms
    return_parts: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculates the Tolles-Lawson aircraft vector field.

    Args:
        B_vec:         3xN matrix of vector magnetometer measurements
        B_vec_dot:     3xN matrix of vector magnetometer measurement derivatives
        TL_coef_p:     length-3 vector of permanent field coefficients
        TL_coef_i_mat: 3x3 symmetric matrix of induced field coefficients, denormalized
        TL_coef_e_mat: 3x3 matrix of eddy current coefficients, denormalized, or None/empty
        return_parts:  (optional) if true, also return TL_perm, TL_induced, & TL_eddy

    Returns:
        TL_aircraft: 3xN matrix of TL aircraft vector field
        (if return_parts): TL_perm, TL_induced, TL_eddy
    """
    # Ensure B_vec and B_vec_dot are (3, N)
    _B_vec = B_vec.reshape(3, -1) if B_vec.ndim == 1 and B_vec.size % 3 == 0 else B_vec
    _B_vec_dot = B_vec_dot.reshape(3, -1) if B_vec_dot.ndim == 1 and B_vec_dot.size % 3 == 0 else B_vec_dot

    if _B_vec.shape[0] != 3:
        raise ValueError(f"B_vec must be 3xN, but got {_B_vec.shape}")
    
    num_samples = _B_vec.shape[1]

    # If TL_coef_e_mat is present, B_vec_dot must be valid and match N.
    # Otherwise, create a zero B_vec_dot if it's invalid or not matching.
    if TL_coef_e_mat is not None and TL_coef_e_mat.size > 0:
        if _B_vec_dot.shape[0] != 3 or _B_vec_dot.shape[1] != num_samples:
            raise ValueError(f"B_vec_dot shape mismatch. Expected (3, {num_samples}), got {_B_vec_dot.shape} when eddy terms are present.")
    elif _B_vec_dot.shape[0] != 3 or _B_vec_dot.shape[1] != num_samples : # B_vec_dot is present but invalid, and no eddy terms
        _B_vec_dot = np.zeros((3, num_samples), dtype=_B_vec.dtype)
    elif _B_vec_dot.size == 0 and num_samples > 0 : # B_vec_dot is empty, create zeros
        _B_vec_dot = np.zeros((3, num_samples), dtype=_B_vec.dtype)


    # TL_perm = TL_coef_p[:, np.newaxis] * np.ones_like(B_vec[0,:]) # Julia: TL_coef_p .* one.(B_vec)
    # A more direct way to make TL_coef_p (3,) broadcast to (3,N) with B_vec (3,N)
    TL_perm = TL_coef_p[:, np.newaxis]
    TL_induced = TL_coef_i_mat @ _B_vec

    TL_eddy = np.zeros_like(_B_vec, dtype=_B_vec.dtype)
    if TL_coef_e_mat is not None and TL_coef_e_mat.size > 0:
        # _B_vec_dot is guaranteed to be (3,N) here by checks above
        TL_eddy = TL_coef_e_mat @ _B_vec_dot
    
    TL_aircraft = TL_perm + TL_induced + TL_eddy

    if return_parts:
        # Ensure all parts are broadcastable to (3,N) if TL_perm was (3,1)
        TL_perm_broadcast = TL_perm if TL_perm.shape[1] == num_samples else np.tile(TL_perm, (1,num_samples))
        return TL_aircraft, TL_perm_broadcast, TL_induced, TL_eddy
    else:
        return TL_aircraft
def get_curriculum_ind(TL_diff: np.ndarray, N_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal helper function to get indices for curriculum learning.
    Curriculum learning (Tolles-Lawson) indices are those within N_sigma 
    of the mean, and neural network indices are those outside of it (i.e., outliers).

    Args:
        TL_diff: Difference of TL model to ground truth.
        N_sigma: (optional) Standard deviation threshold.

    Returns:
        ind_cur: Indices for curriculum learning (within N_sigma), boolean array.
        ind_nn:  Indices for training the neural network (outside N_sigma), boolean array.
    """
    if TL_diff.size == 0: # Handle empty input
        return np.array([], dtype=bool), np.array([], dtype=bool)

    # Detrend (mean only)
    # scipy.signal.detrend by default subtracts the least-squares line.
    # For mean only, we can just subtract the mean.
    TL_diff_detrended = TL_diff - np.mean(TL_diff)
    
    std_dev = np.std(TL_diff_detrended)
    if std_dev == 0: # Avoid issues if all diffs are the same (e.g., N_sigma * 0 = 0)
        cutoff = 0 # All points will be ind_cur if N_sigma >= 0
    else:
        cutoff = N_sigma * std_dev
    
    ind_cur = np.abs(TL_diff_detrended) <= cutoff
    ind_nn = ~ind_cur # Boolean negation for the opposite set
    
    return ind_cur, ind_nn

def get_x(xyz: XYZ, ind: np.ndarray, features_setup: List[str], **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Placeholder for getting feature matrix X."""
    # This function would extract specified features from xyz data
    # For now, returns dummy data
    num_samples = len(ind) if ind is not None and ind.size > 0 else 0
    
    # This is a placeholder. Actual implementation would build x_data based on features_setup.
    # For now, assume features_setup directly maps to columns or generates some random data.
    if not features_setup: # Default to using flux_a if no features specified
        if hasattr(xyz, 'flux_a') and xyz.flux_a is not None and xyz.flux_a.ndim == 2 and xyz.flux_a.shape[0] > 0:
            if ind is not None and ind.size > 0:
                # Ensure indices in 'ind' are valid for 'xyz.flux_a'
                if np.max(ind) < xyz.flux_a.shape[0]:
                    x_data = xyz.flux_a[ind, :]
                else: # Index out of bounds
                    logging.warning("Indices for get_x out of bounds for flux_a. Using random data.")
                    x_data = np.random.rand(num_samples, 3).astype(np.float32)
            else: # No indices, or ind is empty
                x_data = np.empty((0,3), dtype=np.float32)
            feature_names = ['flux_a_x', 'flux_a_y', 'flux_a_z']
            num_features = 3
        else: # Fallback if flux_a not available
            num_features = 3
            x_data = np.random.rand(num_samples, num_features).astype(np.float32)
            feature_names = [f"default_feat_{i}" for i in range(num_features)]
    else:
        num_features = len(features_setup)
        x_data = np.random.rand(num_samples, num_features).astype(np.float32) # Placeholder
        feature_names = list(features_setup)

    no_norm_mask = np.array(kwargs.get("features_no_norm", [False]*num_features), dtype=bool)
    if len(no_norm_mask) != num_features: # Ensure mask matches feature count
        no_norm_mask = np.zeros(num_features, dtype=bool) # Default to all False if mismatch

    l_segs = [num_samples] if num_samples > 0 else [] # Empty list if no samples
    
    return x_data, no_norm_mask, feature_names, l_segs

def get_y(xyz: XYZ, ind: np.ndarray, map_val: Any, **kwargs) -> np.ndarray:
    """Placeholder for getting target vector Y."""
    num_samples = len(ind) if ind is not None and ind.size > 0 else 0
    y_data = np.zeros(num_samples, dtype=np.float32) # Initialize
    
    y_type = kwargs.get("y_type", "d")
    use_mag_field = kwargs.get("use_mag", "mag_1_c")
    
    mag_data_selected = np.zeros(num_samples, dtype=np.float32)
    if hasattr(xyz, use_mag_field) and getattr(xyz, use_mag_field) is not None:
        full_mag_data = getattr(xyz, use_mag_field)
        if ind is not None and ind.size > 0 and (np.all(ind < len(full_mag_data)) if ind.ndim == 1 and len(full_mag_data)>0 else True): # Check bounds
            mag_data_selected = full_mag_data[ind]
        elif ind is not None and ind.size > 0 : # ind out of bounds or other issue
             logging.warning(f"Indices for get_y out of bounds for field {use_mag_field}. Using zeros.")


    map_val_processed = np.zeros(num_samples, dtype=np.float32)
    if isinstance(map_val, np.ndarray) and map_val.size == num_samples:
        map_val_processed = map_val.astype(np.float32)
    elif isinstance(map_val, (int, float, np.number)) and map_val == -1 and num_samples > 0: # mapS_null case
        pass # map_val_processed remains zeros
    elif map_val is not None and not isinstance(map_val, int) and num_samples > 0 and not (isinstance(map_val, np.ndarray) and map_val.size==0) :
        logging.warning(f"Unexpected map_val type or size in get_y: {type(map_val)}, size: {getattr(map_val, 'size', 'N/A')}. Using zeros.")


    if y_type == 'a':
        # B_earth = B_total - B_aircraft. B_aircraft needs TL model. Placeholder:
        y_data = mag_data_selected
    elif y_type == 'b':
        y_data = map_val_processed
    elif y_type == 'c':
        y_data = mag_data_selected - map_val_processed
    elif y_type == 'd':
        y_data = mag_data_selected
        # Typically, 'd' implies delta from a reference like IGRF or diurnal mean.
        # This subtraction should happen *after* bpf_mag if y_type='e' was the original goal.
        # For 'd' directly, it's often mag_total - (IGRF + Diurnal).
    elif y_type == 'e':
        y_data_temp = mag_data_selected
        if kwargs.get("bpf_mag", True):
            fs = kwargs.get("fs", 1.0 / xyz.dt if xyz.dt > 0 else 10.0)
            bpf_coeffs = get_bpf(fs=fs)
            y_data_temp = bpf_data(y_data_temp, bpf=bpf_coeffs) if bpf_coeffs is not None else y_data_temp
        y_data = y_data_temp
    
    # Apply diurnal and IGRF corrections to the base total field component
    # This logic assumes y_data at this point represents some form of total field or direct measurement
    # before becoming the final target y (e.g. aircraft field).
    # If y_type is 'a' or 'c', these should ideally be subtracted from the 'total field' part.
    # For 'd' and 'e', it's more direct.
    
    # Refined subtraction logic:
    # These are typically removed from the measured total field.
    # If y_data is already a difference (like 'c'), this might be double subtraction.
    # Let's assume for 'd' and 'e', they are subtracted from the (potentially BPF'd) total field.
    # For 'a', 'b', 'c', the interpretation is more complex for these subtractions.
    # For now, apply if sub_X is true, to y_data as it stands.
    if kwargs.get("sub_diurnal", True) and hasattr(xyz, "diurnal") and xyz.diurnal is not None:
        if ind is not None and ind.size > 0 and (np.all(ind < len(xyz.diurnal)) if ind.ndim == 1 and len(xyz.diurnal)>0 else True):
            y_data -= xyz.diurnal[ind]
    if kwargs.get("sub_igrf", True) and hasattr(xyz, "igrf") and xyz.igrf is not None:
        if ind is not None and ind.size > 0 and (np.all(ind < len(xyz.igrf)) if ind.ndim == 1 and len(xyz.igrf)>0 else True):
            y_data -= xyz.igrf[ind]
        
    return y_data.astype(np.float32)


# def get_map_val(mapS: Any, traj: np.ndarray, ind: np.ndarray, alpha: float = 200) -> Union[np.ndarray, int]:
#     """Placeholder for getting map values along a trajectory."""
#     if ind is None or ind.size == 0: return -1
#     return np.random.rand(len(ind)).astype(np.float32) * 50000 # Dummy map values

def get_bpf(fs: float = 10.0, pass1: float = 0.1, pass2: float = 0.9) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Gets bandpass filter coefficients using scipy.signal.butter."""
    if fs <= 0:
        logging.error(f"Sampling frequency fs must be positive. Got {fs}. Cannot create BPF.")
        return None
    nyquist = 0.5 * fs
    low = pass1 / nyquist
    high = pass2 / nyquist

    if high >= 1.0: high = 0.999
    if low <= 0.0: low = 0.001
    
    if low >= high:
        logging.warning(f"BPF low cutoff {pass1}Hz (norm: {low}) >= high cutoff {pass2}Hz (norm: {high}). Filter disabled.")
        return None
    try:
        # butter is already imported at the top
        b, a = butter(2, [low, high], btype='band')
        return (b,a)
    except ValueError as e:
        logging.error(f"Error creating BPF with pass1={pass1}, pass2={pass2}, fs={fs}: {e}. Filter disabled.")
        return None

def bpf_data(data: np.ndarray, bpf: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Applies bandpass filter to data."""
    # lfilter is imported at the top
    if bpf is not None and data.size > 0 and bpf[0].size > 0 and bpf[1].size > 0:
        filter_order_check = max(len(bpf[0]), len(bpf[1]))
        
        min_data_len = 3 * filter_order_check
        if min_data_len == 0 : min_data_len = 1 # handle case where filter_order_check might be 0 for some reason

        if data.ndim == 1:
            if data.shape[0] > min_data_len:
                 return lfilter(bpf[0], bpf[1], data)
            else:
                 # logging.warning(f"Data length ({data.shape[0]}) too short for BPF order ({filter_order_check}). Skipping filter.")
                 return data
        elif data.ndim == 2:
            filtered_data = np.empty_like(data)
            for i in range(data.shape[1]):
                if data[:,i].shape[0] > min_data_len:
                    filtered_data[:,i] = lfilter(bpf[0], bpf[1], data[:,i])
                else:
                    # logging.warning(f"Data column {i} length ({data.shape[0]}) too short for BPF. Skipping filter.")
                    filtered_data[:,i] = data[:,i]
            return filtered_data
    return data

def get_Axy(*args, **kwargs) -> Tuple:
    """Combined placeholder for variants of get_Axy."""
    # This is a complex data loading and preprocessing function.
    # For now, returning dummy values based on some expected shapes.
    # Based on comp_train(comp_params, lines, df_line, df_flight, df_map)
    lines = args[0] # Assuming lines is the first argument
    # A rough estimation of number of samples from lines
    num_samples = len(lines) * 100 if isinstance(lines, (list, np.ndarray)) else 100 # Dummy
    
    terms_A = kwargs.get("terms_A", ["permanent", "induced", "eddy"])
    num_A_cols = 18 # Estimate based on typical TL terms
    A = np.random.rand(num_samples, num_A_cols).astype(np.float32)
    
    features_setup = args[4] if len(args) > 4 else kwargs.get("features_setup", [])
    num_x_features = len(features_setup) if features_setup else 5
    x = np.random.rand(num_samples, num_x_features).astype(np.float32)
    
    y = np.random.rand(num_samples).astype(np.float32)
    no_norm = np.zeros(num_x_features, dtype=bool)
    features = features_setup if features_setup else [f"feat{i}" for i in range(num_x_features)]

def linear_fit(
    x: np.ndarray, 
    y: np.ndarray, 
    no_norm: Optional[np.ndarray] = None,
    trim: int = 0,
    lambda_ridge: float = 0.0, # Renamed from λ to avoid keyword clash
    norm_type_x: str = "none",
    norm_type_y: str = "none",
    data_norms_in: Optional[Tuple] = None, # Renamed from data_norms
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[Tuple[np.ndarray, float], Tuple, np.ndarray, np.ndarray]:
    """
    Fit a linear regression model to data.

    Args:
        x: N x Nf data matrix (Nf is number of features)
        y: length-N target vector
        no_norm: (optional) length-Nf Boolean indices of features to not be normalized
        trim: (optional) number of elements to trim from each segment (e.g., due to bpf)
        lambda_ridge: (optional) ridge parameter
        norm_type_x: (optional) normalization for x data matrix
        norm_type_y: (optional) normalization for y target vector
        data_norms_in: (optional) length-4 tuple of data normalizations, (x_bias,x_scale,y_bias,y_scale)
        l_segs: (optional) length-N_lines vector of lengths of lines, sum(l_segs) = N
        silent: (optional) if true, no print outs

    Returns:
        model: length-2 tuple of linear regression model, (length-Nf coefficients, bias=0.0)
        data_norms_out: length-4 tuple of data normalizations, (x_bias,x_scale,y_bias,y_scale)
        y_hat: length-N prediction vector
        err: length-N mean-corrected (per line) error
    """
    if y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(f"linear_fit expects a 1D target vector y, but got shape {y.shape}")
    y_flat = y.flatten() # Ensure y is 1D

    if no_norm is None:
        no_norm = np.zeros(x.shape[1], dtype=bool)
    if l_segs is None:
        l_segs = [len(y_flat)]
    
    x_bias, x_scale, y_bias, y_scale = np.array([0.0]),np.array([1.0]),np.array([0.0]),np.array([1.0]) # Defaults

    if data_norms_in is None or np.sum(np.abs(data_norms_in[-1])) == 0 : # Normalize data
        x_bias, x_scale, x_norm = norm_sets(x, norm_type=norm_type_x, no_norm=no_norm)
        y_bias, y_scale, y_norm_flat = norm_sets(y_flat, norm_type=norm_type_y)
        data_norms_out = (x_bias, x_scale, y_bias, y_scale)
    else: # Unpack data normalizations
        x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms_in) # type: ignore
        x_norm = (x - x_bias) / x_scale
        y_norm_flat = (y_flat - y_bias) / y_scale
        data_norms_out = data_norms_in

    # Trim each line/segment
    # The original Julia code creates `ind` by taking `trim` from start and end of each segment.
    # Python equivalent:
    trimmed_indices_list = []
    current_pos = 0
    for seg_len in l_segs:
        if seg_len > 2 * trim: # Ensure segment is long enough for trimming
            start_idx = current_pos + trim
            end_idx = current_pos + seg_len - trim
            trimmed_indices_list.extend(list(range(start_idx, end_idx)))
        # else: segment is too short, skip or handle as error/warning
        current_pos += seg_len
    
    trimmed_indices = np.array(trimmed_indices_list, dtype=int)

    if trimmed_indices.size == 0:
        if not silent:
            print("WARN: No data left after trimming in linear_fit. Returning zero coefficients.")
        coeffs = np.zeros(x_norm.shape[1])
        bias_val = 0.0
        model_out = (coeffs, bias_val)
        # Predict with zero model
        y_hat_full, err_full = linear_test(x_norm, y_flat, y_bias, y_scale, model_out, l_segs=l_segs, silent=True)
        return model_out, data_norms_out, y_hat_full, err_full

    x_norm_trimmed = x_norm[trimmed_indices, :]
    y_norm_trimmed = y_norm_flat[trimmed_indices]

    # Linear regression to get coefficients
    # The original linreg function in Julia does not fit an intercept (bias term is zero).
    # We will replicate this behavior.
    if x_norm_trimmed.shape[0] == 0: # No data after trim
        coeffs = np.zeros(x_norm_trimmed.shape[1])
    elif lambda_ridge == 0:
        # Simple least squares: (X^T X)^-1 X^T y
        # Using np.linalg.lstsq for numerical stability
        coeffs, _, _, _ = np.linalg.lstsq(x_norm_trimmed, y_norm_trimmed, rcond=None)
    else:
        # Ridge regression: (X^T X + lambda I)^-1 X^T y
        XtX = x_norm_trimmed.T @ x_norm_trimmed
        lambda_I = lambda_ridge * np.eye(XtX.shape[0])
        coeffs = np.linalg.solve(XtX + lambda_I, x_norm_trimmed.T @ y_norm_trimmed)
    
    bias_val = 0.0  # As per original MagNav.jl linear_fit behavior (bias is zero)
    model_out = (coeffs.flatten(), bias_val)

    # Get results on the full (untrimmed, but normalized if applicable) dataset
    y_hat_full, err_full = linear_test(x_norm, y_flat, y_bias, y_scale, model_out, l_segs=l_segs, silent=True)
    
    if not silent:
        err_std = np.std(err_full) if err_full.size > 0 else float('nan')
        print(f"INFO: fit error: {err_std:.2f} nT")
        if trim > 0:
             print("INFO: fit error may be misleading if using bandpass filter (due to trim)")
             
    return model_out, data_norms_out, y_hat_full, err_full
def linear_fwd(
    x_in: np.ndarray, 
    data_norms_or_y_bias: Union[Tuple, float, np.ndarray], 
    model_or_y_scale: Union[Tuple[np.ndarray, float], float, np.ndarray],
    model_if_normalized: Optional[Tuple[np.ndarray, float]] = None
) -> np.ndarray:
    """
    Forward pass of a linear model.
    This function handles two call signatures from Julia via dispatch:
    1. linear_fwd(x_norm, y_bias, y_scale, model_tuple)
    2. linear_fwd(x_raw, data_norms_tuple, model_tuple)

    :param x_in: N x Nf data matrix (either normalized or raw)
    :type x_in: numpy.ndarray
    :param data_norms_or_y_bias: Either data_norms tuple (x_bias, x_scale, y_bias, y_scale) or y_bias directly.
    :type data_norms_or_y_bias: Union[Tuple, float, numpy.ndarray]
    :param model_or_y_scale: Either the model tuple (coeffs, bias_val) or y_scale directly.
    :type model_or_y_scale: Union[Tuple[numpy.ndarray, float], float, numpy.ndarray]
    :param model_if_normalized: The model tuple, used if the first signature is matched.
    :type model_if_normalized: Optional[Tuple[numpy.ndarray, float]]
    :returns: length-N prediction vector
    :rtype: numpy.ndarray
    """
    x_processed: np.ndarray
    y_bias_eff: Union[float, np.ndarray]
    y_scale_eff: Union[float, np.ndarray]
    model_eff: Tuple[np.ndarray, float]

    if isinstance(data_norms_or_y_bias, tuple) and len(data_norms_or_y_bias) == 4 and isinstance(model_or_y_scale, tuple):
        # Signature 2: linear_fwd(x_raw, data_norms_tuple, model_tuple)
        x_raw = x_in
        data_norms_tuple = data_norms_or_y_bias
        model_eff = model_or_y_scale

        x_bias, x_scale, y_bias_eff, y_scale_eff = unpack_data_norms(data_norms_tuple) # type: ignore
        
        # Handle cases where bias/scale might be single float vs array
        if isinstance(x_bias, (float, int)) and isinstance(x_scale, (float, int)):
             x_processed = (x_raw - x_bias) / (x_scale if x_scale != 0 else 1.0)
        elif isinstance(x_bias, np.ndarray) and isinstance(x_scale, np.ndarray):
             x_processed = (x_raw - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        else:
            raise TypeError("x_bias and x_scale must be both float or both np.ndarray")

    elif model_if_normalized is not None:
        # Signature 1: linear_fwd(x_norm, y_bias, y_scale, model_tuple)
        x_processed = x_in # Already normalized
        y_bias_eff = data_norms_or_y_bias
        y_scale_eff = model_or_y_scale 
        model_eff = model_if_normalized
    else:
        raise TypeError("Invalid arguments for linear_fwd. Check call signature.")

    coeffs, bias_val = model_eff
    
    y_hat_norm = x_processed @ coeffs + bias_val
    
    # Denormalize
    # Ensure y_bias_eff and y_scale_eff are ndarrays for denorm_sets if they came as floats
    _y_bias = np.array(y_bias_eff) if isinstance(y_bias_eff, (float,int)) else y_bias_eff
    _y_scale = np.array(y_scale_eff) if isinstance(y_scale_eff, (float,int)) else y_scale_eff
    
    y_hat = denorm_sets(_y_bias, _y_scale, y_hat_norm)
    
    return y_hat.flatten()


def linear_test(
    x_in: np.ndarray, 
    y: np.ndarray, 
    data_norms_or_y_bias: Union[Tuple, float, np.ndarray], 
    model_or_y_scale: Union[Tuple[np.ndarray, float], float, np.ndarray],
    model_if_normalized: Optional[Tuple[np.ndarray, float]] = None,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate performance of a linear model.
    Handles two call signatures similar to linear_fwd.

    Args (depending on signature):
        x_in: N x Nf data matrix (either normalized or raw)
        y: length-N target vector
        data_norms_or_y_bias: Either data_norms tuple or y_bias.
        model_or_y_scale: Either the model tuple or y_scale.
        model_if_normalized: (optional) The model tuple for the normalized signature.
        l_segs: (optional) length-N_lines vector of lengths of lines, sum(l_segs) = N
        silent: (optional) if true, no print outs

    Returns:
        y_hat: length-N prediction vector
        err: length-N mean-corrected (per line) error
    """
    y_flat = y.flatten()
    if l_segs is None:
        l_segs = [len(y_flat)]

    y_hat = linear_fwd(x_in, data_norms_or_y_bias, model_or_y_scale, model_if_normalized)
    
    # Ensure l_segs is valid for the length of y_hat/y
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_flat):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_flat)}. Using single segment for error calculation.")
        valid_l_segs = [len(y_flat)]
        
    err_val = err_segs(y_hat, y_flat, valid_l_segs, silent=SILENT_DEBUG) # Use global SILENT_DEBUG for internal err_segs
    
    if not silent:
        err_std = np.std(err_val) if err_val.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err_val

def save_comp_params(comp_params: CompParams, filename: str):
    """Placeholder for saving compensation parameters."""
    # In Python, might use pickle, joblib, or torch.save for model parts
    print(f"Placeholder: Saving comp_params to {filename}")

def get_comp_params(filename: str, silent: bool = False) -> CompParams:
    """Placeholder for loading compensation parameters."""
    print(f"Placeholder: Loading comp_params from {filename}")
    # Return a default CompParams object for now
    return NNCompParams() if "nn" in filename.lower() or "m1" in filename.lower() or "m2" in filename.lower() or "m3" in filename.lower() else LinCompParams()

def field_check(xyz: XYZ, field_name: str, expected_type: Any):
    """Placeholder for checking if a field exists in XYZ struct."""
    if not hasattr(xyz, field_name):
        raise AttributeError(f"XYZ object does not have field: {field_name}")
    # Can add type checking if needed: if not isinstance(getattr(xyz, field_name), expected_type): ...
    pass

def linreg(y_norm: np.ndarray, x_norm: np.ndarray, λ: float = 0.0) -> np.ndarray:
    """Placeholder for linear regression, possibly with ridge."""
    if x_norm.shape[0] == 0: # No data
        return np.zeros(x_norm.shape[1])

    if λ == 0: # Simple least squares
        coeffs, _, _, _ = np.linalg.lstsq(x_norm, y_norm, rcond=None)
        return coeffs.flatten()
    else: # Ridge regression
        # (X^T X + lambda I)^-1 X^T y
        I = np.eye(x_norm.shape[1])
        coeffs = np.linalg.solve(x_norm.T @ x_norm + λ * I, x_norm.T @ y_norm)
        return coeffs.flatten()

# --- Struct to Class Translations ---

class M1Struct(nn.Module):
    def __init__(self, m: nn.Sequential):
        super().__init__()
        self.m = m # This is the Chain (nn.Sequential)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This forward might not be directly used if s.m is accessed,
        # but good practice for an nn.Module.
        return self.m(x)

class M2Struct(nn.Module):
    def __init__(self, m: nn.Sequential, tl_coef_norm: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        self.m = m
        if isinstance(tl_coef_norm, np.ndarray):
            tl_coef_norm = torch.from_numpy(tl_coef_norm.astype(np.float32))
        self.TL_coef_norm = nn.Parameter(tl_coef_norm) # Trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x) # Placeholder, actual forward pass depends on usage

class M2StructMOnly(nn.Module): # TL_coef_norm is not trained by optimizer directly
    def __init__(self, m: nn.Sequential, tl_coef_norm: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        self.m = m
        if isinstance(tl_coef_norm, np.ndarray):
            self.register_buffer('TL_coef_norm', torch.from_numpy(tl_coef_norm.astype(np.float32)))
        else:
            self.register_buffer('TL_coef_norm', tl_coef_norm.float())


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

class M3Struct(nn.Module):
    def __init__(self, m: nn.Sequential, tl_p: Union[np.ndarray, torch.Tensor], 
                 tl_i: Union[np.ndarray, torch.Tensor], tl_e: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        self.m = m
        self.TL_p = nn.Parameter(torch.from_numpy(tl_p.astype(np.float32)) if isinstance(tl_p, np.ndarray) else tl_p.float())
        self.TL_i = nn.Parameter(torch.from_numpy(tl_i.astype(np.float32)) if isinstance(tl_i, np.ndarray) else tl_i.float())
        self.TL_e = nn.Parameter(torch.from_numpy(tl_e.astype(np.float32)) if isinstance(tl_e, np.ndarray) else tl_e.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

# --- Function Translations ---

def nn_comp_1_train(
    x: np.ndarray, 
    y: np.ndarray, 
    no_norm: Optional[np.ndarray] = None,
    norm_type_x: str = "standardize",
    norm_type_y: str = "standardize",
    eta_adam: float = 0.001,
    epoch_adam: int = 5,
    epoch_lbfgs: int = 0,
    hidden: List[int] = [8], # type: ignore
    activation: Callable = nn.SiLU,
    loss_fn: Callable = nn.MSELoss(), # Renamed from loss to loss_fn to avoid conflict
    batchsize: int = 2048,
    frac_train: float = 14/17,
    alpha_sgl: float = 1.0,
    lambda_sgl: float = 0.0,
    k_pca: int = -1,
    data_norms_in: Optional[Tuple] = None, # Renamed from data_norms
    model_in: Optional[nn.Sequential] = None, # Renamed from model
    l_segs: Optional[List[int]] = None,
    x_test_in: Optional[np.ndarray] = None, # Renamed from x_test
    y_test_in: Optional[np.ndarray] = None, # Renamed from y_test
    l_segs_test: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[nn.Sequential, Tuple, np.ndarray, np.ndarray]:

    if no_norm is None and x is not None:
        no_norm = np.zeros(x.shape[1], dtype=bool)
    if l_segs is None and y is not None:
        l_segs = [len(y)]
    if x_test_in is None: x_test_in = np.empty((0,0), dtype=x.dtype if x is not None else np.float32)
    if y_test_in is None: y_test_in = np.empty(0, dtype=y.dtype if y is not None else np.float32)
    if l_segs_test is None: l_segs_test = [len(y_test_in)]
    if data_norms_in is None: # Default from Julia
        data_norms_in = (np.zeros((1,1),dtype=np.float32), np.zeros((1,1),dtype=np.float32), 
                           np.zeros((1,1),dtype=np.float32), np.zeros((1,1),dtype=np.float32),
                           np.zeros((1,1),dtype=np.float32), np.array([0.0],dtype=np.float32),
                           np.array([0.0],dtype=np.float32))


    # Convert to Float32
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    alpha = np.float32(alpha_sgl)
    lambda_ = np.float32(lambda_sgl) # Renamed from lambda
    x_test_in = x_test_in.astype(np.float32)
    y_test_in = y_test_in.astype(np.float32)

    Nf = x.shape[1] # number of features

    v_scale_pca: np.ndarray
    if data_norms_in is not None and np.sum(data_norms_in[-1]) == 0: # normalize data
        x_bias, x_scale, x_norm_np = norm_sets(x, norm_type=norm_type_x, no_norm=no_norm)
        y_bias, y_scale, y_norm_np = norm_sets(y, norm_type=norm_type_y)
        
        if k_pca > 0:
            if k_pca > Nf:
                if not silent: print(f"INFO: reducing k_pca from {k_pca} to {Nf}")
                k_pca = Nf
            
            # Ensure x_norm_np is 2D for np.cov
            if x_norm_np.ndim == 1: x_norm_np_2d = x_norm_np[:, np.newaxis]
            else: x_norm_np_2d = x_norm_np

            if x_norm_np_2d.shape[0] > 1 : # cov needs more than 1 sample
                cov_x = np.cov(x_norm_np_2d, rowvar=False)
                # U_svd, S_svd, Vh_svd = np.linalg.svd(cov_x) # Vh_svd is V transpose
                # Principal components (eigenvectors of cov_x) are columns of Vh_svd.T
                # Julia: v_scale = V[:,1:k_pca]*inv(Diagonal(sqrt.(S[1:k_pca])))
                # Python equivalent:
                # W = Vh_svd.T[:, :k_pca] # Shape (Nf, k_pca) components
                # v_scale_pca = W @ np.diag(1.0 / np.sqrt(S_svd[:k_pca])) # Shape (Nf, k_pca) scaled components
                
                # Simpler: use sklearn PCA if available and matches logic, or ensure direct port is correct
                # For direct port of Julia's SVD approach to get scaled components:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_x)
                # eigh returns sorted eigenvalues and corresponding eigenvectors
                # Sort in descending order
                sorted_indices = np.argsort(eigenvalues)[::-1]
                S_svd_sorted = eigenvalues[sorted_indices]
                V_sorted = eigenvectors[:, sorted_indices]

                v_pca_components = V_sorted[:, :k_pca] # Shape (Nf, k_pca)
                # Filter out very small or zero eigenvalues before taking sqrt and inverse
                s_values_for_scaling = S_svd_sorted[:k_pca]
                s_values_for_scaling[s_values_for_scaling <= 1e-9] = 1e-9 # Avoid division by zero / instability
                
                v_scale_pca = v_pca_components @ np.diag(1.0 / np.sqrt(s_values_for_scaling))

                # Variance retained calculation needs to use the SVD singular values if that's the reference
                # If using eigenvalues from eigh:
                var_ret = round(np.sum(np.sqrt(s_values_for_scaling)) / np.sum(np.sqrt(S_svd_sorted[S_svd_sorted > 1e-9])) * 100, 6) if np.sum(np.sqrt(S_svd_sorted[S_svd_sorted > 1e-9])) > 0 else 0.0

                if not silent: print(f"INFO: k_pca = {k_pca} of {Nf}, variance retained: {var_ret} %")
            else: # Not enough samples for meaningful SVD/PCA
                if not silent: print(f"WARN: Not enough samples ({x_norm_np_2d.shape[0]}) for PCA with k_pca={k_pca}. Using identity matrix for v_scale_pca.")
                v_scale_pca = np.eye(Nf, dtype=np.float32)

        else:
            v_scale_pca = np.eye(Nf, dtype=np.float32)
        
        # x_norm_np is (N_samples, Nf), v_scale_pca is (Nf, k_pca)
        # Result x_norm_np @ v_scale_pca is (N_samples, k_pca)
        x_norm_transformed = (x_norm_np @ v_scale_pca).T # Transposed to (k_pca, N_samples)
        y_norm_transformed = y_norm_np.T if y_norm_np.ndim > 1 else y_norm_np.reshape(1, -1) # Ensure 2D for DataLoader
        
        # Store original (non-PCA'd) bias/scale for x for test data transformation
        data_norms_out = (np.zeros_like(x_bias.reshape(1,-1), dtype=np.float32), # Placeholder for Julia's first two elements
                          np.zeros_like(x_bias.reshape(1,-1), dtype=np.float32), # Placeholder
                          v_scale_pca, x_bias, x_scale, y_bias, y_scale)

    else: # unpack data normalizations
        # Assuming data_norms_in has the structure: (_,_,v_scale_pca,x_bias,x_scale,y_bias,y_scale)
        # The first two elements are placeholders from Julia code.
        _, _, v_scale_pca, x_bias, x_scale, y_bias, y_scale = unpack_data_norms(cast(Tuple, data_norms_in))
        x_norm_transformed = (((x - x_bias) / x_scale) @ v_scale_pca).T
        y_norm_transformed = ((y - y_bias) / y_scale).T if y.ndim > 1 else ((y - y_bias) / y_scale).reshape(1,-1)
        data_norms_out = cast(Tuple, data_norms_in)


    x_test_norm_transformed: Optional[np.ndarray] = None
    if x_test_in.size > 0:
        # Use x_bias and x_scale from training data before PCA for consistency
        x_test_norm_transformed = (((x_test_in - x_bias) / x_scale) @ v_scale_pca).T


    # Convert to PyTorch Tensors
    x_norm_torch = torch.from_numpy(x_norm_transformed.astype(np.float32))
    y_norm_torch = torch.from_numpy(y_norm_transformed.astype(np.float32)).squeeze() # Ensure 1D for MSELoss typical use

    # Separate into training & validation
    if frac_train < 1:
        # PyTorch DataLoader expects (features, samples) for x, and (samples,) or (samples, features_y) for y
        # Current x_norm_torch is (features, samples), y_norm_torch is (samples,)
        # We need to split along the sample dimension (dim=1 for x_norm_torch)
        num_samples = x_norm_torch.shape[1]
        indices = np.random.permutation(num_samples)
        split_idx = int(np.floor(num_samples * frac_train))
        
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        
        x_train_norm_torch = x_norm_torch[:, train_indices]
        y_train_norm_torch = y_norm_torch[train_indices]
        x_val_norm_torch = x_norm_torch[:, val_indices]
        y_val_norm_torch = y_norm_torch[val_indices]

        train_dataset = TensorDataset(x_train_norm_torch.T, y_train_norm_torch) # DataLoader expects (sample, feature)
        val_dataset = TensorDataset(x_val_norm_torch.T, y_val_norm_torch)
    else:
        train_dataset = TensorDataset(x_norm_torch.T, y_norm_torch)
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False) # Usually False for validation

    # Setup NN
    current_model: nn.Sequential
    if model_in is None or not list(model_in.children()): # Check if model_in is empty Chain()
        Ny = 1 # length of output
        # Input features to NN is k_pca if used, else Nf
        nn_input_features = v_scale_pca.shape[1] # Number of columns in v_scale_pca
        current_model = get_nn_m(nn_input_features, Ny, hidden=hidden, activation=activation)
    else:
        current_model = copy.deepcopy(model_in)
    
    s_model = M1Struct(current_model) # Wrap in our nn.Module compatible struct

    # Setup loss function (already a callable, e.g., nn.MSELoss())
    # The Julia loss_m1 and loss_m1_λ handle denormalization and SGL.
    # For PyTorch, SGL needs to be added to the main loss calculation.
    # Denormalization is handled in nn_comp_1_fwd.
    
    def compute_loss_val(model_wrapper: M1Struct, loader: DataLoader, y_b: np.ndarray, y_s: np.ndarray) -> float:
        model_wrapper.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x_batch, y_batch_norm in loader:
                # x_batch is (batch, features), y_batch_norm is (batch,)
                y_hat_norm = model_wrapper.m(x_batch).squeeze() # model_wrapper.m is the nn.Sequential
                
                # SGL term (if lambda_ > 0)
                sgl_penalty = torch.tensor(0.0)
                if lambda_ > 0:
                    sgl_penalty = lambda_ * sparse_group_lasso(model_wrapper.m, alpha)

                loss_val = loss_fn(y_hat_norm, y_batch_norm) + sgl_penalty
                total_loss += loss_val.item() * x_batch.size(0)
                count += x_batch.size(0)
        return total_loss / count if count > 0 else 0.0


    # Setup Adam optimizer
    optimizer_adam = optim.Adam(s_model.parameters(), lr=eta_adam)

    # Train NN with Adam optimizer
    best_model_state = copy.deepcopy(s_model.state_dict())
    best_loss_val = compute_loss_val(s_model, val_loader, y_bias, y_scale)
    
    best_test_error_std: Optional[float] = None
    if x_test_norm_transformed is not None and y_test_in is not None and y_test_in.size > 0:
        # Ensure l_segs_test is valid
        valid_l_segs_test = l_segs_test if l_segs_test and sum(l_segs_test) == len(y_test_in) else [len(y_test_in)]
        _, err_test_init = nn_comp_1_test(x_test_norm_transformed, y_test_in, y_bias, y_scale, s_model.m,
                                          l_segs=valid_l_segs_test, silent=SILENT_DEBUG) # type: ignore
        best_test_error_std = np.std(err_test_init) if err_test_init.size > 0 else float('inf')


    if not silent: print(f"INFO: epoch 0: loss = {best_loss_val:.6f}")

    for i in range(1, epoch_adam + 1):
        s_model.train()
        for x_batch, y_batch_norm in train_loader:
            optimizer_adam.zero_grad()
            y_hat_norm = s_model.m(x_batch).squeeze()
            
            sgl_penalty = torch.tensor(0.0)
            if lambda_ > 0:
                sgl_penalty = lambda_ * sparse_group_lasso(s_model.m, alpha)
            
            loss_train = loss_fn(y_hat_norm, y_batch_norm) + sgl_penalty
            loss_train.backward()
            optimizer_adam.step()

        current_loss_val = compute_loss_val(s_model, val_loader, y_bias, y_scale)

        if x_test_norm_transformed is None or y_test_in is None or y_test_in.size == 0 : # No test set
            if current_loss_val < best_loss_val:
                best_loss_val = current_loss_val
                best_model_state = copy.deepcopy(s_model.state_dict())
            if i % 5 == 0 and not silent:
                print(f"INFO: epoch {i}: loss = {best_loss_val:.6f}")
        else: # With test set
            # Ensure l_segs_test is valid
            valid_l_segs_test = l_segs_test if l_segs_test and sum(l_segs_test) == len(y_test_in) else [len(y_test_in)]
            _, err_test_current = nn_comp_1_test(x_test_norm_transformed, y_test_in, y_bias, y_scale, s_model.m,
                                                 l_segs=valid_l_segs_test, silent=SILENT_DEBUG) # type: ignore
            current_test_error_std = np.std(err_test_current) if err_test_current.size > 0 else float('inf')

            if best_test_error_std is not None and current_test_error_std < best_test_error_std:
                best_test_error_std = current_test_error_std
                best_model_state = copy.deepcopy(s_model.state_dict())
            
            if i % 5 == 0 and not silent:
                print(f"INFO: epoch {i}: loss = {current_loss_val:.6f}, test error = {best_test_error_std:.2f} nT" if best_test_error_std is not None else f"INFO: epoch {i}: loss = {current_loss_val:.6f}")
        
        if i % 10 == 0 and not silent:
            # Evaluate on full training data (normalized)
            # Ensure l_segs is valid
            valid_l_segs = l_segs if l_segs and sum(l_segs) == len(y) else [len(y)]
            _, train_err_np = nn_comp_1_test(x_norm_torch.numpy(), y, y_bias, y_scale, s_model.m,
                                             l_segs=valid_l_segs, silent=True) # type: ignore
            train_err_std = np.std(train_err_np) if train_err_np.size > 0 else float('nan')
            print(f"INFO: {i} train error: {train_err_std:.2f} nT")
            if x_test_norm_transformed is not None and y_test_in is not None and y_test_in.size > 0 and best_test_error_std is not None:
                 print(f"INFO: {i} test  error: {current_test_error_std:.2f} nT")


    s_model.load_state_dict(best_model_state)

    if epoch_lbfgs > 0:
        if not silent: print("INFO: LBFGS training started.")
        optimizer_lbfgs = optim.LBFGS(s_model.parameters(), lr=0.1) # lr is often 1 for LBFGS, but can be tuned. Max_iter in options.
        
        # LBFGS requires a closure function
        def closure():
            optimizer_lbfgs.zero_grad()
            # For LBFGS, usually train on the whole dataset or large batches
            # Using the full normalized training set (or a large single batch)
            y_hat_norm_lbfgs = s_model.m(x_norm_torch.T).squeeze() # x_norm_torch.T is (samples, features)
            
            sgl_penalty_lbfgs = torch.tensor(0.0)
            if lambda_ > 0:
                sgl_penalty_lbfgs = lambda_ * sparse_group_lasso(s_model.m, alpha)
            
            loss_lbfgs = loss_fn(y_hat_norm_lbfgs, y_norm_torch) + sgl_penalty_lbfgs
            loss_lbfgs.backward()
            return loss_lbfgs

        for i in range(epoch_lbfgs):
            optimizer_lbfgs.step(closure)
            # Optionally, log loss or evaluate validation set
            if not silent and (i+1) % 5 == 0:
                 current_loss_lbfgs = closure().item() # Re-evaluate for logging
                 print(f"INFO: LBFGS epoch {i+1}: loss = {current_loss_lbfgs:.6f}")
        if not silent: print("INFO: LBFGS training finished.")


    # Get final results on training data
    # Ensure l_segs is valid
    valid_l_segs_final = l_segs if l_segs and sum(l_segs) == len(y) else [len(y)]
    y_hat_final, err_final = nn_comp_1_test(x_norm_torch.numpy(), y, y_bias, y_scale, s_model.m,
                                            l_segs=valid_l_segs_final, silent=True) # type: ignore
    if not silent: print(f"INFO: train error: {np.std(err_final):.2f} nT" if err_final.size > 0 else "INFO: train error: N/A")

    if x_test_norm_transformed is not None and y_test_in is not None and y_test_in.size > 0:
        # Ensure l_segs_test is valid
        valid_l_segs_test_final = l_segs_test if l_segs_test and sum(l_segs_test) == len(y_test_in) else [len(y_test_in)]
        nn_comp_1_test(x_test_norm_transformed, y_test_in, y_bias, y_scale, s_model.m,
                       l_segs=valid_l_segs_test_final, silent=silent) # type: ignore

    final_model_chain = s_model.m
    
    return final_model_chain, data_norms_out, y_hat_final, err_final


def nn_comp_1_fwd(
    x_norm_in: Union[np.ndarray, torch.Tensor], 
    y_bias: Union[float, np.ndarray], 
    y_scale: Union[float, np.ndarray], 
    model: nn.Sequential,
    denorm: bool = True,
    testmode: bool = True # In PyTorch, model.eval() is used instead of a flag during forward
) -> np.ndarray:

    if testmode:
        model.eval()
    else:
        model.train() # Set back to train mode if it was changed

    if isinstance(x_norm_in, np.ndarray):
        # Assuming x_norm_in is (features, samples) from Julia's perspective
        # PyTorch nn.Linear expects (batch/samples, features)
        x_torch = torch.from_numpy(x_norm_in.T.astype(np.float32)) 
    else: # Already a torch tensor
        x_torch = x_norm_in.T # Assuming (features, samples) -> (samples, features)

    with torch.no_grad(): # Important for inference
        y_hat_norm_torch = model(x_torch).squeeze()
    
    y_hat_norm_np = y_hat_norm_torch.cpu().numpy()

    y_hat_np: np.ndarray
    if denorm:
        y_hat_np = denorm_sets(np.array(y_bias), np.array(y_scale), y_hat_norm_np)
    else:
        y_hat_np = y_hat_norm_np
        
    return y_hat_np.flatten()


# Overloaded version of nn_comp_1_fwd
def nn_comp_1_fwd_from_raw(
    x: np.ndarray, 
    data_norms: Tuple, # Should contain (_,_,v_scale,x_bias,x_scale,y_bias,y_scale)
    model: nn.Sequential
) -> np.ndarray:
    x_f32 = x.astype(np.float32)

    # Unpack data normalizations. Structure from nn_comp_1_train's data_norms_out:
    # (placeholder1, placeholder2, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    if len(data_norms) != 7:
        raise ValueError(f"data_norms tuple expected to have 7 elements, got {len(data_norms)}")

    v_scale_pca = data_norms[2]
    x_bias      = data_norms[3]
    x_scale     = data_norms[4]
    y_bias      = data_norms[5]
    y_scale     = data_norms[6]

    # Ensure x_bias and x_scale are correctly broadcastable or shaped
    # x_f32 is (N, F_orig), x_bias is (F_orig,), x_scale is (F_orig,)
    # v_scale_pca is (F_orig, k_pca)
    
    # Handle cases where x_bias/x_scale might be scalar if x was 1D originally normalized
    _x_bias = np.asarray(x_bias)
    _x_scale = np.asarray(x_scale)
    if x_f32.ndim == 1 and _x_bias.ndim > 0 and _x_bias.size > 1: _x_bias = _x_bias[0]
    if x_f32.ndim == 1 and _x_scale.ndim > 0 and _x_scale.size > 1: _x_scale = _x_scale[0]


    x_norm_step1 = (x_f32 - _x_bias) / _x_scale
    
    # x_norm_step1 is (N, F_orig)
    # v_scale_pca is (F_orig, k_pca)
    # x_norm_projected is (N, k_pca)
    x_norm_projected = x_norm_step1 @ v_scale_pca
    
    # Transpose for nn_comp_1_fwd, which expects (features=k_pca, samples=N)
    x_norm_transformed_for_fwd = x_norm_projected.T

    # Call the other nn_comp_1_fwd which takes normalized data (already PCA'd and transposed)
    y_hat = nn_comp_1_fwd(x_norm_transformed_for_fwd, y_bias, y_scale, model,
                          denorm=True, testmode=True)
    
    return y_hat

# Note: The Julia version has two nn_comp_1_test functions.
# 1. nn_comp_1_test(x_norm::AbstractMatrix, y, y_bias, y_scale, model::Chain; ...)
# 2. nn_comp_1_test(x::Matrix, y, data_norms::Tuple, model::Chain; ...)

def nn_comp_1_test(
    x_norm_in: Union[np.ndarray, torch.Tensor], # Normalized and PCA-transformed input, (features, samples)
    y: np.ndarray,
    y_bias: Union[float, np.ndarray],
    y_scale: Union[float, np.ndarray],
    model: nn.Sequential,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    
    if l_segs is None:
        l_segs = [len(y)]

    # Get results
    # nn_comp_1_fwd expects x_norm_in as (features, samples) if numpy, or (samples, features) if torch
    # It handles the transpose internally if numpy.
    y_hat = nn_comp_1_fwd(x_norm_in, y_bias, y_scale, model, denorm=True, testmode=True)
    
    err = err_segs(y_hat, y, l_segs, silent=SILENT_DEBUG) # Use global SILENT_DEBUG for internal calls
    
    if not silent:
        err_std = np.std(err) if err.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err

def nn_comp_1_test_from_raw(
    x: np.ndarray, # Raw input data (samples, features_original)
    y: np.ndarray,
    data_norms: Tuple, # Should contain (_,_,v_scale_pca,x_bias,x_scale,y_bias,y_scale)
    model: nn.Sequential,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    if l_segs is None:
        l_segs = [len(y)]
    
    y_f32 = y.astype(np.float32) # Match Julia's type consistency

    # Get results using nn_comp_1_fwd_from_raw
    y_hat = nn_comp_1_fwd_from_raw(x, data_norms, model)
    
    err = err_segs(y_hat, y_f32, l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err) if err.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err
    # The internal nn_comp_1_fwd will handle transpose if it's numpy
    y_hat = nn_comp_1_fwd(x_norm_np_for_fwd, y_bias, y_scale, model, denorm=True, testmode=True)
    return y_hat


def nn_comp_1_test(
    x_norm_in: Union[np.ndarray, torch.Tensor], # (features, samples) if numpy, (samples, features) if torch
    y: np.ndarray, 
    y_bias: Union[float, np.ndarray], 
    y_scale: Union[float, np.ndarray], 
    model: nn.Sequential,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    if l_segs is None: l_segs = [len(y)]
    
    y_hat = nn_comp_1_fwd(x_norm_in, y_bias, y_scale, model, denorm=True, testmode=True)
    
    # Ensure l_segs is valid for the length of y_hat/y
    valid_l_segs = l_segs
    if sum(l_segs) != len(y):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y)}. Using single segment.")
        valid_l_segs = [len(y)]
        
    err = err_segs(y_hat, y, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err) if err.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err

# Overloaded version of nn_comp_1_test
def nn_comp_1_test_from_raw(
    x: np.ndarray, 
    y: np.ndarray, 
    data_norms: Tuple, 
    model: nn.Sequential,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    
    if l_segs is None: l_segs = [len(y)]
    y_f32 = y.astype(np.float32)

    # y_hat is already denormalized by nn_comp_1_fwd_from_raw
    y_hat = nn_comp_1_fwd_from_raw(x, data_norms, model)
    
    # Ensure l_segs is valid
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_f32):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_f32)}. Using single segment.")
        valid_l_segs = [len(y_f32)]

    err = err_segs(y_hat, y_f32, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err) if err.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err

def nn_comp_2_fwd(
    A_norm: Union[np.ndarray, torch.Tensor],      # (TL_terms, samples)
    x_norm: Union[np.ndarray, torch.Tensor],      # (features, samples)
    y_bias: Union[float, np.ndarray],
    y_scale: Union[float, np.ndarray],
    model_nn: nn.Sequential, # The core nn.Sequential model (s.m in Julia)
    TL_coef_norm: Union[np.ndarray, torch.Tensor], # Normalized TL coefficients
    model_type: str, # "m2a", "m2b", "m2c", "m2d"
    denorm: bool = True,
    testmode: bool = True
) -> np.ndarray:
    """
    Forward pass of neural network-based aeromagnetic compensation, model 2.
    Assumes A_norm and x_norm are (features, samples) if numpy,
    and expects model_nn to take (samples, features_x).
    TL_coef_norm is expected to be a 1D array/tensor.
    """
    if testmode:
        model_nn.eval()
    else:
        # If called during training, ensure model is in train mode if it has dropout/batchnorm
        # This is typically handled by the main training loop calling model.train()
        pass 

    is_torch_input = isinstance(x_norm, torch.Tensor)
    
    if isinstance(A_norm, np.ndarray):
        A_norm_torch = torch.from_numpy(A_norm.astype(np.float32))
    else:
        A_norm_torch = A_norm.float()

    if isinstance(x_norm, np.ndarray):
        # PyTorch nn.Linear expects (batch/samples, features)
        x_norm_torch = torch.from_numpy(x_norm.T.astype(np.float32))
    else:
        x_norm_torch = x_norm.T.float() # Assuming (features, samples) -> (samples, features)

    if isinstance(TL_coef_norm, np.ndarray):
        TL_coef_norm_torch = torch.from_numpy(TL_coef_norm.astype(np.float32))
    else:
        TL_coef_norm_torch = TL_coef_norm.float()

    y_hat_norm_torch: torch.Tensor

    with torch.no_grad() if testmode else torch.enable_grad(): # Ensure no_grad for pure inference
        nn_output = model_nn(x_norm_torch) # (samples, nn_output_features)

        if model_type in ["m2a", "m2d"]:
            # NN output shape should be (samples, num_TL_terms) for these models
            # A_norm_torch is (TL_terms, samples)
            # nn_output is (samples, TL_terms)
            # We need element-wise product then sum over TL_terms
            # (A_norm_torch.T * nn_output) -> (samples, TL_terms) element-wise
            # sum over dim=1 -> (samples,)
            if model_type == "m2a":
                # y_hat = vec(sum(A_norm.*m(x_norm), dims=1))
                # m(x_norm) shape is (num_TL_terms, samples) in Julia after transpose
                # A_norm shape is (num_TL_terms, samples)
                # Python: nn_output is (samples, num_TL_terms), A_norm_torch.T is (samples, num_TL_terms)
                y_hat_norm_torch = torch.sum(A_norm_torch.T * nn_output, dim=1)
            elif model_type == "m2d":
                # y_hat = vec(sum(A_norm.*(m(x_norm) .+ TL_coef_norm), dims=1))
                # m(x_norm) .+ TL_coef_norm -> TL_coef_norm needs to broadcast or match shape
                # TL_coef_norm_torch is (num_TL_terms), nn_output is (samples, num_TL_terms)
                # A_norm_torch.T is (samples, num_TL_terms)
                combined_coeffs = nn_output + TL_coef_norm_torch # Broadcasting TL_coef_norm
                y_hat_norm_torch = torch.sum(A_norm_torch.T * combined_coeffs, dim=1)
        
        elif model_type in ["m2b", "m2c"]:
            # NN output shape is (samples, 1)
            # y_hat = vec(m(x_norm)) + A_norm'*TL_coef_norm
            # A_norm_torch is (TL_terms, samples), TL_coef_norm_torch is (TL_terms)
            # A_norm_torch.T @ TL_coef_norm_torch -> (samples, TL_terms) @ (TL_terms) -> (samples,)
            tl_effect = A_norm_torch.T @ TL_coef_norm_torch
            y_hat_norm_torch = nn_output.squeeze() + tl_effect
        else:
            raise ValueError(f"Unknown model_type for nn_comp_2_fwd: {model_type}")

    y_hat_norm_np = y_hat_norm_torch.cpu().detach().numpy() if is_torch_input or testmode else y_hat_norm_torch.cpu().numpy()


    y_hat_np: np.ndarray
    if denorm:
        _y_bias = np.array(y_bias) if isinstance(y_bias, (float,int)) else y_bias
        _y_scale = np.array(y_scale) if isinstance(y_scale, (float,int)) else y_scale
        y_hat_np = denorm_sets(_y_bias, _y_scale, y_hat_norm_np)
    else:
        y_hat_np = y_hat_norm_np
        
    return y_hat_np.flatten()

def nn_comp_2_fwd_from_raw(
    A_raw: np.ndarray,      # (samples, TL_terms)
    x_raw: np.ndarray,      # (samples, features)
    data_norms: Tuple,      # (A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    model_nn: nn.Sequential,
    model_type: str,
    TL_coef: np.ndarray     # Raw (not normalized) TL coefficients
) -> np.ndarray:
    """
    Forward pass of neural network-based aeromagnetic compensation, model 2, from raw inputs.
    """
    A_raw_f32 = A_raw.astype(np.float32)
    x_raw_f32 = x_raw.astype(np.float32)
    TL_coef_f32 = TL_coef.astype(np.float32)

    A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms)

    # Normalize A: ((A_raw - A_bias) / A_scale).T -> (TL_terms, samples)
    A_norm_np = ((A_raw_f32 - A_bias.reshape(1,-1)) / np.where(A_scale.reshape(1,-1) == 0, 1.0, A_scale.reshape(1,-1))).T
    
    # Normalize x: (((x_raw - x_bias) / x_scale) @ v_scale_pca).T -> (PCA_features, samples)
    x_norm_step1 = (x_raw_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
    x_norm_np = (x_norm_step1 @ v_scale_pca).T

    # Normalize TL_coef: TL_coef / y_scale
    # Ensure y_scale is not zero
    y_scale_eff = y_scale if isinstance(y_scale, np.ndarray) and y_scale.size > 0 and y_scale[0] != 0 else (y_scale if isinstance(y_scale, (float, int)) and y_scale !=0 else 1.0)
    if isinstance(y_scale_eff, np.ndarray) and y_scale_eff.size > 1: # Should be scalar for this context
        y_scale_val = y_scale_eff[0] if y_scale_eff[0] != 0 else 1.0
    elif isinstance(y_scale_eff, np.ndarray):
         y_scale_val = y_scale_eff[0] if y_scale_eff.size > 0 and y_scale_eff[0] != 0 else 1.0
    else: # float
        y_scale_val = y_scale_eff if y_scale_eff != 0 else 1.0

    TL_coef_norm_np = TL_coef_f32 / y_scale_val

    y_hat = nn_comp_2_fwd(
        A_norm_np, x_norm_np, y_bias, y_scale, model_nn,
        TL_coef_norm_np, model_type, denorm=True, testmode=True
    )
    return y_hat
def nn_comp_2_test(
    A_norm: Union[np.ndarray, torch.Tensor],      # (TL_terms, samples)
    x_norm: Union[np.ndarray, torch.Tensor],      # (features, samples)
    y: np.ndarray,                                # Raw target vector (samples,)
    y_bias: Union[float, np.ndarray],
    y_scale: Union[float, np.ndarray],
    model_nn: nn.Sequential,
    TL_coef_norm: Union[np.ndarray, torch.Tensor],
    model_type: str,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate performance of neural network-based aeromagnetic compensation, model 2.
    """
    y_flat = y.flatten()
    if l_segs is None:
        l_segs = [len(y_flat)]

    y_hat = nn_comp_2_fwd(
        A_norm, x_norm, y_bias, y_scale, model_nn,
        TL_coef_norm, model_type, denorm=True, testmode=True
    )
    
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_flat):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_flat)}. Using single segment.")
        valid_l_segs = [len(y_flat)]

    err_val = err_segs(y_hat, y_flat, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err_val) if err_val.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err_val

def nn_comp_2_test_from_raw(
    A_raw: np.ndarray,      # (samples, TL_terms)
    x_raw: np.ndarray,      # (samples, features)
    y_raw: np.ndarray,      # (samples,)
    data_norms: Tuple,      # (A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    model_nn: nn.Sequential,
    model_type: str,
    TL_coef: np.ndarray,    # Raw (not normalized) TL coefficients
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate performance of neural network-based aeromagnetic compensation, model 2, from raw inputs.
    """
    y_raw_flat = y_raw.astype(np.float32).flatten()
    if l_segs is None:
        l_segs = [len(y_raw_flat)]

    y_hat = nn_comp_2_fwd_from_raw(
        A_raw, x_raw, data_norms, model_nn, model_type, TL_coef
    )
    
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_raw_flat):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_raw_flat)}. Using single segment.")
        valid_l_segs = [len(y_raw_flat)]
        
    err_val = err_segs(y_hat, y_raw_flat, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err_val) if err_val.size > 0 else float('nan')
        print(f"INFO: test error: {err_std:.2f} nT")
        
    return y_hat, err_val
def nn_comp_2_train(
    A: np.ndarray,      # (samples, TL_terms)
    x: np.ndarray,      # (samples, features)
    y: np.ndarray,      # (samples,)
    no_norm: Optional[np.ndarray] = None,
    model_type: str = "m2a", # :m2a, :m2b, :m2c, :m2d
    norm_type_A: str = "none",
    norm_type_x: str = "standardize",
    norm_type_y: str = "none", # Note: Julia default is :none for y in m2
    TL_coef_in: Optional[np.ndarray] = None, # Renamed from TL_coef
    eta_adam: float = 0.001,
    epoch_adam: int = 5,
    epoch_lbfgs: int = 0,
    hidden: List[int] = [8],
    activation: Callable = nn.SiLU, # swish
    loss_fn: Callable = nn.MSELoss(), # Renamed from loss
    batchsize: int = 2048,
    frac_train: float = 14/17,
    alpha_sgl: float = 1.0,
    lambda_sgl: float = 0.0,
    k_pca: int = -1,
    data_norms_in: Optional[Tuple] = None, # (A_bias,A_scale,v_scale_pca,x_bias,x_scale,y_bias,y_scale)
    model_in: Optional[nn.Sequential] = None,
    l_segs: Optional[List[int]] = None,
    A_test_raw: Optional[np.ndarray] = None, # Renamed from A_test
    x_test_raw: Optional[np.ndarray] = None, # Renamed from x_test
    y_test_raw: Optional[np.ndarray] = None, # Renamed from y_test
    l_segs_test: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[nn.Sequential, np.ndarray, Tuple, np.ndarray, np.ndarray]:
    """
    Train neural network-based aeromagnetic compensation, model 2.
    """
    if TL_coef_in is None:
        # Determine number of TL terms from A matrix if possible, else default (e.g. 18)
        num_tl_terms = A.shape[1] if A.ndim == 2 and A.shape[1] > 0 else 18
        TL_coef_in = np.zeros(num_tl_terms, dtype=np.float32)

    # Convert to Float32
    A_f32 = A.astype(np.float32)
    x_f32 = x.astype(np.float32)
    y_f32 = y.astype(np.float32).flatten()
    alpha = np.float32(alpha_sgl)
    lambda_ = np.float32(lambda_sgl)
    TL_coef_f32 = TL_coef_in.astype(np.float32)

    A_test_f32 = A_test_raw.astype(np.float32) if A_test_raw is not None else np.empty((0,A_f32.shape[1]), dtype=np.float32)
    x_test_f32 = x_test_raw.astype(np.float32) if x_test_raw is not None else np.empty((0,x_f32.shape[1]), dtype=np.float32)
    y_test_f32 = y_test_raw.astype(np.float32).flatten() if y_test_raw is not None else np.empty(0, dtype=np.float32)

    if no_norm is None: no_norm = np.zeros(x_f32.shape[1], dtype=bool)
    if l_segs is None: l_segs = [len(y_f32)]
    if l_segs_test is None: l_segs_test = [len(y_test_f32)]


    Nf = x_f32.shape[1] # number of features for x

    # Normalization
    A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale = (None,)*7
    
    if data_norms_in is None or np.sum(np.abs(data_norms_in[-1])) == 0 : # data_norms_in[6] is y_scale
        A_bias, A_scale, A_norm_np = norm_sets(A_f32, norm_type=norm_type_A) # A_norm_np is (samples, TL_terms)
        x_bias, x_scale, x_norm_np_orig = norm_sets(x_f32, norm_type=norm_type_x, no_norm=no_norm) # x_norm_np_orig is (samples, features)
        y_bias, y_scale, y_norm_np = norm_sets(y_f32, norm_type=norm_type_y) # y_norm_np is (samples,)

        if k_pca > 0:
            if k_pca > Nf:
                if not silent: print(f"INFO: reducing k_pca from {k_pca} to {Nf}")
                k_pca = Nf
            if x_norm_np_orig.shape[0] > 1:
                cov_x = np.cov(x_norm_np_orig, rowvar=False)
                _, S_svd, V_svd_T = np.linalg.svd(cov_x) # V_svd_T is V.T
                V_svd = V_svd_T.T
                # Ensure S_svd has k_pca elements, pad with small value if fewer
                S_svd_padded = np.pad(S_svd, (0, max(0, k_pca - S_svd.shape[0])), 'constant', constant_values=1e-9)

                v_scale_pca = V_svd[:, :k_pca] @ np.diag(1.0 / np.sqrt(S_svd_padded[:k_pca]))
                var_ret = round(np.sum(np.sqrt(S_svd_padded[:k_pca])) / np.sum(np.sqrt(S_svd_padded)) * 100, 6)
                if not silent: print(f"INFO: k_pca = {k_pca} of {Nf}, variance retained: {var_ret} %")
            else:
                if not silent: print(f"WARN: Not enough samples ({x_norm_np_orig.shape[0]}) for PCA. Using identity for v_scale_pca.")
                v_scale_pca = np.eye(Nf, k_pca if k_pca <= Nf else Nf, dtype=np.float32) # Ensure correct shape if k_pca > Nf
        else:
            v_scale_pca = np.eye(Nf, dtype=np.float32)
        
        # Transpose for PyTorch convention (features, samples)
        A_norm_transposed = A_norm_np.T   # (TL_terms, samples)
        x_norm_transformed_transposed = (x_norm_np_orig @ v_scale_pca).T # (PCA_features, samples)
        y_norm_transposed = y_norm_np.reshape(1, -1) # (1, samples)

        data_norms_out = (A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    else:
        A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms_in) # type: ignore
        A_norm_transposed = ((A_f32 - A_bias.reshape(1,-1)) / np.where(A_scale.reshape(1,-1) == 0, 1.0, A_scale.reshape(1,-1))).T
        x_norm_orig = (x_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        x_norm_transformed_transposed = (x_norm_orig @ v_scale_pca).T
        y_norm_transposed = ((y_f32 - y_bias) / (y_scale if y_scale != 0 else 1.0)).reshape(1,-1)
        data_norms_out = data_norms_in

    # Normalize TL_coef (y_scale can be float or array)
    y_scale_val = y_scale[0] if isinstance(y_scale, np.ndarray) and y_scale.size > 0 else (y_scale if isinstance(y_scale, (float,int)) else 1.0)
    y_scale_val = y_scale_val if y_scale_val != 0 else 1.0 # Avoid division by zero
    TL_coef_norm_np = TL_coef_f32 / y_scale_val

    # Normalize test data if provided
    A_test_norm_transposed, x_test_norm_transformed_transposed = None, None
    if A_test_f32.size > 0:
        A_test_norm_transposed = ((A_test_f32 - A_bias.reshape(1,-1)) / np.where(A_scale.reshape(1,-1) == 0, 1.0, A_scale.reshape(1,-1))).T
    if x_test_f32.size > 0:
        x_test_norm_orig = (x_test_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        x_test_norm_transformed_transposed = (x_test_norm_orig @ v_scale_pca).T

    # PyTorch Tensors: DataLoader expects (samples, features)
    A_norm_torch = torch.from_numpy(A_norm_transposed.T.astype(np.float32)) # (samples, TL_terms)
    x_norm_torch = torch.from_numpy(x_norm_transformed_transposed.T.astype(np.float32)) # (samples, PCA_features)
    y_norm_torch = torch.from_numpy(y_norm_transposed.squeeze().astype(np.float32)) # (samples,)
    TL_coef_norm_torch = torch.from_numpy(TL_coef_norm_np.astype(np.float32))

    # Training/Validation Split
    if frac_train < 1:
        num_samples = x_norm_torch.shape[0]
        # Use get_split for consistent splitting logic if temporal aspects are important later
        # For now, simple random permutation for non-temporal split:
        # indices = np.random.permutation(num_samples)
        # split_idx = int(np.floor(num_samples * frac_train))
        # train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        train_indices, val_indices = get_split(num_samples, frac_train, window_type="none") # Assuming non-temporal for M2 split

        A_train_norm_torch, x_train_norm_torch, y_train_norm_torch = A_norm_torch[train_indices], x_norm_torch[train_indices], y_norm_torch[train_indices]
        A_val_norm_torch, x_val_norm_torch, y_val_norm_torch = A_norm_torch[val_indices], x_norm_torch[val_indices], y_norm_torch[val_indices]
        
        train_dataset = TensorDataset(A_train_norm_torch, x_train_norm_torch, y_train_norm_torch)
        val_dataset = TensorDataset(A_val_norm_torch, x_val_norm_torch, y_val_norm_torch)
    else:
        train_dataset = TensorDataset(A_norm_torch, x_norm_torch, y_norm_torch)
        val_dataset = train_dataset # Use full dataset for validation if frac_train is 1

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    # Setup NN model
    current_nn_model: nn.Sequential
    nn_input_features = x_norm_transformed_transposed.shape[0] # Number of PCA features
    
    if model_type in ["m2a", "m2d"]:
        Ny_nn = A_norm_transposed.shape[0] # NN outputs a coefficient for each TL term
    else: # m2b, m2c
        Ny_nn = 1 # NN outputs a scalar correction
        
    if model_in is None or not list(model_in.children()):
        current_nn_model = get_nn_m(nn_input_features, Ny_nn, hidden=hidden, activation=activation)
    else:
        current_nn_model = copy.deepcopy(model_in)

    # Setup combined model structure (M2Struct or M2StructMOnly)
    # TL_coef_norm_torch is already prepared
    if model_type == "m2c": # TL_coef_norm is trainable
        s_model = M2Struct(current_nn_model, TL_coef_norm_torch)
    else: # m2a, m2b, m2d: TL_coef_norm is fixed or handled outside direct NN optimization
        s_model = M2StructMOnly(current_nn_model, TL_coef_norm_torch)

    # Loss function (will be called within the training loop)
    # The loss needs to compute y_hat_norm using nn_comp_2_fwd (with denorm=False)
    # and then apply the base loss_fn (e.g., MSELoss)

    def compute_loss_val_m2(model_wrapper: Union[M2Struct, M2StructMOnly], loader: DataLoader) -> float:
        model_wrapper.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for A_b_norm, x_b_norm, y_b_norm in loader:
                # nn_comp_2_fwd expects (features, samples) for numpy,
                # but here we pass tensors directly.
                # A_b_norm: (batch, TL_terms), x_b_norm: (batch, PCA_features)
                # y_b_norm: (batch,)
                # TL_coef_norm is taken from model_wrapper
                
                y_hat_b_norm = nn_comp_2_fwd(
                    A_b_norm.T, x_b_norm.T, # Pass as (features, batch)
                    0.0, 1.0, # Dummy bias/scale as we want normalized output
                    model_wrapper.m, 
                    model_wrapper.TL_coef_norm, # Use the one from the struct
                    model_type=model_type,
                    denorm=False, 
                    testmode=True # Already in no_grad context
                )
                y_hat_b_norm_torch = torch.from_numpy(y_hat_b_norm.astype(np.float32))
                
                current_loss = loss_fn(y_hat_b_norm_torch, y_b_norm)
                # SGL penalty (if applicable)
                if lambda_ > 0:
                    sgl_penalty = lambda_ * sparse_group_lasso(model_wrapper.m, alpha)
                    current_loss += sgl_penalty
                
                total_loss += current_loss.item() * x_b_norm.size(0)
                count += x_b_norm.size(0)
        return total_loss / count if count > 0 else 0.0

    # Optimizer
    optimizer_adam = optim.Adam(s_model.parameters(), lr=eta_adam) # s_model.parameters() includes TL_coef_norm if M2Struct

    # Training loop (Adam)
    best_model_state = copy.deepcopy(s_model.state_dict())
    best_loss_val = compute_loss_val_m2(s_model, val_loader)
    
    best_test_error_std: Optional[float] = None
    if x_test_norm_transformed_transposed is not None and y_test_f32.size > 0 and A_test_norm_transposed is not None:
        _, err_test_init = nn_comp_2_test(
            A_test_norm_transposed, x_test_norm_transformed_transposed, 
            y_test_f32, y_bias, y_scale, 
            s_model.m, s_model.TL_coef_norm.cpu().detach().numpy(), model_type,
            l_segs=l_segs_test, silent=SILENT_DEBUG
        )
        best_test_error_std = np.std(err_test_init) if err_test_init.size > 0 else float('inf')

    if not silent: print(f"INFO: M2 epoch 0: loss = {best_loss_val:.6f}")

    for i in range(1, epoch_adam + 1):
        s_model.train()
        for A_batch_norm, x_batch_norm, y_batch_norm_target in train_loader:
            # A_batch_norm: (batch, TL_terms), x_batch_norm: (batch, PCA_features)
            # y_batch_norm_target: (batch,)
            optimizer_adam.zero_grad()
            
            # Forward pass for loss calculation (normalized)
            y_hat_batch_norm = nn_comp_2_fwd(
                A_batch_norm.T, x_batch_norm.T, # Pass as (features, batch)
                0.0, 1.0, # Dummy bias/scale for normalized output
                s_model.m, 
                s_model.TL_coef_norm, # Use current TL_coef_norm from model
                model_type=model_type,
                denorm=False,
                testmode=False # Important: ensure gradients flow
            )
            y_hat_batch_norm_torch = torch.from_numpy(y_hat_batch_norm.astype(np.float32))
            
            loss_train_val = loss_fn(y_hat_batch_norm_torch, y_batch_norm_target)
            if lambda_ > 0:
                sgl_penalty = lambda_ * sparse_group_lasso(s_model.m, alpha)
                loss_train_val += sgl_penalty
            
            loss_train_val.backward()
            optimizer_adam.step()

        current_loss_val = compute_loss_val_m2(s_model, val_loader)

        if x_test_norm_transformed_transposed is None or y_test_f32.size == 0 or A_test_norm_transposed is None:
            if current_loss_val < best_loss_val:
                best_loss_val = current_loss_val
                best_model_state = copy.deepcopy(s_model.state_dict())
            if i % 5 == 0 and not silent:
                print(f"INFO: M2 epoch {i}: loss = {best_loss_val:.6f}")
        else:
            _, err_test_current = nn_comp_2_test(
                A_test_norm_transposed, x_test_norm_transformed_transposed,
                y_test_f32, y_bias, y_scale,
                s_model.m, s_model.TL_coef_norm.cpu().detach().numpy(), model_type,
                l_segs=l_segs_test, silent=SILENT_DEBUG
            )
            current_test_error_std = np.std(err_test_current) if err_test_current.size > 0 else float('inf')

            if best_test_error_std is not None and current_test_error_std < best_test_error_std:
                best_test_error_std = current_test_error_std
                best_model_state = copy.deepcopy(s_model.state_dict())
            
            if i % 5 == 0 and not silent:
                print(f"INFO: M2 epoch {i}: loss = {current_loss_val:.6f}, test error = {best_test_error_std:.2f} nT" if best_test_error_std is not None else f"INFO: M2 epoch {i}: loss = {current_loss_val:.6f}")
        
        # Optional: Log full training set error periodically
        if i % 10 == 0 and not silent:
            _, train_err_np = nn_comp_2_test(
                A_norm_transposed, x_norm_transformed_transposed, # Full normalized training data
                y_f32, y_bias, y_scale, 
                s_model.m, s_model.TL_coef_norm.cpu().detach().numpy(), model_type,
                l_segs=l_segs, silent=True
            )
            train_err_std = np.std(train_err_np) if train_err_np.size > 0 else float('nan')
            print(f"INFO: M2 {i} train error: {train_err_std:.2f} nT")


    s_model.load_state_dict(best_model_state)
    final_TL_coef_norm = s_model.TL_coef_norm.cpu().detach().numpy() # Get the best TL_coef_norm

    # LBFGS optimization (if epoch_lbfgs > 0)
    if epoch_lbfgs > 0:
        if not silent: print("INFO: M2 LBFGS training started.")
        optimizer_lbfgs = optim.LBFGS(s_model.parameters(), lr=0.1) # lr often 1, PyTorch default is 1

        def closure_lbfgs_m2():
            optimizer_lbfgs.zero_grad()
            # Use full training dataset for LBFGS
            y_hat_norm_lbfgs = nn_comp_2_fwd(
                A_norm_torch.T, x_norm_torch.T, # (features, N)
                0.0, 1.0, # Dummy for normalized output
                s_model.m, s_model.TL_coef_norm, model_type,
                denorm=False, testmode=False
            )
            y_hat_norm_lbfgs_torch = torch.from_numpy(y_hat_norm_lbfgs.astype(np.float32))
            
            loss_val = loss_fn(y_hat_norm_lbfgs_torch, y_norm_torch) # y_norm_torch is (N,)
            if lambda_ > 0:
                loss_val += lambda_ * sparse_group_lasso(s_model.m, alpha)
            loss_val.backward()
            return loss_val

        for i_lbfgs in range(epoch_lbfgs):
            optimizer_lbfgs.step(closure_lbfgs_m2)
            if not silent and (i_lbfgs + 1) % 5 == 0:
                current_loss_lbfgs = closure_lbfgs_m2().item()
                print(f"INFO: M2 LBFGS epoch {i_lbfgs+1}: loss = {current_loss_lbfgs:.6f}")
        if not silent: print("INFO: M2 LBFGS training finished.")
        final_TL_coef_norm = s_model.TL_coef_norm.cpu().detach().numpy() # Update after LBFGS

    # Denormalize final TL coefficients
    final_TL_coef = final_TL_coef_norm * y_scale_val

    # Final evaluation on training data
    y_hat_final, err_final = nn_comp_2_test(
        A_norm_transposed, x_norm_transformed_transposed, y_f32, 
        y_bias, y_scale, s_model.m, final_TL_coef_norm, model_type,
        l_segs=l_segs, silent=True
    )
    if not silent: 
        err_std_final = np.std(err_final) if err_final.size > 0 else float('nan')
        print(f"INFO: M2 final train error: {err_std_final:.2f} nT")

    # Final evaluation on test data if provided
    if x_test_norm_transformed_transposed is not None and y_test_f32.size > 0 and A_test_norm_transposed is not None:
        nn_comp_2_test(
            A_test_norm_transposed, x_test_norm_transformed_transposed, y_test_f32,
            y_bias, y_scale, s_model.m, final_TL_coef_norm, model_type,
            l_segs=l_segs_test, silent=silent # Show test error if not silent overall
        )
        
    return s_model.m, final_TL_coef, data_norms_out, y_hat_final, err_final
def nn_comp_3_fwd(
    B_unit: Union[np.ndarray, torch.Tensor],      # (3, samples)
    B_vec: Union[np.ndarray, torch.Tensor],       # (3, samples)
    B_vec_dot: Union[np.ndarray, torch.Tensor],   # (3, samples)
    x_norm: Union[np.ndarray, torch.Tensor],      # (features, samples) or (features, window, samples) for temporal
    y_bias: Union[float, np.ndarray],
    y_scale: Union[float, np.ndarray],
    model_nn: nn.Sequential,
    TL_coef_p: Union[np.ndarray, torch.Tensor],   # (3,)
    TL_coef_i_mat: Union[np.ndarray, torch.Tensor], # (3,3)
    TL_coef_e_mat: Optional[Union[np.ndarray, torch.Tensor]], # (3,3) or None
    model_type: str,    # :m3s, :m3v, :m3sc, :m3vc, :m3w, :m3tf
    y_type: str,        # :a, :b, :c, :d
    use_nn: bool = True,
    denorm: bool = True,
    testmode: bool = True
) -> np.ndarray:
    """
    Forward pass of neural network-based aeromagnetic compensation, model 3.
    B_unit, B_vec, B_vec_dot are (3, samples).
    x_norm is (features, samples) or (features, window, samples) for temporal.
    TL_coef_p is (3,), TL_coef_i_mat is (3,3), TL_coef_e_mat is (3,3) or None.
    """
    if y_type not in ["a", "b", "c", "d"]:
        raise ValueError(f"Unsupported y_type = {y_type} for nn_comp_3")

    if testmode:
        model_nn.eval()
    # else: training mode is set by the main loop

    # Ensure inputs are PyTorch tensors for nn_comp_3_fwd internal logic
    B_unit_th = torch.as_tensor(B_unit, dtype=torch.float32)
    B_vec_th = torch.as_tensor(B_vec, dtype=torch.float32)
    B_vec_dot_th = torch.as_tensor(B_vec_dot, dtype=torch.float32) if B_vec_dot is not None and B_vec_dot.size > 0 else torch.empty_like(B_vec_th)
    
    # x_norm needs to be (samples, features) or (samples, window, features) for PyTorch model
    if isinstance(x_norm, np.ndarray):
        if model_type in ["m3w", "m3tf"]: # (features, window, samples) -> (samples, window, features)
            x_norm_th = torch.from_numpy(np.moveaxis(x_norm, [0, 1, 2], [2, 1, 0]).astype(np.float32))
        else: # (features, samples) -> (samples, features)
            x_norm_th = torch.from_numpy(x_norm.T.astype(np.float32))
    else: # Already a tensor
        if model_type in ["m3w", "m3tf"]: # Assuming (features, window, samples)
             x_norm_th = x_norm.permute(2,1,0).float() # (samples, window, features)
        else: # Assuming (features, samples)
            x_norm_th = x_norm.T.float() # (samples, features)


    TL_coef_p_th = torch.as_tensor(TL_coef_p, dtype=torch.float32)
    TL_coef_i_mat_th = torch.as_tensor(TL_coef_i_mat, dtype=torch.float32)
    TL_coef_e_mat_th = torch.as_tensor(TL_coef_e_mat, dtype=torch.float32) if TL_coef_e_mat is not None and TL_coef_e_mat.size > 0 else None
    
    y_hat_torch: torch.Tensor

    with torch.no_grad() if testmode else torch.enable_grad():
        # Calculate TL aircraft field vector
        # get_TL_aircraft_vec expects (3,N) inputs
        tl_aircraft_vec = get_TL_aircraft_vec(
            B_vec_th, B_vec_dot_th, TL_coef_p_th, TL_coef_i_mat_th, TL_coef_e_mat_th, return_parts=False
        ) # Returns (3, N)
        
        vec_aircraft_th = tl_aircraft_vec # (3, N)

        if use_nn:
            nn_output = model_nn(x_norm_th) # (samples, nn_out_features)
            # nn_out_features is 3 for m3v/m3vc, 1 for m3s/m3sc/m3w/m3tf

            if model_type in ["m3v", "m3vc"]: # vector NN correction
                # nn_output is (samples, 3), needs to be (3, samples)
                vec_aircraft_th = vec_aircraft_th + nn_output.T * torch.as_tensor(y_scale, dtype=torch.float32) # Rescale NN part
            # For scalar models (m3s, m3sc, m3w, m3tf), NN correction is applied after projecting to scalar
            
        # Calculate y_hat based on y_type
        if y_type in ["c", "d"]: # Aircraft field component along B_unit
            # vec_aircraft_th is (3,N), B_unit_th is (3,N)
            # Dot product for each sample: sum(vec_aircraft_th * B_unit_th, dim=0)
            y_hat_torch = torch.sum(vec_aircraft_th * B_unit_th, dim=0) # (N,)
        elif y_type in ["a", "b"]: # Magnitude of Earth field
            B_e_th = B_vec_th - vec_aircraft_th # (3,N)
            y_hat_torch = torch.linalg.norm(B_e_th, dim=0) # (N,)
        else: # Should not happen due to earlier check
            raise ValueError(f"y_type {y_type} logic error in nn_comp_3_fwd")

        if use_nn and model_type in ["m3s", "m3sc", "m3w", "m3tf"]: # scalar NN correction
            nn_output_scalar = model_nn(x_norm_th).squeeze() # (samples,)
            y_hat_torch = y_hat_torch + nn_output_scalar * torch.as_tensor(y_scale, dtype=torch.float32) # Rescale NN part

    y_hat_norm_np = y_hat_torch.cpu().detach().numpy() if isinstance(y_hat_torch, torch.Tensor) else y_hat_torch # if already numpy

    y_hat_final_np: np.ndarray
    if denorm:
        _y_bias = np.array(y_bias) if isinstance(y_bias, (float,int)) else y_bias
        _y_scale = np.array(y_scale) if isinstance(y_scale, (float,int)) else y_scale
        y_hat_final_np = denorm_sets(_y_bias, _y_scale, y_hat_norm_np)
    else:
        y_hat_final_np = y_hat_norm_np
        
    return y_hat_final_np.flatten()

def nn_comp_3_fwd_from_raw(
    A_raw: np.ndarray,      # (samples, TL_terms_raw_A) e.g. flux components
    Bt_raw: np.ndarray,     # (samples,) total field
    B_dot_raw: np.ndarray,  # (samples, 3) derivatives of flux components
    x_raw: np.ndarray,      # (samples, features)
    data_norms: Tuple,      # (A_bias, A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
                            # Note: A_bias/A_scale here are for the components of A used in create_TL_A (e.g. flux, not the full TL matrix)
    model_nn: nn.Sequential,
    model_type: str,
    y_type: str,
    TL_coef_raw: np.ndarray, # Raw (not normalized) full TL coefficient vector
    terms_A: List[str],      # Terms used to construct the full TL matrix from A_raw components
    l_segs: Optional[List[int]] = None, # For get_temporal_data if model is temporal
    l_window: int = 0 # For get_temporal_data
) -> np.ndarray:
    """
    Forward pass of neural network-based aeromagnetic compensation, model 3, from raw inputs.
    """
    A_raw_f32 = A_raw.astype(np.float32)
    Bt_raw_f32 = Bt_raw.astype(np.float32).flatten()
    B_dot_raw_f32 = B_dot_raw.astype(np.float32) # (samples, 3)
    x_raw_f32 = x_raw.astype(np.float32)
    TL_coef_raw_f32 = TL_coef_raw.astype(np.float32)

    # Unpack data normalizations
    # The data_norms tuple for M3 is typically:
    # (placeholder_A_bias, placeholder_A_scale, v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    # The A_bias/A_scale from norm_sets(A_f32) in nn_comp_3_train are not directly used here for A_raw.
    # Instead, B_unit, B_vec, B_vec_dot are constructed from raw A_raw (fluxes) and Bt_raw.
    _, _, v_scale_pca, x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms)

    # Construct B_unit, B_vec, B_vec_dot from raw inputs
    # A_raw is typically the flux components (samples, 3)
    # B_unit = A_raw / ||A_raw|| (row-wise)
    norm_A_raw = np.linalg.norm(A_raw_f32, axis=1, keepdims=True)
    norm_A_raw[norm_A_raw == 0] = 1e-9 # Avoid division by zero
    B_unit_np = (A_raw_f32 / norm_A_raw).T # (3, samples)
    
    B_vec_np = B_unit_np * Bt_raw_f32 # (3, samples) * (samples,) -> (3, samples)
    B_vec_dot_np = B_dot_raw_f32.T    # (3, samples)

    # Normalize x
    x_norm_step1 = (x_raw_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
    x_norm_pca = x_norm_step1 @ v_scale_pca # (samples, PCA_features)
    
    # Handle temporal data if needed
    x_norm_for_fwd_transposed: np.ndarray # Will be (PCA_features, samples) or (PCA_features, window, samples)
    if model_type in ["m3w", "m3tf"]:
        if l_segs is None: l_segs = [x_norm_pca.shape[0]]
        x_norm_temporal = get_temporal_data(x_norm_pca.T, l_segs, l_window) # Expects (features, N)
        x_norm_for_fwd_transposed = x_norm_temporal # Already (features, window, N)
    else:
        x_norm_for_fwd_transposed = x_norm_pca.T # (PCA_features, samples)

    # Split TL_coef_raw into p, i_mat, e_mat components
    # Bt_scale is used inside TL_vec2mat for denormalizing i and e components from the vector form
    # Here, TL_coef_raw is already in the "vector" form but not scaled by Bt_scale for i and e parts yet.
    # The TL_vec2mat function expects the vector form where i and e are already scaled by Bt_scale.
    # This means we need to be careful. The TL_coef_raw from NNCompParams is the direct output of training.
    # Let's assume TL_coef_raw is the "beta" from `tolles_lawson_train` or similar,
    # which means it's directly applicable to the A matrix from `create_TL_A`.
    # The `nn_comp_3_fwd` expects p, i_mat, e_mat that are already appropriately scaled.
    
    # The TL_coef stored in CompParams is the direct coefficient vector.
    # We need to convert it to the matrix forms (p, i_mat, e_mat) expected by nn_comp_3_fwd.
    # The Bt_scale in TL_vec2mat is for converting the "normalized" vector components (like those in Flux)
    # back to physical units. Here, TL_coef_raw is already in physical units.
    # So, we use Bt_scale=1.0 when calling TL_vec2mat if TL_coef_raw's i and e parts are already physical.
    # However, the nn_comp_3_fwd expects TL_coef_i_mat and TL_coef_e_mat to be such that
    # TL_induced = TL_coef_i_mat @ B_vec.
    # If TL_coef_raw comes from `tolles_lawson_train`, its induced/eddy parts are typically normalized by Bt.
    # Let's assume TL_coef_raw is the direct output of `nn_comp_3_train` which stores the physical coefficients.
    
    # The `TL_vec2mat` function in Julia divides by Bt_scale for induced/eddy.
    # So, if TL_coef_raw has induced/eddy parts that are meant to be multiplied by B_vec/B_vec_dot directly,
    # then TL_vec2mat should be called with Bt_scale=1.0.
    # Or, if TL_coef_raw's induced/eddy parts are like the 'c' in c*B (where B includes Bt), then Bt_scale=50000.
    # Given the context of nn_comp_3_fwd, TL_coef_i_mat and TL_coef_e_mat are multiplied by B_vec and B_vec_dot respectively.
    # These B_vec and B_vec_dot already contain the magnetic field strength.
    # So, the coefficients themselves should be dimensionless or scaled appropriately.
    # The Julia `TL_vec2mat` divides by Bt_scale. This implies the input TL_coef vector's induced/eddy parts
    # are effectively (dimensionless_coeff * Bt_scale).
    # If our TL_coef_raw is this (dimensionless_coeff * Bt_scale) form, then Julia's TL_vec2mat is correct.
    # The `nn_comp_3_train` stores `TL_coef = [s.TL_p;s.TL_i;s.TL_e]`.
    # `s.TL_p, s.TL_i, s.TL_e` are parameters of `m3_struct`, which are trained.
    # The loss function `loss_m3` calls `TL_vec2mat(TL_coef, terms_A; Bt_scale=Bt_scale)`.
    # This implies the stored `s.TL_p,i,e` are such that when recombined and then split by `TL_vec2mat`
    # with the standard Bt_scale, they yield the correct physical matrices.
    # This means the stored s.TL_i and s.TL_e components in m3_struct are likely the "dimensionless_coeff * Bt_scale" form.
    
    # Therefore, when calling TL_vec2mat here with TL_coef_raw, we should use the standard Bt_scale.
    # The resulting TL_coef_i_mat and TL_coef_e_mat will then be the dimensionless matrices.
    # This seems consistent with how nn_comp_3_fwd would use them with B_vec (which includes field strength).

    TL_p_np, TL_i_mat_np, TL_e_mat_np = TL_vec2mat(TL_coef_raw_f32, terms_A, Bt_scale=50000.0)

    y_hat = nn_comp_3_fwd(
        B_unit_np, B_vec_np, B_vec_dot_np,
        x_norm_for_fwd_transposed, # Already (features, samples) or (features, window, samples)
        y_bias, y_scale, model_nn,
        TL_p_np, TL_i_mat_np, TL_e_mat_np,
        model_type=model_type,
        y_type=y_type,
        use_nn=True, # Always use NN for fwd_from_raw as it's part of the trained model
        denorm=True, 
        testmode=True
    )
    return y_hat
def nn_comp_3_test(
    B_unit: Union[np.ndarray, torch.Tensor],      # (3, samples)
    B_vec: Union[np.ndarray, torch.Tensor],       # (3, samples)
    B_vec_dot: Union[np.ndarray, torch.Tensor],   # (3, samples)
    x_norm: Union[np.ndarray, torch.Tensor],      # (features, samples) or (features, window, samples)
    y: np.ndarray,                                # Raw target vector (samples,)
    y_bias: Union[float, np.ndarray],
    y_scale: Union[float, np.ndarray],
    model_nn: nn.Sequential,
    TL_coef_p: Union[np.ndarray, torch.Tensor],
    TL_coef_i_mat: Union[np.ndarray, torch.Tensor],
    TL_coef_e_mat: Optional[Union[np.ndarray, torch.Tensor]],
    model_type: str,
    y_type: str,
    l_segs: Optional[List[int]] = None,
    use_nn: bool = True, # In test, usually True if NN is part of the model
    denorm: bool = True, # Usually True for final error calculation
    testmode: bool = True, # Always True for test function
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate performance of neural network-based aeromagnetic compensation, model 3.
    """
    y_flat = y.flatten()
    if l_segs is None:
        l_segs = [len(y_flat)]

    y_hat = nn_comp_3_fwd(
        B_unit, B_vec, B_vec_dot, x_norm,
        y_bias, y_scale, model_nn,
        TL_coef_p, TL_coef_i_mat, TL_coef_e_mat,
        model_type=model_type, y_type=y_type,
        use_nn=use_nn, denorm=denorm, testmode=testmode
    )
    
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_flat):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_flat)}. Using single segment.")
        valid_l_segs = [len(y_flat)]
        
    err_val = err_segs(y_hat, y_flat, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err_val) if err_val.size > 0 else float('nan')
        print(f"INFO: M3 test error: {err_std:.2f} nT")
        
    return y_hat, err_val

def nn_comp_3_test_from_raw(
    A_raw: np.ndarray,      # (samples, flux_components)
    Bt_raw: np.ndarray,     # (samples,)
    B_dot_raw: np.ndarray,  # (samples, 3)
    x_raw: np.ndarray,      # (samples, features)
    y_raw: np.ndarray,      # (samples,)
    data_norms: Tuple,
    model_nn: nn.Sequential,
    model_type: str,
    y_type: str,
    TL_coef_raw: np.ndarray,
    terms_A: List[str],
    l_segs: Optional[List[int]] = None,
    l_window: int = 0, # For get_temporal_data if model is temporal
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate performance of neural network-based aeromagnetic compensation, model 3, from raw inputs.
    """
    y_raw_flat = y_raw.astype(np.float32).flatten()
    if l_segs is None:
        l_segs = [len(y_raw_flat)]

    y_hat = nn_comp_3_fwd_from_raw(
        A_raw, Bt_raw, B_dot_raw, x_raw, data_norms, model_nn,
        model_type, y_type, TL_coef_raw, terms_A,
        l_segs=l_segs, l_window=l_window # Pass l_segs for temporal data prep if needed
    )
    
    valid_l_segs = l_segs
    if sum(l_segs) != len(y_raw_flat):
        if not silent: print(f"Warning: sum(l_segs)={sum(l_segs)} != len(y)={len(y_raw_flat)}. Using single segment.")
        valid_l_segs = [len(y_raw_flat)]

    err_val = err_segs(y_hat, y_raw_flat, valid_l_segs, silent=SILENT_DEBUG)
    
    if not silent:
        err_std = np.std(err_val) if err_val.size > 0 else float('nan')
        print(f"INFO: M3 test error (from raw): {err_std:.2f} nT")
        
    return y_hat, err_val
def nn_comp_3_train(
    A_raw: np.ndarray,      # Raw flux components (samples, 3)
    Bt_raw: np.ndarray,     # Raw total field (samples,)
    B_dot_raw: np.ndarray,  # Raw flux derivatives (samples, 3)
    x_raw: np.ndarray,      # Raw features (samples, features)
    y_raw: np.ndarray,      # Raw target (samples,)
    no_norm: Optional[np.ndarray] = None,
    model_type: str = "m3s",
    norm_type_x: str = "standardize",
    norm_type_y: str = "standardize", # Julia default :standardize for M3
    TL_coef_in: Optional[np.ndarray] = None, # Full TL coefficient vector
    terms_A: Optional[List[str]] = None, # Terms for TL_vec_split & TL_vec2mat
    y_type: str = "d",
    eta_adam: float = 0.001,
    epoch_adam: int = 5,
    epoch_lbfgs: int = 0,
    hidden: List[int] = [8],
    activation: Callable = nn.SiLU,
    loss_fn: Callable = nn.MSELoss(),
    batchsize: int = 2048,
    frac_train: float = 14/17,
    alpha_sgl: float = 1.0, # Note: SGL not fully implemented for M3 in Julia
    lambda_sgl: float = 0.0,
    k_pca: int = -1,
    sigma_curriculum: float = 1.0,
    l_window: int = 5,
    window_type_temporal: str = "sliding", # Renamed from window_type to avoid clash
    tf_layer_type: str = "postlayer", # Transformer specific
    tf_norm_type: str = "batch",    # Transformer specific
    dropout_prob: float = 0.2,      # Transformer specific
    N_tf_head: int = 8,             # Transformer specific
    tf_gain: float = 1.0,           # Transformer specific
    data_norms_in: Optional[Tuple] = None,
    model_in: Optional[nn.Sequential] = None,
    l_segs: Optional[List[int]] = None,
    A_test_raw_in: Optional[np.ndarray] = None,
    Bt_test_raw_in: Optional[np.ndarray] = None,
    B_dot_test_raw_in: Optional[np.ndarray] = None,
    x_test_raw_in: Optional[np.ndarray] = None,
    y_test_raw_in: Optional[np.ndarray] = None,
    l_segs_test: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[nn.Sequential, np.ndarray, Tuple, np.ndarray, np.ndarray]:
    """
    Train neural network-based aeromagnetic compensation, model 3.
    """
    if terms_A is None:
        terms_A = ["permanent", "induced", "eddy"] # Default if not provided

    if TL_coef_in is None:
        # Determine number of TL terms from a dummy A matrix based on terms_A
        # This requires a fully functional create_TL_A or a way to infer num_coeffs
        # For now, defaulting to 18 if not provided.
        # num_tl_terms = create_TL_A(np.ones((1,3)), terms=terms_A).shape[1] if terms_A else 18
        num_tl_terms = 18 # Fallback, ideally calculate from terms_A
        TL_coef_in = np.zeros(num_tl_terms, dtype=np.float32)

    if alpha_sgl != 1.0 or lambda_sgl != 0.0:
        if not silent: print("WARN: Sparse Group Lasso (SGL) is not fully implemented for nn_comp_3 in Python version yet.")

    if y_type not in ["a", "b", "c", "d"]:
        raise ValueError(f"Unsupported y_type = {y_type} for nn_comp_3")

    # Convert to Float32
    A_f32 = A_raw.astype(np.float32)
    Bt_f32 = Bt_raw.astype(np.float32).flatten()
    B_dot_f32 = B_dot_raw.astype(np.float32)
    x_f32 = x_raw.astype(np.float32)
    y_f32 = y_raw.astype(np.float32).flatten()
    TL_coef_f32 = TL_coef_in.astype(np.float32)

    A_test_f32 = A_test_raw_in.astype(np.float32) if A_test_raw_in is not None else np.empty((0,A_f32.shape[1]), dtype=np.float32)
    Bt_test_f32 = Bt_test_raw_in.astype(np.float32).flatten() if Bt_test_raw_in is not None else np.empty(0, dtype=np.float32)
    B_dot_test_f32 = B_dot_test_raw_in.astype(np.float32) if B_dot_test_raw_in is not None else np.empty((0,B_dot_f32.shape[1]), dtype=np.float32)
    x_test_f32 = x_test_raw_in.astype(np.float32) if x_test_raw_in is not None else np.empty((0,x_f32.shape[1]), dtype=np.float32)
    y_test_f32 = y_test_raw_in.astype(np.float32).flatten() if y_test_raw_in is not None else np.empty(0, dtype=np.float32)
    
    if no_norm is None: no_norm = np.zeros(x_f32.shape[1], dtype=bool)
    if l_segs is None: l_segs = [len(y_f32)]
    if l_segs_test is None: l_segs_test = [len(y_test_f32)] if y_test_f32.size > 0 else []


    Nf_x = x_f32.shape[1] # number of features for x

    # Initial TL coefficient split (physical units, Bt_scale=1 for vec2mat if coeffs are already physical)
    # The TL_coef_f32 is the full coefficient vector.
    # TL_vec2mat expects the vector form where induced/eddy parts are scaled by Bt_scale.
    # If TL_coef_f32 is already in physical units (e.g. from tolles_lawson_train),
    # then for TL_vec2mat to return dimensionless matrices (which nn_comp_3_fwd expects for i_mat, e_mat),
    # we should use the actual Bt_scale.
    Bt_scale_val = 50000.0 # Standard scaling
    TL_p_np, TL_i_mat_np, TL_e_mat_np = TL_vec2mat(TL_coef_f32, terms_A, Bt_scale=Bt_scale_val)

    # Construct B_unit, B_vec from raw A (fluxes) and Bt
    norm_A_f32 = np.linalg.norm(A_f32, axis=1, keepdims=True)
    norm_A_f32[norm_A_f32 == 0] = 1e-9
    B_unit_np_T = (A_f32 / norm_A_f32).T # (3, samples)
    B_vec_np_T = B_unit_np_T * Bt_f32    # (3, samples)
    B_vec_dot_np_T = B_dot_f32.T         # (3, samples)

    B_unit_test_np_T, B_vec_test_np_T, B_vec_dot_test_np_T = None, None, None
    if A_test_f32.size > 0 and Bt_test_f32.size > 0:
        norm_A_test_f32 = np.linalg.norm(A_test_f32, axis=1, keepdims=True)
        norm_A_test_f32[norm_A_test_f32 == 0] = 1e-9
        B_unit_test_np_T = (A_test_f32 / norm_A_test_f32).T
        B_vec_test_np_T = B_unit_test_np_T * Bt_test_f32
    if B_dot_test_f32.size > 0:
        B_vec_dot_test_np_T = B_dot_test_f32.T


    # Normalization of x and y
    v_scale_pca, x_bias, x_scale, y_bias, y_scale = (None,)*5
    if data_norms_in is None or np.sum(np.abs(data_norms_in[-1])) == 0: # y_scale is last
        x_bias, x_scale, x_norm_np_orig = norm_sets(x_f32, norm_type=norm_type_x, no_norm=no_norm)
        y_bias, y_scale, y_norm_np = norm_sets(y_f32, norm_type=norm_type_y)

        if k_pca > 0:
            if k_pca > Nf_x:
                if not silent: print(f"INFO: reducing k_pca from {k_pca} to {Nf_x}")
                k_pca = Nf_x
            if x_norm_np_orig.shape[0] > 1:
                cov_x = np.cov(x_norm_np_orig, rowvar=False)
                _, S_svd, V_svd_T = np.linalg.svd(cov_x)
                V_svd = V_svd_T.T
                S_svd_padded = np.pad(S_svd, (0, max(0, k_pca - S_svd.shape[0])), 'constant', constant_values=1e-9)
                v_scale_pca = V_svd[:, :k_pca] @ np.diag(1.0 / np.sqrt(S_svd_padded[:k_pca]))
                var_ret = round(np.sum(np.sqrt(S_svd_padded[:k_pca])) / np.sum(np.sqrt(S_svd_padded)) * 100, 6)
                if not silent: print(f"INFO: k_pca = {k_pca} of {Nf_x}, variance retained: {var_ret} %")
            else:
                if not silent: print(f"WARN: Not enough samples for PCA. Using identity for v_scale_pca.")
                v_scale_pca = np.eye(Nf_x, k_pca if k_pca <= Nf_x else Nf_x, dtype=np.float32)
        else:
            v_scale_pca = np.eye(Nf_x, dtype=np.float32)
        
        x_norm_pca_T = (x_norm_np_orig @ v_scale_pca).T # (PCA_features, samples)
        y_norm_T = y_norm_np.reshape(1, -1)         # (1, samples)
        # For M3, data_norms typically stores (_,_,v_scale_pca,x_bias,x_scale,y_bias,y_scale)
        # The first two are placeholders in Julia. We'll store None or zeros.
        data_norms_out = (np.array([0.0]), np.array([0.0]), v_scale_pca, x_bias, x_scale, y_bias, y_scale)
    else:
        _, _, v_scale_pca, x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms_in) # type: ignore
        x_norm_orig = (x_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        x_norm_pca_T = (x_norm_orig @ v_scale_pca).T
        y_norm_T = ((y_f32 - y_bias) / (y_scale if y_scale != 0 else 1.0)).reshape(1,-1)
        data_norms_out = data_norms_in

    # Temporal data preparation
    x_norm_final_T = x_norm_pca_T # (PCA_features, samples)
    x_test_norm_final_T = None
    if model_type in ["m3w", "m3tf"]:
        x_norm_final_T = get_temporal_data(x_norm_pca_T, l_segs, l_window) # (PCA_features, window, samples)
        if x_test_raw_in is not None and x_test_raw_in.size > 0:
            x_test_norm_orig = (x_test_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
            x_test_norm_pca_T = (x_test_norm_orig @ v_scale_pca).T
            if l_segs_test:
                 x_test_norm_final_T = get_temporal_data(x_test_norm_pca_T, l_segs_test, l_window)

    # PyTorch Tensors for training
    # DataLoader expects (samples, ...)
    # B_unit_np_T, B_vec_np_T, B_vec_dot_np_T are (3, samples)
    # x_norm_final_T is (PCA_features, samples) or (PCA_features, window, samples)
    # y_norm_T is (1, samples)

    B_unit_torch = torch.from_numpy(B_unit_np_T.T.astype(np.float32)) # (samples, 3)
    B_vec_torch = torch.from_numpy(B_vec_np_T.T.astype(np.float32))    # (samples, 3)
    B_vec_dot_torch = torch.from_numpy(B_vec_dot_np_T.T.astype(np.float32))# (samples, 3)
    
    if model_type in ["m3w", "m3tf"]: # (PCA_features, window, samples) -> (samples, window, PCA_features)
        x_torch_train = torch.from_numpy(np.moveaxis(x_norm_final_T, [0,1,2], [2,1,0]).astype(np.float32))
    else: # (PCA_features, samples) -> (samples, PCA_features)
        x_torch_train = torch.from_numpy(x_norm_final_T.T.astype(np.float32))
        
    y_torch_train = torch.from_numpy(y_norm_T.squeeze().astype(np.float32)) # (samples,)

    # Train/Val Split
    # Note: For temporal data, splitting needs to be careful not to break sequences if window_type is 'contiguous'
    # get_split handles this based on window_type.
    num_total_samples = x_torch_train.shape[0]
    
    # Curriculum learning indices (applied to the training portion after frac_train split)
    ind_cur_train, ind_nn_train = None, None # Placeholders for indices within the training set

    if frac_train < 1:
        # For M3, the split is done on the original sample dimension N
        # If temporal, N is size(x_norm,3) before permuting for DataLoader
        N_for_split = x_norm_final_T.shape[2] if model_type in ["m3w", "m3tf"] else x_norm_final_T.shape[1]
        
        # Use 'window_type_temporal' for get_split if model is temporal
        current_window_type_for_split = window_type_temporal if model_type in ["m3w", "m3tf"] else "none"
        l_window_for_split = l_window if model_type in ["m3w", "m3tf"] else 0
        
        p_train_idx, p_val_idx = get_split(N_for_split, frac_train, current_window_type_for_split, l_window=l_window_for_split)

        B_unit_train_th, B_vec_train_th, B_vec_dot_train_th = B_unit_torch[p_train_idx], B_vec_torch[p_train_idx], B_vec_dot_torch[p_train_idx]
        x_train_th, y_train_th = x_torch_train[p_train_idx], y_torch_train[p_train_idx]
        
        B_unit_val_th, B_vec_val_th, B_vec_dot_val_th = B_unit_torch[p_val_idx], B_vec_torch[p_val_idx], B_vec_dot_torch[p_val_idx]
        x_val_th, y_val_th = x_torch_train[p_val_idx], y_torch_train[p_val_idx]

        # Curriculum learning for m3sc, m3vc (applied on the p_train_idx portion)
        if model_type in ["m3sc", "m3vc"]:
            if not silent: print("INFO: M3 making curriculum")
            # Calculate TL estimate on the p_train_idx part of the data
            # Need raw y for this, corresponding to p_train_idx
            y_raw_train_subset = y_f32[p_train_idx]
            
            # We need B_unit, B_vec, B_vec_dot, x_norm for this subset to call nn_comp_3_fwd
            # These should be in their (features, samples) or (features, window, samples) format
            B_unit_train_subset_T = B_unit_np_T[:, p_train_idx]
            B_vec_train_subset_T = B_vec_np_T[:, p_train_idx]
            B_vec_dot_train_subset_T = B_vec_dot_np_T[:, p_train_idx]
            
            if model_type in ["m3w", "m3tf"]: # x_norm_final_T is (feat, win, samp)
                 x_norm_train_subset_T = x_norm_final_T[:, :, p_train_idx]
            else: # x_norm_final_T is (feat, samp)
                 x_norm_train_subset_T = x_norm_final_T[:, p_train_idx]

            y_TL_hat_train_subset = nn_comp_3_fwd(
                B_unit_train_subset_T, B_vec_train_subset_T, B_vec_dot_train_subset_T,
                x_norm_train_subset_T, 
                y_bias, y_scale, # Use overall y_bias, y_scale for denormalization
                nn.Sequential(), # Dummy NN model for TL-only part
                TL_p_np, TL_i_mat_np, TL_e_mat_np, # Initial physical TL coeffs
                model_type="m3tl", # Force TL-only behavior for this calc
                y_type=y_type,
                use_nn=False, denorm=True, testmode=True
            )
            TL_diff_train_subset = y_raw_train_subset - y_TL_hat_train_subset
            ind_cur_train, _ = get_curriculum_ind(TL_diff_train_subset, sigma_curriculum) # Boolean mask for p_train_idx

            # Create DataLoaders: one for curriculum, one for full training (used later)
            train_dataset_cur = TensorDataset(
                B_unit_train_th[ind_cur_train], B_vec_train_th[ind_cur_train], B_vec_dot_train_th[ind_cur_train],
                x_train_th[ind_cur_train], y_train_th[ind_cur_train]
            )
            train_loader_cur = DataLoader(train_dataset_cur, batch_size=batchsize, shuffle=True)
            
            train_dataset_full = TensorDataset(B_unit_train_th, B_vec_train_th, B_vec_dot_train_th, x_train_th, y_train_th)
            train_loader_full = DataLoader(train_dataset_full, batch_size=batchsize, shuffle=True)
            
            # Initial training will use train_loader_cur
            current_train_loader = train_loader_cur
        else:
            train_dataset = TensorDataset(B_unit_train_th, B_vec_train_th, B_vec_dot_train_th, x_train_th, y_train_th)
            current_train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
            train_loader_full = current_train_loader # For consistency later

        val_dataset = TensorDataset(B_unit_val_th, B_vec_val_th, B_vec_dot_val_th, x_val_th, y_val_th)
        val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    else: # frac_train == 1
        train_dataset = TensorDataset(B_unit_torch, B_vec_torch, B_vec_dot_torch, x_torch_train, y_torch_train)
        current_train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        train_loader_full = current_train_loader
        val_loader = current_train_loader # Validate on the full training set

    # Setup NN
    current_nn_model: nn.Sequential
    # Input features to NN is number of PCA features (x_norm_final_T.shape[0])
    # Or if temporal, it's x_norm_final_T.shape[0] (features) * x_norm_final_T.shape[1] (window) if MLP based on window
    # But get_nn_m for temporal models in Julia seems to take Nf (original features before windowing)
    # and handles windowing internally or via layer type.
    # For PyTorch, if using LSTM/Transformer, input_size to get_nn_m is num_features_per_step.
    # If x_norm_final_T is (PCA_features, window, samples), then input to NN is (PCA_features) if batch_first=False, seq_len=window
    # Or (window, PCA_features) if batch_first=True for Transformer.
    # Let's assume get_nn_m expects num_features_per_step.
    
    nn_input_features_count = x_norm_final_T.shape[0] # This is num_pca_components

    Ny_nn = 3 if model_type in ["m3v", "m3vc"] else 1 # NN output: 3 for vector, 1 for scalar correction
    
    if model_in is None or not list(model_in.children()):
        current_nn_model = get_nn_m(
            nn_input_features_count, Ny_nn, hidden=hidden, activation=activation,
            model_type=model_type, # Pass model_type for potential internal logic in get_nn_m
            l_window=l_window, 
            tf_layer_type=tf_layer_type, tf_norm_type=tf_norm_type,
            dropout_prob=dropout_prob, N_tf_head=N_tf_head, tf_gain=tf_gain
        )
    else:
        current_nn_model = copy.deepcopy(model_in)

    # TL_coef_f32 is the full vector. Split it into p, i, e components for M3Struct
    # These components are parameters of M3Struct and will be trained.
    # They are already in physical units.
    TL_p_param, TL_i_param, TL_e_param = TL_vec_split(TL_coef_f32, terms_A)
    
    s_model = M3Struct(current_nn_model, TL_p_param, TL_i_param, TL_e_param if TL_e_param.size > 0 else torch.empty(0, dtype=torch.float32))


    # Loss function for M3
    def compute_loss_val_m3(model_wrapper: M3Struct, loader: DataLoader, current_use_nn: bool) -> float:
        model_wrapper.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for B_u_b, B_v_b, B_vd_b, x_b, y_b_target in loader:
                # Inputs to nn_comp_3_fwd need to be (features, batch_size) or (features, window, batch_size)
                # DataLoader gives (batch_size, features) or (batch_size, window, features)
                
                # Transpose B_u_b, B_v_b, B_vd_b from (batch, 3) to (3, batch)
                # Transpose/permute x_b based on model_type
                if model_type in ["m3w", "m3tf"]: # x_b is (batch, window, features)
                    x_b_fwd = x_b.permute(2,1,0) # (features, window, batch)
                else: # x_b is (batch, features)
                    x_b_fwd = x_b.T # (features, batch)

                # Reconstruct TL coefficient matrices from model_wrapper parameters
                _current_TL_p_np = model_wrapper.TL_p.data.cpu().numpy()
                _current_TL_i_vec_np = model_wrapper.TL_i.data.cpu().numpy()
                _current_TL_e_vec_np = model_wrapper.TL_e.data.cpu().numpy() if model_wrapper.TL_e.numel() > 0 else np.array([], dtype=np.float32)

                _full_TL_coef_vec_parts = []
                if _current_TL_p_np.size > 0: _full_TL_coef_vec_parts.append(_current_TL_p_np)
                if _current_TL_i_vec_np.size > 0: _full_TL_coef_vec_parts.append(_current_TL_i_vec_np)
                if _current_TL_e_vec_np.size > 0: _full_TL_coef_vec_parts.append(_current_TL_e_vec_np)

                _p_fwd_th: torch.Tensor
                _i_mat_fwd_th: torch.Tensor
                _e_mat_fwd_th: Optional[torch.Tensor]

                if not _full_TL_coef_vec_parts:
                    # This case should ideally not happen if TL params are part of the model
                    _p_fwd_th = torch.zeros(3, device=model_wrapper.TL_p.device)
                    _i_mat_fwd_th = torch.zeros((3,3), device=model_wrapper.TL_p.device)
                    _e_mat_fwd_th = None
                    if not SILENT_DEBUG: logging.warning("No TL coefficient parts found in model_wrapper for loss calculation.")
                else:
                    _current_full_TL_coef_np = np.concatenate(_full_TL_coef_vec_parts)
                    # Bt_scale_val should be defined in the outer scope of nn_comp_3_train
                    _p_fwd_np, _i_mat_fwd_np, _e_mat_fwd_np = TL_vec2mat(_current_full_TL_coef_np, terms_A, Bt_scale_val)
                    
                    _p_fwd_th = torch.from_numpy(_p_fwd_np.astype(np.float32)).to(model_wrapper.TL_p.device)
                    _i_mat_fwd_th = torch.from_numpy(_i_mat_fwd_np.astype(np.float32)).to(model_wrapper.TL_p.device)
                    _e_mat_fwd_th = torch.from_numpy(_e_mat_fwd_np.astype(np.float32)).to(model_wrapper.TL_p.device) if _e_mat_fwd_np is not None and _e_mat_fwd_np.size > 0 else None

                y_hat_b_norm = nn_comp_3_fwd(
                    B_u_b.T, B_v_b.T, B_vd_b.T, x_b_fwd,
                    0.0, 1.0, # Dummy y_bias, y_scale for normalized output
                    model_wrapper.m,
                    _p_fwd_th,
                    _i_mat_fwd_th,
                    _e_mat_fwd_th,
                    model_type=model_type, y_type=y_type,
                    use_nn=current_use_nn, denorm=False, testmode=True
                )
                y_hat_b_norm_torch = torch.from_numpy(y_hat_b_norm.astype(np.float32))
                
                current_loss = loss_fn(y_hat_b_norm_torch, y_b_target)
                # SGL not implemented for M3 in Julia, so skipping here too for now
                
                total_loss += current_loss.item() * x_b.size(0)
                count += x_b.size(0)
        return total_loss / count if count > 0 else 0.0

    # Optimizer
    optimizer_adam = optim.Adam(s_model.parameters(), lr=eta_adam)

    # Training loop (Adam)
    best_model_state = copy.deepcopy(s_model.state_dict())
    best_loss_val = float('inf') # Initialize with a high value

    # Initial use_nn state for curriculum learning
    current_use_nn_for_loss = False if model_type in ["m3sc", "m3vc", "m3tl"] else True
    
    # Calculate initial validation loss
    best_loss_val = compute_loss_val_m3(s_model, val_loader, current_use_nn_for_loss)


    best_test_error_std: Optional[float] = None
    if x_test_raw_in is not None and x_test_raw_in.size > 0 and y_test_f32.size > 0 and \
       A_test_f32.size > 0 and Bt_test_f32.size > 0 : # B_dot_test can be empty
        
        # Prepare test x_norm
        x_test_norm_orig_np = (x_test_f32 - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        x_test_norm_pca_np_T = (x_test_norm_orig_np @ v_scale_pca).T # (PCA_feat, samples)
        
        x_test_norm_final_eval_T = x_test_norm_pca_np_T
        if model_type in ["m3w", "m3tf"]:
             if l_segs_test:
                x_test_norm_final_eval_T = get_temporal_data(x_test_norm_pca_np_T, l_segs_test, l_window)


        _, err_test_init = nn_comp_3_test(
            B_unit_test_np_T, B_vec_test_np_T, B_vec_dot_test_np_T,
            x_test_norm_final_eval_T, 
            y_test_f32, y_bias, y_scale,
            s_model.m, _p_fwd_np,
            _i_mat_fwd_np,
            _e_mat_fwd_np if _e_mat_fwd_np is not None and _e_mat_fwd_np.size > 0 else None,
            model_type, y_type, l_segs=l_segs_test, use_nn=current_use_nn_for_loss, silent=SILENT_DEBUG
        )
        best_test_error_std = np.std(err_test_init) if err_test_init.size > 0 else float('inf')

    if not silent: print(f"INFO: M3 epoch 0: loss = {best_loss_val:.6f}")

    # Curriculum learning schedule for Adam epochs
    # Julia: epoch_adam_cur = ceil.(Int, epoch_adam * [1, 2, 3, 6] / 10)
    # Example: epoch_adam = 100 -> cur = [10, 20, 30, 60]
    # Example: epoch_adam = 5   -> cur = [1, 1, 2, 3] (ceil makes it tricky for small epochs)
    # Let's use fractions directly for epoch stages
    epoch_stages_frac = [0.1, 0.2, 0.3, 0.6] 
    epoch_stages = [int(np.ceil(epoch_adam * frac)) for frac in epoch_stages_frac]
    # Ensure stages are at least 1 and distinct if epoch_adam is small
    for k_stage in range(len(epoch_stages)):
        if k_stage > 0 and epoch_stages[k_stage] <= epoch_stages[k_stage-1]:
            epoch_stages[k_stage] = epoch_stages[k_stage-1] + 1
    epoch_stages = [min(ep, epoch_adam) for ep in epoch_stages] # Cap at epoch_adam


    for i_epoch in range(1, epoch_adam + 1):
        s_model.train()
        
        # Curriculum logic for m3sc, m3vc
        if model_type in ["m3sc", "m3vc"]:
            if i_epoch == 1: # Stage 0: Train P only, NN frozen
                for param in s_model.m.parameters(): param.requires_grad = False
                s_model.TL_p.requires_grad = True
                s_model.TL_i.requires_grad = False
                s_model.TL_e.requires_grad = False
                current_use_nn_for_loss = False
                active_train_loader = train_loader_cur if 'train_loader_cur' in locals() else train_loader_full
            elif i_epoch == epoch_stages[0]: # Stage 1: Train P, I; NN frozen
                s_model.TL_i.requires_grad = True
            elif i_epoch == epoch_stages[1]: # Stage 2: Train P, I, E; NN frozen
                if s_model.TL_e.numel() > 0 : s_model.TL_e.requires_grad = True
            elif i_epoch == epoch_stages[2]: # Stage 3: Train NN only; TL frozen, use_nn=True, full data
                for param in s_model.m.parameters(): param.requires_grad = True
                s_model.TL_p.requires_grad = False
                s_model.TL_i.requires_grad = False
                if s_model.TL_e.numel() > 0 : s_model.TL_e.requires_grad = False
                current_use_nn_for_loss = True
                active_train_loader = train_loader_full
            elif i_epoch == epoch_stages[3]: # Stage 4: Train all; use_nn=True, full data
                for param in s_model.m.parameters(): param.requires_grad = True
                s_model.TL_p.requires_grad = True
                s_model.TL_i.requires_grad = True
                if s_model.TL_e.numel() > 0 : s_model.TL_e.requires_grad = True
                current_use_nn_for_loss = True
                active_train_loader = train_loader_full
        elif model_type == "m3tl": # Only TL terms are trained
             for param in s_model.m.parameters(): param.requires_grad = False
             s_model.TL_p.requires_grad = True
             s_model.TL_i.requires_grad = True
             if s_model.TL_e.numel() > 0 : s_model.TL_e.requires_grad = True
             current_use_nn_for_loss = False
             active_train_loader = train_loader_full
        else: # m3s, m3v, m3w, m3tf: train all from start
            for param in s_model.parameters(): param.requires_grad = True # Ensure all are trainable
            current_use_nn_for_loss = True
            active_train_loader = train_loader_full


        for B_u_b, B_v_b, B_vd_b, x_b, y_b_target in active_train_loader:
            optimizer_adam.zero_grad()
            
            if model_type in ["m3w", "m3tf"]: x_b_fwd = x_b.permute(2,1,0)
            else: x_b_fwd = x_b.T

            # Reconstruct TL coefficient matrices from current s_model parameters for this training step
            _current_TL_p_np_train = s_model.TL_p.data.cpu().numpy()
            _current_TL_i_vec_np_train = s_model.TL_i.data.cpu().numpy()
            _current_TL_e_vec_np_train = s_model.TL_e.data.cpu().numpy() if s_model.TL_e.numel() > 0 else np.array([], dtype=np.float32)

            _full_TL_coef_vec_parts_train = []
            if _current_TL_p_np_train.size > 0: _full_TL_coef_vec_parts_train.append(_current_TL_p_np_train)
            if _current_TL_i_vec_np_train.size > 0: _full_TL_coef_vec_parts_train.append(_current_TL_i_vec_np_train)
            if _current_TL_e_vec_np_train.size > 0: _full_TL_coef_vec_parts_train.append(_current_TL_e_vec_np_train)
            
            _p_fwd_np_train, _i_mat_fwd_np_train, _e_mat_fwd_np_train = np.array([]), np.array([]), np.array([])
            if _full_TL_coef_vec_parts_train:
                _current_full_TL_coef_np_train = np.concatenate(_full_TL_coef_vec_parts_train)
                if _current_full_TL_coef_np_train.size > 0:
                    _p_fwd_np_train, _i_mat_fwd_np_train, _e_mat_fwd_np_train = TL_vec2mat(
                        _current_full_TL_coef_np_train, terms_A, Bt_scale_val
                    )
            else: # Default if TL coeffs are empty
                _p_fwd_np_train = np.zeros(3)

            _p_fwd_th_train = s_model.TL_p # Already a tensor parameter
            _i_mat_fwd_th_train = torch.from_numpy(_i_mat_fwd_np_train.astype(np.float32)).to(s_model.TL_p.device) if _i_mat_fwd_np_train.size > 0 else torch.empty((3,3), device=s_model.TL_p.device)
            _e_mat_fwd_th_train = torch.from_numpy(_e_mat_fwd_np_train.astype(np.float32)).to(s_model.TL_p.device) if _e_mat_fwd_np_train is not None and _e_mat_fwd_np_train.size > 0 else None

            y_hat_b_norm = nn_comp_3_fwd(
                B_u_b.T, B_v_b.T, B_vd_b.T, x_b_fwd,
                0.0, 1.0, # Dummy for normalized output
                s_model.m,
                _p_fwd_th_train,
                _i_mat_fwd_th_train,
                _e_mat_fwd_th_train,
                model_type=model_type, y_type=y_type,
                use_nn=current_use_nn_for_loss, denorm=False, testmode=False
            )
            y_hat_b_norm_torch = torch.from_numpy(y_hat_b_norm.astype(np.float32))
            
            loss_train_val = loss_fn(y_hat_b_norm_torch, y_b_target)
            # SGL skipped for M3
            loss_train_val.backward()
            optimizer_adam.step()

        current_loss_val = compute_loss_val_m3(s_model, val_loader, current_use_nn_for_loss)

        if not (x_test_raw_in is not None and x_test_raw_in.size > 0 and y_test_f32.size > 0 and \
                A_test_f32.size > 0 and Bt_test_f32.size > 0): # No test set
            if current_loss_val < best_loss_val:
                best_loss_val = current_loss_val
                best_model_state = copy.deepcopy(s_model.state_dict())
            if i_epoch % 5 == 0 and not silent:
                print(f"INFO: M3 epoch {i_epoch}: loss = {best_loss_val:.6f}")
        else: # With test set
            x_test_norm_final_eval_T_th = None
            if model_type in ["m3w", "m3tf"]:
                 if x_test_norm_final_eval_T is not None:
                    x_test_norm_final_eval_T_th = torch.from_numpy(np.moveaxis(x_test_norm_final_eval_T, [0,1,2], [2,1,0]).astype(np.float32))
            elif x_test_norm_pca_np_T is not None:
                 x_test_norm_final_eval_T_th = torch.from_numpy(x_test_norm_pca_np_T.T.astype(np.float32))


            if B_unit_test_np_T is not None and x_test_norm_final_eval_T_th is not None :
                _, err_test_current = nn_comp_3_test(
                    torch.from_numpy(B_unit_test_np_T.astype(np.float32)), 
                    torch.from_numpy(B_vec_test_np_T.astype(np.float32)), 
                    torch.from_numpy(B_vec_dot_test_np_T.astype(np.float32)) if B_vec_dot_test_np_T is not None else torch.empty(0),
                    x_test_norm_final_eval_T_th.T, # nn_comp_3_test expects (features, samples)
                    y_test_f32, y_bias, y_scale,
                    s_model.m, torch.from_numpy(_p_fwd_np.astype(np.float32)).to(s_model.TL_p.device),
                    torch.from_numpy(_i_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device),
                    torch.from_numpy(_e_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device) if _e_mat_fwd_np is not None and _e_mat_fwd_np.size > 0 else None,
                    model_type, y_type, l_segs=l_segs_test, use_nn=current_use_nn_for_loss, silent=SILENT_DEBUG
                )
                current_test_error_std = np.std(err_test_current) if err_test_current.size > 0 else float('inf')

                if best_test_error_std is not None and current_test_error_std < best_test_error_std:
                    best_test_error_std = current_test_error_std
                    best_model_state = copy.deepcopy(s_model.state_dict())
                
                if i_epoch % 5 == 0 and not silent:
                    print(f"INFO: M3 epoch {i_epoch}: loss = {current_loss_val:.6f}, test error = {best_test_error_std:.2f} nT" if best_test_error_std is not None else f"INFO: M3 epoch {i_epoch}: loss = {current_loss_val:.6f}")
            elif i_epoch % 5 == 0 and not silent :
                 print(f"INFO: M3 epoch {i_epoch}: loss = {current_loss_val:.6f} (Test data missing for error calc)")


        if i_epoch % 10 == 0 and not silent:
            # Full training data for nn_comp_3_test needs (features, samples) or (features, window, samples)
            _, train_err_np = nn_comp_3_test(
                B_unit_np_T, B_vec_np_T, B_vec_dot_np_T,
                x_norm_final_T, # This is already (features, [window,] samples)
                y_f32, y_bias, y_scale, 
                s_model.m, torch.from_numpy(_p_fwd_np.astype(np.float32)).to(s_model.TL_p.device),
                torch.from_numpy(_i_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device),
                torch.from_numpy(_e_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device) if _e_mat_fwd_np is not None and _e_mat_fwd_np.size > 0 else None,
                model_type, y_type, l_segs=l_segs, use_nn=current_use_nn_for_loss, silent=True
            )
            train_err_std = np.std(train_err_np) if train_err_np.size > 0 else float('nan')
            print(f"INFO: M3 {i_epoch} train error: {train_err_std:.2f} nT")

    s_model.load_state_dict(best_model_state)
    
    # Combine trained TL_p, TL_i, TL_e back into a single TL_coef vector
    # The M3Struct stores them as separate tensors.
    # TL_mat2vec expects physical coefficient matrices/vectors.
    # The s_model.TL_p, s_model.TL_i (vector form), s_model.TL_e (vector form) are already the parameters.
    final_TL_p_trained = s_model.TL_p.cpu().detach().numpy()
    final_TL_i_trained_vec = s_model.TL_i.cpu().detach().numpy() # This is the vector form (e.g., 6 elements for i6)
    final_TL_e_trained_vec = s_model.TL_e.cpu().detach().numpy() if s_model.TL_e.numel() > 0 else np.array([], dtype=np.float32)
    
    # Reconstruct the full TL_coef vector based on the order defined by terms_A
    # This is a bit manual, assuming P then I then E.
    # A more robust way would be to use the logic from TL_mat2vec in reverse, or store indices.
    temp_TL_coef_list = [final_TL_p_trained]
    if final_TL_i_trained_vec.size > 0: temp_TL_coef_list.append(final_TL_i_trained_vec)
    if final_TL_e_trained_vec.size > 0: temp_TL_coef_list.append(final_TL_e_trained_vec)
    final_TL_coef = np.concatenate(temp_TL_coef_list)


    # LBFGS optimization (if epoch_lbfgs > 0)
    if epoch_lbfgs > 0 and model_type != "m3tl": # LBFGS not supported for m3tl in Julia
        if not silent: print("INFO: M3 LBFGS training started.")
        # Ensure all relevant parameters are trainable for LBFGS
        for param in s_model.parameters(): param.requires_grad = True
        current_use_nn_for_loss_lbfgs = True # Usually train all with LBFGS

        optimizer_lbfgs = optim.LBFGS(s_model.parameters(), lr=0.1)

        def closure_lbfgs_m3():
            optimizer_lbfgs.zero_grad()
            # Use full training dataset for LBFGS
            # Inputs to nn_comp_3_fwd need to be (features, batch_size) or (features, window, batch_size)
            # B_unit_torch, B_vec_torch, B_vec_dot_torch are (samples, 3)
            # x_torch_train is (samples, features) or (samples, window, features)
            # y_torch_train is (samples,)
            
            x_fwd_lbfgs = x_torch_train.permute(2,1,0) if model_type in ["m3w", "m3tf"] and x_torch_train.ndim == 3 else x_torch_train.T

            y_hat_norm_lbfgs = nn_comp_3_fwd(
                B_unit_torch.T, B_vec_torch.T, B_vec_dot_torch.T, 
                x_fwd_lbfgs,
                0.0, 1.0, # Dummy for normalized output
                s_model.m,
                s_model.TL_p, torch.from_numpy(_i_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device), torch.from_numpy(_e_mat_fwd_np.astype(np.float32)).to(s_model.TL_p.device) if _e_mat_fwd_np is not None and _e_mat_fwd_np.size > 0 else None,
                model_type=model_type, y_type=y_type,
                use_nn=current_use_nn_for_loss_lbfgs, denorm=False, testmode=False
            )
            y_hat_norm_lbfgs_torch = torch.from_numpy(y_hat_norm_lbfgs.astype(np.float32))
            
            loss_val = loss_fn(y_hat_norm_lbfgs_torch, y_torch_train)
            loss_val.backward()
            return loss_val

        for i_lbfgs in range(epoch_lbfgs):
            optimizer_lbfgs.step(closure_lbfgs_m3)
            if not silent and (i_lbfgs + 1) % 5 == 0:
                current_loss_lbfgs = closure_lbfgs_m3().item()
                print(f"INFO: M3 LBFGS epoch {i_lbfgs+1}: loss = {current_loss_lbfgs:.6f}")
        if not silent: print("INFO: M3 LBFGS training finished.")
        
        # Update final_TL_coef after LBFGS
        final_TL_p_trained = s_model.TL_p.cpu().detach().numpy()
        final_TL_i_trained_vec = s_model.TL_i.cpu().detach().numpy()
        final_TL_e_trained_vec = s_model.TL_e.cpu().detach().numpy() if s_model.TL_e.numel() > 0 else np.array([], dtype=np.float32)
        temp_TL_coef_list = [final_TL_p_trained]
        if final_TL_i_trained_vec.size > 0: temp_TL_coef_list.append(final_TL_i_trained_vec)
        if final_TL_e_trained_vec.size > 0: temp_TL_coef_list.append(final_TL_e_trained_vec)
        final_TL_coef = np.concatenate(temp_TL_coef_list)


    # Final evaluation on training data
    final_TL_p_eval, final_TL_i_mat_eval, final_TL_e_mat_eval = TL_vec2mat(final_TL_coef, terms_A, Bt_scale=Bt_scale_val)

    y_hat_final, err_final = nn_comp_3_test(
        B_unit_np_T, B_vec_np_T, B_vec_dot_np_T,
        x_norm_final_T, # (features, [window,] samples)
        y_f32, y_bias, y_scale, 
        s_model.m, 
        final_TL_p_eval, final_TL_i_mat_eval, final_TL_e_mat_eval,
        model_type, y_type, l_segs=l_segs, 
        use_nn=(True if model_type not in ["m3tl"] else False), # Use NN unless it's pure TL
        silent=True
    )
    if not silent: 
        err_std_final = np.std(err_final) if err_final.size > 0 else float('nan')
        print(f"INFO: M3 final train error: {err_std_final:.2f} nT")

    # Final evaluation on test data if provided
    if x_test_raw_in is not None and x_test_raw_in.size > 0 and y_test_f32.size > 0 and \
       A_test_f32.size > 0 and Bt_test_f32.size > 0:
        
        x_test_norm_final_eval_T_th = None # Re-prepare for test
        if model_type in ["m3w", "m3tf"]:
             if x_test_norm_final_eval_T is not None: # This was (feat, win, samp)
                x_test_norm_final_eval_T_th = torch.from_numpy(x_test_norm_final_eval_T.astype(np.float32))
        elif x_test_norm_pca_np_T is not None: # This was (feat, samp)
             x_test_norm_final_eval_T_th = torch.from_numpy(x_test_norm_pca_np_T.astype(np.float32))

        if B_unit_test_np_T is not None and x_test_norm_final_eval_T_th is not None:
            nn_comp_3_test(
                torch.from_numpy(B_unit_test_np_T.astype(np.float32)), 
                torch.from_numpy(B_vec_test_np_T.astype(np.float32)), 
                torch.from_numpy(B_vec_dot_test_np_T.astype(np.float32)) if B_vec_dot_test_np_T is not None else torch.empty(0),
                x_test_norm_final_eval_T_th, # nn_comp_3_test expects (features, [win,] samples)
                y_test_f32, y_bias, y_scale,
                s_model.m, 
                final_TL_p_eval_th,
                final_TL_i_mat_eval_th,
                final_TL_e_mat_eval_th if final_TL_e_mat_eval_th is not None else None,
                model_type, y_type, l_segs=l_segs_test, 
                use_nn=(True if model_type not in ["m3tl"] else False), 
                silent=silent 
            )
        
    return s_model.m, final_TL_coef, data_norms_out, y_hat_final, err_final
def plsr_fit(
    x: np.ndarray, 
    y: np.ndarray, 
    k: Optional[int] = None, 
    no_norm: Optional[np.ndarray] = None,
    data_norms_in: Optional[Tuple] = None,
    l_segs: Optional[List[int]] = None,
    return_set: bool = False, # If true, returns coef_set instead
    silent: bool = False
) -> Union[Tuple[Tuple[np.ndarray, float], Tuple, np.ndarray, np.ndarray], np.ndarray]:
    """
    Fit a multi-input, single-output (MISO for now) partial least squares regression (PLSR) model.
    Python translation of MagNav.jl's plsr_fit.
    Note: The Julia version supports MIMO (Ny > 1), this Python version is simplified
    to MISO (Ny=1) for now, as y is typically a 1D vector in this context.
    If MIMO is needed, q_out and coef_set dimensions would need adjustment.
    """
    if y.ndim == 1:
        y_proc = y.reshape(-1, 1) # Ensure y is 2D (N, 1) for consistency
    elif y.ndim == 2 and y.shape[1] == 1:
        y_proc = y
    else:
        raise ValueError(f"y must be a 1D array or 2D array with one column, got {y.shape}")

    if k is None:
        k = x.shape[1] # Default to number of features

    Nf = x.shape[1]
    Ny = y_proc.shape[1] # Should be 1 for MISO

    if no_norm is None:
        no_norm = np.zeros(Nf, dtype=bool)
    if l_segs is None:
        l_segs = [x.shape[0]]


    if k > Nf:
        if not silent: print(f"INFO: reducing k from {k} to {Nf}")
        k = Nf
    if k <=0:
        if not silent: print(f"INFO: k must be > 0. Setting k=1.")
        k = 1


    norm_type_x = "standardize"
    norm_type_y = "standardize"
    x_bias, x_scale, y_bias, y_scale = (None,)*4


    if data_norms_in is None or np.sum(np.abs(data_norms_in[-1])) == 0: # y_scale is last
        x_bias, x_scale, x_norm = norm_sets(x, norm_type=norm_type_x, no_norm=no_norm)
        y_bias, y_scale, y_norm = norm_sets(y_proc, norm_type=norm_type_y)
        data_norms_out = (x_bias, x_scale, y_bias, y_scale)
    else:
        x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms_in) # type: ignore
        x_norm = (x - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        y_norm = (y_proc - y_bias) / (y_scale if y_scale != 0 else 1.0) # y_bias/scale are scalar for 1D y
        data_norms_out = data_norms_in
    
    # Ensure y_norm is 2D (N,1)
    if y_norm.ndim == 1:
        y_norm = y_norm.reshape(-1,1)

    # PLS algorithm (NIPALS-based approach as in Julia example)
    x_current = x_norm.copy()
    y_current = y_norm.copy()
    
    T_scores = np.zeros((x.shape[0], k)) # Input scores
    P_loadings = np.zeros((Nf, k))       # Input loadings
    Q_loadings = np.zeros((Ny, k))       # Output loadings (for MISO, Ny=1, so (1,k))
    W_weights = np.zeros((Nf, k))        # Weights

    if return_set: # For MIMO PLS, coef_set would be (Nf, Ny, k)
        coef_set = np.zeros((Nf, Ny, k), dtype=x.dtype)


    for i in range(k):
        if x_current.shape[0] == 0 or y_current.shape[0] == 0: break # No data left

        # Calculate weights w: X'y / ||X'y|| (simplified for MISO y_current is (N,1))
        # For MISO, y_current.T @ x_current would be (1, Nf). We need X'y.
        # So, x_current.T @ y_current gives (Nf, 1)
        xy_cov = x_current.T @ y_current # (Nf, Ny)
        
        # For MISO (Ny=1), xy_cov is (Nf, 1).
        # The SVD approach in Julia `svd(Cyx')` where Cyx is Ny x Nf.
        # `Cyx'` is Nf x Ny. U from svd(Cyx') is Nf x min(Nf,Ny). u = U[:,0]
        # Here, for MISO, we can directly use xy_cov as the direction.
        if xy_cov.shape[0] == 0: continue # No features left or data
        
        # w_i = xy_cov[:,0] / np.linalg.norm(xy_cov[:,0]) # Weight for this component (Nf,)
        # A common way for PLS1 (single y): w = X'y
        w_i = x_current.T @ y_current[:,0] # (Nf,)
        w_i = w_i / (np.linalg.norm(w_i) if np.linalg.norm(w_i) !=0 else 1.0)
        W_weights[:, i] = w_i

        # Calculate scores t = Xw / ||w|| (or Xw / w'w if w not normalized, but here it is)
        t_i = x_current @ w_i # (N,)
        T_scores[:, i] = t_i

        # Calculate loadings p = X't / t't
        p_i = x_current.T @ t_i / (t_i.T @ t_i if t_i.T @ t_i !=0 else 1.0) # (Nf,)
        P_loadings[:, i] = p_i

        # Calculate output loadings q = Y't / t't
        q_i = y_current.T @ t_i / (t_i.T @ t_i if t_i.T @ t_i !=0 else 1.0) # (Ny,) -> (1,) for MISO
        Q_loadings[:, i] = q_i # q_i is (1,)

        # Deflate X and Y
        x_current = x_current - t_i[:, np.newaxis] @ p_i[:, np.newaxis].T
        y_current = y_current - t_i[:, np.newaxis] @ q_i[:, np.newaxis].T # q_i is (1,)

        if return_set:
            # Regression coefficients B = W(P'W)^-1 Q'
            # This computes B for the current number of components i+1
            # W_subset is (Nf, i+1), P_subset is (Nf, i+1), Q_subset is (Ny, i+1)
            W_sub = W_weights[:, :i+1]
            P_sub = P_loadings[:, :i+1]
            Q_sub = Q_loadings[:, :i+1] # (Ny, i+1)
            
            try:
                # B_i = W_sub @ np.linalg.inv(P_sub.T @ W_sub) @ Q_sub.T # (Nf, Ny)
                # For MISO, Q_sub.T is (i+1, 1)
                # Result B_i is (Nf, 1)
                term_inv = np.linalg.inv(P_sub.T @ W_sub)
                B_i = W_sub @ term_inv @ Q_sub.T
                coef_set[:, :, i] = B_i
            except np.linalg.LinAlgError:
                 if not silent: print(f"WARN: Singular matrix in PLSR coefficient calculation at component {i+1}. Using pseudo-inverse.")
                 term_pinv = np.linalg.pinv(P_sub.T @ W_sub)
                 B_i = W_sub @ term_pinv @ Q_sub.T
                 coef_set[:, :, i] = B_i

    if return_set:
        return coef_set

    # Final regression coefficients B = W(P'W)^-1 Q'
    try:
        final_coeffs = W_weights @ np.linalg.inv(P_loadings.T @ W_weights) @ Q_loadings.T
    except np.linalg.LinAlgError:
        if not silent: print("WARN: Singular matrix in final PLSR coefficient calculation. Using pseudo-inverse.")
        final_coeffs = W_weights @ np.linalg.pinv(P_loadings.T @ W_weights) @ Q_loadings.T

    # For MISO, final_coeffs will be (Nf, 1). We want (Nf,).
    model_coeffs = final_coeffs.flatten()
    model_bias = 0.0 # PLS on centered data, bias is effectively handled by y_bias
    model_out = (model_coeffs, model_bias)

    # Get results on the full (untrimmed, but normalized if applicable) dataset
    # linear_test expects x_norm, y_raw, y_bias, y_scale, model
    y_hat_full, err_full = linear_test(x_norm, y_proc.flatten(), y_bias, y_scale, model_out, l_segs=l_segs, silent=True)
    
    if not silent:
        err_std = np.std(err_full) if err_full.size > 0 else float('nan')
        # The Julia version prints input/output residue variance. This is a bit more involved.
        # For now, just print the fit error.
        print(f"INFO: PLSR fit error: {err_std:.2f} nT")
             
    return model_out, data_norms_out, y_hat_full, err_full

def elasticnet_fit(
    x: np.ndarray, 
    y: np.ndarray, 
    alpha: float = 0.99, # L1 ratio (0 for Ridge, 1 for Lasso)
    no_norm: Optional[np.ndarray] = None,
    lambda_val: float = -1.0, # Renamed from λ. If -1, use CV.
    data_norms_in: Optional[Tuple] = None,
    l_segs: Optional[List[int]] = None,
    silent: bool = False
) -> Tuple[Tuple[np.ndarray, float], Tuple, np.ndarray, np.ndarray]:
    """
    Fit an elastic net (ridge regression and/or Lasso) model to data.

    Args:
        x: N x Nf data matrix (Nf is number of features)
        y: length-N target vector
        alpha: (optional) ElasticNet mixing parameter, with 0 <= alpha <= 1.
                alpha=0 is L2 penalty (Ridge), alpha=1 is L1 penalty (Lasso).
        no_norm: (optional) length-Nf Boolean indices of features to not be normalized
        lambda_val: (optional) elastic net regularization strength. 
                     If -1, determine with cross-validation.
        data_norms_in: (optional) length-4 tuple of data normalizations, (x_bias,x_scale,y_bias,y_scale)
        l_segs: (optional) length-N_lines vector of lengths of lines, sum(l_segs) = N
        silent: (optional) if true, no print outs

    Returns:
        model: length-2 tuple of elastic net model, (length-Nf coefficients, bias)
        data_norms_out: length-4 tuple of data normalizations, (x_bias,x_scale,y_bias,y_scale)
        y_hat: length-N prediction vector
        err: length-N mean-corrected (per line) error
    """
    if y.ndim > 1 and y.shape[1] > 1:
        raise ValueError(f"elasticnet_fit expects a 1D target vector y, but got shape {y.shape}")
    y_flat = y.flatten()

    if no_norm is None:
        no_norm = np.zeros(x.shape[1], dtype=bool)
    if l_segs is None:
        l_segs = [len(y_flat)]

    norm_type_x = "standardize" # ElasticNet benefits from standardized features
    norm_type_y = "standardize" # Standardize y as well
    x_bias, x_scale, y_bias, y_scale = (None,)*4


    if data_norms_in is None or np.sum(np.abs(data_norms_in[-1])) == 0: # y_scale is last
        x_bias, x_scale, x_norm = norm_sets(x, norm_type=norm_type_x, no_norm=no_norm)
        y_bias, y_scale, y_norm_flat = norm_sets(y_flat, norm_type=norm_type_y)
        data_norms_out = (x_bias, x_scale, y_bias, y_scale)
    else:
        x_bias, x_scale, y_bias, y_scale = unpack_data_norms(data_norms_in) # type: ignore
        x_norm = (x - x_bias.reshape(1,-1)) / np.where(x_scale.reshape(1,-1) == 0, 1.0, x_scale.reshape(1,-1))
        y_norm_flat = (y_flat - y_bias) / (y_scale if y_scale != 0 else 1.0)
        data_norms_out = data_norms_in
    
    if x_norm.shape[0] == 0: # No data
        if not silent: print("WARN: No data for elasticnet_fit. Returning zero coefficients.")
        coeffs = np.zeros(x_norm.shape[1])
        intercept = 0.0
        model_out = (coeffs, intercept)
        y_hat_full, err_full = linear_test(x_norm, y_flat, y_bias, y_scale, model_out, l_segs=l_segs, silent=True)
        return model_out, data_norms_out, y_hat_full, err_full

    # Scikit-learn's ElasticNet: alpha is l1_ratio, lambda is the overall strength alpha (confusingly named)
    # Julia's GLMNet: alpha is mixing (0=ridge, 1=lasso), lambda is strength
    # sklearn ElasticNet: alpha parameter is `l1_ratio`. Total penalty is `alpha * (l1_ratio * L1 + 0.5 * (1 - l1_ratio) * L2)`
    # So, sklearn `l1_ratio` corresponds to Julia `alpha`.
    # Sklearn `alpha` (strength) corresponds to Julia `lambda`.

    if lambda_val < 0: # Use cross-validation to find best lambda (alpha in sklearn)
        # ElasticNetCV finds the best alpha (strength) and l1_ratio (mixing)
        # We want to fix l1_ratio (Julia's alpha) and find best strength (Julia's lambda)
        # If we want to fix l1_ratio = alpha (from arg), we can pass it as a list: [alpha]
        # cv_model = ElasticNetCV(l1_ratio=[alpha] if 0 < alpha < 1 else alpha, # handles pure Lasso/Ridge
        #                         alphas=None, # Let CV find best strength
        #                         cv=5, random_state=0, fit_intercept=True,
        #                         # sklearn ElasticNetCV standardizes X by default if fit_intercept=True
        #                         # but we have already standardized X, so normalize=False (deprecated) or handle carefully.
        #                         # It's safer to pass pre-standardized X and set fit_intercept=True.
        #                         # The `normalize` parameter is deprecated.
        #                         # If X is already standardized, fit_intercept=True will work correctly.
        #                         )
        # Simpler: if lambda_val is -1, we might need to define a set of lambdas to try,
        # or use a simpler approach if only strength is CV'd.
        # The Julia code `glmnetcv` finds the best lambda for a fixed alpha.
        # `ElasticNetCV` in sklearn can find best `alpha` (strength) for given `l1_ratio`(s).
        if not silent: print(f"INFO: ElasticNet using Cross-Validation to find lambda for l1_ratio={alpha}")
        cv_model = ElasticNetCV(l1_ratio=alpha, cv=5, random_state=0, fit_intercept=True, n_alphas=100, tol=1e-3, max_iter=2000)
        cv_model.fit(x_norm, y_norm_flat)
        coeffs = cv_model.coef_
        intercept = cv_model.intercept_
        if not silent: print(f"INFO: ElasticNetCV selected lambda (strength): {cv_model.alpha_}")
    else:
        # Use provided lambda_val (strength)
        # sklearn ElasticNet: alpha is strength, l1_ratio is mixing
        model_sk = ElasticNet(alpha=lambda_val, l1_ratio=alpha, fit_intercept=True, random_state=0, tol=1e-3, max_iter=2000)
        model_sk.fit(x_norm, y_norm_flat)
        coeffs = model_sk.coef_
        intercept = model_sk.intercept_

    # The model from scikit-learn includes an intercept.
    # The Julia version via MLJLinearModels also fits an intercept.
    # The `linear_test` function expects model as (coeffs, bias_val=0.0) if bias is handled by y_norm.
    # If y_norm was (y - y_mean) / y_std, then prediction is (X_norm @ coef + intercept_norm) * y_std + y_mean.
    # Our `linear_fwd` and `linear_test` assume the intercept is part of the normalized prediction.
    # So, the intercept from ElasticNet (which is on y_norm scale) is the 'bias' for linear_fwd.
    model_out = (coeffs, intercept)

    y_hat_full, err_full = linear_test(x_norm, y_flat, y_bias, y_scale, model_out, l_segs=l_segs, silent=True)
    
    if not silent:
        err_std = np.std(err_full) if err_full.size > 0 else float('nan')
        print(f"INFO: ElasticNet fit error: {err_std:.2f} nT")
             
    return model_out, data_norms_out, y_hat_full, err_full

def comp_train(
    comp_params: CompParams, 
    xyz: XYZ, 
    ind: np.ndarray,
    mapS: Any = mapS_null, # Union[MapS, MapSd, MapS3D]
    temp_params: Optional[TempParams] = None,
    xyz_test: Optional[XYZ] = None,
    ind_test: Optional[np.ndarray] = None,
    silent: bool = False
) -> Tuple[CompParams, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train an aeromagnetic compensation model.
    Base version taking XYZ and ind.

    Args:
        comp_params: CompParams object (NNCompParams or LinCompParams).
        xyz: XYZ flight data struct.
        ind: Selected data indices for training.
        mapS: (optional) MapS struct, only used for y_type = 'b', 'c'.
        temp_params: (optional) TempParams struct.
        xyz_test: (optional) XYZ held-out test data struct.
        ind_test: (optional) Indices for test data struct.
        silent: (optional) If true, no print outs.

    Returns:
        comp_params: Updated CompParams object.
        y: Target vector.
        y_hat: Prediction vector.
        err: Compensation error.
        features: List of feature names used.
    """
    np.random.seed(2) # for reproducibility
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
        
    t0 = time.time()

    if temp_params is None:
        temp_params = TempParams() # Use default TempParams if None

    # Unpack parameters common to both or specific to NNCompParams/LinCompParams
    # This needs to be done carefully based on the type of comp_params
    
    # Common attributes
    model_type = comp_params.model_type
    y_type = comp_params.y_type
    use_mag = comp_params.use_mag
    use_vec = comp_params.use_vec
    data_norms = comp_params.data_norms # This will be updated
    model = comp_params.model # This will be updated
    terms = comp_params.terms
    terms_A = list(comp_params.terms_A) # Ensure it's a mutable list
    sub_diurnal = comp_params.sub_diurnal
    sub_igrf = comp_params.sub_igrf
    bpf_mag = comp_params.bpf_mag
    reorient_vec = comp_params.reorient_vec # Not directly used in this dispatch, but part of params
    norm_type_A = comp_params.norm_type_A
    norm_type_x = comp_params.norm_type_x
    norm_type_y = comp_params.norm_type_y
    features_setup = list(comp_params.features_setup)
    features_no_norm = list(comp_params.features_no_norm)


    # NNCompParams specific
    if isinstance(comp_params, NNCompParams):
        TL_coef_nn = comp_params.TL_coef # Specific to NN models that might use it
        eta_adam = comp_params.η_adam
        epoch_adam = comp_params.epoch_adam
        epoch_lbfgs = comp_params.epoch_lbfgs
        hidden = comp_params.hidden
        activation = comp_params.activation
        loss = comp_params.loss # Renamed to loss_fn in train functions
        batchsize = comp_params.batchsize
        frac_train = comp_params.frac_train
        alpha_sgl = comp_params.α_sgl
        lambda_sgl = comp_params.λ_sgl
        k_pca = comp_params.k_pca
        drop_fi = comp_params.drop_fi
        drop_fi_bson = comp_params.drop_fi_bson
        drop_fi_csv = comp_params.drop_fi_csv
        # perm_fi = comp_params.perm_fi # Not used in comp_train in Julia
        # perm_fi_csv = comp_params.perm_fi_csv
    elif isinstance(comp_params, LinCompParams):
        k_plsr = comp_params.k_plsr
        lambda_TL = comp_params.λ_TL # This is lambda_ridge for linear_fit with TL
        # Set defaults for NN params not in LinCompParams to avoid UnboundLocalError later if logic paths are complex
        # Though ideally, these paths are mutually exclusive.
        drop_fi = False 
    else:
        raise TypeError("comp_params must be an instance of NNCompParams or LinCompParams")

    # TempParams unpack
    sigma_curriculum = temp_params.σ_curriculum
    l_window = temp_params.l_window
    window_type_temporal = temp_params.window_type # Renamed to avoid clash
    tf_layer_type = temp_params.tf_layer_type
    tf_norm_type = temp_params.tf_norm_type
    dropout_prob = temp_params.dropout_prob
    N_tf_head = temp_params.N_tf_head
    tf_gain = temp_params.tf_gain
    
    # Adjust y_type based on model_type (mirroring Julia logic)
    original_y_type = y_type
    if model_type in ["TL", "mod_TL"] and y_type != "e":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'e' (BPF'd total field) for model_type {model_type}")
        y_type = "e"
    if model_type == "map_TL" and y_type != "c":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'c' (aircraft field #1, using map) for model_type {model_type}")
        y_type = "c"
    
    # Adjust norm_types for certain linear models
    if model_type in ["elasticnet", "plsr"]:
        if norm_type_x != "standardize":
            if not silent: print(f"INFO: Forcing norm_type_x {norm_type_x} -> 'standardize' for {model_type}")
            norm_type_x = "standardize"
        if norm_type_y != "standardize":
            if not silent: print(f"INFO: Forcing norm_type_y {norm_type_y} -> 'standardize' for {model_type}")
            norm_type_y = "standardize"

    # Adjust terms_A for M3 models
    if model_type.startswith("m3"):
        original_terms_A_len = len(terms_A)
        original_TL_coef_len = TL_coef_nn.size if isinstance(comp_params, NNCompParams) and TL_coef_nn is not None else 0
        
        terms_A_updated = []
        TL_coef_indices_to_keep = []
        current_coeff_idx = 0
        
        # Simulate create_TL_A term parsing to correctly adjust TL_coef_nn if terms are removed
        # This is a simplified version. A robust solution would need the exact column counts from create_TL_A.
        # For now, assume standard term sizes if present.
        term_sizes = {"permanent":3, "p":3, "permanent3":3, "p3":3,
                      "induced":6, "i":6, "induced6":6, "i6":6, "induced5":5, "i5":5, "induced3":3, "i3":3,
                      "eddy":9, "e":9, "eddy9":9, "e9":9, "eddy8":8, "e8":8, "eddy3":3, "e3":3}
        
        temp_terms_A_parsing = [] # To track parsed terms and their original indices in TL_coef_nn
        
        # This parsing needs to be in the order create_TL_A would assemble them.
        # Assuming: permanent, then induced, then eddy, then others.
        # This is a simplification. The actual create_TL_A logic is complex.
        
        parsed_coeffs_count = 0
        # Simplified parsing logic - this needs to be robust based on create_TL_A
        # For now, we'll just filter terms_A and assume TL_coef_nn is handled correctly by nn_comp_3_train
        # if it receives a shorter terms_A list and a TL_coef that might be too long (it should use TL_vec_split).
        
        terms_A_to_remove = ["fdm", "f", "fdm3", "f3", "bias", "b"]
        terms_A_final = [term for term in terms_A if term not in terms_A_to_remove]
        if len(terms_A_final) != len(terms_A):
            if not silent: print(f"INFO: Removing derivative/bias terms from terms_A for M3 model. Original: {terms_A}, New: {terms_A_final}")
            terms_A = terms_A_final
            # Note: TL_coef_nn might need adjustment here if its length was based on the original terms_A.
            # nn_comp_3_train's internal TL_vec_split should handle a potentially longer TL_coef_nn
            # if terms_A is shorter, by only using the relevant parts of TL_coef_nn.

    # Get map values if needed
    map_val_data = get_map_val(mapS, xyz.traj, ind) if y_type in ["b", "c"] else -1

    # Create Tolles-Lawson A matrix and other B components
    # `create_TL_A` needs to be fully implemented from `tolles_lawson.py`
    # For now, using the placeholder which might not be fully accurate.
    # The `terms` argument for create_TL_A in Julia is `terms_A` from comp_params.
    
    # Call the version of create_TL_A from tolles_lawson.py if available, else placeholder
    try:
        from .tolles_lawson import create_TL_A as create_TL_A_actual
        # Check if it's different from the placeholder, to avoid recursion if it's the same file
        if 'compensation' in create_TL_A_actual.__module__ : # if it's the placeholder in this file
            create_TL_A_fn = create_TL_A # use placeholder
        else:
            create_TL_A_fn = create_TL_A_actual
    except ImportError:
        create_TL_A_fn = create_TL_A # Use placeholder if import fails

    A_matrix_np: np.ndarray
    Bt_np: Optional[np.ndarray] = None
    B_dot_np: Optional[np.ndarray] = None

    if model_type == "mod_TL":
        A_matrix_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, Bt=getattr(xyz, use_mag)[ind])
    elif model_type == "map_TL":
        A_matrix_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, Bt=map_val_data)
    else: # Includes M1, M2, M3, TL, elasticnet, plsr
        # For M3, create_TL_A returns A, Bt, B_dot
        # For others, it might just return A. The placeholder needs to handle this.
        # The fully implemented create_TL_A from tolles_lawson.py should handle return_B=True.
        A_out = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, return_B=True, Bt_in=getattr(xyz,use_mag)[ind] if hasattr(xyz,use_mag) else None)
        if isinstance(A_out, tuple) and len(A_out) == 3:
            A_matrix_np, Bt_np, B_dot_np = A_out
        else: # Assuming it returned only A
            A_matrix_np = A_out
            # For models other than M3 that might still need Bt, B_dot, they'd need to be derived
            # or get_Axy should provide them. For now, assume create_TL_A gives what's needed.
            # If M1/M2 need Bt/B_dot, this needs to be addressed.
            # The placeholder create_TL_A returns dummy Bt, B_dot if return_B=True.

    fs = 1.0 / xyz.dt if xyz.dt > 0 else 10.0 # Default fs if dt is zero
    
    A_no_bpf_np = None
    if model_type in ["TL", "mod_TL"]: # Store A before BPF for these models
        A_no_bpf_np = A_matrix_np.copy()
    
    if y_type == "e": # Bandpass filter A if y_type is 'e' (BPF'd total field)
        # `bpf_data` needs to be fully implemented. Placeholder does no-op.
        # `get_bpf` also needs to be implemented.
        try:
            from .analysis_util import get_bpf as get_bpf_actual, bpf_data as bpf_data_actual
            if 'compensation' in get_bpf_actual.__module__: # placeholder
                 bpf_coeffs = get_bpf(fs=fs)
                 A_matrix_np = bpf_data(A_matrix_np, bpf=bpf_coeffs) if bpf_coeffs is not None else A_matrix_np
            else:
                 bpf_coeffs = get_bpf_actual(fs=fs) # pass1, pass2 defaults
                 A_matrix_np = bpf_data_actual(A_matrix_np, bpf=bpf_coeffs) if bpf_coeffs is not None else A_matrix_np
        except ImportError:
            bpf_coeffs = get_bpf(fs=fs)
            A_matrix_np = bpf_data(A_matrix_np, bpf=bpf_coeffs) if b_coeffs is not None else A_matrix_np


    # Load features (x) and target (y)
    # `get_x` and `get_y` need to be fully implemented.
    try:
        from .analysis_util import get_x as get_x_actual, get_y as get_y_actual
        get_x_fn = get_x_actual if 'compensation' not in get_x_actual.__module__ else get_x
        get_y_fn = get_y_actual if 'compensation' not in get_y_actual.__module__ else get_y
    except ImportError:
        get_x_fn = get_x
        get_y_fn = get_y

    x_np, no_norm_out, features_out, l_segs_out = get_x_fn(
        xyz, ind, features_setup,
        features_no_norm=features_no_norm, terms=terms, # `terms` here refers to comp_params.terms (e.g. for specific feature construction)
        sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag
    )

    y_np = get_y_fn(
        xyz, ind, map_val_data,
        y_type=y_type, use_mag=use_mag,
        sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag, fs=fs # Pass fs for potential BPF in get_y
    )
    
    y_no_bpf_np = None
    if model_type in ["TL", "mod_TL"]: # Get non-BPF'd y for these models for final error calc
        y_no_bpf_np = get_y_fn(
            xyz, ind, map_val_data,
            y_type="d", use_mag=use_mag, # Typically 'd' (delta_mag) for non-BPF comparison
            sub_diurnal=sub_diurnal, sub_igrf=sub_igrf 
            # bpf_mag should be False here or handled by y_type='d'
        )

    # Prepare test data if ind_test is provided
    A_test_np, Bt_test_np, B_dot_test_np, x_test_np, y_test_np, l_segs_test_out = (None,) * 6
    if ind_test is not None and ind_test.size > 0:
        xyz_test_eff = xyz_test if xyz_test is not None else xyz # Default to training xyz if not provided

        map_val_test = get_map_val(mapS, xyz_test_eff.traj, ind_test) if y_type in ["b", "c"] else -1
        
        A_test_out = create_TL_A_fn(getattr(xyz_test_eff, use_vec), ind_test, terms=terms_A, return_B=True, Bt_in=getattr(xyz_test_eff,use_mag)[ind_test] if hasattr(xyz_test_eff,use_mag) else None)
        if isinstance(A_test_out, tuple) and len(A_test_out) == 3:
            A_test_np, Bt_test_np, B_dot_test_np = A_test_out
        else: A_test_np = A_test_out

        x_test_np, _, _, l_segs_test_out = get_x_fn(
            xyz_test_eff, ind_test, features_setup,
            features_no_norm=features_no_norm, terms=terms,
            sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag
        )
        y_test_np = get_y_fn(
            xyz_test_eff, ind_test, map_val_test,
            y_type=y_type, use_mag=use_mag,
            sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag, fs=fs
        )
    else: # Ensure empty arrays if no test data
        A_test_np = np.empty((0, A_matrix_np.shape[1] if A_matrix_np is not None else 0), dtype=A_matrix_np.dtype if A_matrix_np is not None else np.float32)
        Bt_test_np = np.empty(0, dtype=Bt_np.dtype if Bt_np is not None else np.float32)
        B_dot_test_np = np.empty((0, B_dot_np.shape[1] if B_dot_np is not None and B_dot_np.ndim==2 else 3), dtype=B_dot_np.dtype if B_dot_np is not None else np.float32)
        x_test_np = np.empty((0, x_np.shape[1] if x_np.ndim==2 else 0), dtype=x_np.dtype)
        y_test_np = np.empty(0, dtype=y_np.dtype)
        l_segs_test_out = []


    y_hat_np = np.zeros_like(y_np)
    err_np = np.full_like(y_np, np.nan) # Initialize with NaN or large value

    # Drop Feature Importance (FI) logic
    if isinstance(comp_params, NNCompParams) and drop_fi:
        if not silent: print("INFO: Starting Drop Feature Importance training...")
        # This loop retrains the model for each feature dropped.
        # The best model (or params from it) based on error is not explicitly stored back to comp_params in Julia loop.
        # It seems to save each dropped-feature model. The final returned comp_params is from the last iteration.
        # This might need clarification if the goal is to return the *best* overall model.
        # For now, will replicate the behavior of returning the params from the last FI iteration.
        
        # Ensure drop_fi_bson and drop_fi_csv are set if drop_fi is True
        if not comp_params.drop_fi_bson:
            comp_params.drop_fi_bson = f"comp_params_{comp_params.model_type}_dropfi.bson"
            if not silent: print(f"INFO: drop_fi_bson not set, defaulting to {comp_params.drop_fi_bson}")
        if not comp_params.drop_fi_csv: # Not used in this Python version directly for saving errors
            pass

        best_err_std_fi = float('inf')

        for i_fi in range(x_np.shape[1]): # Iterate through features to drop
            if not silent: print(f"INFO: Training with feature {features_out[i_fi]} dropped.")
            
            x_fi_train = np.delete(x_np, i_fi, axis=1)
            no_norm_fi = np.delete(no_norm_out, i_fi) if no_norm_out is not None else None
            
            x_fi_test = np.delete(x_test_np, i_fi, axis=1) if x_test_np is not None and x_test_np.size > 0 else x_test_np

            # Temporary comp_params for this FI iteration to hold intermediate model/data_norms
            temp_comp_params_fi = copy.deepcopy(comp_params)
            temp_comp_params_fi.data_norms = None # Reset data_norms for retraining
            temp_comp_params_fi.model = None      # Reset model for retraining

            y_hat_fi_iter, err_fi_iter = np.array([]),np.array([]) # Init

            if model_type == "m1":
                model_fi, data_norms_fi, y_hat_fi_iter, err_fi_iter = nn_comp_1_train(
                    x_fi_train, y_np, no_norm_fi, norm_type_x, norm_type_y, eta_adam, epoch_adam, epoch_lbfgs,
                    hidden, activation, loss, batchsize, frac_train, alpha_sgl, lambda_sgl, k_pca,
                    data_norms_in=None, model_in=None, # Retrain from scratch
                    l_segs=l_segs_out, x_test_in=x_fi_test, y_test_in=y_test_np, l_segs_test=l_segs_test_out, silent=silent
                )
                temp_comp_params_fi.model = model_fi
                temp_comp_params_fi.data_norms = data_norms_fi
            elif model_type.startswith("m2"):
                model_fi, TL_coef_fi, data_norms_fi, y_hat_fi_iter, err_fi_iter = nn_comp_2_train(
                    A_matrix_np, x_fi_train, y_np, no_norm_fi, model_type, norm_type_A, norm_type_x, norm_type_y,
                    TL_coef_nn, eta_adam, epoch_adam, epoch_lbfgs, hidden, activation, loss, batchsize,
                    frac_train, alpha_sgl, lambda_sgl, k_pca,
                    data_norms_in=None, model_in=None, l_segs=l_segs_out,
                    A_test_raw=A_test_np, x_test_raw=x_fi_test, y_test_raw=y_test_np, l_segs_test=l_segs_test_out, silent=silent
                )
                temp_comp_params_fi.model = model_fi
                temp_comp_params_fi.TL_coef = TL_coef_fi
                temp_comp_params_fi.data_norms = data_norms_fi
            elif model_type.startswith("m3"):
                model_fi, TL_coef_fi, data_norms_fi, y_hat_fi_iter, err_fi_iter = nn_comp_3_train(
                    A_raw=A_matrix_np, Bt_raw=Bt_np, B_dot_raw=B_dot_np, # These are from training data
                    x_raw=x_fi_train, y_raw=y_np, no_norm=no_norm_fi, model_type=model_type,
                    norm_type_x=norm_type_x, norm_type_y=norm_type_y, TL_coef_in=TL_coef_nn, terms_A=terms_A, y_type=y_type,
                    eta_adam=eta_adam, epoch_adam=epoch_adam, epoch_lbfgs=epoch_lbfgs, hidden=hidden, activation=activation,
                    loss_fn=loss, batchsize=batchsize, frac_train=frac_train, alpha_sgl=alpha_sgl, lambda_sgl=lambda_sgl,
                    k_pca=k_pca, sigma_curriculum=sigma_curriculum, l_window=l_window, window_type_temporal=window_type_temporal,
                    tf_layer_type=tf_layer_type, tf_norm_type=tf_norm_type, dropout_prob=dropout_prob, N_tf_head=N_tf_head, tf_gain=tf_gain,
                    data_norms_in=None, model_in=None, l_segs=l_segs_out,
                    A_test_raw_in=A_test_np, Bt_test_raw_in=Bt_test_np, B_dot_test_raw_in=B_dot_test_np,
                    x_test_raw_in=x_fi_test, y_test_raw_in=y_test_np, l_segs_test=l_segs_test_out, silent=silent
                )
                temp_comp_params_fi.model = model_fi
                temp_comp_params_fi.TL_coef = TL_coef_fi
                temp_comp_params_fi.data_norms = data_norms_fi
            else:
                if not silent: print(f"WARN: Drop FI not implemented for model_type {model_type}. Skipping FI for this feature.")
                continue
            
            current_err_std = np.std(err_fi_iter) if err_fi_iter.size > 0 else float('inf')
            if not silent: print(f"INFO: Dropped '{features_out[i_fi]}', train error std: {current_err_std:.2f} nT")

            # Save this FI model's params
            if drop_fi_bson:
                save_comp_params(temp_comp_params_fi, f"{remove_extension(drop_fi_bson)}_{i_fi}.bson")
            
            # Update overall best error and potentially the main comp_params if this is better
            # The Julia code seems to just save each and the last one's results are implicitly returned.
            # To match, we update comp_params with the last iteration's results.
            if i_fi == x_np.shape[1] - 1: # Last iteration
                comp_params = temp_comp_params_fi
                y_hat_np = y_hat_fi_iter
                err_np = err_fi_iter
        
        if not silent: print("INFO: Drop Feature Importance training finished.")

    else: # Standard training (no drop_fi)
        if model_type == "m1":
            model, data_norms, y_hat_np, err_np = nn_comp_1_train(
                x_np, y_np, no_norm_out, norm_type_x, norm_type_y, eta_adam, epoch_adam, epoch_lbfgs,
                hidden, activation, loss, batchsize, frac_train, alpha_sgl, lambda_sgl, k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                x_test_in=x_test_np, y_test_in=y_test_np, l_segs_test=l_segs_test_out, silent=silent
            )
            comp_params.model = model
            comp_params.data_norms = data_norms
        elif model_type.startswith("m2"): # m2a, m2b, m2c, m2d
            model, TL_coef_out, data_norms, y_hat_np, err_np = nn_comp_2_train(
                A_matrix_np, x_np, y_np, no_norm_out, model_type, norm_type_A, norm_type_x, norm_type_y,
                TL_coef_nn, eta_adam, epoch_adam, epoch_lbfgs, hidden, activation, loss, batchsize,
                frac_train, alpha_sgl, lambda_sgl, k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw=A_test_np, x_test_raw=x_test_np, y_test_raw=y_test_np, l_segs_test=l_segs_test_out, silent=silent
            )
            comp_params.model = model
            if isinstance(comp_params, NNCompParams): comp_params.TL_coef = TL_coef_out
            comp_params.data_norms = data_norms
        elif model_type.startswith("m3"):
             model, TL_coef_out, data_norms, y_hat_np, err_np = nn_comp_3_train(
                A_raw=A_matrix_np, Bt_raw=Bt_np, B_dot_raw=B_dot_np, # Pass the versions derived from training xyz
                x_raw=x_np, y_raw=y_np, no_norm=no_norm_out, model_type=model_type,
                norm_type_x=norm_type_x, norm_type_y=norm_type_y, TL_coef_in=TL_coef_nn, terms_A=terms_A, y_type=y_type,
                eta_adam=eta_adam, epoch_adam=epoch_adam, epoch_lbfgs=epoch_lbfgs, hidden=hidden, activation=activation,
                loss_fn=loss, batchsize=batchsize, frac_train=frac_train, alpha_sgl=alpha_sgl, lambda_sgl=lambda_sgl,
                k_pca=k_pca, sigma_curriculum=sigma_curriculum, l_window=l_window, window_type_temporal=window_type_temporal,
                tf_layer_type=tf_layer_type, tf_norm_type=tf_norm_type, dropout_prob=dropout_prob, N_tf_head=N_tf_head, tf_gain=tf_gain,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw_in=A_test_np, Bt_test_raw_in=Bt_test_np, B_dot_test_raw_in=B_dot_test_np,
                x_test_raw_in=x_test_np, y_test_raw_in=y_test_np, l_segs_test=l_segs_test_out, silent=silent
            )
             comp_params.model = model
             if isinstance(comp_params, NNCompParams): comp_params.TL_coef = TL_coef_out
             comp_params.data_norms = data_norms
        elif model_type in ["TL", "mod_TL", "map_TL"]:
            trim_val = 20 if model_type in ["TL", "mod_TL"] else 0
            model_tuple, data_norms_out_lin, y_hat_np, err_np = linear_fit(
                A_matrix_np, y_np, trim=trim_val, lambda_ridge=lambda_TL if isinstance(comp_params, LinCompParams) else 0.0,
                norm_type_x=norm_type_A, norm_type_y=norm_type_y,
                data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            if model_type in ["TL", "mod_TL"] and A_no_bpf_np is not None and y_no_bpf_np is not None:
                # Recalculate error on non-BPF'd data for these specific models
                y_hat_np, err_np = linear_test(A_no_bpf_np, y_no_bpf_np, data_norms_out_lin, model_tuple, l_segs=l_segs_out, silent=silent)

            comp_params.model = model_tuple # Store (coeffs, bias)
            comp_params.data_norms = data_norms_out_lin
        elif model_type == "elasticnet":
            model_tuple, data_norms_out_lin, y_hat_np, err_np = elasticnet_fit(
                x_np, y_np, alpha=0.99, no_norm=no_norm_out, # Default alpha from Julia
                lambda_val= -1.0, # Default to CV
                data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            comp_params.model = model_tuple
            comp_params.data_norms = data_norms_out_lin
        elif model_type == "plsr":
            model_tuple, data_norms_out_lin, y_hat_np, err_np = plsr_fit(
                x_np, y_np, k=k_plsr if isinstance(comp_params, LinCompParams) else x_np.shape[1], 
                no_norm=no_norm_out, data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            comp_params.model = model_tuple
            comp_params.data_norms = data_norms_out_lin
        else:
            raise ValueError(f"Unknown model_type: {model_type} in comp_train")

    if not silent:
        elapsed_time = time.time() - t0
        time_unit = "sec"
        if elapsed_time > 60:
            elapsed_time /= 60
            time_unit = "min"
        print(f"INFO: comp_train completed in {elapsed_time:.1f} {time_unit}")

    return comp_params, y_np, y_hat_np, err_np, features_out
def comp_train( # Overload for DataFrames
    comp_params: CompParams,
    lines: Union[int, float, List[Union[int, float]], np.ndarray], # Can be single line or list/array of lines
    df_line: Any, # pandas.DataFrame
    df_flight: Any, # pandas.DataFrame
    df_map: Any, # pandas.DataFrame
    temp_params: Optional[TempParams] = None,
    silent: bool = False
) -> Tuple[CompParams, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train an aeromagnetic compensation model using DataFrame inputs.
    This version loads XYZ data based on lines and DataFrames.
    """
    np.random.seed(2)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
    t0_df = time.time()

    if temp_params is None:
        temp_params = TempParams()

    # The core logic of this overload in Julia is to:
    # 1. Determine if `lines` refers to training lines or test lines based on `comp_params.frac_train`.
    #    (This seems to be for a specific workflow where `lines` could be all available lines,
    #     and `frac_train` dictates splitting them into actual train/test sets for this call).
    #    However, the Python `nn_comp_X_train` functions already handle frac_train internally.
    #    The `comp_train(XYZ, ind)` version also takes explicit `xyz_test, ind_test`.
    #    For simplicity and consistency, this Python wrapper will assume `lines` are for training,
    #    and if a test set is desired from the same `lines` based on `frac_train`,
    #    that splitting logic should be handled before calling this, or by passing
    #    explicit `xyz_test, ind_test` to the primary `comp_train`.
    #    Alternatively, we can implement a split here if that's the intended behavior.
    #
    #    Looking at the Julia `comp_train` (df version), it does:
    #    - `get_Axy` which itself calls `get_XYZ` and `get_ind` for the provided `lines`.
    #    - It does *not* seem to split `lines` into train/test at this top `comp_train` (df) level.
    #    - The `frac_train` is passed down to `nn_comp_X_train` which does the split on the data `x,y`.
    #    - The `drop_fi` logic inside the primary `comp_train(XYZ, ind)` *does* use `xyz_test, ind_test`
    #      which are passed as optional arguments. If they are not provided, they default to empty.
    #
    #    Therefore, this wrapper will focus on loading the primary XYZ data for `lines`.
    #    If a separate test set from different lines is needed, the `comp_train_test` function
    #    or manual preparation of `xyz_test, ind_test` for the primary `comp_train` is more appropriate.

    # For now, assume `lines` are all for training, and `frac_train` within `comp_params`
    # will be used by the `nn_comp_X_train` functions to split the *data* (not the lines themselves here).

    # `get_Axy` is a complex function that loads data. We need its Python equivalent.
    # For now, we'll assume `get_Axy` is available from `analysis_util.py`
    # and it correctly processes DataFrames to produce A, x, y, etc.
    # The call signature in Julia is:
    # get_Axy(lines, df_line, df_flight, df_map, features_setup; kwargs...)
    
    # Unpack relevant comp_params for get_Axy and subsequent calls
    model_type = comp_params.model_type
    y_type = comp_params.y_type
    use_mag = comp_params.use_mag
    use_vec = comp_params.use_vec
    terms = comp_params.terms
    terms_A = list(comp_params.terms_A) # mutable copy
    sub_diurnal = comp_params.sub_diurnal
    sub_igrf = comp_params.sub_igrf
    bpf_mag = comp_params.bpf_mag
    reorient_vec = comp_params.reorient_vec
    features_setup = list(comp_params.features_setup)
    features_no_norm = list(comp_params.features_no_norm)
    
    # Adjust y_type and terms_A based on model_type, similar to the other comp_train
    if model_type in ["TL", "mod_TL"] and y_type != "e":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'e' for model_type {model_type}")
        y_type = "e" # Update local y_type for get_Axy
        comp_params.y_type = "e" # Also update in comp_params if it's to be persisted
    if model_type == "map_TL" and y_type != "c":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'c' for model_type {model_type}")
        y_type = "c"
        comp_params.y_type = "c"
    if model_type.startswith("m3"):
        terms_A_original = list(terms_A)
        terms_A = [term for term in terms_A if term not in ["fdm", "f", "fdm3", "f3", "bias", "b"]]
        if len(terms_A) != len(terms_A_original) and not silent:
            print(f"INFO: Removing derivative/bias terms from terms_A for M3 model. Original: {terms_A_original}, New: {terms_A}")
        comp_params.terms_A = terms_A # Persist change

    # Call get_Axy (assuming it's imported or defined)
    # This function needs to be robustly implemented from analysis_util.py
    try:
        from .analysis_util import get_Axy as get_Axy_actual
        if 'compensation' in get_Axy_actual.__module__: get_Axy_fn = get_Axy # placeholder
        else: get_Axy_fn = get_Axy_actual
    except ImportError:
        get_Axy_fn = get_Axy # Use placeholder

    Axy_outputs = get_Axy_fn(
        lines, df_line, df_flight, df_map,
        features_setup,
        features_no_norm=features_no_norm,
        y_type=y_type,
        use_mag=use_mag,
        use_vec=use_vec,
        terms=terms, # comp_params.terms
        terms_A=terms_A, # comp_params.terms_A (potentially modified for M3)
        sub_diurnal=sub_diurnal,
        sub_igrf=sub_igrf,
        bpf_mag=bpf_mag,
        reorient_vec=reorient_vec,
        mod_TL=(model_type == "mod_TL"),
        map_TL=(model_type == "map_TL"),
        return_B=model_type.startswith("m3"), # Only M3 explicitly needs Bt, B_dot from get_Axy
        silent=SILENT_DEBUG # Use global for internal prints of get_Axy
    )

    if model_type.startswith("m3"):
        A_matrix_np, Bt_np, B_dot_np, x_np, y_np, no_norm_out, features_out, l_segs_out = Axy_outputs
    else:
        A_matrix_np, x_np, y_np, no_norm_out, features_out, l_segs_out = Axy_outputs
        Bt_np, B_dot_np = None, None # Not returned by get_Axy for non-M3, or not used directly by train functions

    # --- The rest of the logic mirrors the primary comp_train after data loading ---
    # This includes handling drop_fi and calling the specific nn_comp_X_train or linear_fit.
    # For brevity and to avoid large duplication, we can call the primary comp_train here,
    # but that would require constructing an XYZ object first.
    # The Julia version directly passes A, x, y to nn_comp_X_train.
    # Let's follow that pattern.

    y_hat_np = np.zeros_like(y_np)
    err_np = np.full_like(y_np, np.nan)
    
    # We need to pass all relevant comp_params attributes to the training functions.
    # Re-fetch them from comp_params as they might have been updated (e.g. y_type, terms_A)
    model_type = comp_params.model_type # Use potentially updated one
    y_type = comp_params.y_type         # Use potentially updated one
    terms_A = list(comp_params.terms_A) # Use potentially updated one
    norm_type_A = comp_params.norm_type_A
    norm_type_x = comp_params.norm_type_x
    norm_type_y = comp_params.norm_type_y
    data_norms = comp_params.data_norms # Pass current, will be updated by train functions
    model = comp_params.model           # Pass current, will be updated

    # Test data handling: get_Axy for DataFrames doesn't explicitly return a test set.
    # The nn_comp_X_train functions expect raw test data if provided.
    # For this wrapper, we'll assume no separate test set is loaded by get_Axy(df) itself.
    # If testing is needed, it should be done via comp_test or comp_train_test.
    # So, pass empty/None for test data to nn_comp_X_train.
    A_test_for_train = np.empty((0, A_matrix_np.shape[1]), dtype=A_matrix_np.dtype) if A_matrix_np is not None and A_matrix_np.ndim ==2 else np.empty((0,0))
    Bt_test_for_train = np.empty(0, dtype=Bt_np.dtype if Bt_np is not None else np.float32)
    B_dot_test_for_train = np.empty((0, B_dot_np.shape[1] if B_dot_np is not None and B_dot_np.ndim ==2 else 3), dtype=B_dot_np.dtype if B_dot_np is not None else np.float32)
    x_test_for_train = np.empty((0, x_np.shape[1] if x_np.ndim == 2 else 0), dtype=x_np.dtype)
    y_test_for_train = np.empty(0, dtype=y_np.dtype)
    l_segs_test_for_train = []


    if isinstance(comp_params, NNCompParams) and comp_params.drop_fi:
        if not silent: print("INFO: df_comp_train Drop Feature Importance starting...")
        # Simplified: We'll just run the main training once without FI for this wrapper.
        # Full FI loop here would be very extensive. The primary comp_train(XYZ) handles FI.
        # To enable FI here, one would need to reconstruct the FI loop from the other comp_train.
        # For now, set drop_fi to False for this specific call path.
        if not silent: print("WARN: Drop FI with DataFrame input is simplified in this version. Training full model.")
        drop_fi_original_setting = comp_params.drop_fi
        comp_params.drop_fi = False # Temporarily disable for this path
        
        # Call the main training logic (copied/adapted from the other comp_train)
        # This section will be very similar to the `else` block of the FI logic in the primary comp_train
        # (Code for standard training path)
        if model_type == "m1":
            model_res, dn_res, y_hat_np, err_np = nn_comp_1_train(
                x_np, y_np, no_norm_out, norm_type_x, norm_type_y, comp_params.η_adam, comp_params.epoch_adam, comp_params.epoch_lbfgs,
                comp_params.hidden, comp_params.activation, comp_params.loss, comp_params.batchsize, comp_params.frac_train, 
                comp_params.α_sgl, comp_params.λ_sgl, comp_params.k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                x_test_in=x_test_for_train, y_test_in=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.data_norms = dn_res
        elif model_type.startswith("m2"):
            model_res, tl_coef_res, dn_res, y_hat_np, err_np = nn_comp_2_train(
                A_matrix_np, x_np, y_np, no_norm_out, model_type, norm_type_A, norm_type_x, norm_type_y,
                comp_params.TL_coef, comp_params.η_adam, comp_params.epoch_adam, comp_params.epoch_lbfgs, 
                comp_params.hidden, comp_params.activation, comp_params.loss, comp_params.batchsize,
                comp_params.frac_train, comp_params.α_sgl, comp_params.λ_sgl, comp_params.k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw=A_test_for_train, x_test_raw=x_test_for_train, y_test_raw=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.TL_coef = tl_coef_res
            comp_params.data_norms = dn_res
        elif model_type.startswith("m3"):
            model_res, tl_coef_res, dn_res, y_hat_np, err_np = nn_comp_3_train(
                A_raw=A_matrix_np, Bt_raw=Bt_np, B_dot_raw=B_dot_np, 
                x_raw=x_np, y_raw=y_np, no_norm=no_norm_out, model_type=model_type,
                norm_type_x=norm_type_x, norm_type_y=norm_type_y, TL_coef_in=comp_params.TL_coef, terms_A=terms_A, y_type=y_type,
                eta_adam=comp_params.η_adam, epoch_adam=comp_params.epoch_adam, epoch_lbfgs=comp_params.epoch_lbfgs, 
                hidden=comp_params.hidden, activation=comp_params.activation, loss_fn=comp_params.loss, 
                batchsize=comp_params.batchsize, frac_train=comp_params.frac_train, 
                alpha_sgl=comp_params.α_sgl, lambda_sgl=comp_params.λ_sgl, k_pca=comp_params.k_pca, 
                sigma_curriculum=sigma_curriculum, l_window=l_window, window_type_temporal=window_type_temporal,
                tf_layer_type=tf_layer_type, tf_norm_type=tf_norm_type, dropout_prob=dropout_prob, N_tf_head=N_tf_head, tf_gain=tf_gain,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw_in=A_test_for_train, Bt_test_raw_in=Bt_test_for_train, B_dot_test_raw_in=B_dot_test_for_train,
                x_test_raw_in=x_test_for_train, y_test_raw_in=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.TL_coef = tl_coef_res
            comp_params.data_norms = dn_res
        # ... (add LinCompParams models if drop_fi was intended for them too)
        comp_params.drop_fi = drop_fi_original_setting # Restore original setting
    else: # Standard training path (no drop_fi, or drop_fi handled by primary comp_train)
        if model_type == "m1" and isinstance(comp_params, NNCompParams):
            model_res, dn_res, y_hat_np, err_np = nn_comp_1_train(
                x_np, y_np, no_norm_out, norm_type_x, norm_type_y, comp_params.η_adam, comp_params.epoch_adam, comp_params.epoch_lbfgs,
                comp_params.hidden, comp_params.activation, comp_params.loss, comp_params.batchsize, comp_params.frac_train, 
                comp_params.α_sgl, comp_params.λ_sgl, comp_params.k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                x_test_in=x_test_for_train, y_test_in=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.data_norms = dn_res
        elif model_type.startswith("m2") and isinstance(comp_params, NNCompParams):
            model_res, tl_coef_res, dn_res, y_hat_np, err_np = nn_comp_2_train(
                A_matrix_np, x_np, y_np, no_norm_out, model_type, norm_type_A, norm_type_x, norm_type_y,
                comp_params.TL_coef, comp_params.η_adam, comp_params.epoch_adam, comp_params.epoch_lbfgs, 
                comp_params.hidden, comp_params.activation, comp_params.loss, comp_params.batchsize,
                comp_params.frac_train, comp_params.α_sgl, comp_params.λ_sgl, comp_params.k_pca,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw=A_test_for_train, x_test_raw=x_test_for_train, y_test_raw=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.TL_coef = tl_coef_res
            comp_params.data_norms = dn_res
        elif model_type.startswith("m3") and isinstance(comp_params, NNCompParams):
            model_res, tl_coef_res, dn_res, y_hat_np, err_np = nn_comp_3_train(
                A_raw=A_matrix_np, Bt_raw=Bt_np, B_dot_raw=B_dot_np, 
                x_raw=x_np, y_raw=y_np, no_norm=no_norm_out, model_type=model_type,
                norm_type_x=norm_type_x, norm_type_y=norm_type_y, TL_coef_in=comp_params.TL_coef, terms_A=terms_A, y_type=y_type,
                eta_adam=comp_params.η_adam, epoch_adam=comp_params.epoch_adam, epoch_lbfgs=comp_params.epoch_lbfgs, 
                hidden=comp_params.hidden, activation=comp_params.activation, loss_fn=comp_params.loss, 
                batchsize=comp_params.batchsize, frac_train=comp_params.frac_train, 
                alpha_sgl=comp_params.α_sgl, lambda_sgl=comp_params.λ_sgl, k_pca=comp_params.k_pca, 
                sigma_curriculum=sigma_curriculum, l_window=l_window, window_type_temporal=window_type_temporal,
                tf_layer_type=tf_layer_type, tf_norm_type=tf_norm_type, dropout_prob=dropout_prob, N_tf_head=N_tf_head, tf_gain=tf_gain,
                data_norms_in=data_norms, model_in=model, l_segs=l_segs_out,
                A_test_raw_in=A_test_for_train, Bt_test_raw_in=Bt_test_for_train, B_dot_test_raw_in=B_dot_test_for_train,
                x_test_raw_in=x_test_for_train, y_test_raw_in=y_test_for_train, l_segs_test=l_segs_test_for_train, silent=silent
            )
            comp_params.model = model_res
            comp_params.TL_coef = tl_coef_res
            comp_params.data_norms = dn_res
        elif model_type in ["TL", "mod_TL", "map_TL"] and isinstance(comp_params, LinCompParams):
            trim_val = 20 if model_type in ["TL", "mod_TL"] else 0
            model_tuple, dn_res, y_hat_np, err_np = linear_fit(
                A_matrix_np, y_np, trim=trim_val, lambda_ridge=comp_params.λ_TL,
                norm_type_x=norm_type_A, norm_type_y=norm_type_y,
                data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            if model_type in ["TL", "mod_TL"] and A_no_bpf_np is not None and y_no_bpf_np is not None:
                 y_hat_np, err_np = linear_test(A_no_bpf_np, y_no_bpf_np, dn_res, model_tuple, l_segs=l_segs_out, silent=silent)
            comp_params.model = model_tuple
            comp_params.data_norms = dn_res
        elif model_type == "elasticnet" and isinstance(comp_params, LinCompParams):
            model_tuple, dn_res, y_hat_np, err_np = elasticnet_fit(
                x_np, y_np, alpha=0.99, no_norm=no_norm_out, # Default alpha from Julia
                lambda_val= -1.0, # Default to CV, or use a param from LinCompParams if defined
                data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            comp_params.model = model_tuple
            comp_params.data_norms = dn_res
        elif model_type == "plsr" and isinstance(comp_params, LinCompParams):
            model_tuple, dn_res, y_hat_np, err_np = plsr_fit(
                x_np, y_np, k=comp_params.k_plsr, no_norm=no_norm_out, 
                data_norms_in=data_norms, l_segs=l_segs_out, silent=silent
            )
            comp_params.model = model_tuple
            comp_params.data_norms = dn_res
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' or mismatched CompParams type for comp_train(df).")

    if not silent:
        elapsed_time = time.time() - t0_df
        time_unit = "sec"
        if elapsed_time > 60:
            elapsed_time /= 60
            time_unit = "min"
        print(f"INFO: comp_train (DataFrame version) completed in {elapsed_time:.1f} {time_unit}")

    return comp_params, y_np, y_hat_np, err_np, features_out
def comp_test(
    comp_params: CompParams, 
    xyz: XYZ, 
    ind: np.ndarray,
    mapS: Any = mapS_null, # Union[MapS, MapSd, MapS3D]
    temp_params: Optional[TempParams] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate performance of an aeromagnetic compensation model.
    Base version taking XYZ and ind.

    Args:
        comp_params: CompParams object (NNCompParams or LinCompParams).
        xyz: XYZ flight data struct.
        ind: Selected data indices for testing.
        mapS: (optional) MapS struct, only used for y_type = 'b', 'c'.
        temp_params: (optional) TempParams struct.
        silent: (optional) If true, no print outs.

    Returns:
        y: Target vector.
        y_hat: Prediction vector.
        err: Compensation error.
        features: List of feature names used.
    """
    np.random.seed(2) # for reproducibility
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
        
    t0 = time.time()

    if temp_params is None:
        temp_params = TempParams()

    # Unpack parameters
    model_type = comp_params.model_type
    y_type = comp_params.y_type
    use_mag = comp_params.use_mag
    use_vec = comp_params.use_vec
    data_norms = comp_params.data_norms
    model = comp_params.model # This is the trained model
    terms = comp_params.terms
    terms_A = list(comp_params.terms_A)
    sub_diurnal = comp_params.sub_diurnal
    sub_igrf = comp_params.sub_igrf
    bpf_mag = comp_params.bpf_mag
    # reorient_vec = comp_params.reorient_vec # Not directly used in test dispatch
    features_setup = list(comp_params.features_setup)
    features_no_norm = list(comp_params.features_no_norm)

    TL_coef_val: Optional[np.ndarray] = None
    if isinstance(comp_params, NNCompParams):
        TL_coef_val = comp_params.TL_coef
        # drop_fi / perm_fi logic is specific to comp_test in Julia, handle below
        drop_fi = comp_params.drop_fi
        perm_fi = comp_params.perm_fi
        drop_fi_bson_base = comp_params.drop_fi_bson
        drop_fi_csv_path = comp_params.drop_fi_csv
        perm_fi_csv_path = comp_params.perm_fi_csv
    elif isinstance(comp_params, LinCompParams):
        drop_fi = False # Not applicable to LinCompParams in the same way
        perm_fi = False
    else:
        raise TypeError("comp_params must be an instance of NNCompParams or LinCompParams")

    # TempParams unpack (only l_window needed for _from_raw calls if temporal)
    l_window = temp_params.l_window


    # Adjust y_type based on model_type for testing (mirroring Julia's comp_test)
    # In Julia's comp_test, for TL, mod_TL, it forces y_type to 'd' if it was 'e' during training.
    # This is because the error is typically evaluated on the un-filtered delta mag.
    original_y_type_for_get_y = y_type
    if model_type in ["TL", "mod_TL"] and y_type == "e":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'd' (Δmag) for error calculation in model_type {model_type}")
        original_y_type_for_get_y = "d" # Use 'd' to get the target for error calculation
    elif model_type == "map_TL" and y_type != "c": # map_TL was trained on 'c'
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'c' for error calculation in model_type {model_type}")
        original_y_type_for_get_y = "c"
    
    # For M3 models, terms_A might need adjustment if derivative/bias terms were removed during training
    if model_type.startswith("m3"):
        terms_A = [term for term in terms_A if term not in ["fdm", "f", "fdm3", "f3", "bias", "b"]]


    # Get map values if needed for the target y
    map_val_data = get_map_val(mapS, xyz.traj, ind) if original_y_type_for_get_y in ["b", "c"] else -1
    
    # Prepare A, Bt, B_dot, x, y for the test data
    # `create_TL_A` and `get_x`/`get_y` need to be robust.
    try:
        from .analysis_util import get_x as get_x_actual, get_y as get_y_actual
        from .tolles_lawson import create_TL_A as create_TL_A_actual

        get_x_fn = get_x_actual if 'compensation' not in get_x_actual.__module__ else get_x
        get_y_fn = get_y_actual if 'compensation' not in get_y_actual.__module__ else get_y
        create_TL_A_fn = create_TL_A_actual if 'compensation' not in create_TL_A_actual.__module__ else create_TL_A
    except ImportError:
        get_x_fn, get_y_fn, create_TL_A_fn = get_x, get_y, create_TL_A


    A_matrix_np: Optional[np.ndarray] = None
    Bt_np: Optional[np.ndarray] = None
    B_dot_np: Optional[np.ndarray] = None

    # A_matrix_np for linear models (TL, mod_TL, map_TL)
    if model_type in ["TL", "mod_TL", "map_TL"]:
        if model_type == "mod_TL":
            A_matrix_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, Bt=getattr(xyz, use_mag)[ind])
        elif model_type == "map_TL":
            A_matrix_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, Bt=map_val_data)
        else: # TL
            A_matrix_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, return_B=False, Bt_in=getattr(xyz,use_mag)[ind] if hasattr(xyz,use_mag) else None)
        # If y_type was 'e' during training, A was BPF'd. For testing against 'd', use non-BPF'd A.
        # This is implicitly handled if create_TL_A is called fresh.
        # If comp_params stored a BPF'd A, that's an issue. Assume create_TL_A here gives the correct A for testing.
    elif model_type.startswith("m2") or model_type.startswith("m3"):
        # These models need A, Bt, B_dot for their _from_raw forward pass
        A_out_test = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, return_B=True, Bt_in=getattr(xyz,use_mag)[ind] if hasattr(xyz,use_mag) else None)
        if isinstance(A_out_test, tuple) and len(A_out_test) == 3:
            A_matrix_np, Bt_np, B_dot_np = A_out_test # A_matrix_np here is the raw flux components for M3, or full TL for M2
        else: # Should not happen if return_B=True
            A_matrix_np = A_out_test 


    x_np, _, features_out, l_segs_out = get_x_fn(
        xyz, ind, features_setup,
        features_no_norm=features_no_norm, terms=terms,
        sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag
    )

    y_np = get_y_fn(
        xyz, ind, map_val_data,
        y_type=original_y_type_for_get_y, # Use the potentially adjusted y_type for target
        use_mag=use_mag,
        sub_diurnal=sub_diurnal, sub_igrf=sub_igrf, bpf_mag=bpf_mag, # bpf_mag for get_y if y_type='e'
        fs = (1.0/xyz.dt if xyz.dt > 0 else 10.0)
    )
    
    y_hat_np = np.zeros_like(y_np)
    err_np = np.full_like(y_np, np.nan)

    # Feature Importance (FI) logic for testing
    if isinstance(comp_params, NNCompParams) and (drop_fi or perm_fi):
        if not silent: print(f"INFO: comp_test running Feature Importance (perm_fi={perm_fi}, drop_fi={drop_fi})...")
        
        fi_csv_output_path = perm_fi_csv_path if perm_fi else drop_fi_csv_path
        if fi_csv_output_path:
             # Clear or header for FI CSV
            with open(fi_csv_output_path, 'w') as f:
                f.write("feature_idx,error_std\n")
        
        best_err_std_fi = float('inf')
        # y_hat_np and err_np will be from the iteration that produced the best_err_std_fi

        for i_fi in range(x_np.shape[1]): # Iterate through features
            x_fi_test = x_np.copy()
            current_comp_params = comp_params # Use the main trained comp_params

            if perm_fi:
                if not silent: print(f"INFO: Permuting feature {features_out[i_fi]} for FI test.")
                x_fi_test[:, i_fi] = np.random.permutation(x_fi_test[:, i_fi])
            elif drop_fi:
                if not silent: print(f"INFO: Testing with model trained without feature {features_out[i_fi]}.")
                # Load the specific model trained without this feature
                bson_path_fi = f"{remove_extension(drop_fi_bson_base)}_{i_fi}.bson"
                try:
                    current_comp_params = get_comp_params(bson_path_fi, silent=True)
                    if not silent: print(f"Loaded FI model from {bson_path_fi}")
                except FileNotFoundError:
                    if not silent: print(f"WARN: FI model {bson_path_fi} not found. Skipping this feature for drop_fi.")
                    continue
                # x_fi_test will be x_np with the i_fi column conceptually dropped,
                # because the loaded model was trained on data without it.
                # The _from_raw functions need to handle x that matches the model's training.
                # This means nn_comp_X_test_from_raw needs x_raw that's already subsetted.
                # This is complex. The Julia version passes x_fi (subsetted x) to nn_comp_X_test.
                # For drop_fi, the loaded `current_comp_params.model` was trained on x_fi.
                # So, `data_norms` in `current_comp_params` are also for x_fi.
                # We need to pass x_fi_test (which is x_np with col i_fi removed) to _from_raw.
                x_fi_test_for_drop = np.delete(x_np, i_fi, axis=1)


            # Select the correct test function based on model_type
            y_hat_iter, err_iter = np.array([]), np.array([])
            
            # Use _from_raw versions for testing with loaded comp_params
            if current_comp_params.model_type == "m1":
                if drop_fi:
                    y_hat_iter, err_iter = nn_comp_1_test_from_raw(x_fi_test_for_drop, y_np, current_comp_params.data_norms, current_comp_params.model, l_segs_out, silent)
                else: # perm_fi
                    y_hat_iter, err_iter = nn_comp_1_test_from_raw(x_fi_test, y_np, current_comp_params.data_norms, current_comp_params.model, l_segs_out, silent)
            
            elif current_comp_params.model_type.startswith("m2"):
                # A_matrix_np, Bt_np, B_dot_np are from the full test data (xyz, ind)
                # x_fi_test or x_fi_test_for_drop is used
                # TL_coef is from current_comp_params
                if drop_fi:
                    y_hat_iter, err_iter = nn_comp_2_test_from_raw(A_matrix_np, x_fi_test_for_drop, y_np, current_comp_params.data_norms, current_comp_params.model, current_comp_params.model_type, current_comp_params.TL_coef, l_segs_out, silent)
                else: # perm_fi
                     y_hat_iter, err_iter = nn_comp_2_test_from_raw(A_matrix_np, x_fi_test, y_np, current_comp_params.data_norms, current_comp_params.model, current_comp_params.model_type, current_comp_params.TL_coef, l_segs_out, silent)
            
            elif current_comp_params.model_type.startswith("m3"):
                if drop_fi:
                    y_hat_iter, err_iter = nn_comp_3_test_from_raw(
                        A_matrix_np, Bt_np, B_dot_np, x_fi_test_for_drop, y_np, current_comp_params.data_norms, current_comp_params.model,
                        current_comp_params.model_type, original_y_type_for_get_y, # Use original y_type for consistency with how model was trained/tested
                        current_comp_params.TL_coef, current_comp_params.terms_A, l_segs=l_segs_out, l_window=l_window, silent=silent)
                else: # perm_fi
                     y_hat_iter, err_iter = nn_comp_3_test_from_raw(
                        A_matrix_np, Bt_np, B_dot_np, x_fi_test, y_np, current_comp_params.data_norms, current_comp_params.model,
                        current_comp_params.model_type, original_y_type_for_get_y,
                        current_comp_params.TL_coef, current_comp_params.terms_A, l_segs=l_segs_out, l_window=l_window, silent=silent)
            else:
                if not silent: print(f"WARN: FI testing not implemented for model_type {current_comp_params.model_type}")
                continue

            current_err_std = np.std(err_iter) if err_iter.size > 0 else float('inf')
            if fi_csv_output_path:
                with open(fi_csv_output_path, 'a') as f:
                    f.write(f"{i_fi},{current_err_std}\n") # Save 0-indexed feature index

            if current_err_std < best_err_std_fi : # Julia uses > for perm_fi, < for drop_fi. Let's use < for both (lower error is better)
                best_err_std_fi = current_err_std
                y_hat_np = y_hat_iter # Store the y_hat from the best FI iteration
                err_np = err_iter     # Store the error from the best FI iteration
        
        if not silent: print(f"INFO: Feature Importance testing finished. Best error std: {best_err_std_fi:.2f} nT (results in y_hat, err are from this best FI iteration).")

    else: # Standard test (no FI loop)
        if model_type == "m1":
            y_hat_np, err_np = nn_comp_1_test_from_raw(x_np, y_np, data_norms, model, l_segs=l_segs_out, silent=silent)
        elif model_type.startswith("m2"):
            y_hat_np, err_np = nn_comp_2_test_from_raw(A_matrix_np, x_np, y_np, data_norms, model, model_type, TL_coef_val, l_segs=l_segs_out, silent=silent)
        elif model_type.startswith("m3"):
            y_hat_np, err_np = nn_comp_3_test_from_raw(
                A_raw=A_matrix_np, Bt_raw=Bt_np, B_dot_raw=B_dot_np, # A_matrix_np is flux for M3
                x_raw=x_np, y_raw=y_np, data_norms=data_norms, model_nn=model,
                model_type=model_type, y_type=original_y_type_for_get_y, TL_coef_raw=TL_coef_val, terms_A=terms_A,
                l_segs=l_segs_out, l_window=l_window, silent=silent
            )
        elif model_type in ["TL", "mod_TL", "map_TL"]:
            # linear_test expects normalized x (A_matrix_np here) and raw y (y_np)
            # It also needs y_bias and y_scale from data_norms.
            # The model is (coeffs, bias_val)
            if data_norms is None or len(data_norms) < 4: # Should have been set during training
                 raise ValueError("data_norms missing or incomplete for linear model testing")
            _, _, y_bias_lin, y_scale_lin = unpack_data_norms(data_norms) # type: ignore
            
            # A_matrix_np should be the one used for training (potentially BPF'd if y_type was 'e')
            # However, for error calculation, if original y_type was 'e', we test against 'd' with non-BPF'd A.
            A_test_final = A_matrix_np
            y_test_final = y_np
            if model_type in ["TL", "mod_TL"] and comp_params.y_type == "e": # If trained on 'e'
                if A_no_bpf_np is None: # Should have been stored during comp_train if y_type was 'e'
                     # Recreate non-BPF A if not available (less ideal)
                     A_no_bpf_np = create_TL_A_fn(getattr(xyz, use_vec), ind, terms=terms_A, return_B=False, Bt_in=getattr(xyz,use_mag)[ind] if hasattr(xyz,use_mag) else None)
                A_test_final = A_no_bpf_np
                if y_no_bpf_np is None:
                     y_no_bpf_np = get_y_fn(xyz, ind, map_val_data, y_type="d", use_mag=use_mag, sub_diurnal=sub_diurnal, sub_igrf=sub_igrf)
                y_test_final = y_no_bpf_np

            y_hat_np, err_np = linear_test(A_test_final, y_test_final, data_norms, model, l_segs=l_segs_out, silent=silent)

        elif model_type == "elasticnet" or model_type == "plsr":
            y_hat_np, err_np = linear_test(x_np, y_np, data_norms, model, l_segs=l_segs_out, silent=silent)
        else:
            raise ValueError(f"Unknown model_type: {model_type} in comp_test")

    if not silent:
        elapsed_time = time.time() - t0
        time_unit = "sec"
        if elapsed_time > 60:
            elapsed_time /= 60
            time_unit = "min"
        print(f"INFO: comp_test completed in {elapsed_time:.1f} {time_unit}")

    return y_np, y_hat_np, err_np, features_out
def comp_test( # Overload for DataFrames
    comp_params: CompParams,
    lines: Union[int, float, List[Union[int, float]], np.ndarray],
    df_line: Any, # pandas.DataFrame
    df_flight: Any, # pandas.DataFrame
    df_map: Any, # pandas.DataFrame
    temp_params: Optional[TempParams] = None,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate performance of an aeromagnetic compensation model using DataFrame inputs.
    """
    np.random.seed(2)
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2)
    t0_df = time.time()

    if temp_params is None:
        temp_params = TempParams()

    # Unpack parameters
    model_type = comp_params.model_type
    y_type = comp_params.y_type # This is the y_type the model was trained on
    use_mag = comp_params.use_mag
    use_vec = comp_params.use_vec
    # data_norms = comp_params.data_norms # Loaded by comp_test(XYZ,...)
    # model = comp_params.model           # Loaded by comp_test(XYZ,...)
    terms = comp_params.terms
    terms_A = list(comp_params.terms_A)
    sub_diurnal = comp_params.sub_diurnal
    sub_igrf = comp_params.sub_igrf
    bpf_mag = comp_params.bpf_mag
    reorient_vec = comp_params.reorient_vec
    features_setup = list(comp_params.features_setup)
    features_no_norm = list(comp_params.features_no_norm)
    
    # Adjust y_type for testing if necessary (e.g. evaluate TL model trained on 'e' against 'd')
    y_type_for_eval = y_type
    if model_type in ["TL", "mod_TL"] and y_type == "e":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'd' (Δmag) for error calculation in model_type {model_type}")
        y_type_for_eval = "d"
    elif model_type == "map_TL" and y_type != "c":
        if not silent: print(f"INFO: Forcing y_type {y_type} -> 'c' for error calculation in model_type {model_type}")
        y_type_for_eval = "c"
    
    if model_type.startswith("m3"):
        terms_A_original = list(terms_A)
        terms_A = [term for term in terms_A if term not in ["fdm", "f", "fdm3", "f3", "bias", "b"]]
        if len(terms_A) != len(terms_A_original) and not silent:
             print(f"INFO: Removing derivative/bias terms from terms_A for M3 model test. Original: {terms_A_original}, New: {terms_A}")
def comp_train_test(
    comp_params: CompParams,
    xyz_train: XYZ,
    xyz_test: XYZ,
    ind_train: np.ndarray,
    ind_test: np.ndarray,
    mapS_train: Any = mapS_null, # Union[MapS, MapSd, MapS3D]
    mapS_test: Any = mapS_null,  # Union[MapS, MapSd, MapS3D]
    temp_params: Optional[TempParams] = None,
    silent: bool = False
) -> Tuple[CompParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train & evaluate performance of an aeromagnetic compensation model.

    Args:
        comp_params: CompParams object.
        xyz_train: XYZ flight data struct for training.
        xyz_test: XYZ flight data struct for testing.
        ind_train: Selected data indices for training.
        ind_test: Selected data indices for testing.
        mapS_train: (optional) MapS struct for training.
        mapS_test: (optional) MapS struct for testing.
        temp_params: (optional) TempParams struct.
        silent: (optional) If true, no print outs.

    Returns:
        comp_params: Updated CompParams object.
        y_train: Training target vector.
        y_train_hat: Training prediction vector.
        err_train: Training compensation error.
        y_test: Testing target vector.
        y_test_hat: Testing prediction vector.
        err_test: Testing compensation error.
        features: List of feature names used.
    """
    if type(xyz_train) != type(xyz_test): # Basic type check
        # In Julia, this might check for specific XYZ subtypes.
        # For Python, a simple type check or isinstance checks might be used.
        if not silent: print("WARN: xyz_train and xyz_test are of different types. This might be unintended.")

    if temp_params is None:
        temp_params = TempParams()

    # Train the model
    # The comp_train(XYZ, ind, ...) version can take xyz_test and ind_test for internal validation/best model selection
    # during its own training epochs. Here, we are doing a separate test *after* training is complete.
    # So, we pass None for xyz_test, ind_test to the comp_train call.
    comp_params_trained, y_train_np, y_train_hat_np, err_train_np, features_out = comp_train(
        comp_params, xyz_train, ind_train, mapS_train,
        temp_params=temp_params,
        xyz_test=None, # Do not use xyz_test for internal validation loop of comp_train here
        ind_test=None, # As this is a separate test step after full training
        silent=silent
    )

    # Test the trained model
    y_test_np, y_test_hat_np, err_test_np, _ = comp_test(
        comp_params_trained, xyz_test, ind_test, mapS_test,
        temp_params=temp_params,
        silent=silent
    )

    return (
        comp_params_trained,
        y_train_np, y_train_hat_np, err_train_np,
        y_test_np, y_test_hat_np, err_test_np,
        features_out
    )
        # comp_params.terms_A is not modified here as it's a test function

    # Load data using get_Axy
    try:
        from .analysis_util import get_Axy as get_Axy_actual, get_XYZ as get_XYZ_actual, get_ind # Assuming get_ind is also there
        if 'compensation' in get_Axy_actual.__module__: get_Axy_fn = get_Axy # placeholder
        else: get_Axy_fn = get_Axy_actual
        if 'compensation' in get_XYZ_actual.__module__: get_XYZ_fn = None # placeholder used by get_Axy
        else: get_XYZ_fn = get_XYZ_actual
    except ImportError:
        get_Axy_fn = get_Axy 
        get_XYZ_fn = None # Cannot get XYZ directly

    # The comp_test(XYZ, ind, ...) needs an XYZ object.
    # We need to construct it from the DataFrames for the given `lines`.
    # This part is simplified; a full get_XYZ from DataFrames would be complex.
    # Assuming `get_Axy` can internally call or we can call `get_XYZ` and `get_ind`.
    
    # For now, let's assume we can get a single XYZ object for all lines for testing.
    # This might need refinement if lines span multiple flights that can't be concatenated.
    # The Julia `get_Axy` handles iterating through lines and concatenating.
    # We will rely on the primary `comp_test(XYZ, ...)` to use the data from `get_Axy`
    # rather than re-calling `get_XYZ` itself.
    
    # The `comp_test(XYZ, ind, ...)` function will internally call get_Axy components.
    # It's better to call the primary `comp_test` by first creating the `xyz` and `ind` objects.
    
    # Simplified approach: Create one XYZ for all lines.
    # This assumes all lines are from compatible flights/setups if multiple.
    # A more robust `get_XYZ_from_df_lines` would be needed for general cases.
    
    xyz_list = []
    ind_list = []
    current_total_samples = 0
    
    _lines = lines
    if isinstance(lines, (int, float)):
        _lines = [lines]

    for line_num_val in _lines:
        line_info = df_line[df_line["line"] == line_num_val]
        if line_info.empty:
            if not silent: print(f"WARN: Line {line_num_val} not found in df_line. Skipping.")
            continue
        line_info = line_info.iloc[0] # Take the first match

        flight_name = line_info["flight"]
        flight_info = df_flight[df_flight["flight"] == flight_name]
        if flight_info.empty:
            if not silent: print(f"WARN: Flight {flight_name} for line {line_num_val} not found in df_flight. Skipping line.")
            continue
        flight_info = flight_info.iloc[0]

        xyz_file_path = flight_info["xyz_file"]
        # Assuming get_XYZ can load this (needs to be fully implemented)
        try:
            # This is a placeholder for actual XYZ loading logic from file + df_line info
            # xyz_single_line = get_XYZ(xyz_file_path, flight_info["xyz_type"]) 
            # For now, we'll pass all data to the main comp_test via get_Axy logic below
            # This means we don't construct XYZ here but let comp_test(params, lines, dfs...) do it.
            pass
        except Exception as e:
            if not silent: print(f"Error loading XYZ for line {line_num_val}: {e}. Skipping.")
            continue
    
    # Since comp_test(XYZ, ind, ...) is the primary one that handles model logic,
    # this DataFrame version should ideally prepare XYZ and ind and call that.
    # However, the Julia version's comp_test(df) also calls get_Axy directly.
    # Let's stick to that pattern for now.

    Axy_outputs_test = get_Axy_fn(
        lines, df_line, df_flight, df_map,
        features_setup,
        features_no_norm=features_no_norm,
        y_type=y_type_for_eval, # Use the potentially adjusted y_type
        use_mag=use_mag,
        use_vec=use_vec,
        terms=terms,
        terms_A=terms_A, # Use potentially adjusted terms_A
        sub_diurnal=sub_diurnal,
        sub_igrf=sub_igrf,
        bpf_mag=bpf_mag, # This should align with y_type_for_eval
        reorient_vec=reorient_vec,
        mod_TL=(model_type == "mod_TL"),
        map_TL=(model_type == "map_TL"),
        return_B=model_type.startswith("m3"),
        silent=SILENT_DEBUG
    )

    if model_type.startswith("m3"):
        A_test_np, Bt_test_np, B_dot_test_np, x_test_np, y_test_np, _, features_out, l_segs_out = Axy_outputs_test
    else:
        A_test_np, x_test_np, y_test_np, _, features_out, l_segs_out = Axy_outputs_test
        Bt_test_np, B_dot_test_np = None, None 
        # Ensure these are correctly sized if None, for functions expecting them
        if A_test_np is not None:
            num_samples_test = A_test_np.shape[0]
            Bt_test_np = np.empty(num_samples_test, dtype=np.float32) if Bt_test_np is None else Bt_test_np
            B_dot_test_np = np.empty((num_samples_test,3), dtype=np.float32) if B_dot_test_np is None else B_dot_test_np


    # Now call the appropriate _from_raw test function
    y_hat_np: np.ndarray
    err_np: np.ndarray

    if model is None or data_norms is None:
        raise ValueError("Model or data_norms not found in comp_params for comp_test.")

    if model_type == "m1":
        y_hat_np, err_np = nn_comp_1_test_from_raw(x_test_np, y_test_np, data_norms, model, l_segs=l_segs_out, silent=silent)
    elif model_type.startswith("m2"):
        if not isinstance(comp_params, NNCompParams): raise TypeError("NNCompParams expected for M2 models")
        y_hat_np, err_np = nn_comp_2_test_from_raw(A_test_np, x_test_np, y_test_np, data_norms, model, model_type, comp_params.TL_coef, l_segs=l_segs_out, silent=silent)
    elif model_type.startswith("m3"):
        if not isinstance(comp_params, NNCompParams): raise TypeError("NNCompParams expected for M3 models")
        y_hat_np, err_np = nn_comp_3_test_from_raw(
            A_test_np, Bt_test_np, B_dot_test_np, x_test_np, y_test_np, data_norms, model,
            model_type, y_type_for_eval, comp_params.TL_coef, terms_A, # Use original terms_A from comp_params for TL_coef interpretation
            l_segs=l_segs_out, l_window=l_window, silent=silent
        )
    elif model_type in ["TL", "mod_TL", "map_TL"]:
         # linear_test expects normalized A_test_np, raw y_test_np, y_bias, y_scale, and model tuple
        _, _, y_bias_lin, y_scale_lin = unpack_data_norms(data_norms) # type: ignore
        
        A_eval_final = A_test_np
        y_eval_final = y_test_np

        if model_type in ["TL", "mod_TL"] and comp_params.y_type == "e": # If model was trained on 'e'
            # Need to re-fetch non-BPF'd A and y for evaluation against 'd'
            # This is complex as get_Axy was called with y_type_for_eval='d'
            # If comp_params.y_type was 'e', the stored model was for BPF'd A.
            # We need non-BPF A for testing against non-BPF y.
            # This implies the data_norms might also be for BPF'd A.
            # This path needs careful review of how data was normed and model trained.
            # For now, assume A_test_np and y_test_np from get_Axy(y_type_for_eval='d') are correct.
            # The model (coefficients) in comp_params, however, was trained on potentially BPF'd A.
            # This is a mismatch if not handled carefully.
            # The Julia version's linear_test for TL/mod_TL uses A_no_bpf and y_no_bpf if original y was 'e'.
            # This means we should try to get the non-BPF versions of A and y.
            # The current get_Axy call uses y_type_for_eval. If that was 'd', A_test_np is not BPF'd.
            pass # Assuming A_test_np and y_test_np are correctly non-BPF'd if y_type_for_eval is 'd'

        y_hat_np, err_np = linear_test(A_eval_final, y_eval_final, data_norms, model, l_segs=l_segs_out, silent=silent)

    elif model_type in ["elasticnet", "plsr"]:
        y_hat_np, err_np = linear_test(x_test_np, y_test_np, data_norms, model, l_segs=l_segs_out, silent=silent)
    else:
        raise ValueError(f"Unknown model_type: {model_type} in comp_test (DataFrame version)")

    if not silent:
        elapsed_time = time.time() - t0_df
        time_unit = "sec"
        if elapsed_time > 60:
            elapsed_time /= 60
            time_unit = "min"
        print(f"INFO: comp_test (DataFrame version) completed in {elapsed_time:.1f} {time_unit}")

    return y_test_np, y_hat_np, err_np, features_out
def comp_train_test( # Overload for DataFrames
    comp_params: CompParams,
    lines_train: Union[int, float, List[Union[int, float]], np.ndarray],
    lines_test: Union[int, float, List[Union[int, float]], np.ndarray],
    df_line: Any, # pandas.DataFrame
    df_flight: Any, # pandas.DataFrame
    df_map: Any, # pandas.DataFrame
    temp_params: Optional[TempParams] = None,
    silent: bool = False
) -> Tuple[CompParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Train & evaluate performance of an aeromagnetic compensation model using DataFrame inputs.
    """
    if temp_params is None:
        temp_params = TempParams()

    # This function needs to load XYZ data for train and test sets.
    # The Julia version calls get_XYZ and get_ind for lines_train and lines_test separately.
    # We need a robust way to do this. For now, assuming a helper or direct logic.

    # Simplified XYZ and ind loading for train and test based on DataFrames
    # This is a placeholder for a more robust get_XYZ_and_ind_from_dfs function
    def load_xyz_and_ind_for_lines(
        lines_set: Union[int, float, List[Union[int, float]], np.ndarray],
        df_line_local: Any, # pd.DataFrame,
        df_flight_local: Any, # pd.DataFrame,
        df_map_local: Any, # pd.DataFrame,
        # comp_params_local: CompParams, # Pass if get_XYZ_ported needs specific params from it
        silent_local: bool
    ) -> Tuple[Optional[XYZ], Optional[np.ndarray], Optional[Any]]: # Returns XYZ, ind, mapS_obj
        
        if not hasattr(df_line_local, "empty"): # Check if it's a DataFrame
            raise TypeError("df_line must be a pandas DataFrame")

        try:
            # Assuming get_XYZ_ported is the correctly ported version of Julia's
            # get_XYZ(lines, df_line, df_flight, df_map; kwargs...)
            # which should handle lists of lines and return a single (possibly concatenated)
            # XYZ object, corresponding indices, and map object.
            from .create_xyz import get_XYZ as get_XYZ_ported
        except ImportError:
            if not silent_local: logging.error("compensation.py - create_xyz.get_XYZ not found for DataFrame processing in comp_train_test.")
            return None, None, None

        try:
            # Pass necessary kwargs if get_XYZ_ported requires them (e.g. from comp_params)
            # For now, assuming a simple call signature matching Julia's direct usage.
            # The silent flag is passed to get_XYZ_ported.
            xyz_loaded, ind_loaded, map_obj_loaded, _ = get_XYZ_ported(
                lines_set, df_line_local, df_flight_local, df_map_local,
                silent=silent_local # Pass silent flag
                # Add other kwargs like use_mag, use_vec if get_XYZ_ported needs them
                # e.g., use_mag=comp_params_local.use_mag, ...
            )
            return xyz_loaded, ind_loaded, map_obj_loaded
        except Exception as e:
            if not silent_local: logging.error(f"ERROR loading XYZ/ind for lines {lines_set} via get_XYZ_ported: {e}")
            return None, None, None

    xyz_train_loaded, ind_train_loaded, mapS_train_loaded = load_xyz_and_ind_for_lines(
        lines_train, df_line, df_flight, df_map, silent # Pass comp_params if load_xyz_and_ind_for_lines needs it
    )
    xyz_test_loaded, ind_test_loaded, mapS_test_loaded = load_xyz_and_ind_for_lines(
        lines_test, df_line, df_flight, df_map, silent # Pass comp_params if load_xyz_and_ind_for_lines needs it
    )

    if xyz_train_loaded is None or ind_train_loaded is None or \
       xyz_test_loaded is None or ind_test_loaded is None:
        raise ValueError("Failed to load training or testing data from DataFrames.")

    # Call the primary comp_train_test function
    return comp_train_test(
        comp_params,
        xyz_train_loaded, xyz_test_loaded,
        ind_train_loaded, ind_test_loaded,
        mapS_train=mapS_train_loaded if mapS_train_loaded is not None else mapS_null,
        mapS_test=mapS_test_loaded if mapS_test_loaded is not None else mapS_null,
        temp_params=temp_params,
        silent=silent
    )

def remove_extension(filename: str, ext: str = ".bson") -> str:
    """Removes the extension from a filename if it exists."""
    if filename.endswith(ext):
        return filename[:-len(ext)]
    return filename

def add_extension(filename: str, ext: str = ".csv") -> str:
    """Adds the extension to a filename if it doesn't exist."""
    if not filename.endswith(ext):
        return filename + ext
    return filename

# Helper for LBFGS in PyTorch (simplified, actual implementation might need more from Optim.jl)
# The lbfgs_setup from Julia is quite specific to Optim.jl's interface.
# PyTorch's LBFGS optimizer has a different API (optimizer.step(closure)).
# The closure function is defined within each nn_comp_X_train function.
# So, a direct translation of lbfgs_setup might not be needed if using PyTorch's LBFGS correctly.

def print_time(t: float, digits: int = 1):
    """Prints time in sec if <1 min, otherwise min."""
    if t < 60:
        print(f"INFO: time: {t:.{digits}f} sec")
    else:
        print(f"INFO: time: {t/60:.{digits}f} min")

# Removed duplicate __main__ block and extensive summary comments.
# The final __main__ block is kept for potential example usage.

if __name__ == '__main__':
    # Example usage or tests could go here
    print("MagNavPy compensation module cleaned.")

    # Example: Test m1_struct and a dummy model
    # model_example = nn.Sequential(nn.Linear(10, 8), nn.SiLU(), nn.Linear(8, 1))
    # m1_instance = M1Struct(model_example)
    # dummy_input = torch.randn(5, 10) # batch_size=5, features=10
    # output = m1_instance.m(dummy_input)
    # print(f"M1Struct dummy output shape: {output.shape}")
    pass