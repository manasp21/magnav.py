import numpy as np
import math
from typing import Tuple, Union

def skew(v: np.ndarray) -> np.ndarray:
    """
    Create a skew-symmetric matrix from a 3-element vector.

    Args:
        v: 3-element numpy array.

    Returns:
        3x3 skew-symmetric matrix.
    """
    if not isinstance(v, np.ndarray) or v.shape != (3,):
        raise ValueError("Input must be a 3-element numpy array.")
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def dcm2euler(Cnb: np.ndarray, order: str = 'zyx') -> np.ndarray:
    """
    Convert Direction Cosine Matrix (DCM) to Euler angles.

    Args:
        Cnb: 3x3xN (or 3x3) Direction Cosine Matrix.
             If 3x3xN, returns N Euler angle sets.
        order: (optional) Euler angle rotation order and interpretation.
               Default is 'zyx'.
               Supported orders:
               - 'zyx': Assumes Cnb = Rz(yaw)Ry(pitch)Rx(roll).
                        Returns (yaw, pitch, roll).
               - 'xyz': Assumes Cnb is a DCM from which roll, pitch, yaw are to be extracted
                        using formulas typically applied to Cbn = (Rz(yaw)Ry(pitch)Rx(roll))^T.
                        Effectively, if Cnb is Cbn_julia, this returns (roll, pitch, yaw).
                        Returns (roll, pitch, yaw).
               (Other sequences like 'xzy', 'yxz', 'yzx', 'zxy' can be added if needed).

    Returns:
        np.ndarray:
        If Cnb is 3x3, returns a 1D NumPy array of 3 elements: [angle1, angle2, angle3].
        If Cnb is 3x3xN, returns a 2D NumPy array of shape (N, 3): [[a1,a2,a3], ...].
        The interpretation of angles depends on 'order':
        - For 'zyx': (yaw, pitch, roll)
        - For 'xyz': (roll, pitch, yaw)
        - For 'body2nav': (roll, pitch, yaw)

    Julia MagNav.jl Mapping:
        - To match Julia `dcm2euler(Cnb, order=:body2nav)` which returns `(roll, pitch, yaw)`:
          `yaw_py, pitch_py, roll_py = dcm2euler(Cnb, order='zyx')`
          Result: `(roll_py, pitch_py, yaw_py)`
        - To match Julia `dcm2euler(Cbn, order=:nav2body)` which returns `(roll, pitch, yaw)`
          (where Cbn is the DCM from navigation to body frame, i.e., Cnb_julia.T):
          `roll_py, pitch_py, yaw_py = dcm2euler(Cbn, order='xyz')`
          Result: `(roll_py, pitch_py, yaw_py)`
    """
    if Cnb.ndim == 2:
        Cnb_proc = Cnb[:, :, np.newaxis] # Make it 3x3x1 for consistent processing
    elif Cnb.ndim == 3:
        Cnb_proc = Cnb
    else:
        raise ValueError("Input DCM must be 3x3 or 3x3xN")

    N = Cnb_proc.shape[2]
    angle1_out = np.zeros(N)
    angle2_out = np.zeros(N)
    angle3_out = np.zeros(N)

    for i in range(N):
        C = Cnb_proc[:, :, i]
        if order == 'zyx': # Yaw, Pitch, Roll from Cnb = Rz(yaw)Ry(pitch)Rx(roll)
            pitch_val = -C[2, 0]
            if pitch_val >= 1.0:
                pitch = math.pi / 2.0
            elif pitch_val <= -1.0:
                pitch = -math.pi / 2.0
            else:
                pitch = math.asin(pitch_val)

            if abs(abs(pitch) - math.pi / 2.0) < 1e-9: # Near pi/2 or -pi/2 (Gimbal lock)
                roll = 0.0 
                yaw = math.atan2(C[0, 1], C[1, 1]) 
            else:
                roll = math.atan2(C[2, 1], C[2, 2])
                yaw = math.atan2(C[1, 0], C[0, 0])

            angle1_out[i] = yaw
            angle2_out[i] = pitch
            angle3_out[i] = roll
        elif order == 'xyz': # Extracts (roll, pitch, yaw) assuming C is effectively Cbn_julia
                             # Cbn_julia[1,3] (0-indexed C[0,2]) = -sin(pitch)
                             # Cbn_julia[2,3] (0-indexed C[1,2]) = sin(roll)cos(pitch)
                             # Cbn_julia[3,3] (0-indexed C[2,2]) = cos(roll)cos(pitch)
                             # Cbn_julia[1,2] (0-indexed C[0,1]) = cos(pitch)sin(yaw)
                             # Cbn_julia[1,1] (0-indexed C[0,0]) = cos(pitch)cos(yaw)
            pitch_val = -C[0, 2] 
            if pitch_val >= 1.0:
                pitch = math.pi / 2.0
            elif pitch_val <= -1.0:
                pitch = -math.pi / 2.0
            else:
                pitch = math.asin(pitch_val)

            if abs(abs(pitch) - math.pi / 2.0) < 1e-9: # Gimbal lock
                # For Cbn, if pitch = pi/2 (sp=1, cp=0):
                # Cbn = [[0,0,-1],[-cr*sy+sr*cy, cr*cy+sr*sy, 0],[sr*sy+cr*cy, -sr*cy+cr*sy,0]]
                # yaw and roll are not uniquely defined. Set yaw = 0.
                # roll = atan2(C[1,0], C[2,0]) (if C[1,0]=sin(roll-yaw), C[2,0]=cos(roll-yaw))
                # roll = atan2(C[1,0], C[1,1]) is not directly applicable here.
                # From Julia dcm2euler for nav2body, if pitch is +/-pi/2,
                # roll = atan2(c23,c33) and yaw = atan2(c12,c11) would have cos(pitch)=0 in denominator.
                # A common convention: yaw = 0, roll = atan2(C[1,0], C[2,0]) (if C is from XYZ)
                # Or, roll = 0, yaw = atan2(C[1,0],C[0,0]) (if C is from ZYX)
                # Let's follow a convention for Cbn: yaw=0, roll = atan2(C[1,0], C[1,1])
                # C[1,0] = -cr*sy+sr*sp*cy ; C[1,1] = cr*cy+sr*sp*sy
                # If pitch = pi/2 (sp=1): C[1,0] = -cr*sy+sr*cy; C[1,1] = cr*cy+sr*sy
                # roll = atan2(-cr*sy+sr*cy, cr*cy+sr*sy) = atan2(sin(roll-yaw_orig), cos(roll-yaw_orig))
                # This is roll - yaw_original. If yaw_original is not necessarily 0.
                # For simplicity and to match typical gimbal lock handling where one angle is arbitrary:
                yaw_extracted = 0.0 
                roll_extracted = math.atan2(C[1,0], C[1,1]) # This is roll_orig - yaw_orig if C=Cnb.T
                                                          # If C=Cbn, this is atan2(-cr*sy+sr*cy, cr*cy+sr*sy)
            else:
                roll_extracted = math.atan2(C[1, 2], C[2, 2]) 
                yaw_extracted = math.atan2(C[0, 1], C[0, 0]) 

            angle1_out[i] = roll_extracted 
            angle2_out[i] = pitch      
            angle3_out[i] = yaw_extracted 
        elif order == 'body2nav': # Roll, Pitch, Yaw from Cnb = Rz(yaw)Ry(pitch)Rx(roll)
            # Calculations are the same as 'zyx' for yaw_calc, pitch_calc, roll_calc
            # but the output order is (roll, pitch, yaw)
            pitch_val = -C[2, 0]
            if pitch_val >= 1.0:
                current_pitch = math.pi / 2.0
            elif pitch_val <= -1.0:
                current_pitch = -math.pi / 2.0
            else:
                current_pitch = math.asin(pitch_val)

            if abs(abs(current_pitch) - math.pi / 2.0) < 1e-9: # Gimbal lock
                # Following 'zyx' gimbal lock convention:
                current_roll = 0.0
                current_yaw = math.atan2(C[0, 1], C[1, 1])
            else:
                current_roll = math.atan2(C[2, 1], C[2, 2])
                current_yaw = math.atan2(C[1, 0], C[0, 0])

            angle1_out[i] = current_roll  # roll
            angle2_out[i] = current_pitch # pitch
            angle3_out[i] = current_yaw   # yaw
        else:
            raise ValueError(f"Unsupported Euler angle order for dcm2euler: {order}")

    if Cnb.ndim == 2: # If input was 3x3, return a 1D array
        return np.array([angle1_out[0], angle2_out[0], angle3_out[0]])
    # If input was 3x3xN, return an Nx3 array
    return np.stack((angle1_out, angle2_out, angle3_out), axis=1)

def euler2dcm(angle1: Union[float, np.ndarray],
              angle2: Union[float, np.ndarray],
              angle3: Union[float, np.ndarray],
              order: str = 'zyx') -> np.ndarray:
    """
    Convert Euler angles to Direction Cosine Matrix (DCM).

    Args:
        angle1, angle2, angle3: Euler angles in radians. Can be scalars or 1D arrays.
                                 The interpretation depends on 'order'.
        order: (optional) Euler angle rotation order. Default is 'zyx'.
               Supported orders:
               - 'zyx': angle1=yaw, angle2=pitch, angle3=roll. DCM = Rz(yaw)Ry(pitch)Rx(roll).
               - 'xyz': angle1=roll, angle2=pitch, angle3=yaw. DCM = Rx(roll)Ry(pitch)Rz(yaw).
               (Other sequences like 'xzy', 'yxz', 'yzx', 'zxy' can be added if needed).

    Returns:
        np.ndarray: 3x3xN (if inputs are arrays) or 3x3 (if inputs are scalars) DCM.

    Julia MagNav.jl Mapping:
        - To match Julia `euler2dcm(roll, pitch, yaw, order=:body2nav)` which produces Cnb:
          `Cnb_py = euler2dcm(yaw, pitch, roll, order='zyx')`
        - To match Julia `euler2dcm(roll, pitch, yaw, order=:nav2body)` which produces Cbn (Cnb_julia.T):
          `Cnb_temp = euler2dcm(yaw, pitch, roll, order='zyx')`
          If Cnb_temp is 3x3: `Cbn_py = Cnb_temp.T`
          If Cnb_temp is 3x3xN: `Cbn_py = np.transpose(Cnb_temp, axes=(1,0,2))`
    """
    is_scalar = not (isinstance(angle1, np.ndarray) and angle1.ndim > 0) and \
                  not (isinstance(angle2, np.ndarray) and angle2.ndim > 0) and \
                  not (isinstance(angle3, np.ndarray) and angle3.ndim > 0)

    if is_scalar:
        # Ensure inputs are float for math functions
        a1_arr = np.array([float(angle1)])
        a2_arr = np.array([float(angle2)])
        a3_arr = np.array([float(angle3)])
    else:
        a1_arr = np.atleast_1d(np.asarray(angle1, dtype=float))
        a2_arr = np.atleast_1d(np.asarray(angle2, dtype=float))
        a3_arr = np.atleast_1d(np.asarray(angle3, dtype=float))


    if not (len(a1_arr) == len(a2_arr) == len(a3_arr)):
        raise ValueError("Input angle arrays must have the same length.")

    N = len(a1_arr)
    C_out = np.zeros((3, 3, N))

    for i in range(N):
        a1_val, a2_val, a3_val = a1_arr[i], a2_arr[i], a3_arr[i]

        if order == 'zyx': # angle1=yaw (Z), angle2=pitch (Y'), angle3=roll (X'')
            Rz = np.array([
                [math.cos(a1_val), -math.sin(a1_val), 0],
                [math.sin(a1_val),  math.cos(a1_val), 0],
                [0,                 0,                1]
            ])
            Ry = np.array([
                [math.cos(a2_val),  0, math.sin(a2_val)],
                [0,                 1, 0               ],
                [-math.sin(a2_val), 0, math.cos(a2_val)]
            ])
            Rx = np.array([
                [1, 0,                 0               ],
                [0, math.cos(a3_val), -math.sin(a3_val)],
                [0, math.sin(a3_val),  math.cos(a3_val)]
            ])
            C_out[:, :, i] = Rz @ Ry @ Rx
        elif order == 'xyz': # angle1=roll (X), angle2=pitch (Y'), angle3=yaw (Z'')
            Rx = np.array([
                [1, 0,                 0               ],
                [0, math.cos(a1_val), -math.sin(a1_val)],
                [0, math.sin(a1_val),  math.cos(a1_val)]
            ])
            Ry = np.array([
                [math.cos(a2_val),  0, math.sin(a2_val)],
                [0,                 1, 0               ],
                [-math.sin(a2_val), 0, math.cos(a2_val)]
            ])
            Rz = np.array([
                [math.cos(a3_val), -math.sin(a3_val), 0],
                [math.sin(a3_val),  math.cos(a3_val), 0],
                [0,                 0,                1]
            ])
            C_out[:, :, i] = Rx @ Ry @ Rz
        else:
            raise ValueError(f"Unsupported Euler angle order for euler2dcm: {order}")

    if is_scalar:
        return C_out[:, :, 0]
    return C_out

def correct_Cnb(Cnb: np.ndarray, tilt_err: np.ndarray) -> np.ndarray:
    """
    Corrects a body-to-navigation DCM (Cnb) with tilt angle errors.
    The resulting DCM is the "estimated" or "in error" Cnb.
    Cnb_estimate = R_error * Cnb_true
    where R_error = expm(skew(-tilt_err)) is the rotation matrix from the tilt_err vector.

    This function ports the logic from MagNav.jl/src/dcm.jl correct_Cnb.
    The formula for B (R_error) used in Julia:
    B = I - (sin(m)/m)*S_psi - ((1-cos(m))/m^2)*(S_psi @ S_psi)
    where m = ||tilt_err|| and S_psi = skew(tilt_err).
    This B corresponds to expm(skew(-tilt_err)).

    Args:
        Cnb: 3x3xN (or 3x3) true Direction Cosine Matrix (body to navigation).
        tilt_err: 3xN (or 3x1 or 1D array of 3) tilt angle error vector [rad].
                  Each column is an [X, Y, Z] tilt error.

    Returns:
        np.ndarray: 3x3xN (or 3x3) "in error" Direction Cosine Matrix.
    """
    if Cnb.ndim == 2:
        Cnb_proc = Cnb[:, :, np.newaxis]
    elif Cnb.ndim == 3:
        Cnb_proc = Cnb
    else:
        raise ValueError("Input Cnb DCM must be 3x3 or 3x3xN")

    # Process tilt_err to be 3xN
    if tilt_err.ndim == 1:
        if tilt_err.shape == (3,):
            tilt_err_proc = tilt_err[:, np.newaxis]
        else:
            raise ValueError("1D tilt_err must have 3 elements.")
    elif tilt_err.ndim == 2:
        if tilt_err.shape[0] == 3:
            tilt_err_proc = tilt_err
        else:
            raise ValueError("2D tilt_err must be of shape 3xN.")
    else:
        raise ValueError("tilt_err must be a 3-element vector or a 3xN array.")


    num_Cnb = Cnb_proc.shape[2]
    num_tilt = tilt_err_proc.shape[1]

    if num_Cnb != num_tilt:
        if num_Cnb == 1 and num_tilt > 1: # Apply single Cnb to multiple errors
             Cnb_proc = np.repeat(Cnb_proc, num_tilt, axis=2)
        elif num_tilt == 1 and num_Cnb > 1: # Apply single error to multiple Cnbs
             tilt_err_proc = np.repeat(tilt_err_proc, num_Cnb, axis=1)
        else:
            raise ValueError(
                f"Number of Cnb matrices ({num_Cnb}) and tilt_err vectors ({num_tilt}) must match, "
                "or one of them must be singular (N=1)."
            )
    
    N = Cnb_proc.shape[2]
    Cnb_estimate = np.zeros_like(Cnb_proc)

    for i in range(N):
        psi = tilt_err_proc[:, i]
        m = np.linalg.norm(psi)
        
        S_psi = skew(psi)

        if np.isclose(m, 0.0): # Handles m == 0 case
            B = np.eye(3)
        else:
            # Formula from Julia: B = I - sin(m)/m*s - (1-cos(m))/m^2*s^2
            # This corresponds to expm(skew(-psi))
            term_sin_m_div_m = math.sin(m) / m
            term_1_minus_cos_m_div_m_sq = (1 - math.cos(m)) / (m * m)
            
            B = np.eye(3) - term_sin_m_div_m * S_psi - \
                term_1_minus_cos_m_div_m_sq * (S_psi @ S_psi)
        
        Cnb_estimate[:, :, i] = B @ Cnb_proc[:, :, i]

    if Cnb.ndim == 2 and tilt_err.ndim == 1 : # If original inputs were singular
        return Cnb_estimate[:, :, 0]
    return Cnb_estimate