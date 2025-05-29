import numpy as np
import math
from typing import Tuple, Union

def dcm2euler(Cnb: np.ndarray, order: str = 'zyx') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Direction Cosine Matrix (DCM) to Euler angles.

    Args:
        Cnb: 3x3xN (or 3x3) Direction Cosine Matrix (body to navigation).
             If 3x3xN, returns N Euler angle sets.
        order: (optional) Euler angle rotation order. Default is 'zyx' (yaw, pitch, roll).
               Other options: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (angle1, angle2, angle3) in radians.
                                                   The interpretation of angles depends on 'order'.
                                                   For 'zyx': (yaw, pitch, roll).
    """
    if Cnb.ndim == 2:
        Cnb = Cnb[:, :, np.newaxis] # Make it 3x3x1 for consistent processing
    
    N = Cnb.shape[2]
    angle1 = np.zeros(N)
    angle2 = np.zeros(N)
    angle3 = np.zeros(N)

    for i in range(N):
        C = Cnb[:, :, i]
        if order == 'zyx': # Yaw, Pitch, Roll (common for navigation)
            pitch_val = -C[2, 0]
            if pitch_val >= 1.0:
                pitch = math.pi / 2.0
            elif pitch_val <= -1.0:
                pitch = -math.pi / 2.0
            else:
                pitch = math.asin(pitch_val)
            
            if abs(pitch) > (math.pi / 2.0 - 1e-6): # Gimbal lock or near singularity
                roll = 0.0 # Arbitrary, often set to 0
                yaw = math.atan2(C[0, 1], C[1, 1])
            else:
                roll = math.atan2(C[2, 1], C[2, 2])
                yaw = math.atan2(C[1, 0], C[0, 0])
            
            angle1[i] = yaw
            angle2[i] = pitch
            angle3[i] = roll
        elif order == 'xyz': # Roll, Pitch, Yaw
            pitch_val = -C[0, 2]
            if pitch_val >= 1.0:
                pitch = math.pi / 2.0
            elif pitch_val <= -1.0:
                pitch = -math.pi / 2.0
            else:
                pitch = math.asin(pitch_val)
            
            if abs(pitch) > (math.pi / 2.0 - 1e-6):
                yaw = 0.0
                roll = math.atan2(C[1, 0], C[2, 0])
            else:
                yaw = math.atan2(C[0, 1], C[0, 0])
                roll = math.atan2(C[1, 2], C[2, 2])
            
            angle1[i] = roll
            angle2[i] = pitch
            angle3[i] = yaw
        # Add other orders as needed (xzy, yxz, yzx, zxy)
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
            
    if Cnb.shape[2] == 1: # If input was 3x3, return scalars
        return angle1[0], angle2[0], angle3[0]
    return angle1, angle2, angle3

def euler2dcm(angle1: Union[float, np.ndarray], angle2: Union[float, np.ndarray], angle3: Union[float, np.ndarray], order: str = 'zyx') -> np.ndarray:
    """
    Convert Euler angles to Direction Cosine Matrix (DCM).

    Args:
        angle1, angle2, angle3: Euler angles in radians. Can be scalars or 1D arrays.
                                The interpretation depends on 'order'.
                                For 'zyx': (yaw, pitch, roll).
        order: (optional) Euler angle rotation order. Default is 'zyx' (yaw, pitch, roll).
               Other options: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy'.

    Returns:
        np.ndarray: 3x3xN (or 3x3) Direction Cosine Matrix.
    """
    is_scalar = not isinstance(angle1, np.ndarray)
    if is_scalar:
        angle1 = np.array([angle1])
        angle2 = np.array([angle2])
        angle3 = np.array([angle3])

    N = len(angle1)
    Cnb = np.zeros((3, 3, N))

    for i in range(N):
        a1, a2, a3 = angle1[i], angle2[i], angle3[i]
        
        if order == 'zyx': # Yaw (Z), Pitch (Y'), Roll (X'')
            Rz = np.array([
                [math.cos(a1), -math.sin(a1), 0],
                [math.sin(a1), math.cos(a1),  0],
                [0,            0,             1]
            ])
            Ry = np.array([
                [math.cos(a2),  0, math.sin(a2)],
                [0,             1, 0           ],
                [-math.sin(a2), 0, math.cos(a2)]
            ])
            Rx = np.array([
                [1, 0,            0           ],
                [0, math.cos(a3), -math.sin(a3)],
                [0, math.sin(a3), math.cos(a3) ]
            ])
            Cnb[:, :, i] = Rz @ Ry @ Rx
        elif order == 'xyz': # Roll (X), Pitch (Y'), Yaw (Z'')
            Rx = np.array([
                [1, 0,            0           ],
                [0, math.cos(a1), -math.sin(a1)],
                [0, math.sin(a1), math.cos(a1) ]
            ])
            Ry = np.array([
                [math.cos(a2),  0, math.sin(a2)],
                [0,             1, 0           ],
                [-math.sin(a2), 0, math.cos(a2)]
            ])
            Rz = np.array([
                [math.cos(a3), -math.sin(a3), 0],
                [math.sin(a3), math.cos(a3),  0],
                [0,            0,             1]
            ])
            Cnb[:, :, i] = Rx @ Ry @ Rz
        # Add other orders as needed
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
            
    if is_scalar:
        return Cnb[:, :, 0]
    return Cnb