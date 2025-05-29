import numpy as np
from typing import Union

def fdm(x: np.ndarray, dt: Union[float, np.ndarray] = 1.0, scheme: str = "central") -> np.ndarray:
    """
    Finite difference method for derivatives.

    Args:
        x: 1D array of data.
        dt: (optional) time step or array of time steps. Default is 1.0.
        scheme: (optional) finite difference scheme ('central', 'forward', 'backward', 'fourth').
                Default is 'central'.

    Returns:
        np.ndarray: Derivative of x.
    """
    N = len(x)
    dx = np.zeros_like(x, dtype=float)

    if isinstance(dt, np.ndarray) and dt.ndim == 0:
        dt = dt.item() # Convert 0-dim array to scalar
    elif isinstance(dt, np.ndarray) and dt.ndim == 1 and len(dt) == N:
        pass # dt is already an array of correct size
    elif isinstance(dt, (float, int)):
        dt = np.full(N, dt, dtype=float) # Broadcast scalar dt to array
    else:
        raise ValueError("dt must be a scalar or a 1D array of the same length as x.")

    if scheme == "central":
        if N < 3:
            raise ValueError("Central difference requires at least 3 data points.")
        dx[0] = (x[1] - x[0]) / dt[0] # Forward difference for first point
        dx[N-1] = (x[N-1] - x[N-2]) / dt[N-1] # Backward difference for last point
        for i in range(1, N - 1):
            dx[i] = (x[i+1] - x[i-1]) / (dt[i+1] + dt[i-1]) # dt[i+1] + dt[i-1] for non-uniform dt
    elif scheme == "forward":
        if N < 2:
            raise ValueError("Forward difference requires at least 2 data points.")
        for i in range(N - 1):
            dx[i] = (x[i+1] - x[i]) / dt[i]
        dx[N-1] = (x[N-1] - x[N-2]) / dt[N-1] # Backward difference for last point
    elif scheme == "backward":
        if N < 2:
            raise ValueError("Backward difference requires at least 2 data points.")
        dx[0] = (x[1] - x[0]) / dt[0] # Forward difference for first point
        for i in range(1, N):
            dx[i] = (x[i] - x[i-1]) / dt[i-1] # dt[i-1] for previous interval
    elif scheme == "fourth":
        if N < 5:
            raise ValueError("Fourth-order central difference requires at least 5 data points.")
        # Fourth-order central difference for interior points
        for i in range(2, N - 2):
            dx[i] = (-x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2]) / (12 * dt[i]) # Assuming uniform dt for simplicity here
        # Use lower-order schemes for boundary points
        dx[0] = (x[1] - x[0]) / dt[0]
        dx[1] = (x[2] - x[0]) / (dt[1] + dt[0])
        dx[N-2] = (x[N-1] - x[N-3]) / (dt[N-2] + dt[N-3])
        dx[N-1] = (x[N-1] - x[N-2]) / dt[N-1]
    else:
        raise ValueError(f"Unsupported finite difference scheme: {scheme}")
        
    return dx