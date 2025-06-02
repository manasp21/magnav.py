import numpy as np
from typing import Union, Callable

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

def fdm_gradient(func: Callable[[np.ndarray], float],
                 x0: np.ndarray,
                 h: float = 1e-6,
                 method: str = "central") -> np.ndarray:
    """
    Computes the gradient of a scalar-valued function using finite differences.

    Args:
        func: The function R^n -> R to differentiate.
        x0: The point (1D NumPy array) at which to compute the gradient.
        h: Step size for finite differences. Default is 1e-6.
        method: Finite difference method ('central', 'forward', 'backward').
                Default is 'central'.

    Returns:
        np.ndarray: The gradient vector of the function at x0.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    grad = np.zeros(n, dtype=float)

    if method not in ["central", "forward", "backward"]:
        raise ValueError("Method must be 'central', 'forward', or 'backward'.")

    for i in range(n):
        x_plus_h = x0.copy()
        x_minus_h = x0.copy()
        x_plus_h[i] += h
        x_minus_h[i] -= h

        if method == "central":
            grad[i] = (func(x_plus_h) - func(x_minus_h)) / (2 * h)
        elif method == "forward":
            grad[i] = (func(x_plus_h) - func(x0)) / h
        elif method == "backward":
            grad[i] = (func(x0) - func(x_minus_h)) / h

    return grad

def fdm_jacobian(func: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray,
                 h: float = 1e-6,
                 method: str = "central") -> np.ndarray:
    """
    Computes the Jacobian of a vector-valued function using finite differences.

    Args:
        func: The function R^n -> R^m to differentiate.
        x0: The point (1D NumPy array of size n) at which to compute the Jacobian.
        h: Step size for finite differences. Default is 1e-6.
        method: Finite difference method ('central', 'forward', 'backward') for gradient calculation.
                Default is 'central'.

    Returns:
        np.ndarray: The Jacobian matrix (m x n) of the function at x0.
    """
    x0 = np.asarray(x0, dtype=float)
    n_input = len(x0)

    # Evaluate function at x0 to determine output dimension m
    f_x0 = func(x0)
    f_x0 = np.asarray(f_x0) # Ensure it's an array
    if f_x0.ndim == 0: # Scalar output
        m_output = 1
        # Reshape to (1,) if it's a scalar, so jacobian is (1, n_input)
        f_x0_reshaped = f_x0.reshape(1)
    elif f_x0.ndim == 1: # Vector output
        m_output = len(f_x0)
        f_x0_reshaped = f_x0
    else:
        raise ValueError("Function output must be scalar or 1D array.")

    jacobian = np.zeros((m_output, n_input), dtype=float)

    if method not in ["central", "forward", "backward"]:
        raise ValueError("Method must be 'central', 'forward', or 'backward'.")

    for j in range(n_input): # Iterate over input variables x_j
        x_plus_h = x0.copy()
        x_minus_h = x0.copy()
        x_plus_h[j] += h
        x_minus_h[j] -= h

        if method == "central":
            f_plus_h = np.asarray(func(x_plus_h)).reshape(m_output)
            f_minus_h = np.asarray(func(x_minus_h)).reshape(m_output)
            jacobian[:, j] = (f_plus_h - f_minus_h) / (2 * h)
        elif method == "forward":
            f_plus_h = np.asarray(func(x_plus_h)).reshape(m_output)
            jacobian[:, j] = (f_plus_h - f_x0_reshaped) / h
        elif method == "backward":
            f_minus_h = np.asarray(func(x_minus_h)).reshape(m_output)
            jacobian[:, j] = (f_x0_reshaped - f_minus_h) / h

    return jacobian

def fdm_hessian(func: Callable[[np.ndarray], float],
                x0: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
    """
    Computes the Hessian of a scalar-valued function using central finite differences.

    Args:
        func: The function R^n -> R to differentiate.
        x0: The point (1D NumPy array) at which to compute the Hessian.
        h: Step size for finite differences. Default is 1e-5.

    Returns:
        np.ndarray: The Hessian matrix (n x n) of the function at x0.
    """
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    hessian = np.zeros((n, n), dtype=float)
    fx0 = func(x0)

    for i in range(n):
        # Diagonal elements: d^2f / dx_i^2
        x_plus_h_i = x0.copy()
        x_plus_h_i[i] += h
        x_minus_h_i = x0.copy()
        x_minus_h_i[i] -= h
        hessian[i, i] = (func(x_plus_h_i) - 2 * fx0 + func(x_minus_h_i)) / (h**2)

        # Off-diagonal elements: d^2f / dx_i dx_j
        for j in range(i + 1, n):
            x_pp = x0.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x0.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x0.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x0.copy(); x_mm[i] -= h; x_mm[j] -= h

            val = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
            hessian[i, j] = val
            hessian[j, i] = val # Hessian is symmetric

    return hessian