"""
Wiener Filter (WF) Precoding
Borrowed from https://github.com/quantizedmassivemimo/1bit_precoding
"""

import numpy as np
import torch

def _wf_numpy(s, H, N0):
    """
    Wiener Filter precoding using NumPy
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    # number of UEs
    U = H.shape[0]
    
    # precoding matrix (before normalization)
    T = H.conj().T @ np.linalg.inv(H @ H.conj().T + U * N0 * np.eye(U))
    
    # precoding factor
    beta = np.sqrt(np.real(np.trace(T @ T.conj().T)))
    
    # precoding matrix
    P = (1 / beta) * T
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def _wf_torch(s, H, N0):
    """
    Wiener Filter precoding using PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    
    # number of UEs
    U = H.shape[0]
    
    # precoding matrix (before normalization)
    T = H.conj().T @ torch.linalg.inv(H @ H.conj().T + U * N0 * torch.eye(U, device=H.device, dtype=H.dtype))
    
    # precoding factor
    beta = torch.sqrt(torch.real(torch.trace(T @ T.conj().T)))
    
    # precoding matrix
    P = (1 / beta) * T
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def wiener(s, H, N0):
    """
    Wiener Filter precoding - automatically detects NumPy or PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    # Check if inputs are PyTorch tensors
    try:
        if isinstance(H, torch.Tensor):
            return _wf_torch(s, H, N0)
    except ImportError:
        pass
    
    # Default to NumPy
    return _wf_numpy(s, H, N0)

"""
Zero-Forcing (ZF) Precoding
Borrowed from https://github.com/quantizedmassivemimo/1bit_precoding
"""

def _zf_numpy(s, H):
    """
    Zero-Forcing precoding using NumPy
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    # precoding factor
    beta = np.sqrt(np.trace(np.linalg.inv(H @ H.conj().T)))
    
    # precoding matrix
    P = (1 / beta) * H.conj().T @ np.linalg.inv(H @ H.conj().T)
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def _zf_torch(s, H):
    """
    Zero-Forcing precoding using PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    
    # precoding factor
    beta = torch.sqrt(torch.trace(torch.linalg.inv(H @ H.conj().T)))
    
    # precoding matrix
    P = (1 / beta) * H.conj().T @ torch.linalg.inv(H @ H.conj().T)
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def zeroForcing(s, H):
    """
    Zero-Forcing precoding - automatically detects NumPy or PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
    
    Returns:
        x: precoded vector (N,) or (N, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (N, U)
    """
    # Check if inputs are PyTorch tensors
    try:
        if isinstance(H, torch.Tensor):
            return _zf_torch(s, H)
    except ImportError:
        pass
    
    # Default to NumPy
    return _zf_numpy(s, H)

"""
Maximum Ratio Transmission (MRT) Precoding
Borrowed from https://github.com/quantizedmassivemimo/1bit_precoding
"""

def _mrt_numpy(s, H):
    """
    Maximum Ratio Transmission precoding using NumPy
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, B) where B is number of BS antennas
    
    Returns:
        x: precoded vector (B,) or (B, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (B, U)
    """
    # number of BS antennas
    B = H.shape[1]
    
    # precoding factor
    beta = np.sqrt(np.trace(H.conj().T @ H)) / B
    
    # precoding matrix
    P = (1 / B / beta) * H.conj().T
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def _mrt_torch(s, H):
    """
    Maximum Ratio Transmission precoding using PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, B) where B is number of BS antennas
    
    Returns:
        x: precoded vector (B,) or (B, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (B, U)
    """
    
    # number of BS antennas
    B = H.shape[1]
    
    # precoding factor
    beta = torch.sqrt(torch.trace(H.conj().T @ H)) / B
    
    # precoding matrix
    P = (1 / B / beta) * H.conj().T
    
    # precoded vector
    x = P @ s
    
    return x, beta, P

def mrt(s, H):
    """
    Maximum Ratio Transmission precoding - automatically detects NumPy or PyTorch
    
    Args:
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, B) where B is number of BS antennas
    
    Returns:
        x: precoded vector (B,) or (B, 1)
        beta: precoding factor (scalar)
        P: precoding matrix (B, U)
    """
    # Check if inputs are PyTorch tensors
    try:
        if isinstance(H, torch.Tensor):
            return _mrt_torch(s, H)
    except ImportError:
        pass
    
    # Default to NumPy
    return _mrt_numpy(s, H)

"""
SQUID: Sum-of-Quadratic Iterative Algorithm for 1-bit Precoding
"""

def _prox_infinity_norm2_numpy(w, lmbda):
    """
    Proximal mapping of the infinity-norm-squared.
    Perform prox operator: min lambda*||x||_inf^2 + ||x-w||^2
    
    Args:
        w: input vector
        lmbda: lambda parameter
    
    Returns:
        xk: output of proximal operator
    """
    N = len(w)
    wabs = np.abs(w)
    ws = np.cumsum(np.sort(wabs)[::-1]) / (lmbda + np.arange(1, N + 1))
    alphaopt = np.max(ws)
    
    if alphaopt > 0:
        xk = np.minimum(wabs, alphaopt) * np.sign(w)  # truncation step
    else:
        xk = np.zeros_like(w)  # if lambda is big, then solution is zero
    
    return xk

def _prox_infinity_norm2_torch(w, lmbda):
    """
    Proximal mapping of the infinity-norm-squared (PyTorch version).
    
    Args:
        w: input tensor
        lmbda: lambda parameter
    
    Returns:
        xk: output of proximal operator
    """
    
    N = len(w)
    wabs = torch.abs(w)
    ws = torch.cumsum(torch.sort(wabs, descending=True)[0], dim=0) / (lmbda + torch.arange(1, N + 1, device=w.device, dtype=w.dtype))
    alphaopt = torch.max(ws)
    
    if alphaopt > 0:
        xk = torch.minimum(wabs, alphaopt) * torch.sign(w)  # truncation step
    else:
        xk = torch.zeros_like(w)  # if lambda is big, then solution is zero
    
    return xk

def _squid_numpy(par, s, H, N0):
    """
    SQUID algorithm for 1-bit precoding using NumPy
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (only 1-bit supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - 1-bit quantized
        beta: output gain (scalar)
    """
    N = par['N']
    U = par['U']
    b = par.get('b', 1)
    
    if b != 1:
        raise ValueError('SQUID: only 1-bit DACs supported!')
    
    # convert to real-valued channel
    HR = np.vstack([
        np.hstack([np.real(H), -np.imag(H)]),
        np.hstack([np.imag(H), np.real(H)])
    ])
    sR = np.concatenate([np.real(s), np.imag(s)])
    
    # initialize
    x = np.zeros(N * 2)
    y = np.zeros(N * 2)
    
    # set gain based on problem size
    gain = 1.0 if N > 16 else 0.05
    epsilon = 1e-5
    
    # pre-processing
    A = HR.T @ HR + 0.5 / gain * np.eye(N * 2)
    sREG = np.linalg.solve(A, HR.T @ sR)
    
    # SQUID loop
    for t in range(100):
        u = sREG + 0.5 / gain * np.linalg.solve(A, 2 * x - y)
        xold = x.copy()
        x = _prox_infinity_norm2_numpy(y + u - x, 2 * 2 * U * N * N0)
        
        if np.linalg.norm(x - xold) / np.linalg.norm(x) < epsilon:
            break
        
        y = y + u - x
    
    # extract binary solution
    xRest = np.sign(x)
    x_complex = (1 / np.sqrt(2 * N)) * (xRest[:N] + 1j * xRest[N:])
    
    # compute output gains
    beta = np.real(x_complex.conj() @ H.conj().T @ s) / (np.linalg.norm(H @ x_complex) ** 2 + U * N0)
    
    # check (and fix) if beta is negative
    if beta < 0:
        x_complex = -x_complex
        beta = -beta
    
    return x_complex, beta

def _squid_torch(par, s, H, N0):
    """
    SQUID algorithm for 1-bit precoding using PyTorch
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (only 1-bit supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - 1-bit quantized
        beta: output gain (scalar)
    """
    
    N = par['N']
    U = par['U']
    b = par.get('b', 1)
    
    if b != 1:
        raise ValueError('SQUID: only 1-bit DACs supported!')
    
    # convert to real-valued channel
    HR = torch.cat([
        torch.cat([H.real, -H.imag], dim=1),
        torch.cat([H.imag, H.real], dim=1)
    ], dim=0)
    sR = torch.cat([s.real, s.imag])
    
    # initialize
    x = torch.zeros(N * 2, device=H.device, dtype=H.real.dtype)
    y = torch.zeros(N * 2, device=H.device, dtype=H.real.dtype)
    
    # set gain based on problem size
    gain = 1.0 if N > 16 else 0.05
    epsilon = 1e-5
    
    # pre-processing
    A = HR.T @ HR + 0.5 / gain * torch.eye(N * 2, device=H.device, dtype=H.real.dtype)
    sREG = torch.linalg.solve(A, HR.T @ sR)
    
    # SQUID loop
    for t in range(100):
        u = sREG + 0.5 / gain * torch.linalg.solve(A, 2 * x - y)
        xold = x.clone()
        x = _prox_infinity_norm2_torch(y + u - x, 2 * 2 * U * N * N0)
        
        if torch.linalg.norm(x - xold) / torch.linalg.norm(x) < epsilon:
            break
        
        y = y + u - x
    
    # extract binary solution
    xRest = torch.sign(x)
    x_complex = (1 / torch.sqrt(torch.tensor(2 * N, device=H.device, dtype=H.real.dtype))) * (xRest[:N] + 1j * xRest[N:])
    
    # compute output gains
    beta = torch.real(x_complex.conj() @ H.conj().T @ s) / (torch.linalg.norm(H @ x_complex) ** 2 + U * N0)
    
    # check (and fix) if beta is negative
    if beta < 0:
        x_complex = -x_complex
        beta = -beta
    
    return x_complex, beta

def squid(par, s, H, N0):
    """
    SQUID algorithm for 1-bit precoding - automatically detects NumPy or PyTorch
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (only 1-bit supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - 1-bit quantized
        beta: output gain (scalar)
    """
    # Check if inputs are PyTorch tensors
    try:
        if isinstance(H, torch.Tensor):
            return _squid_torch(par, s, H, N0)
    except ImportError:
        pass
    
    # Default to NumPy
    return _squid_numpy(par, s, H, N0)

"""
ADMM-based precoding for quantized massive MIMO
Developed by Lei Chu and Fei Wen
"""

def _admm_leo_numpy(par, s, H, N0):
    """
    ADMM algorithm for multi-bit DAC precoding using NumPy
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (1, 2, or 3 bits supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - quantized
        beta: output gain (scalar)
        zr: convergence history for z
        vr: convergence history for v
    """
    N = par['N']
    U = par['U']
    b = par['b']
    
    # Convert to real-valued channel
    HR0 = np.vstack([
        np.hstack([np.real(H), -np.imag(H)]),
        np.hstack([np.imag(H), np.real(H)])
    ])
    sR = np.concatenate([np.real(s), np.imag(s)])
    
    # Create quantization matrix C based on bit depth
    if b == 1:
        C = np.eye(2 * N)
        bps = 2
    elif b == 2:
        Q1 = np.sqrt(5)
        C = np.hstack([2 * np.eye(2 * N), np.eye(2 * N)]) / Q1
        bps = 4
    elif b == 3:
        Q1 = np.sqrt(21)
        C = np.hstack([4 * np.eye(2 * N), 2 * np.eye(2 * N), np.eye(2 * N)]) / Q1
        bps = 6
    else:
        raise ValueError('Not supported at current version!!!')
    
    HR = HR0 @ C
    
    # Initialize variables
    v = np.zeros(N * bps)
    z = np.zeros(N * bps)
    w = np.zeros(N * bps)
    
    # Algorithm parameters
    max_iter = 500
    c = U * N0
    epsilon = 1e-6
    vr = []
    zr = []
    rho_0 = 1.0
    rho = rho_0
    rho_t = 2.1 * np.linalg.norm(HR.T @ HR, 2)
    
    # Pre-compute SVD
    uu, ss, _ = np.linalg.svd(HR.T @ HR, full_matrices=False)
    ss = np.diag(ss)
    Hs = HR.T @ sR
    
    # ADMM iterations
    for k in range(max_iter):
        vm1 = v.copy()
        zm1 = z.copy()
        
        # Update rho
        if rho < rho_t:
            rho = rho_0 * (1.15 ** (k + 1))
        
        # Update v
        gg = 2 * np.diag(ss) + (2 * c + rho)
        v = uu @ ((uu.T @ (2 * Hs + rho * z + w)) / gg)
        
        # Update z (soft thresholding)
        v_minus_w_rho = v - w / rho
        z = np.sign(v_minus_w_rho) * np.linalg.norm(v_minus_w_rho, 1) / (N * bps)
        
        # Update w (dual variable)
        w = w - rho * (v - z)
        
        # Track convergence
        vr.append(np.linalg.norm(v - vm1) / np.linalg.norm(v))
        zr.append(np.linalg.norm(z - zm1) / np.linalg.norm(z))
        
        # Check convergence (commented out for convergence analysis)
        if np.linalg.norm(v - vm1) < epsilon:
            break
    
    # Extract quantized solution
    xRest = np.sign(v)
    x0 = C @ xRest
    x = (1 / np.sqrt(2 * N)) * (x0[:N] + 1j * x0[N:])
    
    # Compute output gain
    beta = np.real(x.conj() @ H.conj().T @ s) / (np.linalg.norm(H @ x) ** 2 + U * N0)
    
    return x, beta, zr, vr


def _admm_leo_torch(par, s, H, N0):
    """
    ADMM algorithm for multi-bit DAC precoding using PyTorch
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (1, 2, or 3 bits supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - quantized
        beta: output gain (scalar)
        zr: convergence history for z
        vr: convergence history for v
    """
    
    N = par['N']
    U = par['U']
    b = par['b']
    
    device = H.device
    dtype = H.real.dtype
    
    # Convert to real-valued channel
    HR0 = torch.cat([
        torch.cat([H.real, -H.imag], dim=1),
        torch.cat([H.imag, H.real], dim=1)
    ], dim=0)
    sR = torch.cat([s.real, s.imag])
    
    # Create quantization matrix C based on bit depth
    if b == 1:
        C = torch.eye(2 * N, device=device, dtype=dtype)
        bps = 2
    elif b == 2:
        Q1 = torch.sqrt(torch.tensor(5.0, device=device, dtype=dtype))
        C = torch.cat([2 * torch.eye(2 * N, device=device, dtype=dtype), 
                       torch.eye(2 * N, device=device, dtype=dtype)], dim=1) / Q1
        bps = 4
    elif b == 3:
        Q1 = torch.sqrt(torch.tensor(21.0, device=device, dtype=dtype))
        C = torch.cat([4 * torch.eye(2 * N, device=device, dtype=dtype), 
                       2 * torch.eye(2 * N, device=device, dtype=dtype),
                       torch.eye(2 * N, device=device, dtype=dtype)], dim=1) / Q1
        bps = 6
    else:
        raise ValueError('Not supported at current version!!!')
    
    HR = HR0 @ C
    
    # Initialize variables
    v = torch.zeros(N * bps, device=device, dtype=dtype)
    z = torch.zeros(N * bps, device=device, dtype=dtype)
    w = torch.zeros(N * bps, device=device, dtype=dtype)
    
    # Algorithm parameters
    max_iter = 500
    c = U * N0
    epsilon = 1e-6
    vr = []
    zr = []
    rho_0 = 1.0
    rho = rho_0
    rho_t = 2.1 * torch.linalg.norm(HR.T @ HR, 2)
    
    # Pre-compute SVD
    uu, ss, _ = torch.linalg.svd(HR.T @ HR, full_matrices=False)
    ss = torch.diag(ss)
    Hs = HR.T @ sR
    
    # ADMM iterations
    for k in range(max_iter):
        vm1 = v.clone()
        zm1 = z.clone()
        
        # Update rho
        if rho < rho_t:
            rho = rho_0 * (1.15 ** (k + 1))
        
        # Update v
        gg = 2 * torch.diag(ss) + (2 * c + rho)
        v = uu @ ((uu.T @ (2 * Hs + rho * z + w)) / gg)
        
        # Update z (soft thresholding)
        v_minus_w_rho = v - w / rho
        z = torch.sign(v_minus_w_rho) * torch.linalg.norm(v_minus_w_rho, 1) / (N * bps)
        
        # Update w (dual variable)
        w = w - rho * (v - z)
        
        # Track convergence
        vr.append((torch.linalg.norm(v - vm1) / torch.linalg.norm(v)).item())
        zr.append((torch.linalg.norm(z - zm1) / torch.linalg.norm(z)).item())
        
        # Check convergence (commented out for convergence analysis)
        if torch.linalg.norm(v - vm1) < epsilon:
            break
    
    # Extract quantized solution
    xRest = torch.sign(v)
    x0 = C @ xRest
    x = (1 / torch.sqrt(torch.tensor(2 * N, device=device, dtype=dtype))) * (x0[:N] + 1j * x0[N:])
    
    # Compute output gain
    beta = torch.real(x.conj() @ H.conj().T @ s) / (torch.linalg.norm(H @ x) ** 2 + U * N0)
    
    return x, beta, zr, vr


def admm_leo(par, s, H, N0):
    """
    ADMM algorithm for multi-bit DAC precoding - automatically detects NumPy or PyTorch
    
    Args:
        par: parameters dict with keys:
            - N: number of BS antennas
            - U: number of users
            - b: number of DAC bits (1, 2, or 3 bits supported)
        s: symbol vector (U,) or (U, 1)
        H: channel matrix (U, N)
        N0: noise variance
    
    Returns:
        x: precoded vector (N,) - quantized
        beta: output gain (scalar)
        zr: convergence history for z
        vr: convergence history for v
    """
    # Check if inputs are PyTorch tensors
    try:
        if isinstance(H, torch.Tensor):
            return _admm_leo_torch(par, s, H, N0)
    except ImportError:
        pass
    
    # Default to NumPy
    return _admm_leo_numpy(par, s, H, N0)