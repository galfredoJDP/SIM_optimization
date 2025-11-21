# Technical Details & Formulas

## Channel Model Mathematics

### Rician Fading Formula
Used in both CLT and geometric modes:
```
H[k,:] = sqrt(K/(K+1)) * H_LOS + sqrt(1/(K+1)) * H_scattered
```
- K: Rician K-factor (linear, converted from dB: K_linear = 10^(K_dB/10))
- H_LOS: Line-of-sight component
- H_scattered: Rayleigh fading component

**Power interpretation** (from Tse 2005 MIMO book, eq 2.54):
- LOS power fraction: K/(K+1)
- Scattered power fraction: 1/(K+1)
- Total power normalized

### Path Loss Model

**In dB**:
```
PL(d) = PL(d0) + 10*α*log10(d/d0) + X_σ
```
- PL(d0): Reference path loss (path_loss_at_reference)
- α: Path loss exponent (per-user: path_loss_exponents[k])
- d0: Reference distance (reference_distance)
- X_σ: Log-normal shadowing ~ N(0, path_loss_variance²)

**In linear** (for channel coefficients):
```
path_loss = sqrt(PL_ref_linear * distance_factor * shadowing_linear)
```
- PL_ref_linear = 10^(path_loss_at_reference/10)  # Power
- distance_factor = (d/d0)^(-α)
- shadowing_linear = 10^(shadowing_dB/10)  # Power

**Important**: Use /10 for power, /20 for amplitude

### Complex Gaussian Generation
For Rayleigh fading (scattered component):
```python
H_scattered = (randn(A) + 1j*randn(A)) / sqrt(2) * path_loss
```
- Each component (real, imag) has variance σ²/2
- Combined: CN(0, σ²) where σ = path_loss
- Division by sqrt(2) ensures proper power scaling

### Rayleigh-Sommerfeld Diffraction
Used for near-field propagation (antenna ↔ SIM, SIM ↔ users):
```
H[n,k] = (dx*dy * cos(ξ) / d) * (1/(2πd) - j/λ) * exp(j*2π*d/λ)
```
- dx, dy: Meta-atom dimensions
- ξ: Angle between propagation and normal direction
- d: Distance between elements
- λ: Wavelength

## Signal Model

### Received Signal (per user k)
```
y_k = h_k^H * Ψ * Σ(w_k' * sqrt(p_k') * s_k') + n_k
```
- h_k: SIM last layer → user k channel
- Ψ: SIM configuration matrix (phases)
- w_k': Antenna → SIM first layer
- p_k': Transmit power to user k'
- s_k': Information symbol (unit variance)
- n_k: AWGN noise (variance σ²)

### SINR (Paper's M=K Architecture)
From paper Equation 8:
```
SINR_k = p_k * |h_k^H * Ψ * w_k^[1]|² / (Σ_{k'≠k}(p_k' * |h_k^H * Ψ * w_k'^[1]|²) + σ²)
```
- w_k^[1]: Channel from antenna k to SIM first layer (Rayleigh-Sommerfeld)
- Signal: antenna k → Ψ → user k (k-th antenna dedicated to user k)
- Interference: antennas k'≠k → Ψ → user k (other antennas interfere)

### SINR Computation in Code
With M=K (one antenna per user), effective channel H_eff is **(K, K)**:
- H_eff[k, k]: Signal path from antenna k to user k
- H_eff[k, j] for j≠k: Interference path from antenna j to user k

Implementation in `wireless/transceiver.py:180-221`:
```python
# H_eff is (K, K) after SIM or digital beamforming
for k in range(K):
    signal_power = p_k * |H_eff[k, k]|²        # Diagonal: desired signal
    interference = Σ_{j≠k} p_j * |H_eff[k, j]|²  # Off-diagonal: interference
    SINR_k = signal_power / (interference + σ²)
```

**Important:** This assumes M=K architecture. For M≠K without digital beamforming weights, a different SINR model is needed.

### Sum-Rate
```
R = Σ log₂(1 + SINR_k)
```
Units: bits/s/Hz

## SIM Configuration Matrix

### Structure
```
Ψ = Θ[L] * W[L] * Θ[L-1] * ... * Θ[2] * W[2] * Θ[1]
```
- Θ[ℓ]: Diagonal phase matrix for layer ℓ
- W[ℓ]: Transmission matrix (ℓ-1) → ℓ (Rayleigh-Sommerfeld)
- L: Number of layers

### Phase Matrix
```
Θ[ℓ] = diag(exp(j*θ[ℓ]))
```
- θ[ℓ]: Phase vector for layer ℓ
- Elements from quantized set Q_b = {0, 2π/2^b, 4π/2^b, ..., 2π(2^b-1)/2^b}
- b: Number of bits (typically 2)

## Optimization Problem

### Original (from paper)
```
P1: max_{p,θ} Σ log₂(1 + SINR_k)
    s.t. Σ p_k ≤ P_T
         p_k ∈ P_U  (discrete power set)
         θ_n^[ℓ] ∈ Q_b  (discrete phase set)
```

### Alternative Optimization (AO)
Iterate:
1. Fix p, optimize θ using Filled Function (FF) method
2. Fix θ, optimize p using modified FF (mFF) method
3. Repeat until convergence (I_AO iterations)

## Numerical Stability

### Complex Number Handling
- Use `torch.complex64` or `np.complex64`
- Phase wrapping: automatic with exp(j*θ)
- Magnitude extraction: `torch.abs()` or `np.abs()`

### Power Conversions
- **dB to linear (power)**: 10^(x/10)
- **dB to linear (amplitude)**: 10^(x/20)
- **Linear to dB (power)**: 10*log10(x)
- **Linear to dB (amplitude)**: 20*log10(x)

### Typical Values
- path_loss_at_reference: -30 dB (at 1m)
- noise_power: -80 dBm
- Rician K-factor: 0 dB (Rayleigh) to 10+ dB (strong LOS)
- path_loss_variance: 0-8 dB (shadowing)

## Device Handling

### Pattern
```python
# In UserChannel/Transceiver
if isinstance(input, torch.Tensor):
    input = input.cpu().numpy()
# ... numpy operations ...
output_torch = torch.tensor(output, dtype=torch.complex64, device=self.device)
return output_torch
```

Always return torch tensors on correct device from UserChannel methods.

## Key Assumptions

1. **Single coherence time**: Channel constant during optimization
2. **Perfect CSI**: Channel known at transmitter
3. **Narrowband**: Single frequency operation
4. **Far-field approximation** (when not using Rayleigh-Sommerfeld): d >> wavelength
5. **CLT validity**: Large number of scatterers, random phases

## References to Paper Sections

- Channel model: Section II-A
- SIM configuration: Section II-A (equations 2, 5-7)
- SINR: Equation 8
- Problem formulation: Section II-B (P1, P2)
- Rayleigh-Sommerfeld: Equation 2, 6
- Rician fading power: Tse 2005 MIMO book, eq 2.54