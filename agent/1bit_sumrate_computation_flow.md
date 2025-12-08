# 1-Bit CG-MC1bit Sum-Rate Computation Flow

**Date:** 2025-12-05
**Reference:** One-Bit Downlink Precoding for Massive MIMO OFDM System (Wen et al., 2023)
**Purpose:** Implementation guide for computing sum-rate with 1-bit quantized CG-MC1bit precoding

---

## System Overview

```
Symbols s[k] → CG-MC1bit → x[k] (1-bit) → Channel → y[k] → SINR → Sum-Rate
              (Precoding)   (Quantized)    H[k]    (Received)  (Compute)  (Evaluate)
```

---

## Step-by-Step Flow

### **STEP 1: Generate Input Symbols**

```python
num_users = 4          # U
num_subcarriers = 1024 # N (OFDM)
num_antennas = 4       # Nt (M = K case from paper)

# Generate random constellation symbols (e.g., QPSK, 16QAM)
# Shape: (U, N) - for all users across all subcarriers
symbols = generate_qpsk_symbols(num_users, num_subcarriers)

# For each subcarrier k:
# s[k] = symbols[:, k]  # Shape: (U,)
```

**Input Requirements:**
- Constellation: QPSK, 16QAM, etc.
- One symbol per user per subcarrier
- Complex-valued

---

### **STEP 2: Obtain Channel Information**

```python
# Frequency domain channel for each subcarrier
# H[k] ∈ C^(U×Nt) for subcarrier k
# Shape overall: (U, Nt, N)

H_freq = torch.zeros(num_users, num_antennas, num_subcarriers, dtype=torch.complex64)

for k in range(num_subcarriers):
    H_freq[:, :, k] = get_frequency_domain_channel(k)
```

**Channel Properties:**
- Assumed constant within each OFDM symbol period (single coherence time)
- Known at transmitter (CSI available)
- Each subcarrier has independent channel realization

---

### **STEP 3: Run CG-MC1bit ADMM Optimization**

```python
x_precoded = CG_MC1bit(
    symbols=symbols,           # (U, N)
    H=H_freq,                  # (U, Nt, N)
    noise_power=noise_power,   # σ²
    total_power=total_power,   # P_T
    num_iterations=100,        # T from Algorithm 1
    lambda_penalty=0.01        # Must satisfy λ > √(2*Lϕ)
)

# Output: x_precoded (Nt, N)
# Each element: x[k]_m ∈ {-1-j, -1+j, 1-j, 1+j}
# All elements have magnitude √2 (constant envelope)
```

**Key Output Properties:**
- **Discrete values only:** Each output element from 1-bit set A
- **Constant envelope:** All outputs have same magnitude
- **Frequency domain:** Before IFFT to time domain
- **Per-subcarrier:** Shape (Nt, N), one (Nt,) vector per subcarrier

---

### **STEP 4: Simulate Channel Transmission**

```python
# Initialize received signal
y_received = torch.zeros(num_users, num_subcarriers, dtype=torch.complex64)

# For each subcarrier k:
for k in range(num_subcarriers):
    # Transmitted signal (1-bit from CG-MC1bit)
    x_k = x_precoded[:, k]  # (Nt,)

    # Channel matrix for this subcarrier
    H_k = H_freq[:, :, k]   # (U, Nt)

    # AWGN noise
    noise_k = torch.randn(num_users, dtype=torch.complex64)
    noise_k = noise_k * sqrt(noise_power / 2)  # Complex Gaussian CN(0, σ²)

    # Received signal equation: y[k] = H[k] @ x[k] + z[k]
    y_received[:, k] = H_k @ x_k + noise_k
    # Result shape: (U,)
```

**Received Signal Properties:**
- Observed at U users
- Includes inter-user interference
- Includes AWGN noise

---

### **STEP 5: Compute SINR (Critical Step)**

The SINR computation is **key** for sum-rate calculation. Multiple approaches:

#### **Approach A: Signal Power vs Interference + Noise**

```python
sinr_per_user_subcarrier = torch.zeros(num_users, num_subcarriers)

for k in range(num_subcarriers):
    H_k = H_freq[:, :, k]  # (U, Nt)
    x_k = x_precoded[:, k]  # (Nt,)

    # Received signal at each user (no noise for SINR calculation)
    y_k = H_k @ x_k  # (U,)

    for u in range(num_users):
        # Decompose y_k into signal and interference
        # This assumes x[k] was optimized to serve user u as primary

        # Signal: Power directed to user u
        signal_power = torch.abs(y_k[u])**2 / num_users

        # Interference: Power leaking to user u from other users' signals
        interference_power = 0
        for j in range(num_users):
            if j != u:
                interference_power += torch.abs(y_k[j])**2 / num_users

        # SINR
        sinr_per_user_subcarrier[u, k] = signal_power / (
            interference_power + noise_power
        )
```

**Note:** This is a heuristic - CG-MC1bit doesn't have explicit per-user decomposition like linear precoding.

#### **Approach B: Effective SNR (Simpler)**

```python
for k in range(num_subcarriers):
    H_k = H_freq[:, :, k]
    x_k = x_precoded[:, k]

    received_power = torch.abs(H_k @ x_k)**2  # (U,)

    for u in range(num_users):
        # Simplified: treat received power as signal
        sinr_per_user_subcarrier[u, k] = received_power[u] / noise_power
```

#### **Approach C: Capacity Lower Bound (From Paper)**

```python
# Use MSE-based capacity approximation
# C ≈ log₂(1 + P_signal / P_noise)
# where P_signal is average received power

for k in range(num_subcarriers):
    H_k = H_freq[:, :, k]
    x_k = x_precoded[:, k]

    for u in range(num_users):
        # Channel from antennas to user u
        h_u = H_k[u, :]  # (Nt,)

        # Received signal
        received = h_u @ x_k
        received_power = torch.abs(received)**2

        # Signal-to-noise ratio
        snr = received_power / noise_power
        sinr_per_user_subcarrier[u, k] = snr
```

**Recommendation:** Start with Approach B or C for simplicity; refine later if needed.

---

### **STEP 6: Compute Sum-Rate**

```python
# Sum-rate = Σ_{k,u} log₂(1 + SINR_{u,k})
sum_rate = torch.sum(torch.log2(1 + sinr_per_user_subcarrier))

print(f"Sum-Rate: {sum_rate.item():.4f} bits/s/Hz")

# Optional: Break down by user
sum_rate_per_user = torch.sum(torch.log2(1 + sinr_per_user_subcarrier), dim=1)
print(f"Per-user rates: {sum_rate_per_user}")

# Optional: Break down by subcarrier
sum_rate_per_subcarrier = torch.sum(torch.log2(1 + sinr_per_user_subcarrier), dim=0)
print(f"Mean rate per subcarrier: {torch.mean(sum_rate_per_subcarrier).item():.4f}")
```

---

## Complete Implementation Template

```python
def compute_sum_rate_1bit(symbols, H_freq, x_precoded, noise_power):
    """
    Compute sum-rate for 1-bit CG-MC1bit precoding.

    Args:
        symbols: (U, N) - input symbols
        H_freq: (U, Nt, N) - channel per subcarrier
        x_precoded: (Nt, N) - 1-bit precoded signals from CG-MC1bit
        noise_power: σ² - AWGN noise power

    Returns:
        sum_rate: Total sum-rate in bits/s/Hz
        sinr_matrix: (U, N) - SINR per user per subcarrier
    """
    U, Nt, N = H_freq.shape
    sinr_matrix = torch.zeros(U, N)

    # For each subcarrier
    for k in range(N):
        H_k = H_freq[:, :, k]      # (U, Nt)
        x_k = x_precoded[:, k]     # (Nt,)

        # Received signal at each user (no noise in SINR calc)
        y_k = H_k @ x_k            # (U,)

        # SINR for each user
        for u in range(U):
            received_power = torch.abs(y_k[u])**2
            sinr_matrix[u, k] = received_power / noise_power

    # Sum-rate
    sum_rate = torch.sum(torch.log2(1 + sinr_matrix))

    return sum_rate, sinr_matrix
```

---

## Comparison: Continuous vs 1-Bit

| Aspect | Continuous Precoding | 1-Bit CG-MC1bit |
|--------|----------------------|-----------------|
| **Precoding** | W = H^H(HH^H)^{-1} | ADMM optimization |
| **Output Constraint** | Continuous ℂ | Discrete {-1-j, -1+j, 1-j, 1+j} |
| **Sum-Rate Curve** | Smooth monotonic | Possibly non-monotonic |
| **Performance Gap** | ~3-4 dB vs 1-bit | Baseline (1-bit aware) |
| **Computation** | Analytical | Iterative (100+ iterations) |

---

## Key Implementation Notes

1. **CSI Assumption:** Channel is known and constant during OFDM symbol
2. **Noise Model:** AWGN, CN(0, σ²) at each user per subcarrier
3. **Power Constraint:** Total transmit power = P_T across all antennas, all subcarriers
4. **Quantization:** 1-bit means 2 bits per complex number (1 for I, 1 for Q)
5. **SINR Definition:** Depends on how precoding decomposes signal vs interference
   - With linear precoding: Clear diagonal/off-diagonal decomposition
   - With nonlinear precoding: Heuristic or empirical measurement recommended

---

## Reference Equations from Paper

**Received Signal (Eq. 4):**
```
y[k] = H[k]x[k] + z[k]
```

**MSE Minimization (Eq. 8):**
```
MSE = Σ_k ||s[k] - A·H[k]x[k]||²₂ + Nσ²Σ α_i²
```

**1-Bit Quantization (Eq. 21):**
```
P_A(w_i) = sign(ℜ(w_i)) + j·sign(ℑ(w_i))
```

**Sum-Rate:**
```
R = Σ_k Σ_u log₂(1 + SINR_{k,u})
```

---

## Next Steps for Implementation

- [ ] Implement CG-MC1bit algorithm (Algorithm 1 from paper)
- [ ] Test with simple 2-user, 2-antenna case
- [ ] Validate SINR computation against paper's SER results
- [ ] Generate sum-rate vs power plots (comparable to paper Figure 4)
- [ ] Compare: (WF, ZF) vs (WF, PGA) vs (WF, CG-MC1bit)

---

**Last Updated:** 2025-12-05
**Status:** Ready for implementation