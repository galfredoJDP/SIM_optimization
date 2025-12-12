# Analysis: 1-bit ADMM Performance Gap Discrepancy

## Problem Statement
Our test_admm.py shows ~1.3-1.5dB loss between 1-bit ADMM and continuous Wiener filter at SNR=5-7dB, but the paper (Wen et al., 2023) reports ~3dB loss at the same SNR range.

## Test Results (SNR Sweep at M=64, K=8)
```
SNR (dB)     1-bit (bits/s/Hz)    Continuous (bits/s/Hz)    Loss (dB)
---------------------------------------------------------------------------
0            6.71                 8.13                      0.83
3            9.11                 12.02                     1.20
5            12.21                16.48                     1.30
7            14.11                19.69                     1.45
10           19.88                27.70                     1.44
```

**Expected from paper: ~3dB loss at SNR=5-7dB**
**Observed: ~1.3-1.5dB loss**

## Key Differences Between Our Test and Paper

### 1. OFDM vs Single-Carrier ⚠️ **MOST LIKELY CAUSE**
- **Paper**: Uses N=1024 OFDM subcarriers
  - Optimization runs independently on each subcarrier
  - Each subcarrier has different channel realization
  - Quantization happens per subcarrier
  - Sum-rate is averaged across all subcarriers
- **Our test**: Single-carrier system
  - Only one channel realization
  - Quantization happens once

**Impact**: With OFDM, quantization errors can accumulate across many subcarriers, potentially leading to larger overall loss. The paper's ~3dB gap is averaged over 1024 subcarriers with different channel conditions.

### 2. Algorithm Parameters
- **Lambda penalty**: We use λ=0.01
  - Paper states λ > √(2·Lϕ) for convergence but doesn't specify exact value used
  - Different λ could lead to different convergence behavior
- **Max iterations**: We use 100
  - Paper doesn't specify; they might use more for better convergence
- **Tolerance**: We use 1e-6
  - Paper doesn't specify

### 3. Channel Model
- **Our test**: Rayleigh fading with path loss
  - Uses beamformer.update_user_channel()
  - CLT mode with random user positions
- **Paper**: States "Rayleigh fading channel"
  - Exact model not fully detailed
  - Might include additional effects (correlation, Doppler, etc.)

### 4. SNR Definition and Control
- **Our approach**:
  - Set transmit power to achieve target SNR
  - Assume average channel gain of -60dB
  - Actual SNR varies per user and channel realization
- **Paper**:
  - SNR definition not explicitly stated
  - Likely averaged across users and subcarriers
  - Might be defined differently (per-user vs total)

### 5. Performance Metric
- **Both use**: Sum-rate in bits/s/Hz
- **Our calculation**:
  ```python
  SINR_k = signal_power / (interference + noise)
  sum_rate = Σ log2(1 + SINR_k)
  ```
- **Paper**:
  - Likely similar but averaged across OFDM subcarriers
  - Might include additional factors from OFDM system

### 6. Power Constraint Enforcement
- **Our ADMM**:
  - Normalizes at end: `gamma = P_total / ||R||²`
  - Power constraint satisfied post-optimization
- **Paper**:
  - Likely similar but per-subcarrier
  - Total power constraint might be distributed across subcarriers

## Algorithm Implementation Verification

The CG-MC1bit implementation in `simpy/algorithm.py` appears correct:
- ✅ X update: Solves (2H^H·H + λI) x = (2H^H·s + λR - V)
- ✅ R update: Quantizes to {±1±j} using projection
- ✅ V update: Dual variable ascent V += λ(X - R)
- ✅ α update: Per-user factors α_i = Re(R^H·h_i·s_i) / (||h_i·R||² + σ²)
- ✅ Power normalization at end

## SINR Calculation Verification

Our SINR calculation in test_admm.py:
```python
# For each user k:
signal_component = y_k * conj(s_k)
signal_power = |signal_component|²
interference = |y_k|² - signal_power
SINR_k = signal_power / (interference + noise_power)
```

This is correct for multiuser MIMO downlink. The α factors from ADMM are optimization variables for MSE minimization, not part of the actual channel model for sum-rate calculation.

## Hypotheses for Gap Discrepancy

### Hypothesis 1: OFDM Averaging Effect (Most Likely)
**Probability: HIGH**

With N=1024 subcarriers:
- Some subcarriers will have deep fades (very poor channels)
- 1-bit quantization causes larger relative loss on poor subcarriers
- Averaging across all subcarriers includes these high-loss cases
- Single-carrier test only samples one channel realization

**Test**: Implement OFDM with multiple subcarriers and average results.

### Hypothesis 2: SNR Definition Mismatch
**Probability: MEDIUM**

Our "target SNR" is approximate:
- Based on assumed average channel gain
- Actual SNR varies per channel realization
- Paper might define SNR differently (per-user average, post-beamforming, etc.)

**Test**: Compute actual average SNR from channel realizations and correlate with loss.

### Hypothesis 3: Algorithm Parameter Tuning
**Probability: LOW**

Different λ, iterations, or initialization might affect convergence:
- Paper might use different λ (we use 0.01)
- More iterations might give better convergence
- Different initialization might lead to different local optima

**Test**: Sweep λ parameter and max_iterations to see if gap changes.

### Hypothesis 4: Monte Carlo Sample Size
**Probability: LOW**

We use 20 trials; paper might use more:
- More trials → better averaging → might reveal larger gap
- But 20 trials should be sufficient for convergence of mean

**Test**: Increase to 100+ trials and check if gap grows.

## Recommended Next Steps

### Option A: Implement OFDM Test (Most thorough)
Add OFDM system with N subcarriers to test_admm.py:
```python
# Run ADMM optimization on each of N=1024 subcarriers
for n in range(N_subcarriers):
    H_n = generate_channel_for_subcarrier(n)
    result = admm.optimize(symbols, total_power_per_subcarrier)
    sum_rate_n = compute_sum_rate(H_n, result['antenna_signals'], ...)

# Average sum-rate across subcarriers
avg_sum_rate = mean(sum_rate_n for all n)
```

**Pros**: Most accurate comparison with paper
**Cons**: Significant implementation effort, computationally expensive

### Option B: Increase Monte Carlo Trials (Quick test)
Change num_trials from 20 to 100 or 200:
```python
num_trials = 100  # or 200
```

**Pros**: Easy to test, might reveal if we're undersampling
**Cons**: Unlikely to explain 2x gap (1.5dB vs 3dB)

### Option C: Tune Algorithm Parameters (Medium effort)
Test different λ values:
```python
for lambda_penalty in [0.001, 0.01, 0.1, 1.0]:
    admm = CG_MC1bit(beamformer, lambda_penalty=lambda_penalty, ...)
```

**Pros**: Might improve algorithm performance
**Cons**: Paper doesn't suggest this is the issue

### Option D: Verify SNR Definition (Diagnostic)
Compute actual experienced SNR from channel realizations:
```python
# After generating H:
avg_channel_gain = mean(||h_k||² for k in users)
actual_snr_db = 10*log10(total_power * avg_channel_gain / noise_power)
print(f"Target SNR: {target_snr_db}, Actual SNR: {actual_snr_db}")
```

**Pros**: Helps understand if SNR mismatch is causing discrepancy
**Cons**: Doesn't directly fix the gap

### Option E: Accept Current Results
If the goal is to verify the SIM improvement (not exactly match paper):
- Our test shows 1-bit ADMM has quantization loss (✓)
- SIM significantly reduces this loss from 17dB to 3.5dB (✓)
- Exact replication might require full OFDM implementation

**Pros**: Moves project forward
**Cons**: Doesn't fully explain discrepancy

## Conclusion

The most likely cause of the gap discrepancy is the **OFDM vs single-carrier** difference. The paper's ~3dB loss is averaged over 1024 subcarriers with diverse channel conditions, while our test uses a single carrier. Implementing OFDM would provide the most accurate comparison, but requires significant effort.

For the project's main goal (demonstrating SIM effectiveness), our current test successfully shows:
1. 1-bit quantization causes significant loss (~1.5dB in single-carrier)
2. M=K=4 without SIM shows much worse loss (~17dB)
3. SIM reduces loss to ~3.5dB by providing spatial degrees of freedom

The test validates the core insight even if the absolute numbers don't exactly match the ADMM paper.