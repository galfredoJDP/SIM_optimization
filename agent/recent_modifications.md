# Recent Modifications Log

## 2025-11-20 (Part 5): SINR Computation and Beamforming Architecture Verification

### Problem
User questioned whether `compute_sinr_downlink` correctly handles three scenarios:
1. Digital beamforming only (phases=None, digital_beamforming_weights provided)
2. SIM only (phases provided, M antennas, no digital weights)
3. Hybrid (phases + digital weights)

Initial concern: Matrix dimension mismatches, especially with M=1 antenna + SIM.

### Paper's Architecture (Verified from ICC 2025 paper)

**Key Finding:** Paper uses **M = K** (one antenna per user), not M = 1!

From paper page 2:
> "Therefore, in this work, we focus on M = K and aim to maximize sum-rate solely through power allocation and SIM element optimization."

**Paper's System:**
- K antennas, one dedicated to each user (not traditional beamforming!)
- No digital beamforming - only SIM phases + power allocation
- Each antenna k transmits to user k through the SIM
- SINR formula (Equation 8): Signal from antenna k to user k (diagonal), interference from antennas k'≠k (off-diagonal)

**Matrix Dimensions for M = K:**
- A: (N, K) - K antennas → N meta-atoms at layer 0
- Psi: (N, N) - SIM propagation
- H: (K, N) - N meta-atoms at layer (L-1) → K users
- H_eff = H @ Psi @ A = **(K, K)** naturally

### Current Implementation Status

**✅ WORKS CORRECTLY:**

1. **M=K, no digital weights** (paper's approach):
   - H_eff is (K, K)
   - `effective_channel = channel` (line 198-201)
   - SINR computation correct: diagonal = signal, off-diagonal = interference
   - Matches paper's Equation 8 exactly

2. **Any M, with digital weights**:
   - H_eff is (K, M)
   - weights is (M, K)
   - `effective_channel = channel @ weights` = (K, K)
   - SINR computation works correctly

**❌ WOULD FAIL (not currently used):**

3. **M≠K, no digital weights**:
   - M=1: H_eff is (K, 1), accessing `effective_channel[k,k]` fails for k>0
   - M>K: H_eff is (K, M), wrong SINR formula (off-diagonals don't represent interference)

### Beamforming Architectures Explained

**Traditional Digital Beamforming:**
- All M antennas work together to serve each user
- Beamforming weights W (M, K) where column k serves user k
- All antennas transmit to all users with different phase/amplitude combinations
- Provides spatial multiplexing and array gain

**Paper's SIM Approach (M=K):**
- Each antenna "assigned" to one user (simplified architecture)
- SIM does beamforming in EM domain instead of digital domain
- Avoids: high-resolution DACs, complex DSP, multiple RF chains
- Trade-off: Simpler hardware, but not traditional beamforming

**Can 1 Antenna + SIM Serve Multiple Users?**
- YES - SIM can create spatial beams from single input (acts like programmable lens)
- BUT: Limited degrees of freedom (1 DOF vs K DOF)
- Cannot do zero-forcing or full interference cancellation
- Much worse performance than M=K

### Code Location
- `wireless/transceiver.py:180-221` - `compute_sinr_downlink()` method
- `Beamformer.py:151-175` - `compute_sinr()` method
- `Beamformer.py:177-192` - `compute_sum_rate()` method

### Conclusions

1. **Current implementation is CORRECT for paper's M=K case** - no changes needed
2. Paper's M=K architecture is NOT traditional beamforming (one antenna per user)
3. To support M=1 or M≠K without weights would require:
   - Different SINR model (broadcast or modified interference calculation)
   - Handling (K, M) effective channels differently
   - Not needed for paper replication

### User Preferences
- Focus on paper's M=K approach (verified correct)
- Understanding different beamforming architectures documented
- No implementation changes needed at this time

## 2025-11-20 (Part 4): Critical Directionality Fix and Bug Fixes

### Problem 1: Incorrect SIM Channel Direction (CRITICAL)
The code was using `Psi^H` (conjugate transpose) for downlink when Psi already represented the downlink direction. This effectively computed the **uplink** channel instead of downlink, which is incorrect for the paper's downlink sum-rate maximization problem.

### Root Cause Analysis
- **Psi construction**: Correctly builds Layer 0 → Layer (L-1) propagation (BS side → User side) = **downlink direction**
- **Bug**: Code applied Hermitian transpose, reversing direction to uplink
- **Impact**: Computed wrong channel for optimization, leading to incorrect SINR and sum-rate

### Solution - Directionality Fixes
**1. Fixed `Beamformer.py:138` (compute_end_to_end_channel)**
```python
# Before (WRONG - uplink channel):
H_eff = self.H @ torch.conj(Psi).T @ self.A

# After (CORRECT - downlink channel):
H_eff = self.H @ Psi @ self.A
```

**2. Fixed `wireless/sim.py:248` (downlink method)**
```python
# Before (WRONG):
output_field = torch.conj(self.Psi).T @ input_field

# After (CORRECT):
output_field = self.Psi @ input_field
```

**3. Fixed `wireless/sim.py:234` (uplink method)**
```python
# Before (WRONG - was using downlink):
output_field = self.Psi @ input_field

# After (CORRECT - uses reciprocity):
output_field = torch.conj(self.Psi).T @ input_field
```

**4. Added direction parameter to compute_end_to_end_channel()**
- Default: `direction='down'` for downlink
- Optional: `direction='up'` applies Hermitian for uplink using reciprocity

### Verification
- Signal flow: Antennas → A → Layer 0 → Psi → Layer (L-1) → H → Users
- Downlink: `y = H @ Psi @ A @ s` ✓
- Uplink (reciprocity): `y = A^H @ Psi^H @ H^H @ s` ✓
- Dimensions: H_eff = (K, M) for downlink, (M, K) for uplink ✓

### Problem 2: dtype Mismatch in Zero-Forcing Weights

**Issue**: `wireless/transceiver.py:317` used `np.linalg.inv` and `np.eye` with torch tensors, causing dtype mismatch between complex float and complex double.

**Fix**:
```python
# Before:
HHH_inv = np.linalg.inv(HHH + 1e-8 * np.eye(K, device=self.device))

# After:
HHH_inv = torch.linalg.inv(HHH + 1e-8 * torch.eye(K, dtype=H.dtype, device=self.device))
```

### Problem 3: Divide by Zero in Rayleigh-Sommerfeld

**Issue**: `util/util.py:68` divided by `distances` which could be zero when source and target positions coincide.

**Fix**:
```python
# Added safety epsilon:
distances_safe = distances + 1e-10
cos_xi = np.abs(np.dot(directions, normal)) / distances_safe
amplitude = (aperture_area * cos_xi / distances_safe) * (1 / (2 * np.pi * distances_safe) - 1j / wavelength)
```

### Key Insights from Session

**Electromagnetic Reciprocity:**
- For passive, linear systems: H_uplink = H_downlink^H
- Psi represents physical propagation in one direction (downlink)
- For reverse direction (uplink), apply conjugate transpose

**Matrix Multiplication Order:**
- Order matters! Matrix multiplication is NOT commutative
- Order `H @ Psi @ A` follows physical signal path
- Cannot arbitrarily swap order

**Sum-Rate Calculation:**
- Paper maximizes **sum** of rates, not average
- Sum-rate = Σ_k log_2(1 + SINR_k)
- NOT divided by K - represents total system throughput

**Paper Parameters (from Section V, page 5):**
- K = 4 users
- σ² = -80 dBm (noise power)
- P_T = 26 dBm (total power budget, baseline)
- b = 2 bit (phase shifter resolution)
- α = 2 (path loss exponent)
- C_0 = -30 dB (reference path loss at 1m)
- λ = 0.125m (wavelength)
- I_AO = 5 (AO iterations)
- 50 Monte Carlo trials

### Files Modified
- `Beamformer.py`: Lines 126-141 (compute_end_to_end_channel with direction parameter)
- `wireless/sim.py`: Lines 222-249 (uplink/downlink methods with correct directions)
- `wireless/transceiver.py`: Line 317 (torch instead of numpy for matrix operations)
- `util/util.py`: Lines 64-70 (added epsilon to prevent divide by zero)

### Testing
- `test_beamformer_flow.py` runs successfully without errors
- No more dtype mismatch errors
- No more divide by zero warnings
- Channels computed in correct directions

### Backward Compatibility
⚠️ **Breaking change** - Previous results computed with wrong channel direction are invalid. This fix corrects the fundamental signal propagation, so:
- Old optimization results should be re-run
- Any saved channels/results from before this fix are incorrect
- New results will differ significantly (correct downlink vs. incorrect uplink)

## 2025-11-20 (Part 3): Test Script for Beamforming Flow

### Created File
`test/test_beamformer_flow.py` - Demonstration script showing SIM vs Digital beamforming flow

### Purpose
Educational script that shows:
1. How to set up SIM-based beamformer
2. How to set up Digital beamformer (no SIM)
3. How channels are generated in each case
4. Where optimization algorithms should be inserted
5. How to compute metrics (SINR, sum-rate)

### Key Features
- **Clear separation**: Two cases side-by-side for comparison
- **Dummy values**: Uses random phases/weights to show flow without optimization
- **Placeholders**: Marked with `>>> OPTIMIZATION GOES HERE <<<`
- **Complete flow**: From initialization to metric computation
- **Comments**: Explains what happens at each step

### What It Shows

**SIM-Based Flow:**
1. Create SIM model
2. Create Beamformer with SIM (channels auto-generated)
3. **[OPTIMIZE phases here]** ← Insert optimization algorithm
4. Compute end-to-end channel with phases
5. Compute SINR and sum-rate

**Digital Flow:**
1. Create Beamformer without SIM
2. Generate channel manually
3. **[OPTIMIZE weights here]** ← Insert optimization algorithm
4. Apply weights: H @ W
5. Compute SINR and sum-rate

### Usage
```bash
python test/test_beamformer_flow.py
```

Outputs detailed step-by-step execution showing channel shapes, where optimization goes, and example metrics.

## 2025-11-20 (Part 2): Per-User Distance-Dependent Path Loss in CLT Mode

### Problem
The CLT mode was using the same reference path loss for all users, not matching the paper's model where β_k = C_0 * d_k^(-α) with different distances d_k per user.

### Solution
Added `min_user_distance` and `max_user_distance` parameters to:
1. `UserChannel.__init__()` (wireless/channel.py:22-23)
2. `Beamformer.__init__()` (Beamformer.py:33-34)

Modified CLT mode (wireless/channel.py:218-243):
- Sample user distances uniformly: `d_k ~ U(min_user_distance, max_user_distance)`
- Compute per-user path loss: `β_k = C_0 * (d_k / d_0)^(-α_k)`
- Apply log-normal shadowing per user
- Apply Rician fading per user

### Now Matches Paper
From paper (page 3): "β_k = C_0 * d_k^(-α)"
- C_0: path_loss_at_reference = -30 dB (at 1m reference)
- d_k: user distance (randomly sampled in [10, 100]m by default)
- α: path_loss_exponents[k]

Each user now has:
1. Different distance → different path loss
2. Random shadowing → additional per-user variation
3. Rician fading → per-user LOS/scattered mixing

### Default Values
- min_user_distance = 10.0 m
- max_user_distance = 100.0 m
- Matches paper's "users randomly distributed within an area of 100m²"

## 2025-11-20 (Part 1): Channel Generation Enhancement

### Problem
Original `generate_channel()` required user positions to be set, throwing ValueError if None. Needed ability to generate channels without geometric information for cases where exact positions unknown or using statistical assumptions.

### Solution
Modified `wireless/channel.py` lines 185-311 to support two modes:

#### Mode 1: CLT-based (No Positions)
When `user_positions = None`:
1. Generate i.i.d. complex Gaussian coefficients
2. Apply path loss from reference distance
3. Apply per-user log-normal shadowing
4. Apply Rician fading (LOS + scattered)

**Key implementation details**:
- Path loss baseline: `Co = sqrt(10^(path_loss_at_reference/10))` (line 215)
- LOS component: Random phases since no geometry available
- Scattered component: Rayleigh fading scaled by path_loss
- Mixing uses same formula as geometric case for consistency

#### Mode 2: Geometric (With Positions)
Original behavior preserved - uses distance, angles, and geometry.

### Rationale
- **CLT assumption**: Standard in massive MIMO when many antennas and random user distribution
- **Rician fading inclusion**: User explicitly requested to maintain all three channel effects (path loss, shadowing, fading)
- **Consistent mixing**: Both modes use identical Rician formula for code clarity

### Testing Status
- Code written and verified
- Not yet tested with actual optimization runs
- Geometric mode unchanged, should work as before

## Implementation Notes for Future Sessions

### When to Use Each Mode
- **CLT (no positions)**:
  - Statistical analysis
  - When positions unavailable
  - Faster computation (no geometry calculations)
  - Following paper's approach for FR3 frequencies

- **Geometric (with positions)**:
  - Need spatial information
  - Lower frequency bands
  - Specific deployment scenarios
  - Not fully implemented per user's note (needs cluster delay line)

### Code Comments from User
- Line 215: `#Co in the paper` - refers to reference path loss constant
- Lines 253-257: User notes geometric model "Not 100% correct yet", needs cluster delay line for FR3 frequencies
- User prioritized CLT mode for efficiency

### Related User Questions Answered

**Q: Should channels regenerate during optimization?**
A: No, per reference paper. Generate once, optimize, then run multiple independent trials.

**Q: Why CLT assumption?**
A: When positions unknown, CLT gives i.i.d. Gaussian channels - standard massive MIMO assumption.

**Q: Which approach does paper use?**
A: Fixed channel per optimization run. Paper states "all operations within single coherence time."

## Files Modified
- `wireless/channel.py`: Lines 185-311 (generate_channel method)
- Documentation comment added explaining CLT mode

## Dependencies
None changed. Uses existing numpy/torch operations.

## Backward Compatibility
✅ Fully compatible - geometric mode unchanged, new CLT mode only activates when positions not set.