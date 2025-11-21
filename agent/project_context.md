# SIM Optimization Project Context

## Project Overview
Beamforming optimization for Stacked Intelligent Metasurfaces (SIM) in holographic MIMO communications. Implements sum-rate maximization following the paper by Nassirpour et al. (2025 IEEE ICC).

## Core Architecture

### Main Classes
1. **UserChannel** (`wireless/channel.py`): Multi-user channel model
2. **Transceiver** (`wireless/transceiver.py`): Antenna array management
3. **sim** (`wireless/sim.py`): Stacked Intelligent Metasurface model
4. **Beamformer** (`Beamformer.py`): Combines all components, inherits from both Transceiver and UserChannel

### System Flow
```
Antennas [A] → SIM [Ψ(phases)] → [H] → Users [K]
```

### SIM Directionality (IMPORTANT - Fixed 2025-11-20)

**Psi Matrix Direction:**
- **Psi** represents propagation from **Layer 0 → Layer (L-1)** (BS side → User side)
- This is the **DOWNLINK** direction
- Constructed via Rayleigh-Sommerfeld: Layer ℓ (source) → Layer ℓ+1 (target)

**Signal Propagation:**
- **Downlink** (BS transmits to users): `y = H @ Psi @ A @ s`
  - Effective channel: `H_eff = H @ Psi @ A` (K, M)
  - Use `Psi` directly (no transpose)

- **Uplink** (Users transmit to BS): `y = A^H @ Psi^H @ H^H @ s`
  - Effective channel: `H_eff = torch.conj(H @ Psi @ A).T` (M, K)
  - Use `Psi^H` (conjugate transpose) via electromagnetic reciprocity

**Matrix Dimensions:**
- **A**: (N, M) - M antennas → N meta-atoms at Layer 0
- **Psi**: (N, N) - Layer 0 → Layer (L-1) propagation through SIM
- **H**: (K, N) - N meta-atoms at Layer (L-1) → K users

**Critical Notes:**
- Matrix multiplication order follows physical signal path and is NOT commutative
- For downlink sum-rate maximization (paper's problem), use `H @ Psi @ A`
- Reciprocity: For passive SIM, uplink channel = downlink channel^H

## Key Design Decisions

### Channel Generation (Modified: 2025-11-20)

**UserChannel.generate_channel()** now handles two cases:

#### Case 1: No User Positions (user_positions = None)
- Uses Central Limit Theorem (CLT) assumption
- Generates i.i.d. complex Gaussian channels: CN(0, σ²)
- Applies three channel effects:
  1. **Path loss**: Based on reference distance (path_loss_at_reference)
  2. **Log-normal shadowing**: Per-user variance (path_loss_variance)
  3. **Rician fading**: Mixes LOS + scattered components using rician_k_factors
- LOS component uses random phases (no geometry)
- Located: `wireless/channel.py:210-251`

#### Case 2: With User Positions
- Geometric channel model
- Distance-dependent path loss with per-user exponents
- Array phase response based on angles of arrival
- Rician fading with geometry-based LOS
- Located: `wireless/channel.py:253-311`

### Channel Regeneration Strategy

**Based on reference paper**: Channels are FIXED during optimization runs.

**Paper approach**:
- All operations within single coherence time
- CSI constant and globally available
- Generate one channel → optimize → repeat for multiple trials (Monte Carlo)

**Implementation**:
- Channel generated once in `Beamformer._compute_static_channels()` (line 78-112)
- Stays fixed unless `update_user_channel(time)` explicitly called
- For learning algorithms: Keep channel fixed per optimization run, then run multiple independent trials

**Alternative (not used in paper)**:
- Regenerating channels during training → for robust learning across channel realizations
- Use for: RL, neural networks, time-varying scenarios

## Important Code Patterns

### Device Management
- All tensors use `self.device` (cpu/cuda/mps)
- Convert numpy → torch when returning from UserChannel

### Phase Conventions
- SIM phases: [0, 2π) discrete levels based on b-bit resolution
- Rician K-factor: in dB, converted to linear for calculations

### Power Normalization
- Reference path loss: dB → linear with 10^(dB/10)
- Amplitude path loss: 10^(dB/20)
- Rician mixing: sqrt(K/(K+1)) for LOS, sqrt(1/(K+1)) for scattered

## File Locations

### Core Implementation
- `wireless/channel.py`: Channel models (modified recently)
- `wireless/sim.py`: SIM/metasurface
- `Beamformer.py`: Integrated beamformer system
- `wireless/algorithm.py`: Optimization algorithms

### Reference Materials
- Paper: `files/Sum-Rate_Maximization_in_Holographic_MIMO_Communications_with_Stacked_Intelligent_Metasurfaces.pdf`
- Additional: `files/Phased Array Antennas...pdf`

### Tests
- `test_pga.py`: PGA optimization test
- `test/test_channel.py`: Channel model tests
- `test/test_sim.py`: SIM tests with paper parameters

## Parameter Conventions (from paper)

### From test_sim.py
- Layers: L = 2
- Meta-atoms per layer: N = 25
- Wavelength: λ = 0.125m
- Layer spacing: 0.1m
- Meta-atom spacing: 0.05m
- Meta-atom area: 0.01 × 0.01 m²

### From paper simulations
- **Antennas: M = K** (one antenna per user - key architecture choice!)
- Users: K = 4
- Noise power: σ² = -80 dBm
- Phase shifter bits: b = 2 bit
- Path loss exponent: α = 2
- Reference loss: C₀ = -30 dB at 1m
- AO iterations: I_AO = 5
- Monte Carlo trials: 50

### Paper's Beamforming Architecture (IMPORTANT)
**The paper uses M = K (one antenna per user), NOT traditional beamforming!**

From paper page 2:
> "Therefore, in this work, we focus on M = K and aim to maximize sum-rate solely through power allocation and SIM element optimization."

**This is different from traditional digital beamforming where:**
- All M antennas work together to serve each user
- Each user gets a beamforming vector w_k (length M)
- Requires digital signal processing and high-resolution DACs

**Paper's simplified approach:**
- Each antenna k is "dedicated" to user k
- SIM does beamforming in EM domain (not digital)
- Signal: antenna k → SIM → user k
- Interference: antennas k'≠k → SIM → user k
- Results in (K, K) effective channel naturally

## Common Operations

### Initialize Beamformer
```python
# SIM-based beamforming
beamformer = Beamformer(
    Nx=Nx, Ny=Ny,
    wavelength=wavelength,
    num_users=K,
    sim_model=sim_instance,  # Provide SIM
    user_positions=None,  # Use CLT mode
    min_user_distance=10.0,
    max_user_distance=100.0
)

# Digital beamforming (no SIM)
beamformer = Beamformer(
    Nx=Nx, Ny=Ny,
    wavelength=wavelength,
    num_users=K,
    sim_model=None,  # No SIM
    user_positions=None  # Use CLT mode
)
```

### Generate/Update Channel
- **SIM mode**: Automatic on init via `_compute_static_channels()`
  - Creates A (Antenna→SIM) and H (SIM→Users)
- **Digital mode**: Manual generation
  - `H = beamformer.generate_channel(antenna_positions, time=0.0)`
- Uses CLT if no positions set

### Compute Metrics
```python
# SIM-based
H_eff = beamformer.compute_end_to_end_channel(phases)
sinr = beamformer.compute_sinr(phases, power_allocation)
sum_rate = beamformer.compute_sum_rate(phases, power_allocation)

# Digital (manual)
H_effective = H @ W  # W: beamforming weights
# Compute SINR from H_effective
```

## Optimization Flow

### SIM-Based Beamforming
```python
# 1. Initialize (channels auto-generated)
beamformer = Beamformer(..., sim_model=sim, ...)

# 2. OPTIMIZE: phases (L, N) - discrete values from Q_b
#    >>> Optimization algorithm here <<<
#    objective: maximize compute_sum_rate(phases, powers)

# 3. Evaluate
sum_rate = beamformer.compute_sum_rate(phases_optimal, powers_optimal)
```

### Digital Beamforming
```python
# 1. Initialize and generate channel
beamformer = Beamformer(..., sim_model=None, ...)
H = beamformer.generate_channel(antenna_positions)

# 2. OPTIMIZE: W (M, K) - continuous complex weights
#    >>> Optimization algorithm here <<<
#    methods: Zero-Forcing, MRT, learned weights, etc.

# 3. Evaluate
H_eff = H @ W
# Compute SINR and sum-rate from H_eff
```

## Test Scripts

### Flow Demonstration
- `test/test_beamformer_flow.py`: Shows SIM vs Digital beamforming flow
  - Demonstrates channel generation for both cases
  - Shows where optimization happens (with dummy values)
  - Clear placeholders for adding optimization algorithms