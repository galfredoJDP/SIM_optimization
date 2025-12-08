"""
Detailed Comparison Script: Old Code vs New Code
Ensures identical channel realizations and provides comprehensive diagnostics.
Run from SIM_optimization directory.
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

# ==================== SETUP PATHS ====================
# Add old code path
old_code_path = Path(__file__).parent.parent / "SIM-assisted network_WF_PGA"
sys.path.insert(0, str(old_code_path))

# ==================== IMPORTS ====================
# Old Code
from Parameters import Network_Parameters
from Generate_W import W_mat, W_vec
from Gradient_method import Gradient_ascent_solution
from Performance_matrics import sum_rate, bisection_method, SINR

# New Code
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA, WaterFilling as WF_New

# ==================== CONFIGURATION ====================
# Physical Parameters
N = 25
L = 2
K = 4
M = 4
FREQ = 2.4e9
C = 3e8
WAVELENGTH = C / FREQ
LAYER_SPACING = 0.25 * WAVELENGTH
METAATOM_SPACING = 0.5 * WAVELENGTH
POWER_DBM = 26
POWER_WATTS = 10**(POWER_DBM/10) / 1000
NOISE_DBM = -80
NOISE_WATTS = 10**(NOISE_DBM/10) / 1000

AO_ITERATIONS = 5
SEED = 42

print("="*80)
print("DETAILED COMPARISON: Old Code vs New Code")
print("="*80)
print(f"Configuration:")
print(f"  N = {N} meta-atoms per layer")
print(f"  L = {L} layers")
print(f"  K = {K} users")
print(f"  M = {M} antennas")
print(f"  Power = {POWER_DBM} dBm ({POWER_WATTS*1000:.4f} mW)")
print(f"  Noise = {NOISE_DBM} dBm ({NOISE_WATTS*1e12:.4f} pW)")
print(f"  AO Iterations = {AO_ITERATIONS}")
print(f"  Seed = {SEED}")
print("="*80)

# ==================== GENERATE SHARED CHANNELS ====================
print("\n[1] Generating Shared Channel Realizations...")

# Generate H (SIM to Users)
np.random.seed(SEED)
max_dist = 100
distances = np.random.uniform(0, max_dist, K)
path_loss_ref = 10**(-30/10)
alpha = 2
distances = np.maximum(distances, 0.1)
beta = path_loss_ref * (distances ** (-alpha))

H = np.zeros((N, K), dtype=complex)
for k in range(K):
    var = beta[k] / 2
    h_real = np.random.normal(0, np.sqrt(var), N)
    h_imag = np.random.normal(0, np.sqrt(var), N)
    H[:, k] = h_real + 1j * h_imag

print(f"  H shape: {H.shape} (N x K)")
print(f"  H norm: {np.linalg.norm(H):.4f}")
print(f"  User distances: {distances}")

# Generate W_1 / A (Antennas to SIM)
param_temp = Network_Parameters(M=M, K=K)
param_temp.N = N
param_temp.comm_lambda = WAVELENGTH
param_temp.d_layer = LAYER_SPACING
param_temp.SIM_space = METAATOM_SPACING
param_temp.BS_antenna_space = METAATOM_SPACING
param_temp.element_size = [METAATOM_SPACING, METAATOM_SPACING]
W_1 = W_vec(param_temp)

print(f"  W_1/A shape: {W_1.shape} (N x M)")
print(f"  W_1/A norm: {np.linalg.norm(W_1):.4f}")

# Generate Initial Phases
np.random.seed(SEED)
initial_phases = np.random.uniform(0, 2*np.pi, (L, N))
print(f"  Initial phases shape: {initial_phases.shape} (L x N)")
print(f"  Phase range: [{initial_phases.min():.4f}, {initial_phases.max():.4f}]")

# ==================== VERIFICATION: Check if channels are identical ====================
print("\n[2] Verifying Channel Sharing...")

# For new code, H needs to be transposed and conjugated
H_new = H.conj().T  # (K, N)
A_new = W_1  # (N, M)

print(f"  Old H shape: {H.shape}")
print(f"  New H shape: {H_new.shape}")
print(f"  Transformation check: H_new = H_old.conj().T")

# Verify a random element
test_n, test_k = 5, 2
print(f"  Sample element [n={test_n}, k={test_k}]:")
print(f"    Old H[{test_n},{test_k}] = {H[test_n, test_k]:.6f}")
print(f"    New H[{test_k},{test_n}] = {H_new[test_k, test_n]:.6f}")
print(f"    Old H[{test_n},{test_k}].conj() = {H[test_n, test_k].conj():.6f}")
print(f"    Match: {np.allclose(H[test_n, test_k].conj(), H_new[test_k, test_n])}")

# ==================== RUN OLD CODE ====================
print("\n" + "="*80)
print("[3] Running OLD CODE (Coordinate Descent)")
print("="*80)

# Setup parameters
param = Network_Parameters()
param.N = N
param.L = L
param.K = K
param.M = M
param.comm_lambda = WAVELENGTH
param.d_layer = LAYER_SPACING
param.SIM_space = METAATOM_SPACING
param.BS_antenna_space = METAATOM_SPACING
param.element_size = [METAATOM_SPACING, METAATOM_SPACING]
param.noisy_channel_enable = 0
param.final_calc_option = 0
param.power_budget = POWER_WATTS
param.N0 = NOISE_WATTS * np.ones(K)

# Inject channels
param.H = H
param.W_1 = W_1

# W matrices (inter-layer)
param.W_total = np.zeros((L - 1, N, N), dtype=complex)
for l in range(L - 1):
    param.W_total[l, :, :] = W_mat(param)

# Initial phases and switches
param.Phi_total = np.zeros((L, N, N), dtype=complex)
param.Phi_total_Gradient = np.zeros((L, N, N), dtype=complex)
param.SW_total = np.zeros((L, N, N), dtype=complex)

for l in range(L):
    for n in range(N):
        param.Phi_total[l, n, n] = np.exp(1j * initial_phases[l, n])
        param.SW_total[l, n, n] = 1
        param.Phi_total_Gradient[l, n, n] = param.Phi_total[l, n, n]

# Initial power (equal)
param.power_tx = (param.power_budget / K) * np.ones(K)
initial_sum_rate_old = -1 * sum_rate(param)
print(f"Initial Sum-Rate (equal power, random phases): {initial_sum_rate_old:.6f} bits/s/Hz")

# Optimization parameters
param.step_size_Gradient = 0.001
param.beta_gradient = 0.5

print(f"\nOptimization Settings:")
print(f"  Gradient step size: {param.step_size_Gradient}")
print(f"  Backtracking beta: {param.beta_gradient}")
print(f"  AO iterations: {AO_ITERATIONS}")

# Run AO iterations
old_history = {'sum_rate': [initial_sum_rate_old], 'power': [param.power_tx.copy()]}

for ao in range(AO_ITERATIONS):
    print(f"\n--- AO Iteration {ao+1}/{AO_ITERATIONS} ---")

    # Phase optimization (Gradient Ascent)
    print(f"  Running Gradient Ascent...")
    R_val, _ = Gradient_ascent_solution(param)
    rate_after_phases = -1 * sum_rate(param)
    print(f"    Sum-Rate after phase opt: {rate_after_phases:.6f} bits/s/Hz")

    # Power optimization (Water Filling)
    print(f"  Running Water Filling (max_iter=10)...")
    pWF, power_tx_new = bisection_method(param)
    param.power_tx = power_tx_new
    rate_after_power = -1 * sum_rate(param)
    print(f"    Sum-Rate after power opt: {rate_after_power:.6f} bits/s/Hz")
    print(f"    Power allocation: {param.power_tx}")

    old_history['sum_rate'].append(rate_after_power)
    old_history['power'].append(param.power_tx.copy())

final_sum_rate_old = -1 * sum_rate(param)
final_sinr_old = SINR(param)

print(f"\n{'='*40}")
print(f"OLD CODE FINAL RESULTS:")
print(f"  Sum-Rate: {final_sum_rate_old:.6f} bits/s/Hz")
print(f"  SINR per user: {final_sinr_old}")
print(f"  Power per user: {param.power_tx}")
print(f"  Total power: {np.sum(param.power_tx):.6f} W")
print(f"{'='*40}")

# ==================== RUN NEW CODE ====================
print("\n" + "="*80)
print("[4] Running NEW CODE (PGA + Water Filling)")
print("="*80)

device = 'cpu'

# Setup SIM
sim = Sim(layers=L, metaAtoms=N, layerSpacing=LAYER_SPACING,
          metaAtomSpacing=METAATOM_SPACING, metaAtomArea=METAATOM_SPACING**2,
          wavelength=WAVELENGTH, device=device)

# Setup Beamformer
beamformer = Beamformer(Nx=2, Ny=2, wavelength=WAVELENGTH, device=device,
                        num_users=K, noise_power=NOISE_WATTS, total_power=POWER_WATTS,
                        sim_model=sim)

# Inject channels
beamformer.H = torch.tensor(H_new, dtype=torch.complex64, device=device)
beamformer.A = torch.tensor(A_new, dtype=torch.complex64, device=device)

# Initial state
phases = torch.tensor(initial_phases, dtype=torch.float32, device=device)
power_allocation = torch.ones(K, device=device) * (POWER_WATTS / K)

initial_sum_rate_new = beamformer.compute_sum_rate(phases, power_allocation).item()
print(f"Initial Sum-Rate (equal power, random phases): {initial_sum_rate_new:.6f} bits/s/Hz")

print(f"\nOptimization Settings:")
print(f"  PGA learning rate: 0.1")
print(f"  PGA max iterations: 100")
print(f"  WF max iterations: 100")
print(f"  AO iterations: {AO_ITERATIONS}")

# Run AO iterations
new_history = {'sum_rate': [initial_sum_rate_new], 'power': [power_allocation.cpu().numpy().copy()]}

for ao in range(AO_ITERATIONS):
    print(f"\n--- AO Iteration {ao+1}/{AO_ITERATIONS} ---")

    # Phase optimization (PGA)
    print(f"  Running PGA...")
    optimizer = PGA(beamformer,
                    objective_fn=lambda p: beamformer.compute_sum_rate(p, power_allocation),
                    learning_rate=0.001, max_iterations=100, verbose=False)
    res = optimizer.optimize(phases)
    phases = res['optimal_params']
    rate_after_phases = beamformer.compute_sum_rate(phases, power_allocation).item()
    print(f"    Sum-Rate after phase opt: {rate_after_phases:.6f} bits/s/Hz")
    print(f"    PGA iterations: {res['iterations']}")

    # Power optimization (Water Filling)
    print(f"  Running Water Filling (max_iter=100)...")
    H_eff = beamformer.compute_end_to_end_channel(phases)
    wf = WF_New(H_eff, NOISE_WATTS, POWER_WATTS, device=device)
    wf_res = wf.optimize(initial_power=power_allocation)
    power_allocation = wf_res['optimal_power']
    rate_after_power = beamformer.compute_sum_rate(phases, power_allocation).item()
    print(f"    Sum-Rate after power opt: {rate_after_power:.6f} bits/s/Hz")
    print(f"    WF iterations: {wf_res['iterations']}")
    print(f"    Power allocation: {power_allocation.cpu().numpy()}")

    new_history['sum_rate'].append(rate_after_power)
    new_history['power'].append(power_allocation.cpu().numpy().copy())

final_sum_rate_new = beamformer.compute_sum_rate(phases, power_allocation).item()
final_sinr_new = beamformer.compute_sinr(phases, power_allocation).cpu().numpy()

print(f"\n{'='*40}")
print(f"NEW CODE FINAL RESULTS:")
print(f"  Sum-Rate: {final_sum_rate_new:.6f} bits/s/Hz")
print(f"  SINR per user: {final_sinr_new}")
print(f"  Power per user: {power_allocation.cpu().numpy()}")
print(f"  Total power: {power_allocation.sum().item():.6f} W")
print(f"{'='*40}")

# ==================== COMPARISON ====================
print("\n" + "="*80)
print("[5] COMPARISON SUMMARY")
print("="*80)

print(f"\nSum-Rate Comparison:")
print(f"  Old Code: {final_sum_rate_old:.6f} bits/s/Hz")
print(f"  New Code: {final_sum_rate_new:.6f} bits/s/Hz")
print(f"  Difference: {abs(final_sum_rate_old - final_sum_rate_new):.6f} bits/s/Hz")
print(f"  Relative Diff: {abs(final_sum_rate_old - final_sum_rate_new)/final_sum_rate_old*100:.2f}%")

print(f"\nSINR Comparison (dB):")
sinr_old_db = 10*np.log10(final_sinr_old)
sinr_new_db = 10*np.log10(final_sinr_new)
for k in range(K):
    print(f"  User {k}: Old = {sinr_old_db[k]:7.2f} dB, New = {sinr_new_db[k]:7.2f} dB, Diff = {sinr_old_db[k]-sinr_new_db[k]:+7.2f} dB")

print(f"\nPower Allocation Comparison (mW):")
power_old_mw = param.power_tx * 1000
power_new_mw = power_allocation.cpu().numpy() * 1000
for k in range(K):
    print(f"  User {k}: Old = {power_old_mw[k]:7.4f} mW, New = {power_new_mw[k]:7.4f} mW, Diff = {power_old_mw[k]-power_new_mw[k]:+7.4f} mW")

print(f"\nConvergence Pattern:")
print(f"  Old Code AO iterations: Initial → ", end="")
for i in range(1, len(old_history['sum_rate'])):
    improvement = old_history['sum_rate'][i] - old_history['sum_rate'][i-1]
    print(f"{old_history['sum_rate'][i]:.4f} (+{improvement:.4f})", end="")
    if i < len(old_history['sum_rate']) - 1:
        print(" → ", end="")
print()

print(f"  New Code AO iterations: Initial → ", end="")
for i in range(1, len(new_history['sum_rate'])):
    improvement = new_history['sum_rate'][i] - new_history['sum_rate'][i-1]
    print(f"{new_history['sum_rate'][i]:.4f} (+{improvement:.4f})", end="")
    if i < len(new_history['sum_rate']) - 1:
        print(" → ", end="")
print()

# ==================== VISUALIZATION ====================
print("\n[6] Generating Plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sum-Rate Convergence
ax = axes[0, 0]
ax.plot(range(len(old_history['sum_rate'])), old_history['sum_rate'],
        'o-', label='Old Code', linewidth=2, markersize=8)
ax.plot(range(len(new_history['sum_rate'])), new_history['sum_rate'],
        's-', label='New Code', linewidth=2, markersize=8)
ax.set_xlabel('AO Iteration')
ax.set_ylabel('Sum-Rate (bits/s/Hz)')
ax.set_title('Sum-Rate Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: SINR Comparison
ax = axes[0, 1]
x = np.arange(K)
width = 0.35
ax.bar(x - width/2, sinr_old_db, width, label='Old Code', alpha=0.8)
ax.bar(x + width/2, sinr_new_db, width, label='New Code', alpha=0.8)
ax.set_xlabel('User Index')
ax.set_ylabel('SINR (dB)')
ax.set_title('SINR per User')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Power Allocation Convergence (Old)
ax = axes[1, 0]
for k in range(K):
    powers = [p[k]*1000 for p in old_history['power']]
    ax.plot(range(len(powers)), powers, 'o-', label=f'User {k}', linewidth=2, markersize=6)
ax.set_xlabel('AO Iteration')
ax.set_ylabel('Power (mW)')
ax.set_title('Power Allocation Convergence - Old Code')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Power Allocation Convergence (New)
ax = axes[1, 1]
for k in range(K):
    powers = [p[k]*1000 for p in new_history['power']]
    ax.plot(range(len(powers)), powers, 's-', label=f'User {k}', linewidth=2, markersize=6)
ax.set_xlabel('AO Iteration')
ax.set_ylabel('Power (mW)')
ax.set_title('Power Allocation Convergence - New Code')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'detailed_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
