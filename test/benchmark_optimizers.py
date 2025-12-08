"""
Benchmark: Which optimizer finds better solutions?
Tests both methods across multiple channel realizations.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

# Setup paths
old_code_path = Path(__file__).parent.parent / "SIM-assisted network_WF_PGA"
sys.path.insert(0, str(old_code_path))

# Imports
from Parameters import Network_Parameters
from Generate_W import W_mat, W_vec
from Gradient_method import Gradient_ascent_solution
from Performance_matrics import sum_rate, bisection_method

from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA, WaterFilling as WF_New

# Configuration
N, L, K, M = 25, 2, 4, 4
FREQ, C = 2.4e9, 3e8
WAVELENGTH = C / FREQ
LAYER_SPACING = 0.25 * WAVELENGTH
METAATOM_SPACING = 0.5 * WAVELENGTH
POWER_DBM = 26
POWER_WATTS = 10**(POWER_DBM/10) / 1000
NOISE_DBM = -80
NOISE_WATTS = 10**(-80/10) / 1000

NUM_TRIALS = 10
AO_ITERATIONS = 10  # Run more iterations to see convergence

print("="*80)
print("BENCHMARK: Which Optimizer Finds Better Solutions?")
print("="*80)
print(f"Configuration:")
print(f"  Trials: {NUM_TRIALS}")
print(f"  AO Iterations: {AO_ITERATIONS}")
print(f"  Power: {POWER_DBM} dBm")
print("="*80)

results = {
    'old': [],
    'new': [],
    'old_history': [],
    'new_history': []
}

for trial in range(NUM_TRIALS):
    seed = 42 + trial
    print(f"\n{'='*80}")
    print(f"Trial {trial+1}/{NUM_TRIALS} (seed={seed})")
    print(f"{'='*80}")

    # Generate channels
    np.random.seed(seed)
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

    param_temp = Network_Parameters(M=M, K=K)
    param_temp.N = N
    param_temp.comm_lambda = WAVELENGTH
    param_temp.d_layer = LAYER_SPACING
    param_temp.SIM_space = METAATOM_SPACING
    param_temp.BS_antenna_space = METAATOM_SPACING
    param_temp.element_size = [METAATOM_SPACING, METAATOM_SPACING]
    W_1 = W_vec(param_temp)

    np.random.seed(seed)
    initial_phases = np.random.uniform(0, 2*np.pi, (L, N))

    # ==================== OLD CODE ====================
    print("\nOLD CODE:")
    param = Network_Parameters()
    param.N, param.L, param.K, param.M = N, L, K, M
    param.comm_lambda = WAVELENGTH
    param.d_layer = LAYER_SPACING
    param.SIM_space = METAATOM_SPACING
    param.BS_antenna_space = METAATOM_SPACING
    param.element_size = [METAATOM_SPACING, METAATOM_SPACING]
    param.noisy_channel_enable = 0
    param.final_calc_option = 0
    param.power_budget = POWER_WATTS
    param.N0 = NOISE_WATTS * np.ones(K)
    param.H = H
    param.W_1 = W_1

    param.W_total = np.zeros((L - 1, N, N), dtype=complex)
    for l in range(L - 1):
        param.W_total[l, :, :] = W_mat(param)

    param.Phi_total = np.zeros((L, N, N), dtype=complex)
    param.Phi_total_Gradient = np.zeros((L, N, N), dtype=complex)
    param.SW_total = np.zeros((L, N, N), dtype=complex)
    for l in range(L):
        for n in range(N):
            param.Phi_total[l, n, n] = np.exp(1j * initial_phases[l, n])
            param.SW_total[l, n, n] = 1
            param.Phi_total_Gradient[l, n, n] = param.Phi_total[l, n, n]

    param.power_tx = (param.power_budget / K) * np.ones(K)
    param.step_size_Gradient = 0.001
    param.beta_gradient = 0.5

    old_hist = []
    for ao in range(AO_ITERATIONS):
        Gradient_ascent_solution(param)
        bisection_method(param)
        rate = -1 * sum_rate(param)
        old_hist.append(rate)
        if ao % 2 == 0 or ao == AO_ITERATIONS - 1:
            print(f"  AO iter {ao+1}: {rate:.4f} bits/s/Hz")

    final_old = old_hist[-1]
    results['old'].append(final_old)
    results['old_history'].append(old_hist)

    # ==================== NEW CODE ====================
    print("\nNEW CODE:")
    device = 'cpu'
    sim = Sim(layers=L, metaAtoms=N, layerSpacing=LAYER_SPACING,
              metaAtomSpacing=METAATOM_SPACING, metaAtomArea=METAATOM_SPACING**2,
              wavelength=WAVELENGTH, device=device)

    beamformer = Beamformer(Nx=2, Ny=2, wavelength=WAVELENGTH, device=device,
                            num_users=K, noise_power=NOISE_WATTS, total_power=POWER_WATTS,
                            sim_model=sim)

    H_new = H.conj().T
    A_new = W_1
    beamformer.H = torch.tensor(H_new, dtype=torch.complex64, device=device)
    beamformer.A = torch.tensor(A_new, dtype=torch.complex64, device=device)

    phases = torch.tensor(initial_phases, dtype=torch.float32, device=device)
    power_allocation = torch.ones(K, device=device) * (POWER_WATTS / K)

    new_hist = []
    for ao in range(AO_ITERATIONS):
        optimizer = PGA(beamformer,
                        objective_fn=lambda p: beamformer.compute_sum_rate(p, power_allocation),
                        learning_rate=0.001, max_iterations=100, verbose=False)
        res = optimizer.optimize(phases)
        phases = res['optimal_params']

        H_eff = beamformer.compute_end_to_end_channel(phases)
        wf = WF_New(H_eff, NOISE_WATTS, POWER_WATTS, device=device, verbose=False)
        wf_res = wf.optimize(initial_power=power_allocation)
        power_allocation = wf_res['optimal_power']

        rate = beamformer.compute_sum_rate(phases, power_allocation).item()
        new_hist.append(rate)
        if ao % 2 == 0 or ao == AO_ITERATIONS - 1:
            print(f"  AO iter {ao+1}: {rate:.4f} bits/s/Hz")

    final_new = new_hist[-1]
    results['new'].append(final_new)
    results['new_history'].append(new_hist)

    print(f"\nTrial {trial+1} Results:")
    print(f"  Old Code: {final_old:.4f} bits/s/Hz")
    print(f"  New Code: {final_new:.4f} bits/s/Hz")
    print(f"  Winner: {'NEW' if final_new > final_old else 'OLD'} (+{abs(final_new-final_old):.4f} bits/s/Hz)")

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)

old_results = np.array(results['old'])
new_results = np.array(results['new'])

print(f"\nSum-Rate Statistics ({NUM_TRIALS} trials):")
print(f"{'Method':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print(f"{'-'*65}")
print(f"{'Old Code':<15} {old_results.mean():<12.4f} {old_results.std():<12.4f} {old_results.min():<12.4f} {old_results.max():<12.4f}")
print(f"{'New Code':<15} {new_results.mean():<12.4f} {new_results.std():<12.4f} {new_results.min():<12.4f} {new_results.max():<12.4f}")

improvement = new_results - old_results
print(f"\nImprovement (New - Old):")
print(f"  Mean improvement: {improvement.mean():.4f} bits/s/Hz ({improvement.mean()/old_results.mean()*100:.2f}%)")
print(f"  Std of improvement: {improvement.std():.4f} bits/s/Hz")
print(f"  New wins: {np.sum(new_results > old_results)}/{NUM_TRIALS} trials")
print(f"  Old wins: {np.sum(old_results > new_results)}/{NUM_TRIALS} trials")
print(f"  Ties: {np.sum(np.isclose(old_results, new_results))}/{NUM_TRIALS} trials")

# Statistical test (simple version without scipy)
print(f"\nStatistical Summary:")
print(f"  Effect size (Cohen's d): {improvement.mean() / improvement.std():.4f}")
if abs(improvement.mean()) > 2 * (improvement.std() / np.sqrt(NUM_TRIALS)):
    winner = "NEW CODE" if improvement.mean() > 0 else "OLD CODE"
    print(f"  Result: {winner} is likely significantly better (mean > 2*SEM)")
else:
    print(f"  Result: Difference may not be statistically significant")

# ==================== VISUALIZATION ====================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Box plot comparison
ax = axes[0, 0]
box_data = [old_results, new_results]
bp = ax.boxplot(box_data, labels=['Old Code', 'New Code'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Sum-Rate (bits/s/Hz)')
ax.set_title('Sum-Rate Distribution Across Trials')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Trial-by-trial comparison
ax = axes[0, 1]
trials = np.arange(1, NUM_TRIALS + 1)
ax.plot(trials, old_results, 'o-', label='Old Code', linewidth=2, markersize=8)
ax.plot(trials, new_results, 's-', label='New Code', linewidth=2, markersize=8)
ax.set_xlabel('Trial')
ax.set_ylabel('Sum-Rate (bits/s/Hz)')
ax.set_title('Trial-by-Trial Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Improvement histogram
ax = axes[1, 0]
ax.hist(improvement, bins=15, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero improvement')
ax.axvline(improvement.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {improvement.mean():.2f}')
ax.set_xlabel('Improvement (bits/s/Hz)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Improvements (New - Old)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Convergence curves (average)
ax = axes[1, 1]
old_hist_avg = np.mean(results['old_history'], axis=0)
new_hist_avg = np.mean(results['new_history'], axis=0)
old_hist_std = np.std(results['old_history'], axis=0)
new_hist_std = np.std(results['new_history'], axis=0)

iters = np.arange(1, AO_ITERATIONS + 1)
ax.plot(iters, old_hist_avg, 'o-', label='Old Code', linewidth=2, markersize=6)
ax.fill_between(iters, old_hist_avg - old_hist_std, old_hist_avg + old_hist_std, alpha=0.2)
ax.plot(iters, new_hist_avg, 's-', label='New Code', linewidth=2, markersize=6)
ax.fill_between(iters, new_hist_avg - new_hist_std, new_hist_avg + new_hist_std, alpha=0.2)
ax.set_xlabel('AO Iteration')
ax.set_ylabel('Sum-Rate (bits/s/Hz)')
ax.set_title(f'Average Convergence (±1 std, {NUM_TRIALS} trials)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = 'optimizer_benchmark.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if improvement.mean() > 0.5:
    print("NEW CODE (Batch Gradient Ascent) consistently finds BETTER solutions!")
    print(f"Average improvement: {improvement.mean():.2f} bits/s/Hz ({improvement.mean()/old_results.mean()*100:.1f}%)")
elif improvement.mean() < -0.5:
    print("OLD CODE (Sequential Coordinate Descent) consistently finds BETTER solutions!")
    print(f"Average improvement: {abs(improvement.mean()):.2f} bits/s/Hz ({abs(improvement.mean())/new_results.mean()*100:.1f}%)")
else:
    print("Both methods find SIMILAR solutions on average.")
    print(f"Difference: {abs(improvement.mean()):.2f} bits/s/Hz")
print("="*80)
