"""
Test script to verify SIM beamforming with PGA+Waterfilling scales correctly with power.

This test diagnoses why SIM-based sum-rate might not be increasing with power.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA, WaterFilling

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Parameters ==========
num_users = 4
wavelength = 0.125  # meters
device = 'cpu'

# Antenna array
Nx_antenna = 2
Ny_antenna = 2
num_antennas = Nx_antenna * Ny_antenna

# SIM parameters
sim_layers = 2
sim_metaatoms = 25
sim_layer_spacing = wavelength / 4
sim_metaatom_spacing = wavelength / 2
sim_metaatom_area = sim_metaatom_spacing ** 2

# Channel parameters
min_user_distance = 10.0
max_user_distance = 100.0
path_loss_at_reference = -30.0
reference_distance = 1.0

# Noise power (fixed)
noise_power = 10**(-80/10) / 1000  # Watts
print(f"Noise power: {noise_power:.4e} W ({10*np.log10(noise_power*1000):.2f} dBm)")

# Power sweep
power_values_db = np.array([10, 15, 20, 25, 30])  # dBm
power_values_linear = 10**(power_values_db/10) / 1000  # Watts

print("\n" + "="*80)
print("SIM BEAMFORMING POWER SCALING TEST (PGA + Waterfilling)")
print("="*80)
print(f"Users (K): {num_users}")
print(f"Antennas (M): {num_antennas}")
print(f"SIM: {sim_layers} layers, {sim_metaatoms} meta-atoms per layer")
print(f"Noise power: {noise_power:.4e} W")
print("="*80)

# ========== Create SIM ==========
print("\n1. Creating SIM...")
sim_model = Sim(
    layers=sim_layers,
    metaAtoms=sim_metaatoms,
    layerSpacing=sim_layer_spacing,
    metaAtomSpacing=sim_metaatom_spacing,
    metaAtomArea=sim_metaatom_area,
    wavelength=wavelength,
    device=device
)

# ========== Create SIM Beamformer ==========
print("\n2. Creating SIM Beamformer...")
sim_beamformer = Beamformer(
    Nx=Nx_antenna,
    Ny=Ny_antenna,
    wavelength=wavelength,
    device=device,
    num_users=num_users,
    user_positions=None,  # CLT mode
    reference_distance=reference_distance,
    path_loss_at_reference=path_loss_at_reference,
    min_user_distance=min_user_distance,
    max_user_distance=max_user_distance,
    sim_model=sim_model,
    noise_power=noise_power,
    total_power=1.0,  # Will be updated in loop
)

print(f"   Channel H (SIM to users) shape: {sim_beamformer.H.shape}")
print(f"   Channel H norm: {torch.norm(sim_beamformer.H).item():.6f}")
print(f"   Channel A (antennas to SIM) shape: {sim_beamformer.A.shape}")
print(f"   Channel A norm: {torch.norm(sim_beamformer.A).item():.6f}")

# ========== Power Sweep ==========
print("\n" + "="*80)
print("POWER SWEEP")
print("="*80)

results = {
    'power_db': [],
    'power_linear': [],
    'equal_power_capacity': [],
    'optimized_capacity': [],
    'equal_power_sinr': [],
    'optimized_sinr': [],
    'optimal_phases': [],
    'optimal_power_allocation': []
}

# Test with fixed random phases first
print("\n[Testing with FIXED random phases across all power levels]")
fixed_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

for power_idx, power_w in enumerate(power_values_linear):
    power_db = power_values_db[power_idx]

    print(f"\n{'='*80}")
    print(f"Power: {power_db:.1f} dBm ({power_w:.4e} W)")
    print(f"{'='*80}")

    # Update beamformer with new power
    sim_beamformer.total_power = power_w

    # ========== Case 1: Fixed phases, equal power ==========
    equal_power = torch.ones(num_users, device=device) * (power_w / num_users)

    # Compute end-to-end channel with fixed phases
    H_eff_fixed = sim_beamformer.compute_end_to_end_channel(fixed_phases)

    print(f"\n[Fixed Phases] End-to-end channel H_eff:")
    print(f"   H_eff shape: {H_eff_fixed.shape}")
    print(f"   H_eff norm: {torch.norm(H_eff_fixed).item():.6f}")
    print(f"   H_eff diagonal: {torch.abs(torch.diag(H_eff_fixed)).cpu().numpy()[:4]}")

    # Compute SINR with equal power
    sinr_equal = sim_beamformer.compute_sinr(
        phases=fixed_phases,
        power_allocation=equal_power,
        digital_beamforming_weights=None,
        debug=(power_idx < 2)
    )

    capacity_equal = torch.sum(torch.log2(1 + sinr_equal)).item()

    print(f"\n[Fixed Phases + Equal Power]")
    print(f"   Power per user: {equal_power[0].item():.4e} W")
    print(f"   SINR: {sinr_equal.cpu().numpy()}")
    print(f"   SINR (dB): {10*np.log10(sinr_equal.cpu().numpy())}")
    print(f"   Sum capacity: {capacity_equal:.6f} bits/s/Hz")

    # ========== Case 2: Fixed phases, waterfilling power ==========
    print(f"\n[Fixed Phases + Waterfilling]")
    waterfilling_fixed = WaterFilling(
        H_eff=H_eff_fixed,
        noise_power=noise_power,
        total_power=power_w,
        max_iterations=200,
        tolerance=1e-6,
        verbose=False,
        device=device
    )

    wf_results_fixed = waterfilling_fixed.optimize()
    optimal_power_fixed = wf_results_fixed['optimal_power']

    sinr_wf = sim_beamformer.compute_sinr(
        phases=fixed_phases,
        power_allocation=optimal_power_fixed,
        digital_beamforming_weights=None
    )

    capacity_wf = torch.sum(torch.log2(1 + sinr_wf)).item()

    print(f"   Power allocation: {optimal_power_fixed.cpu().numpy()}")
    print(f"   SINR: {sinr_wf.cpu().numpy()}")
    print(f"   SINR (dB): {10*np.log10(sinr_wf.cpu().numpy())}")
    print(f"   Sum capacity: {capacity_wf:.6f} bits/s/Hz")

    # ========== Case 3: Optimize phases with PGA, then waterfilling ==========
    print(f"\n[PGA Phase Optimization + Waterfilling]")

    # Initialize random phases for optimization
    init_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

    # Phase optimization with PGA (use equal power for optimization)
    equal_power_for_pga = torch.ones(num_users, device=device) * (power_w / num_users)

    optimizer = PGA(
        beamformer=sim_beamformer,
        objective_fn=lambda phases: sim_beamformer.compute_sum_rate(
            phases=phases,
            power_allocation=equal_power_for_pga
        ),
        learning_rate=0.001,
        max_iterations=500,
        verbose=False,
    )

    pga_results = optimizer.optimize(init_phases)
    optimal_phases = pga_results['optimal_params']
    sumrate_after_pga = pga_results['optimal_objective']

    print(f"   After PGA (equal power): {sumrate_after_pga:.6f} bits/s/Hz")

    # Apply waterfilling with optimized phases
    H_eff_opt = sim_beamformer.compute_end_to_end_channel(optimal_phases)

    waterfilling_opt = WaterFilling(
        H_eff=H_eff_opt,
        noise_power=noise_power,
        total_power=power_w,
        max_iterations=200,
        tolerance=1e-6,
        verbose=False,
        device=device
    )

    wf_results_opt = waterfilling_opt.optimize()
    optimal_power = wf_results_opt['optimal_power']
    capacity_opt = wf_results_opt['optimal_sum_rate']

    # Verify SINR
    sinr_opt = sim_beamformer.compute_sinr(
        phases=optimal_phases,
        power_allocation=optimal_power,
        digital_beamforming_weights=None
    )

    print(f"   After PGA + WF:")
    print(f"      Power allocation: {optimal_power.cpu().numpy()}")
    print(f"      SINR: {sinr_opt.cpu().numpy()}")
    print(f"      SINR (dB): {10*np.log10(sinr_opt.cpu().numpy())}")
    print(f"      Sum capacity: {capacity_opt:.6f} bits/s/Hz")

    # Store results
    results['power_db'].append(power_db)
    results['power_linear'].append(power_w)
    results['equal_power_capacity'].append(capacity_equal)
    results['optimized_capacity'].append(capacity_opt)
    results['equal_power_sinr'].append(sinr_equal.cpu().numpy())
    results['optimized_sinr'].append(sinr_opt.cpu().numpy())
    results['optimal_phases'].append(optimal_phases.cpu().numpy())
    results['optimal_power_allocation'].append(optimal_power.cpu().numpy())

# ========== Plot Results ==========
print("\n" + "="*80)
print("PLOTTING RESULTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Capacity vs Power
ax = axes[0, 0]
ax.plot(results['power_db'], results['equal_power_capacity'], 'bo-',
        label='Fixed Phases + Equal Power', linewidth=2, markersize=8)
ax.plot(results['power_db'], results['optimized_capacity'], 'rs-',
        label='PGA + Waterfilling', linewidth=2, markersize=8)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Sum Capacity (bits/s/Hz)', fontsize=12)
ax.set_title('SIM Beamforming: Capacity vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: SINR vs Power (Equal power, fixed phases)
ax = axes[0, 1]
sinr_equal_array = np.array(results['equal_power_sinr'])
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(sinr_equal_array[:, user_idx]),
            'o-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('SINR (dB)', fontsize=12)
ax.set_title('Fixed Phases + Equal Power: SINR vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: SINR vs Power (Optimized)
ax = axes[1, 0]
sinr_opt_array = np.array(results['optimized_sinr'])
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(sinr_opt_array[:, user_idx]),
            's-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('SINR (dB)', fontsize=12)
ax.set_title('PGA + Waterfilling: SINR vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Power allocation (Optimized)
ax = axes[1, 1]
power_alloc_array = np.array(results['optimal_power_allocation'])
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(power_alloc_array[:, user_idx]*1000),
            's-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Allocated Power per User (dBm)', fontsize=12)
ax.set_title('PGA + Waterfilling: Power Allocation', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test/test_sim_power_scaling.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to: test/test_sim_power_scaling.png")
plt.show()

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nExpected behavior:")
print("  ✓ Capacity should INCREASE with power (logarithmically)")
print("  ✓ SINR should INCREASE linearly with power (in dB scale)")
print("  ✓ Optimized phases + waterfilling should outperform fixed phases")

print("\nActual results:")
capacity_increase_equal = results['equal_power_capacity'][-1] - results['equal_power_capacity'][0]
capacity_increase_opt = results['optimized_capacity'][-1] - results['optimized_capacity'][0]
power_increase_db = results['power_db'][-1] - results['power_db'][0]

print(f"  Power increase: {power_increase_db:.1f} dB")
print(f"  Capacity increase (Fixed + Equal): {capacity_increase_equal:.4f} bits/s/Hz")
print(f"  Capacity increase (PGA + WF): {capacity_increase_opt:.4f} bits/s/Hz")

if capacity_increase_opt > 1.0:
    print("  ✓ Capacity increases with power as expected!")
else:
    print("  ✗ WARNING: Capacity not increasing properly!")
    print("    → Possible issues:")
    print("       1. PGA optimizing to different solutions at each power level")
    print("       2. Waterfilling converging incorrectly")
    print("       3. Channel normalization issues")
    print("       4. Phase optimization not accounting for power scaling")

print("="*80)