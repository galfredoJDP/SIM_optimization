"""
Test SIM with LONG PGA optimization (5000 iterations) and verify power scaling.

Based on test_sim_aggressive_pga.py results showing that longer optimization
achieves much better interference cancellation.
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

# Channel parameters (matching paper)
min_user_distance = 1.0
max_user_distance = 7.0
path_loss_at_reference = -30.0
reference_distance = 1.0

# Noise power (fixed)
noise_power = 10**(-80/10) / 1000  # Watts

# Power sweep
power_values_db = np.array([10, 15, 20, 25, 30])  # dBm
power_values_linear = 10**(power_values_db/10) / 1000  # Watts

# Reference power for optimization
reference_power_db = 20  # dBm
reference_power_w = 10**(reference_power_db/10) / 1000

print("\n" + "="*80)
print("SIM LONG PGA POWER SCALING TEST")
print("="*80)
print(f"Strategy: Use 5000 PGA iterations for better phase optimization")
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
    total_power=reference_power_w,
)

print(f"   Channel H norm: {torch.norm(sim_beamformer.H).item():.6f}")
print(f"   Channel A norm: {torch.norm(sim_beamformer.A).item():.6f}")
print(f"   Channel scaling factor: {sim_beamformer.channel_scale:.6f}")

# ========== Optimize Phases at Reference Power ==========
print(f"\n3. Optimizing phases at reference power ({reference_power_db} dBm) with LONG PGA...")
print("   Using 5000 iterations with 3 random restarts...")

# Equal power allocation for optimization
equal_power_ref = torch.ones(num_users, device=device) * (reference_power_w / num_users)

# Multiple restarts to find best solution
num_restarts = 3
best_result = None
best_sumrate = -np.inf

for restart_idx in range(num_restarts):
    print(f"\n   Restart {restart_idx + 1}/{num_restarts}...")

    # Initialize random phases
    init_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

    # Run PGA with LONG optimization
    optimizer = PGA(
        beamformer=sim_beamformer,
        objective_fn=lambda phases: sim_beamformer.compute_sum_rate(
            phases=phases,
            power_allocation=equal_power_ref
        ),
        learning_rate=0.001,
        max_iterations=5000,
        verbose=False,
    )

    pga_results = optimizer.optimize(init_phases)
    optimal_phases = pga_results['optimal_params']
    sumrate_at_ref = pga_results['optimal_objective']

    # Compute effective channel
    H_eff_opt = sim_beamformer.compute_end_to_end_channel(optimal_phases)
    H_eff_diag = torch.abs(torch.diag(H_eff_opt))
    H_eff_offdiag_max = torch.max(torch.abs(H_eff_opt - torch.diag(torch.diag(H_eff_opt)))).item()
    diag_min = torch.min(H_eff_diag).item()
    interference_ratio = H_eff_offdiag_max / diag_min if diag_min > 0 else np.inf

    print(f"      Sum-rate: {sumrate_at_ref:.4f} bits/s/Hz")
    print(f"      Interference ratio: {interference_ratio:.4f}")

    if sumrate_at_ref > best_sumrate:
        best_sumrate = sumrate_at_ref
        best_result = {
            'phases': optimal_phases,
            'H_eff': H_eff_opt,
            'sumrate': sumrate_at_ref,
            'interference_ratio': interference_ratio
        }

optimal_phases = best_result['phases']
H_eff_opt = best_result['H_eff']

print(f"\n   Best sum-rate at {reference_power_db} dBm: {best_result['sumrate']:.4f} bits/s/Hz")
print(f"   Best interference ratio: {best_result['interference_ratio']:.4f}")
print(f"\n   Optimized H_eff:")
print(f"      H_eff shape: {H_eff_opt.shape}")
print(f"      H_eff norm: {torch.norm(H_eff_opt).item():.6f}")
print(f"      H_eff diagonal: {torch.abs(torch.diag(H_eff_opt)).cpu().numpy()}")
print(f"      H_eff off-diagonal max: {torch.max(torch.abs(H_eff_opt - torch.diag(torch.diag(H_eff_opt)))).item():.6e}")

# ========== Test Fixed Phases Across Power Levels ==========
print("\n" + "="*80)
print("TESTING OPTIMIZED PHASES ACROSS POWER LEVELS")
print("="*80)

results = {
    'power_db': [],
    'power_linear': [],
    'equal_power_sinr': [],
    'equal_power_capacity': [],
    'waterfilling_sinr': [],
    'waterfilling_capacity': [],
    'waterfilling_power_allocation': []
}

for power_idx, power_w in enumerate(power_values_linear):
    power_db = power_values_db[power_idx]

    print(f"\n--- Power: {power_db:.1f} dBm ({power_w:.4e} W) ---")

    # Update beamformer power
    sim_beamformer.total_power = power_w

    # ========== Equal Power Allocation ==========
    equal_power = torch.ones(num_users, device=device) * (power_w / num_users)

    sinr_equal = sim_beamformer.compute_sinr(
        phases=optimal_phases,
        power_allocation=equal_power,
        digital_beamforming_weights=None,
        debug=(power_idx == 0)  # Debug first power level only
    )

    capacity_equal = torch.sum(torch.log2(1 + sinr_equal)).item()

    print(f"  Equal Power:")
    print(f"    Power per user: {equal_power[0].item():.4e} W")
    print(f"    SINR: {sinr_equal.cpu().numpy()}")
    print(f"    SINR (dB): {10*np.log10(sinr_equal.cpu().numpy())}")
    print(f"    Sum capacity: {capacity_equal:.6f} bits/s/Hz")

    # ========== Waterfilling Power Allocation ==========
    waterfilling = WaterFilling(
        H_eff=H_eff_opt,
        noise_power=noise_power,
        total_power=power_w,
        max_iterations=200,
        tolerance=1e-6,
        verbose=False,
        device=device
    )

    wf_results = waterfilling.optimize()
    optimal_power = wf_results['optimal_power']

    sinr_wf = sim_beamformer.compute_sinr(
        phases=optimal_phases,
        power_allocation=optimal_power,
        digital_beamforming_weights=None
    )

    capacity_wf = torch.sum(torch.log2(1 + sinr_wf)).item()

    print(f"\n  Waterfilling:")
    print(f"    Power allocation: {optimal_power.cpu().numpy()}")
    print(f"    SINR: {sinr_wf.cpu().numpy()}")
    print(f"    SINR (dB): {10*np.log10(sinr_wf.cpu().numpy())}")
    print(f"    Sum capacity: {capacity_wf:.6f} bits/s/Hz")

    # Store results
    results['power_db'].append(power_db)
    results['power_linear'].append(power_w)
    results['equal_power_sinr'].append(sinr_equal.cpu().numpy())
    results['equal_power_capacity'].append(capacity_equal)
    results['waterfilling_sinr'].append(sinr_wf.cpu().numpy())
    results['waterfilling_capacity'].append(capacity_wf)
    results['waterfilling_power_allocation'].append(optimal_power.cpu().numpy())

# ========== Analysis ==========
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\nCapacity vs Power:")
for i, power_db in enumerate(results['power_db']):
    print(f"  {power_db:5.1f} dBm: Equal={results['equal_power_capacity'][i]:8.4f}, WF={results['waterfilling_capacity'][i]:8.4f} bits/s/Hz")

capacity_increase_equal = results['equal_power_capacity'][-1] - results['equal_power_capacity'][0]
capacity_increase_wf = results['waterfilling_capacity'][-1] - results['waterfilling_capacity'][0]
power_increase_db = results['power_db'][-1] - results['power_db'][0]

print(f"\nCapacity increase over {power_increase_db:.1f} dB:")
print(f"  Equal power: {capacity_increase_equal:.4f} bits/s/Hz")
print(f"  Waterfilling: {capacity_increase_wf:.4f} bits/s/Hz")

# Expected vs actual
print(f"\nTheoretical expectation:")
print(f"  For {power_increase_db:.1f} dB power increase → ~{power_increase_db * 0.3:.1f}-{power_increase_db * 0.5:.1f} bits/s/Hz increase")
print(f"  (depending on SNR regime)")

if capacity_increase_wf > power_increase_db * 0.2:
    print(f"\n✓ Capacity IS increasing with power properly!")
    print(f"  Waterfilling gain: {capacity_increase_wf:.2f} bits/s/Hz over {power_increase_db:.0f} dB")
else:
    print(f"\n✗ Capacity still not increasing properly")

# ========== Plot Results ==========
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Capacity vs Power
ax = axes[0, 0]
ax.plot(results['power_db'], results['equal_power_capacity'], 'bo-',
        label='Equal Power', linewidth=2, markersize=8)
ax.plot(results['power_db'], results['waterfilling_capacity'], 'rs-',
        label='Waterfilling', linewidth=2, markersize=8)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Sum Capacity (bits/s/Hz)', fontsize=12)
ax.set_title('SIM (Long PGA 5000 iter): Capacity vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: SINR vs Power (Equal power)
ax = axes[0, 1]
sinr_equal_array = np.array(results['equal_power_sinr'])
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(sinr_equal_array[:, user_idx]),
            'o-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('SINR (dB)', fontsize=12)
ax.set_title('Equal Power: SINR vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: SINR vs Power (Waterfilling)
ax = axes[1, 0]
sinr_wf_array = np.array(results['waterfilling_sinr'])
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(sinr_wf_array[:, user_idx]),
            's-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('SINR (dB)', fontsize=12)
ax.set_title('Waterfilling: SINR vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Power allocation (Waterfilling)
ax = axes[1, 1]
power_alloc_array = np.array(results['waterfilling_power_allocation'])
for user_idx in range(num_users):
    # Avoid log of zero
    power_alloc_dbm = np.where(power_alloc_array[:, user_idx] > 0,
                                10*np.log10(power_alloc_array[:, user_idx]*1000),
                                -np.inf)
    ax.plot(results['power_db'], power_alloc_dbm,
            's-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Allocated Power per User (dBm)', fontsize=12)
ax.set_title('Waterfilling: Power Allocation', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test/test_sim_long_pga_power_scaling.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to: test/test_sim_long_pga_power_scaling.png")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)