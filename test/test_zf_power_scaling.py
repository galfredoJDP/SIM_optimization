"""
Test script to verify ZF beamforming capacity scales correctly with transmit power.

This test isolates the ZF beamforming and waterfilling to diagnose why capacity
might not be increasing with power.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.beamformer import Beamformer
from simpy.algorithm import WaterFilling

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Parameters ==========
num_users = 4
num_antennas = 4  # M = K for square system
wavelength = 0.125  # meters
device = 'cpu'

# Channel parameters
min_user_distance = 10.0
max_user_distance = 100.0
path_loss_at_reference = -30.0
reference_distance = 1.0

# Noise power (fixed)
noise_power = 10**(-80/10) / 1000  # Watts
print(f"Noise power: {noise_power:.4e} W ({10*np.log10(noise_power*1000):.2f} dBm)")

# Power sweep
power_values_db = np.array([10, 15, 20, 25, 30, 35])  # dBm
power_values_linear = 10**(power_values_db/10) / 1000  # Watts

print("\n" + "="*80)
print("ZF BEAMFORMING POWER SCALING TEST")
print("="*80)
print(f"Users (K): {num_users}")
print(f"Antennas (M): {num_antennas}")
print(f"Noise power: {noise_power:.4e} W")
print("="*80)

# ========== Create Digital Beamformer ==========
beamformer = Beamformer(
    Nx=2,  # 2x2 = 4 antennas
    Ny=2,
    wavelength=wavelength,
    device=device,
    num_users=num_users,
    user_positions=None,  # CLT mode
    reference_distance=reference_distance,
    path_loss_at_reference=path_loss_at_reference,
    min_user_distance=min_user_distance,
    max_user_distance=max_user_distance,
    noise_power=noise_power,
    total_power=1.0,  # Will be updated in loop
)

print(f"\n1. Channel H shape: {beamformer.H.shape}")
print(f"   Channel norm: {torch.norm(beamformer.H).item():.6f}")

# ========== Compute ZF Weights (once) ==========
print("\n2. Computing ZF weights...")
zf_weights = beamformer.compute_zf_weights(beamformer.H)
print(f"   ZF weights shape: {zf_weights.shape}")
print(f"   ZF weights norm: {torch.norm(zf_weights).item():.6f}")

# Compute effective channel after ZF
H_eff = beamformer.H @ zf_weights
print(f"\n3. Effective channel H_eff = H @ W:")
print(f"   H_eff shape: {H_eff.shape}")
print(f"   H_eff diagonal: {torch.abs(torch.diag(H_eff)).cpu().numpy()}")
print(f"   H_eff off-diagonal max: {torch.max(torch.abs(H_eff - torch.diag(torch.diag(H_eff)))).item():.6e}")
print(f"   ^ Should be ~0 for perfect ZF (interference eliminated)")

# Check if ZF is working (off-diagonal should be ~0)
off_diag_max = torch.max(torch.abs(H_eff - torch.diag(torch.diag(H_eff)))).item()
diag_min = torch.min(torch.abs(torch.diag(H_eff))).item()
if off_diag_max < 1e-6 * diag_min:
    print("   ✓ ZF working correctly (interference eliminated)")
else:
    print(f"   ⚠ WARNING: ZF may not be eliminating interference properly!")
    print(f"     Off-diagonal / diagonal ratio: {off_diag_max / diag_min:.6e}")

# ========== Power Sweep ==========
print("\n" + "="*80)
print("POWER SWEEP")
print("="*80)

results = {
    'power_db': [],
    'power_linear': [],
    'equal_power_capacity': [],
    'waterfilling_capacity': [],
    'equal_power_sinr': [],
    'waterfilling_sinr': [],
    'waterfilling_power_allocation': []
}

for power_idx, power_w in enumerate(power_values_linear):
    power_db = power_values_db[power_idx]

    print(f"\n--- Power: {power_db:.1f} dBm ({power_w:.4e} W) ---")

    # ========== Equal Power Allocation ==========
    equal_power = torch.ones(num_users, device=device) * (power_w / num_users)

    # Compute SINR with equal power (enable debug for first two power levels)
    sinr_equal = beamformer.compute_sinr(
        phases=None,
        power_allocation=equal_power,
        digital_beamforming_weights=zf_weights,
        debug=(power_idx < 2)
    )

    capacity_equal = torch.sum(torch.log2(1 + sinr_equal)).item()

    print(f"Equal Power Allocation:")
    print(f"  Power per user: {equal_power[0].item():.4e} W")
    print(f"  SINR: {sinr_equal.cpu().numpy()}")
    print(f"  SINR (dB): {10*np.log10(sinr_equal.cpu().numpy())}")
    print(f"  Individual rates: {torch.log2(1 + sinr_equal).cpu().numpy()}")
    print(f"  Sum capacity: {capacity_equal:.6f} bits/s/Hz")

    # ========== Waterfilling Power Allocation ==========
    waterfilling = WaterFilling(
        H_eff=H_eff,
        noise_power=noise_power,
        total_power=power_w,
        max_iterations=200,
        tolerance=1e-6,
        verbose=False,
        device=device
    )

    wf_results = waterfilling.optimize()
    optimal_power = wf_results['optimal_power']

    # Compute SINR with waterfilling
    sinr_wf = beamformer.compute_sinr(
        phases=None,
        power_allocation=optimal_power,
        digital_beamforming_weights=zf_weights
    )

    capacity_wf = torch.sum(torch.log2(1 + sinr_wf)).item()

    print(f"\nWaterfilling Power Allocation:")
    print(f"  Power allocation: {optimal_power.cpu().numpy()}")
    print(f"  Total power: {optimal_power.sum().item():.4e} W")
    print(f"  SINR: {sinr_wf.cpu().numpy()}")
    print(f"  SINR (dB): {10*np.log10(sinr_wf.cpu().numpy())}")
    print(f"  Individual rates: {torch.log2(1 + sinr_wf).cpu().numpy()}")
    print(f"  Sum capacity: {capacity_wf:.6f} bits/s/Hz")

    # Store results
    results['power_db'].append(power_db)
    results['power_linear'].append(power_w)
    results['equal_power_capacity'].append(capacity_equal)
    results['waterfilling_capacity'].append(capacity_wf)
    results['equal_power_sinr'].append(sinr_equal.cpu().numpy())
    results['waterfilling_sinr'].append(sinr_wf.cpu().numpy())
    results['waterfilling_power_allocation'].append(optimal_power.cpu().numpy())

# ========== Plot Results ==========
print("\n" + "="*80)
print("PLOTTING RESULTS")
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
ax.set_title('ZF Beamforming: Capacity vs Power', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: SINR vs Power (Equal power)
ax = axes[0, 1]
sinr_equal_array = np.array(results['equal_power_sinr'])  # Shape: (num_powers, num_users)
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
sinr_wf_array = np.array(results['waterfilling_sinr'])  # Shape: (num_powers, num_users)
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
power_alloc_array = np.array(results['waterfilling_power_allocation'])  # (num_powers, num_users)
for user_idx in range(num_users):
    ax.plot(results['power_db'], 10*np.log10(power_alloc_array[:, user_idx]*1000),
            's-', label=f'User {user_idx}', linewidth=2, markersize=6)
ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
ax.set_ylabel('Allocated Power per User (dBm)', fontsize=12)
ax.set_title('Waterfilling: Power Allocation', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test/test_zf_power_scaling.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to: test/test_zf_power_scaling.png")
plt.show()

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nExpected behavior:")
print("  ✓ Capacity should INCREASE with power (logarithmically)")
print("  ✓ SINR should INCREASE linearly with power (in dB scale)")
print("  ✓ Waterfilling should allocate more power to users with better channels")
print("  ✓ Off-diagonal elements should be ~0 (ZF eliminates interference)")

print("\nActual results:")
capacity_increase_equal = results['equal_power_capacity'][-1] - results['equal_power_capacity'][0]
capacity_increase_wf = results['waterfilling_capacity'][-1] - results['waterfilling_capacity'][0]
power_increase_db = results['power_db'][-1] - results['power_db'][0]

print(f"  Power increase: {power_increase_db:.1f} dB")
print(f"  Capacity increase (Equal): {capacity_increase_equal:.4f} bits/s/Hz")
print(f"  Capacity increase (WF): {capacity_increase_wf:.4f} bits/s/Hz")

if capacity_increase_equal > 1.0:
    print("  ✓ Capacity increases with power as expected!")
else:
    print("  ✗ WARNING: Capacity not increasing properly!")
    print("    → Check your implementation for normalization issues")

print("="*80)