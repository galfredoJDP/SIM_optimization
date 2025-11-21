"""
Test script to analyze the impact of meta-atom area (relative to wavelength)
on channel characteristics and sum-rate performance.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.sim import Sim
from simpy.beamformer import Beamformer

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("Meta-Atom Area vs Wavelength Analysis")
print("="*70)

# ========== Fixed Parameters ==========
wavelength = 0.125  # meters (2.4 GHz)
num_users = 4
device = 'cpu'

# Antenna array
Nx_antenna = 8
Ny_antenna = 8

# SIM parameters
sim_layers = 2
sim_metaatoms = 25
sim_layer_spacing = 0.1  # meters
sim_metaatom_spacing = 0.05  # meters

# Channel parameters
min_user_distance = 10.0
max_user_distance = 100.0
path_loss_at_reference = -30.0
reference_distance = 1.0

# Power parameters
noise_power = 1e-10
total_power = 1.0

print(f"\nFixed Parameters:")
print(f"  Wavelength (λ): {wavelength} m")
print(f"  Frequency: {3e8/wavelength/1e9:.2f} GHz")
print(f"  SIM: {sim_layers}L × {sim_metaatoms}N")
print(f"  Users: {num_users}")
print(f"  Meta-atom spacing: {sim_metaatom_spacing} m = {sim_metaatom_spacing/wavelength:.3f}λ")

# ========== Vary Meta-Atom Area ==========
print("\n" + "="*70)
print("Varying Meta-Atom Area")
print("="*70)

# Define meta-atom sizes relative to wavelength
# Typical range: λ/20 to λ/2
lambda_fractions = np.array([1/20, 1/15, 1/10, 1/8, 1/5, 1/4, 1/3, 1/2])
metaatom_sizes = lambda_fractions * wavelength  # in meters
metaatom_areas = metaatom_sizes ** 2  # Assuming square meta-atoms

print(f"\nTesting {len(lambda_fractions)} different meta-atom sizes:")
print(f"  Range: λ/{int(1/lambda_fractions[0])} to λ/{int(1/lambda_fractions[-1])}")
print(f"  Size range: {metaatom_sizes[0]*1000:.2f} mm to {metaatom_sizes[-1]*1000:.2f} mm")
print(f"  Area range: {metaatom_areas[0]*1e6:.4f} cm² to {metaatom_areas[-1]*1e6:.4f} cm²")

# Storage for results
channel_power_avg = []
channel_power_std = []
channel_frobenius_norm = []
sum_rates = []

print("\n" + "-"*70)
print(f"{'Size (λ)':<12} {'Size (mm)':<12} {'Area (cm²)':<12} {'H Power':<12} {'H Norm':<12} {'Sum-Rate':<12}")
print("-"*70)

for idx, (lam_frac, size, area) in enumerate(zip(lambda_fractions, metaatom_sizes, metaatom_areas)):

    # Create SIM with this meta-atom area
    sim_model = Sim(
        layers=sim_layers,
        metaAtoms=sim_metaatoms,
        layerSpacing=sim_layer_spacing,
        metaAtomSpacing=sim_metaatom_spacing,
        metaAtomArea=area,
        wavelength=wavelength,
        device=device
    )

    # Create beamformer
    beamformer = Beamformer(
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
        total_power=total_power,
        use_nearfield_user_channel=False
    )

    # Analyze channel H (SIM → Users)
    H = beamformer.H  # (K, N) complex matrix

    # Channel power per user
    H_power_per_user = torch.abs(H) ** 2  # Power per element
    H_power_per_user_sum = H_power_per_user.sum(dim=1)  # Sum over antennas

    avg_power = H_power_per_user_sum.mean().item()
    std_power = H_power_per_user_sum.std().item()

    # Frobenius norm of H (total channel power)
    H_frobenius = torch.norm(H, p='fro').item()

    # Compute sum-rate with random phases (just for comparison)
    random_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi
    power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
    sum_rate = beamformer.compute_sum_rate(random_phases, power_allocation).item()

    # Store results
    channel_power_avg.append(avg_power)
    channel_power_std.append(std_power)
    channel_frobenius_norm.append(H_frobenius)
    sum_rates.append(sum_rate)

    print(f"λ/{int(1/lam_frac):<10} {size*1000:>10.2f} {area*1e4:>10.4f} {avg_power:>10.4e} {H_frobenius:>10.4e} {sum_rate:>10.4f}")

print("-"*70)

# ========== Analysis and Plotting ==========
print("\n" + "="*70)
print("Analysis")
print("="*70)

# Convert to numpy
lambda_fractions = np.array(lambda_fractions)
metaatom_sizes_mm = metaatom_sizes * 1000
metaatom_areas_cm2 = metaatom_areas * 1e4
channel_power_avg = np.array(channel_power_avg)
channel_power_std = np.array(channel_power_std)
channel_frobenius_norm = np.array(channel_frobenius_norm)
sum_rates = np.array(sum_rates)

print(f"\nChannel Power Scaling:")
print(f"  Min power: {channel_power_avg.min():.4e} (at λ/{int(1/lambda_fractions[np.argmin(channel_power_avg)])})")
print(f"  Max power: {channel_power_avg.max():.4e} (at λ/{int(1/lambda_fractions[np.argmax(channel_power_avg)])})")
print(f"  Ratio: {channel_power_avg.max() / channel_power_avg.min():.2f}x")

print(f"\nSum-Rate (with random phases):")
print(f"  Min: {sum_rates.min():.4f} bits/s/Hz (at λ/{int(1/lambda_fractions[np.argmin(sum_rates)])})")
print(f"  Max: {sum_rates.max():.4f} bits/s/Hz (at λ/{int(1/lambda_fractions[np.argmax(sum_rates)])})")
print(f"  Improvement: {(sum_rates.max() - sum_rates.min())/sum_rates.min()*100:.1f}%")

print(f"\nTheoretical Scaling:")
print(f"  Area scales as (dx*dy), so channel amplitude scales linearly with dx*dy")
print(f"  Channel power scales as (dx*dy)²")
print(f"  Expected power ratio: {(metaatom_areas[-1]/metaatom_areas[0]):.2f}x")
print(f"  Observed power ratio: {(channel_power_avg[-1]/channel_power_avg[0]):.2f}x")

# ========== Plotting ==========
print("\n" + "="*70)
print("Generating Plots...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Meta-Atom Area Impact on Channel and Performance', fontsize=14, fontweight='bold')

# Plot 1: Channel Power vs Meta-Atom Size
ax1 = axes[0, 0]
ax1.plot(lambda_fractions, channel_power_avg, 'o-', linewidth=2, markersize=8, label='Avg Power per User')
ax1.fill_between(lambda_fractions,
                  channel_power_avg - channel_power_std,
                  channel_power_avg + channel_power_std,
                  alpha=0.3, label='±1 std')
ax1.set_xlabel('Meta-Atom Size (fraction of λ)', fontsize=11)
ax1.set_ylabel('Channel Power (average per user)', fontsize=11)
ax1.set_title('Channel Power vs Meta-Atom Size', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

# Add custom x-tick labels
xticks = lambda_fractions
xticklabels = [f'λ/{int(1/lf)}' for lf in lambda_fractions]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels, rotation=45)

# Plot 2: Frobenius Norm vs Area
ax2 = axes[0, 1]
ax2.plot(metaatom_areas_cm2, channel_frobenius_norm, 's-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Meta-Atom Area (cm²)', fontsize=11)
ax2.set_ylabel('Frobenius Norm ||H||_F', fontsize=11)
ax2.set_title('Total Channel Power vs Meta-Atom Area', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

# Plot 3: Sum-Rate vs Meta-Atom Size
ax3 = axes[1, 0]
ax3.plot(lambda_fractions, sum_rates, '^-', linewidth=2, markersize=8, color='red')
ax3.set_xlabel('Meta-Atom Size (fraction of λ)', fontsize=11)
ax3.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=11)
ax3.set_title('Sum-Rate vs Meta-Atom Size (Random Phases)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')

# Add custom x-tick labels
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticklabels, rotation=45)

# Plot 4: Normalized comparison
ax4 = axes[1, 1]
# Normalize to [0, 1] for comparison
power_norm = (channel_power_avg - channel_power_avg.min()) / (channel_power_avg.max() - channel_power_avg.min())
rate_norm = (sum_rates - sum_rates.min()) / (sum_rates.max() - sum_rates.min())

ax4.plot(lambda_fractions, power_norm, 'o-', linewidth=2, markersize=8, label='Channel Power (normalized)')
ax4.plot(lambda_fractions, rate_norm, '^-', linewidth=2, markersize=8, label='Sum-Rate (normalized)')
ax4.set_xlabel('Meta-Atom Size (fraction of λ)', fontsize=11)
ax4.set_ylabel('Normalized Value', fontsize=11)
ax4.set_title('Normalized Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xscale('log')

# Add custom x-tick labels
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticklabels, rotation=45)

plt.tight_layout()
plt.savefig('metaatom_area_impact.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: metaatom_area_impact.png")

# ========== Recommendations ==========
print("\n" + "="*70)
print("Recommendations")
print("="*70)

best_idx = np.argmax(sum_rates)
best_size = lambda_fractions[best_idx]
best_area = metaatom_areas[best_idx]

print(f"\nBased on this analysis (with random phases):")
print(f"  Best size: λ/{int(1/best_size)} = {metaatom_sizes[best_idx]*1000:.2f} mm")
print(f"  Best area: {best_area*1e4:.4f} cm²")
print(f"  Sum-rate: {sum_rates[best_idx]:.4f} bits/s/Hz")

print(f"\nCurrent implementation uses:")
print(f"  Size: {0.01*1000:.2f} mm = {0.01/wavelength:.3f}λ = λ/{wavelength/0.01:.1f}")
print(f"  Area: {0.01**2 * 1e4:.4f} cm²")

print(f"\nKey Insights:")
print(f"  • Larger meta-atoms → stronger channel → higher sum-rate")
print(f"  • BUT: meta-atom spacing ({sim_metaatom_spacing/wavelength:.3f}λ) limits size")
print(f"  • Typical range: λ/10 to λ/5 is reasonable")
print(f"  • Trade-off: size vs spacing vs mutual coupling")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
