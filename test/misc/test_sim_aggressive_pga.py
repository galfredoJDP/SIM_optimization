"""
Test SIM with more aggressive PGA optimization to see if we can achieve better interference cancellation.

Strategy:
- Try different learning rates
- More iterations
- Multiple random restarts
- Check if we can achieve more diagonal H_eff
"""

import torch
import numpy as np
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA

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

# Power
power_db = 20  # dBm
power_w = 10**(power_db/10) / 1000
noise_power = 10**(-80/10) / 1000  # Watts

print("\n" + "="*80)
print("AGGRESSIVE PGA OPTIMIZATION TEST")
print("="*80)
print(f"Goal: Achieve diagonal H_eff with minimal off-diagonal interference")
print(f"Users (K): {num_users}")
print(f"Antennas (M): {num_antennas}")
print(f"SIM: {sim_layers} layers, {sim_metaatoms} meta-atoms per layer")
print(f"Power: {power_db} dBm")
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
    total_power=power_w,
)

print(f"   Channel H norm: {torch.norm(sim_beamformer.H).item():.6f}")
print(f"   Channel A norm: {torch.norm(sim_beamformer.A).item():.6f}")
print(f"   Channel scaling factor: {sim_beamformer.channel_scale:.6f}")

# ========== Test Different PGA Configurations ==========
print("\n" + "="*80)
print("TESTING DIFFERENT PGA CONFIGURATIONS")
print("="*80)

equal_power = torch.ones(num_users, device=device) * (power_w / num_users)

configs = [
    {"lr": 0.001, "iterations": 1000, "name": "Baseline (lr=0.001, iter=1000)"},
    {"lr": 0.01, "iterations": 1000, "name": "Higher LR (lr=0.01, iter=1000)"},
    {"lr": 0.001, "iterations": 5000, "name": "More iterations (lr=0.001, iter=5000)"},
    {"lr": 0.01, "iterations": 5000, "name": "Aggressive (lr=0.01, iter=5000)"},
]

best_result = None
best_score = -np.inf

for config in configs:
    print(f"\n{'='*80}")
    print(f"{config['name']}")
    print(f"{'='*80}")

    # Try multiple random restarts
    num_restarts = 3
    restart_results = []

    for restart_idx in range(num_restarts):
        init_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

        optimizer = PGA(
            beamformer=sim_beamformer,
            objective_fn=lambda phases: sim_beamformer.compute_sum_rate(
                phases=phases,
                power_allocation=equal_power
            ),
            learning_rate=config["lr"],
            max_iterations=config["iterations"],
            verbose=False,
        )

        pga_results = optimizer.optimize(init_phases)
        optimal_phases = pga_results['optimal_params']
        sumrate = pga_results['optimal_objective']

        # Compute H_eff and analyze
        H_eff = sim_beamformer.compute_end_to_end_channel(optimal_phases)
        H_eff_diag = torch.abs(torch.diag(H_eff))
        H_eff_offdiag_max = torch.max(torch.abs(H_eff - torch.diag(torch.diag(H_eff)))).item()
        diag_min = torch.min(H_eff_diag).item()

        # Compute interference ratio
        interference_ratio = H_eff_offdiag_max / diag_min if diag_min > 0 else np.inf

        restart_results.append({
            'sumrate': sumrate,
            'phases': optimal_phases,
            'H_eff': H_eff,
            'diag_min': diag_min,
            'offdiag_max': H_eff_offdiag_max,
            'interference_ratio': interference_ratio
        })

        print(f"  Restart {restart_idx+1}: Sum-rate={sumrate:.6f}, "
              f"Diag_min={diag_min:.6e}, Offdiag_max={H_eff_offdiag_max:.6e}, "
              f"Ratio={interference_ratio:.4f}")

    # Select best result from restarts
    best_restart = max(restart_results, key=lambda x: x['sumrate'])

    print(f"\n  Best from {num_restarts} restarts:")
    print(f"    Sum-rate: {best_restart['sumrate']:.6f} bits/s/Hz")
    print(f"    H_eff diagonal: {torch.abs(torch.diag(best_restart['H_eff'])).cpu().numpy()}")
    print(f"    H_eff off-diagonal max: {best_restart['offdiag_max']:.6e}")
    print(f"    Interference ratio (offdiag/diag): {best_restart['interference_ratio']:.4f}")

    if best_restart['sumrate'] > best_score:
        best_score = best_restart['sumrate']
        best_result = best_restart
        best_config = config['name']

# ========== Best Result Analysis ==========
print("\n" + "="*80)
print("BEST RESULT")
print("="*80)
print(f"Configuration: {best_config}")
print(f"Sum-rate: {best_result['sumrate']:.6f} bits/s/Hz")
print(f"\nH_eff characteristics:")
print(f"  Diagonal: {torch.abs(torch.diag(best_result['H_eff'])).cpu().numpy()}")
print(f"  Off-diagonal max: {best_result['offdiag_max']:.6e}")
print(f"  Interference ratio: {best_result['interference_ratio']:.4f}")

# Compute SINR with best phases
sinr = sim_beamformer.compute_sinr(
    phases=best_result['phases'],
    power_allocation=equal_power,
    digital_beamforming_weights=None,
    debug=True
)

print(f"\nSINR: {sinr.cpu().numpy()}")
print(f"SINR (dB): {10*np.log10(sinr.cpu().numpy())}")

# ========== Target Analysis ==========
print("\n" + "="*80)
print("TARGET FOR GOOD PERFORMANCE")
print("="*80)
print("For capacity to scale properly with power:")
print("  ✓ We need NOISE-limited regime, not interference-limited")
print("  ✓ Target: interference_ratio < 0.1 (offdiag < 10% of diag)")
print("  ✓ Target: SINR dominated by noise_power, not interference_power")
print(f"\nCurrent:")
print(f"  Interference ratio: {best_result['interference_ratio']:.4f}")
if best_result['interference_ratio'] < 0.1:
    print("  ✓ GOOD: Low interference, should be noise-limited")
else:
    print(f"  ✗ BAD: High interference ({best_result['interference_ratio']:.1f}x), interference-limited")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)