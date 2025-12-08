"""
Comprehensive test for both DDPG and TD3 with reduced iterations for validation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import DDPG, TD3

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# System parameters
num_users = 4
wavelength = 0.125
device = 'mps'

# SIM parameters
sim_layers = 2
sim_metaatoms = 25
sim_layer_spacing = wavelength/4
sim_metaatom_spacing = wavelength/2
sim_metaatom_area = (sim_metaatom_spacing)**2

# Channel parameters
min_user_distance = 10.0
max_user_distance = 100.0
path_loss_at_reference = -30.0
reference_distance = 1.0

# Power parameters
noise_power = 10**(-80/10)/1000
total_power = 10**(26/10)/1000

print("\n" + "="*80)
print("COMPREHENSIVE RL ALGORITHMS TEST (DDPG & TD3)")
print("="*80)

# Create SIM model
sim_model = Sim(
    layers=sim_layers,
    metaAtoms=sim_metaatoms,
    layerSpacing=sim_layer_spacing,
    metaAtomSpacing=sim_metaatom_spacing,
    metaAtomArea=sim_metaatom_area,
    wavelength=wavelength,
    device=device
)

# Create beamformer
sim_beamformer = Beamformer(
    Nx=2, Ny=2,
    wavelength=wavelength,
    device=device,
    num_users=num_users,
    user_positions=None,
    reference_distance=reference_distance,
    path_loss_at_reference=path_loss_at_reference,
    min_user_distance=min_user_distance,
    max_user_distance=max_user_distance,
    sim_model=sim_model,
    noise_power=noise_power,
    total_power=total_power,
)

# Setup for RL
power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
state_dim = 2 * num_users * sim_metaatoms + num_users  # 204
action_dim = sim_layers * sim_metaatoms  # 50

print(f"\nSystem Configuration:")
print(f"   Users: {num_users}, SIM: {sim_layers}x{sim_metaatoms}")
print(f"   State dim: {state_dim}, Action dim: {action_dim}")
print(f"   Total power: {total_power:.4f} W, Noise power: {noise_power:.2e} W")

# Test parameters
num_runs = 5  # Reduced for quick test
num_episodes = 20  # Reduced episodes
steps_per_episode = 10  # Reduced steps

torch.seed()  # Allow randomization

# Test DDPG
print("\n" + "="*80)
print("TESTING DDPG ALGORITHM")
print("="*80)

ddpg_results = []
for i in range(num_runs):
    print(f"\nDDPG Run {i+1}/{num_runs}...")
    ddpg_agent = DDPG(
        beamformer=sim_beamformer,
        state_dim=state_dim,
        action_dim=action_dim,
        verbose=False
    )

    results = ddpg_agent.optimize(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        power_allocation=power_allocation
    )
    ddpg_results.append(results['optimal_objective'])
    print(f"   Sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")

print(f"\nDDPG Summary:")
print(f"   Mean: {np.mean(ddpg_results):.4f} ± {np.std(ddpg_results):.4f}")
print(f"   Min: {np.min(ddpg_results):.4f}, Max: {np.max(ddpg_results):.4f}")

# Test TD3
print("\n" + "="*80)
print("TESTING TD3 ALGORITHM")
print("="*80)

td3_results = []
for i in range(num_runs):
    print(f"\nTD3 Run {i+1}/{num_runs}...")
    td3_agent = TD3(
        beamformer=sim_beamformer,
        state_dim=state_dim,
        action_dim=action_dim,
        verbose=False
    )

    results = td3_agent.optimize(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        power_allocation=power_allocation
    )
    td3_results.append(results['optimal_objective'])
    print(f"   Sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")

print(f"\nTD3 Summary:")
print(f"   Mean: {np.mean(td3_results):.4f} ± {np.std(td3_results):.4f}")
print(f"   Min: {np.min(td3_results):.4f}, Max: {np.max(td3_results):.4f}")

# Plot comparison
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
x = range(1, num_runs+1)
plt.plot(x, ddpg_results, 'bo-', label='DDPG', markersize=8)
plt.plot(x, td3_results, 'gs-', label='TD3', markersize=8)
plt.xlabel('Run Number')
plt.ylabel('Sum-Rate (bits/s/Hz)')
plt.title('RL Algorithm Performance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
data = [ddpg_results, td3_results]
bp = plt.boxplot(data, labels=['DDPG', 'TD3'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('green')
bp['boxes'][1].set_alpha(0.5)
plt.ylabel('Sum-Rate (bits/s/Hz)')
plt.title('Distribution Comparison')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rl_comparison.png', dpi=150, bbox_inches='tight')

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print(f"✓ DDPG: Mean {np.mean(ddpg_results):.4f} bits/s/Hz")
print(f"✓ TD3:  Mean {np.mean(td3_results):.4f} bits/s/Hz")
print(f"\nComparison plot saved to 'rl_comparison.png'")
print("="*80)

# plt.show()  # Commented out to avoid blocking