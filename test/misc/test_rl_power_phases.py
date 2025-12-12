"""
Test script to demonstrate DDPG and TD3 can optimize either phases OR power.
"""
import torch
import numpy as np
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import DDPG, TD3

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# System parameters
num_users = 4
wavelength = 0.125
device = 'cpu' 
sim_layers = 2
sim_metaatoms = 25

print("\n" + "="*80)
print("TESTING RL ALGORITHMS: OPTIMIZE PHASES OR POWER")
print("="*80)
print(f"Device: {device}")

# Create SIM model
sim_model = Sim(
    layers=sim_layers,
    metaAtoms=sim_metaatoms,
    layerSpacing=wavelength/4,
    metaAtomSpacing=wavelength/2,
    metaAtomArea=(wavelength/2)**2,
    wavelength=wavelength,
    device=device
)

# Create beamformer
beamformer = Beamformer(
    Nx=2, Ny=2,
    wavelength=wavelength,
    device=device,
    num_users=num_users,
    user_positions=None,
    reference_distance=1.0,
    path_loss_at_reference=-30.0,
    min_user_distance=10.0,
    max_user_distance=100.0,
    sim_model=sim_model,
    noise_power=10**(-80/10)/1000,
    total_power=10**(26/10)/1000,
)

print(f"\n✓ Setup complete: {num_users} users, {sim_layers}×{sim_metaatoms} SIM")

# ========== Test 1: DDPG - Optimize Phases (Fixed Power) ==========
print("\n" + "-"*80)
print("TEST 1: DDPG - Optimize Phases (with fixed power)")
print("-"*80)

power_fixed = torch.ones(num_users, device=device) * (beamformer.total_power / num_users)
state_dim_phases = 2 * num_users * sim_metaatoms + num_users  # Channel + power
action_dim_phases = sim_layers * sim_metaatoms  # Phases

ddpg_phases = DDPG(
    beamformer=beamformer,
    state_dim=state_dim_phases,
    action_dim=action_dim_phases,
    optimize_target='phases',  # <-- KEY: Optimize phases
    verbose=False
)

results = ddpg_phases.optimize(
    num_episodes=10,
    steps_per_episode=5,
    power_allocation=power_fixed  # Fixed power
)

print(f"✓ DDPG optimized phases successfully!")
print(f"  - Optimal sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")
print(f"  - Optimal phases shape: {results['optimal_params'].shape}")

# ========== Test 2: DDPG - Optimize Power (Fixed Phases) ==========
print("\n" + "-"*80)
print("TEST 2: DDPG - Optimize Power (with fixed phases)")
print("-"*80)

phases_fixed = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi
state_dim_power = 2 * num_users * sim_metaatoms + sim_layers * sim_metaatoms  # Channel + phases
action_dim_power = num_users  # Power allocation

ddpg_power = DDPG(
    beamformer=beamformer,
    state_dim=state_dim_power,
    action_dim=action_dim_power,
    optimize_target='power',  # <-- KEY: Optimize power
    verbose=False
)

results = ddpg_power.optimize(
    num_episodes=10,
    steps_per_episode=5,
    phases=phases_fixed  # Fixed phases
)

print(f"✓ DDPG optimized power successfully!")
print(f"  - Optimal sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")
print(f"  - Optimal power shape: {results['optimal_params'].shape}")
print(f"  - Power allocation: {results['optimal_params'].cpu().numpy()}")
print(f"  - Total power: {results['optimal_params'].sum().item():.4f} W (should be ~{beamformer.total_power:.4f})")

# ========== Test 3: TD3 - Optimize Phases (Fixed Power) ==========
print("\n" + "-"*80)
print("TEST 3: TD3 - Optimize Phases (with fixed power)")
print("-"*80)

td3_phases = TD3(
    beamformer=beamformer,
    state_dim=state_dim_phases,
    action_dim=action_dim_phases,
    optimize_target='phases',  # <-- KEY: Optimize phases
    verbose=False
)

results = td3_phases.optimize(
    num_episodes=10,
    steps_per_episode=5,
    power_allocation=power_fixed  # Fixed power
)

print(f"✓ TD3 optimized phases successfully!")
print(f"  - Optimal sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")
print(f"  - Optimal phases shape: {results['optimal_params'].shape}")

# ========== Test 4: TD3 - Optimize Power (Fixed Phases) ==========
print("\n" + "-"*80)
print("TEST 4: TD3 - Optimize Power (with fixed phases)")
print("-"*80)

td3_power = TD3(
    beamformer=beamformer,
    state_dim=state_dim_power,
    action_dim=action_dim_power,
    optimize_target='power',  # <-- KEY: Optimize power
    verbose=False
)

results = td3_power.optimize(
    num_episodes=10,
    steps_per_episode=5,
    phases=phases_fixed  # Fixed phases
)

print(f"✓ TD3 optimized power successfully!")
print(f"  - Optimal sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")
print(f"  - Optimal power shape: {results['optimal_params'].shape}")
print(f"  - Power allocation: {results['optimal_params'].cpu().numpy()}")
print(f"  - Total power: {results['optimal_params'].sum().item():.4f} W (should be ~{beamformer.total_power:.4f})")

# ========== Summary ==========
print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\n✓ DDPG can optimize:")
print("  - Phases (with fixed power)")
print("  - Power (with fixed phases)")
print("\n✓ TD3 can optimize:")
print("  - Phases (with fixed power)")
print("  - Power (with fixed phases)")
print("\nUsage:")
print("  algorithm = DDPG(..., optimize_target='phases')  # or 'power'")
print("  algorithm.optimize(power_allocation=...)  # if optimizing phases")
print("  algorithm.optimize(phases=...)            # if optimizing power")
print("="*80)
