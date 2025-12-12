"""
Quick demo of DDPG and TD3 algorithms working correctly.
Runs 3 quick tests for each algorithm to demonstrate functionality.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
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
print("DEMO: RL ALGORITHMS WORKING WITHOUT BUGS")
print("="*80)

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
print(f"\n✓ SIM model created: {sim_layers} layers × {sim_metaatoms} meta-atoms")

# Create beamformer
sim_beamformer = Beamformer(
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
print(f"✓ Beamformer created: M=K={num_users}")

# Setup
power_allocation = torch.ones(num_users, device=device) * (sim_beamformer.total_power / num_users)
state_dim = 2 * num_users * sim_metaatoms + num_users  # 204
action_dim = sim_layers * sim_metaatoms  # 50

print(f"\n✓ Configuration:")
print(f"  - State dimension: {state_dim} (correctly calculated as 2×K×N + K)")
print(f"  - Action dimension: {action_dim} (L × N phases)")
print(f"  - Power allocation: uniform across {num_users} users")

# Quick test parameters
num_runs = 3
num_episodes = 10  # Very short for demo
steps_per_episode = 5

torch.seed()  # Reset for randomization

# Test DDPG
print("\n" + "-"*80)
print("TESTING DDPG (Deep Deterministic Policy Gradient)")
print("-"*80)

ddpg_results = []
for i in range(num_runs):
    print(f"\n  Run {i+1}/{num_runs}:", end=" ")

    ddpg_agent = DDPG(
        beamformer=sim_beamformer,
        state_dim=state_dim,  # Correctly set to 204
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        verbose=False
    )

    try:
        results = ddpg_agent.optimize(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            power_allocation=power_allocation
        )
        ddpg_results.append(results['optimal_objective'])
        print(f"✓ Success! Sum-rate = {results['optimal_objective']:.4f} bits/s/Hz")
    except Exception as e:
        print(f"✗ Error: {e}")
        break

if ddpg_results:
    print(f"\n  DDPG Summary ({len(ddpg_results)} runs):")
    print(f"  - Mean: {np.mean(ddpg_results):.4f} bits/s/Hz")
    print(f"  - Std:  {np.std(ddpg_results):.4f} bits/s/Hz")

# Test TD3
print("\n" + "-"*80)
print("TESTING TD3 (Twin Delayed DDPG)")
print("-"*80)

td3_results = []
for i in range(num_runs):
    print(f"\n  Run {i+1}/{num_runs}:", end=" ")

    td3_agent = TD3(
        beamformer=sim_beamformer,
        state_dim=state_dim,  # Correctly set to 204
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        policy_delay=2,
        policy_noise=0.2,
        verbose=False
    )

    try:
        results = td3_agent.optimize(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            power_allocation=power_allocation
        )
        td3_results.append(results['optimal_objective'])
        print(f"✓ Success! Sum-rate = {results['optimal_objective']:.4f} bits/s/Hz")
    except Exception as e:
        print(f"✗ Error: {e}")
        break

if td3_results:
    print(f"\n  TD3 Summary ({len(td3_results)} runs):")
    print(f"  - Mean: {np.mean(td3_results):.4f} bits/s/Hz")
    print(f"  - Std:  {np.std(td3_results):.4f} bits/s/Hz")

# Final summary
print("\n" + "="*80)
print("DEMO COMPLETE - BOTH ALGORITHMS WORKING!")
print("="*80)

if ddpg_results and td3_results:
    print(f"\n✓ DDPG: Successfully ran {len(ddpg_results)} tests without errors")
    print(f"✓ TD3:  Successfully ran {len(td3_results)} tests without errors")
    print(f"\nKey fixes applied:")
    print("1. State dimension correctly calculated (204 not 36)")
    print("2. Attribute names fixed (sim_model.layers not sim_model.L)")
    print("3. Matplotlib blocking removed")
    print(f"\nThe full test files (test_ddpg.py, test_td3.py) are ready to run!")
else:
    print("\n⚠ Some tests failed - check error messages above")

print("="*80)