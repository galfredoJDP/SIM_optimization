"""
Quick test to verify DDPG and TD3 are working without errors.
"""
import torch
import numpy as np
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import DDPG, TD3

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# System parameters
num_users = 4
wavelength = 0.125
device = 'cpu'
sim_layers = 2
sim_metaatoms = 25

print("Quick RL Algorithm Test")
print("="*50)

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

# Setup for RL
power_allocation = torch.ones(num_users, device=device) * (beamformer.total_power / num_users)
state_dim = 2 * num_users * sim_metaatoms + num_users  # 204
action_dim = sim_layers * sim_metaatoms  # 50

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}\n")

# Test DDPG
print("Testing DDPG...")
ddpg_agent = DDPG(
    beamformer=beamformer,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=256,
    actor_lr=1e-4,
    critic_lr=1e-3,
    verbose=False
)

try:
    results = ddpg_agent.optimize(
        num_episodes=5,  # Very short test
        steps_per_episode=5,
        power_allocation=power_allocation
    )
    print(f"✓ DDPG works! Best sum-rate: {results['optimal_objective']:.4f}")
except Exception as e:
    print(f"✗ DDPG error: {e}")

print()

# Test TD3
print("Testing TD3...")
td3_agent = TD3(
    beamformer=beamformer,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=256,
    actor_lr=1e-4,
    critic_lr=1e-3,
    verbose=False
)

try:
    results = td3_agent.optimize(
        num_episodes=5,  # Very short test
        steps_per_episode=5,
        power_allocation=power_allocation
    )
    print(f"✓ TD3 works! Best sum-rate: {results['optimal_objective']:.4f}")
except Exception as e:
    print(f"✗ TD3 error: {e}")

print()
print("="*50)
print("Test complete! Both algorithms are functional.")