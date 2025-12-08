"""
Test script for TD3 algorithm on SIM-based beamforming.
Follows the same flow as main.py but uses Twin Delayed DDPG.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import TD3

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Parameters ==========

# System parameters
num_users = 4
wavelength = 0.125  # meters
device = 'cpu'

# Antenna array (for beamformer)
Nx_antenna = 2
Ny_antenna = 2
num_antennas = Nx_antenna * Ny_antenna

# SIM parameters
sim_layers = 2
sim_metaatoms = 25
sim_layer_spacing = wavelength/4  # meters
sim_metaatom_spacing = wavelength/2  # meters
sim_metaatom_area = (sim_metaatom_spacing)**2  # m^2

# Channel parameters (CLT mode - no user positions)
min_user_distance = 10.0  # meters
max_user_distance = 100.0  # meters
path_loss_at_reference = -30.0  # dB
reference_distance = 1.0  # meters

# Power parameters
noise_power = 10**(-80/10)/1000  # Watts
total_power = 10**(26/10)/1000  # Watts

print("\n" + "="*80)
print("TD3 TESTING FOR SIM-BASED BEAMFORMING")
print("="*80)
print(f"   Users (K): {num_users}")
print(f"   Wavelength: {wavelength} m")
print(f"   Digital antennas (M): {Nx_antenna}x{Ny_antenna} = {num_antennas}")
print(f"   SIM: {sim_layers} layers (L), {sim_metaatoms} meta-atoms per layer (N)")
print(f"   User distances: [{min_user_distance}, {max_user_distance}] m")

# ========== Create SIM Model ==========
print("\n" + "="*80)
print("CREATING SIM MODEL AND BEAMFORMER")
print("="*80)

print("\n1. Creating SIM model...")
sim_model = Sim(
    layers=sim_layers,
    metaAtoms=sim_metaatoms,
    layerSpacing=sim_layer_spacing,
    metaAtomSpacing=sim_metaatom_spacing,
    metaAtomArea=sim_metaatom_area,
    wavelength=wavelength,
    device=device
)
print(f"   SIM created: {sim_layers}L x {sim_metaatoms}N")

print("\n2. Creating SIM Beamformer (M=K=4 architecture)...")
sim_beamformer = Beamformer(
    # Transceiver params - NOTE: Using M=K=4 antennas (not 64!)
    Nx=2,  # 2x2 = 4 antennas to match K=4 users
    Ny=2,
    wavelength=wavelength,
    device=device,
    # Channel params (CLT mode - no user_positions)
    num_users=num_users,
    user_positions=None,  # Will use CLT mode
    reference_distance=reference_distance,
    path_loss_at_reference=path_loss_at_reference,
    min_user_distance=min_user_distance,
    max_user_distance=max_user_distance,
    # SIM
    sim_model=sim_model,
    # System params
    noise_power=noise_power,
    total_power=total_power,
)
print(f"   Beamformer created with M=K={num_antennas}")

# ========== Setup TD3 ==========
print("\n" + "="*80)
print("CONFIGURING TD3 ALGORITHM")
print("="*80)

# Power allocation - uniform for now
power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
print(f"\n1. Power allocation: {power_allocation}")

# State and action dimensions for RL
# State: Channel matrix (H) flattened (real + imag parts) + power allocation
# H is K x N (users x meta-atoms), so state = 2*K*N + K
state_dim = 2 * num_users * sim_metaatoms + num_users  # 2*K*N + K = 2*4*25 + 4 = 204
action_dim = sim_layers * sim_metaatoms  # L * N = 2 * 25 = 50

print(f"\n2. RL dimensions:")
print(f"   State dimension:  {state_dim} (channel real + imag + power)")
print(f"   Action dimension: {action_dim} (SIM phases: {sim_layers} x {sim_metaatoms})")

# Create TD3 agent
print(f"\n3. Creating TD3 agent...")
td3_agent = TD3(
    beamformer=sim_beamformer,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=256,
    actor_lr=1e-4,
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    buffer_size=100000,
    batch_size=64,
    verbose=True
)
print(f"   TD3 agent created")
print(f"   Actor network: {state_dim} -> 256 -> 256 -> {action_dim}")
print(f"   Twin Critic networks: ({state_dim} + {action_dim}) -> 256 -> 256 -> 1 (x2)")
print(f"   TD3 improvements: Twin critics + Delayed updates + Target smoothing")

# ========== Run TD3 Optimization Multiple Times ==========
print("\n" + "="*80)
print("RUNNING TD3 OPTIMIZATION (50 RUNS)")
print("="*80)

sumrate = []
torch.seed()  # Reset seed to allow randomization

num_runs = 50
num_episodes = 100  # Fewer episodes per run for testing
steps_per_episode = 20  # Fewer steps per episode

print(f"\nConfiguration:")
print(f"   Number of runs: {num_runs}")
print(f"   Episodes per run: {num_episodes}")
print(f"   Steps per episode: {steps_per_episode}")
print()

for i in range(num_runs):
    print(f"\n{'='*80}")
    print(f"Optimization run {i+1}/{num_runs}")
    print(f"{'='*80}")

    # Reset TD3 agent for each run (fresh networks)
    td3_agent = TD3(
        beamformer=sim_beamformer,
        state_dim=state_dim,  # Now correctly set to 204
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        buffer_size=100000,
        batch_size=64,
        verbose=False  # Set to False for cleaner output during multiple runs
    )

    # Run optimization
    results = td3_agent.optimize(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        power_allocation=power_allocation
    )

    sumrate.append(results['optimal_objective'])
    print(f"Run {i+1} completed | Best sum-rate: {results['optimal_objective']:.4f} bits/s/Hz")

# ========== Plot Results ==========
print("\n" + "="*80)
print("PLOTTING RESULTS")
print("="*80)

plt.figure(figsize=(12, 6))

# Plot 1: Sum-rate across runs
plt.subplot(1, 2, 1)
plt.scatter(range(num_runs), sumrate, alpha=0.6, s=50, color='green')
plt.axhline(y=np.mean(sumrate), color='r', linestyle='--',
            label=f'Mean: {np.mean(sumrate):.4f}')
plt.xlabel('Run Number')
plt.ylabel('Sum-Rate (bits/s/Hz)')
plt.title(f'TD3: Sum-Rate across {num_runs} Runs')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Histogram
plt.subplot(1, 2, 2)
plt.hist(sumrate, bins=20, alpha=0.7, edgecolor='black', color='green')
plt.axvline(x=np.mean(sumrate), color='r', linestyle='--',
            label=f'Mean: {np.mean(sumrate):.4f}')
plt.xlabel('Sum-Rate (bits/s/Hz)')
plt.ylabel('Frequency')
plt.title('Distribution of Sum-Rates')
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

plt.tight_layout()
plt.savefig('td3_results.png', dpi=150, bbox_inches='tight')
print("\n   Results plot saved to 'td3_results.png'")

# ========== Summary Statistics ==========
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"   Mean sum-rate:   {np.mean(sumrate):.4f} bits/s/Hz")
print(f"   Std deviation:   {np.std(sumrate):.4f} bits/s/Hz")
print(f"   Min sum-rate:    {np.min(sumrate):.4f} bits/s/Hz")
print(f"   Max sum-rate:    {np.max(sumrate):.4f} bits/s/Hz")
print("="*80)

# plt.show()  # Commented out to avoid blocking
