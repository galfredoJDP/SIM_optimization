"""
Offline Training Script for DDPG and TD3

This script trains DDPG and TD3 agents on random channel realizations
and saves the trained weights for later use in main.py.

Usage:
    python train.py
"""

import torch
import numpy as np
import os
import copy
from datetime import datetime

from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import DDPG, TD3

# ============================================================
# CONFIGURATION - Modify these parameters
# ============================================================

# Training parameters
NUM_EPISODES = 3000              # Number of training episodes, each episode is a different channel realization 
STEPS_PER_EPISODE = 500         # Steps per episode
DEVICE = 'mps'                  # 'cpu', 'cuda', or 'mps'
SAVE_DIR = 'weights'            # Directory to save weights

# What to train
TRAIN_DDPG = True               # Train DDPG agent
TRAIN_TD3 = False                # Train TD3 agent

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================

# Set random seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def create_system(device='cpu'):
    """Create the SIM beamforming system."""
    # System parameters (same as main.py)
    frequency = 28e9
    wavelength = 3e8 / frequency

    # Antenna array
    Nx, Ny = 2, 2
    num_antennas = Nx * Ny

    # SIM parameters
    sim_layers = 2
    sim_metaatoms = 25
    layer_spacing = 0.5 * wavelength
    metaatom_spacing = 0.5 * wavelength
    metaatom_area = (0.5 * wavelength) ** 2

    # User parameters
    num_users = 4
    user_positions = None

    # Create SIM
    sim_model = Sim(
        layers=sim_layers,
        metaAtoms=sim_metaatoms,
        layerSpacing=layer_spacing,
        metaAtomSpacing=metaatom_spacing,
        metaAtomArea=metaatom_area,
        wavelength=wavelength,
        device=device
    )

    # Create beamformer
    beamformer = Beamformer(
        Nx=Nx,
        Ny=Ny,
        wavelength=wavelength,
        device=device,
        num_users=num_users,
        user_positions=user_positions,
        sim_model=sim_model,
        noise_power=1e-10,
        total_power=1.0
    )

    return beamformer, sim_model


def train_ddpg(beamformer : Beamformer, num_episodes=1000, steps_per_episode=50,
               device='cpu', verbose=True):
    """
    Train DDPG agent over many channel realizations.

    Args:
        beamformer: Beamformer instance
        num_episodes: Total training episodes
        steps_per_episode: Steps per episode
        device: Training device
        verbose: Print progress
    """
    sim_model = beamformer.sim_model
    num_users = beamformer.num_users
    sim_metaatoms = sim_model.metaAtoms
    sim_layers = sim_model.layers

    # State and action dimensions
    # State = [H.real, H.imag, power, current_phases] - includes current phases for iterative refinement
    state_dim = 2 * num_users * sim_metaatoms + num_users + (sim_layers * sim_metaatoms)
    action_dim = sim_layers * sim_metaatoms

    # Create DDPG agent
    ddpg = DDPG(
        beamformer=beamformer,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        actor_lr=1e-3,    # Standard learning rate (was 1e-5)
        critic_lr=1e-2,   # Standard learning rate (was 1e-5)
        gamma=0.99,       # Standard discount factor (was 0.85)
        tau=0.005,        # Standard soft update rate (was 0.001)
        buffer_size=100000,
        batch_size=64,
        noise_std=0.01,    # Increased exploration noise (was 0.005)
        verbose=False,  # We'll handle printing ourselves
        device=device
    )

    # Equal power allocation
    power_allocation = torch.ones(num_users, device=device) * (beamformer.total_power / num_users)

    print("=" * 70)
    print("DDPG Offline Training (Multiple Channel Realizations)")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    print()

    # Pre-generate random times for channel realizations

    best_reward = -float('inf')
    best_episode = 0

    for episode in range(num_episodes):

        beamformer.update_user_channel()
        # print("Channel Updated")
        # print(torch.diag(beamformer.H/beamformer.H_scale).cpu().numpy().round(3))

        H_norm = beamformer.H / beamformer.H_scale
        H_eff = H_norm.flatten() 

        # Initialize with random phases for this episode
        current_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

        # State now includes current phases: [H.real, H.imag, power, current_phases]
        state = torch.cat([H_eff.real, H_eff.imag, power_allocation, current_phases.flatten()]).cpu().numpy()

        episode_reward = 0
        step_rewards = []

        for step in range(steps_per_episode):
            # Agent outputs NEW phases (not adjustment, full phases)
            action = ddpg.select_action(state, add_noise=True)

            # Convert action to phases
            new_phases = torch.FloatTensor(action).reshape(sim_layers, sim_metaatoms).to(device)
            reward = beamformer.compute_sum_rate(new_phases, power_allocation).item()

            episode_reward += reward
            step_rewards.append(reward)

            # Next state includes the NEW phases (state transition!)
            next_state = torch.cat([H_eff.real, H_eff.imag, power_allocation, new_phases.flatten()]).cpu().numpy()

            # Push to replay buffer with proper state transition
            ddpg.replay_buffer.push(state, action, reward, next_state, step == steps_per_episode - 1)
            ddpg.update()

            # Update current state for next step
            state = next_state
            current_phases = new_phases

            # Print step-level info every 10 steps
            if verbose and episode % 10 == 0 and step % 10 == 0:
                print(f"  Ep {episode:4d} Step {step:3d} | Sum-rate: {reward:8.4f}")
        breakpoint()
        avg_reward = episode_reward / steps_per_episode
        max_reward_in_episode = max(step_rewards)
        min_reward_in_episode = min(step_rewards)
        ddpg.history['episode_rewards'].append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_episode = episode

        if verbose and episode % 10 == 0:
            print(f"Episode {episode:4d} | Avg: {avg_reward:8.4f} | Max: {max_reward_in_episode:8.4f} | Min: {min_reward_in_episode:8.4f} | Best: {best_reward:8.4f}")

    print(f"Best reward: {best_reward:.4f} at episode {best_episode}")

    return ddpg


def train_td3(beamformer : Beamformer, num_episodes=1000, steps_per_episode=50,
              device='cpu', verbose=True):
    """
    Train TD3 agent over many channel realizations.

    Args:
        beamformer: Beamformer instance
        num_episodes: Total training episodes
        steps_per_episode: Steps per episode
        device: Training device
        verbose: Print progress
    """
    sim_model = beamformer.sim_model
    num_users = beamformer.num_users
    sim_metaatoms = sim_model.metaAtoms
    sim_layers = sim_model.layers

    # State and action dimensions
    # State = [H.real, H.imag, power, current_phases] - includes current phases for iterative refinement
    state_dim = 2 * num_users * sim_metaatoms + num_users + (sim_layers * sim_metaatoms)
    action_dim = sim_layers * sim_metaatoms

    # Create TD3 agent
    td3 = TD3(
        beamformer=beamformer,
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
        noise_std=0.2,    # Increased exploration noise (was 0.1)
        verbose=False,  # We'll handle printing ourselves
        device=device
    )

    # Equal power allocation
    power_allocation = torch.ones(num_users, device=device) * (beamformer.total_power / num_users)

    print("=" * 70)
    print("TD3 Offline Training (Multiple Channel Realizations)")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    print()

    # Pre-generate random times for channel realizations

    best_reward = -float('inf')
    best_episode = 0

    for episode in range(num_episodes):
        # Cycle through channel realizations
        beamformer.update_user_channel()

        # Initialize with random phases for this episode
        current_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

        H_eff = beamformer.H.flatten()
        # State now includes current phases
        state = torch.cat([H_eff.real, H_eff.imag, power_allocation, current_phases.flatten()]).cpu().numpy()

        episode_reward = 0
        step_rewards = []

        for step in range(steps_per_episode):
            action = td3.select_action(state, add_noise=True)

            # New phases from action
            new_phases = torch.FloatTensor(action).reshape(sim_layers, sim_metaatoms).to(device)
            reward = beamformer.compute_sum_rate(new_phases, power_allocation).item()

            episode_reward += reward
            step_rewards.append(reward)

            # Next state includes NEW phases (proper state transition!)
            next_state = torch.cat([H_eff.real, H_eff.imag, power_allocation, new_phases.flatten()]).cpu().numpy()
            td3.replay_buffer.push(state, action, reward, next_state, step == steps_per_episode - 1)
            td3.update()

            # Update for next step
            state = next_state
            current_phases = new_phases

            # Print step-level info every 10 steps
            if verbose and episode % 10 == 0 and step % 10 == 0:
                print(f"  Ep {episode:4d} Step {step:3d} | Sum-rate: {reward:8.4f}")

        avg_reward = episode_reward / steps_per_episode
        max_reward_in_episode = max(step_rewards)
        min_reward_in_episode = min(step_rewards)
        td3.history['episode_rewards'].append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_episode = episode

        if verbose and episode % 10 == 0:
            print(f"Episode {episode:4d} | Avg: {avg_reward:8.4f} | Max: {max_reward_in_episode:8.4f} | Min: {min_reward_in_episode:8.4f} | Best: {best_reward:8.4f}")

    print(f"Best reward: {best_reward:.4f} at episode {best_episode}")

    return td3


def save_weights(agent, agent_name, save_dir='weights'):
    """Save agent weights to disk."""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save actor
    actor_path = os.path.join(save_dir, f'{agent_name}_actor_{timestamp}.pth')
    torch.save({
        'model_state_dict': agent.actor.state_dict(),
        'optimizer_state_dict': agent.actor_optimizer.state_dict(),
    }, actor_path)

    # Save actor target
    actor_target_path = os.path.join(save_dir, f'{agent_name}_actor_target_{timestamp}.pth')
    torch.save({
        'model_state_dict': agent.actor_target.state_dict(),
    }, actor_target_path)

    # Save critic(s)
    critic_path = os.path.join(save_dir, f'{agent_name}_critic_{timestamp}.pth')
    torch.save({
        'model_state_dict': agent.critic.state_dict(),
        'optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, critic_path)

    # Save critic target
    critic_target_path = os.path.join(save_dir, f'{agent_name}_critic_target_{timestamp}.pth')
    torch.save({
        'model_state_dict': agent.critic_target.state_dict(),
    }, critic_target_path)

    # Save training history
    history_path = os.path.join(save_dir, f'{agent_name}_history_{timestamp}.npy')
    np.save(history_path, agent.history)

    # Save a "latest" version for easy loading
    latest_path = os.path.join(save_dir, f'{agent_name}_latest.pth')
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'actor_target_state_dict': agent.actor_target.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'history': agent.history,
        'timestamp': timestamp,
    }, latest_path)

    print(f"Saved {agent_name} weights to {save_dir}/")
    print(f"  - Latest: {latest_path}")

    return latest_path


def load_weights(agent, checkpoint_path, device='cpu'):
    """Load agent weights from disk."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    if 'history' in checkpoint:
        agent.history = checkpoint['history']

    print(f"Loaded weights from {checkpoint_path}")
    if 'timestamp' in checkpoint:
        print(f"  - Trained on: {checkpoint['timestamp']}")

    return agent


def plot_training_curves(ddpg_history, td3_history, save_path='weights/training_curves.png'):
    """Plot training curves for both agents."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # DDPG rewards
    ax = axes[0]
    rewards = ddpg_history['episode_rewards']
    ax.plot(rewards, 'b-', alpha=0.3, label='Episode reward')
    # Smoothed curve
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, 'b-', linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (Sum-Rate)')
    ax.set_title('DDPG Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TD3 rewards
    ax = axes[1]
    rewards = td3_history['episode_rewards']
    ax.plot(rewards, 'r-', alpha=0.3, label='Episode reward')
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (Sum-Rate)')
    ax.set_title('TD3 Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def main():
    print("=" * 70)
    print("SIM Beamforming - Offline RL Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Training DDPG: {TRAIN_DDPG}")
    print(f"Training TD3: {TRAIN_TD3}")
    print()

    # Create system
    beamformer, _ = create_system(device=DEVICE)

    ddpg_history = None
    td3_history = None

    # Train DDPG
    if TRAIN_DDPG:
        print("\n" + "=" * 70)
        print("Training DDPG...")
        print("=" * 70 + "\n")

        beamformer_ddpg = copy.deepcopy(beamformer)
        ddpg = train_ddpg(
            beamformer_ddpg,
            num_episodes=NUM_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            device=DEVICE,
            verbose=True
                            )
        save_weights(ddpg, 'ddpg', SAVE_DIR)
        ddpg_history = ddpg.history

    # Train TD3
    if TRAIN_TD3:
        print("\n" + "=" * 70)
        print("Training TD3...")
        print("=" * 70 + "\n")

        beamformer_td3 = copy.deepcopy(beamformer)
        td3 = train_td3(
            beamformer_td3,
            num_episodes=NUM_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            device=DEVICE,
            verbose=True
        )
        save_weights(td3, 'td3', SAVE_DIR)
        td3_history = td3.history

    # Plot training curves
    if ddpg_history and td3_history:
        plot_training_curves(ddpg_history, td3_history,
                            save_path=os.path.join(SAVE_DIR, 'training_curves.png'))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nWeights saved to: {SAVE_DIR}/")
    print("\nTo use in main.py, load weights with:")
    print("  ddpg_agent.load_weights('weights/ddpg_latest.pth')")
    print("  td3_agent.load_weights('weights/td3_latest.pth')")


if __name__ == '__main__':
    main()