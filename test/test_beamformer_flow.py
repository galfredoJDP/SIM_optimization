"""
Test script to demonstrate the flow of SIM-based vs Digital beamforming.
Shows where channel generation happens and where optimization would go.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simpy.sim import Sim
from simpy.beamformer import Beamformer

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Visualization Functions ==========

def visualize_sim_with_antennas(sim_model, antenna_positions, save_path=None):
    """
    Visualize the SIM structure along with antenna positions.

    Args:
        sim_model: The SIM model object
        antenna_positions: Tensor of antenna positions (M, 3)
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 7))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot SIM layers
    colors = plt.cm.viridis(np.linspace(0, 1, sim_model.layers))

    for layer in range(sim_model.layers):
        positions = sim_model.metaAtomPositions[layer].cpu().numpy()
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=[colors[layer]], s=50, alpha=0.7,
                   label=f'SIM Layer {layer}')

    # Plot antennas
    ant_pos = antenna_positions.cpu().numpy()
    ax1.scatter(ant_pos[:, 0], ant_pos[:, 1], ant_pos[:, 2],
               c='red', s=100, marker='^', alpha=0.9,
               label='Antennas (BS)', edgecolors='black', linewidths=1.5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title('SIM-Based Beamforming: Antennas + SIM Structure')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(122)

    for layer in range(sim_model.layers):
        positions = sim_model.metaAtomPositions[layer].cpu().numpy()
        marker = 'o' if layer == 0 else ('s' if layer == sim_model.layers-1 else '^')
        label_suffix = ' (BS side)' if layer == 0 else (' (User side)' if layer == sim_model.layers-1 else '')

        ax2.scatter(positions[:, 0], positions[:, 1],
                   c=[colors[layer]], s=50, alpha=0.7,
                   marker=marker, label=f'SIM Layer {layer}{label_suffix}')

    # Plot antennas on top view
    ax2.scatter(ant_pos[:, 0], ant_pos[:, 1],
               c='red', s=100, marker='^', alpha=0.9,
               label='Antennas (BS)', edgecolors='black', linewidths=1.5)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Top View (XY Plane)')
    ax2.set_aspect('equal')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add origin marker
    ax2.plot(0, 0, 'k+', markersize=15, markeredgewidth=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_antenna_array(antenna_positions, array_config, save_path=None):
    """
    Visualize antenna array positions for digital beamforming.

    Args:
        antenna_positions: Tensor of antenna positions (M, 3)
        array_config: Tuple (Nx, Ny) for array configuration
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 7))

    ant_pos = antenna_positions.cpu().numpy()
    Nx, Ny = array_config

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.scatter(ant_pos[:, 0], ant_pos[:, 1], ant_pos[:, 2],
               c='blue', s=100, marker='o', alpha=0.9,
               edgecolors='black', linewidths=1.5)

    # Add lines connecting antennas to show array structure
    for i in range(len(ant_pos)):
        ax1.plot([ant_pos[i, 0], ant_pos[i, 0]],
                [ant_pos[i, 1], ant_pos[i, 1]],
                [ant_pos[i, 2], -0.01],
                'k--', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title(f'Digital Beamforming: Antenna Array ({Nx}x{Ny} = {len(ant_pos)} antennas)')
    ax1.grid(True, alpha=0.3)

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(122)

    ax2.scatter(ant_pos[:, 0], ant_pos[:, 1],
               c='blue', s=100, marker='o', alpha=0.9,
               edgecolors='black', linewidths=1.5)

    # Add grid lines to show array structure
    # Vertical lines (constant x)
    unique_x = np.unique(np.round(ant_pos[:, 0], decimals=6))
    for x_val in unique_x:
        mask = np.abs(ant_pos[:, 0] - x_val) < 1e-6
        y_vals = ant_pos[mask, 1]
        if len(y_vals) > 1:
            ax2.plot([x_val, x_val], [y_vals.min(), y_vals.max()],
                    'k--', alpha=0.3, linewidth=0.5)

    # Horizontal lines (constant y)
    unique_y = np.unique(np.round(ant_pos[:, 1], decimals=6))
    for y_val in unique_y:
        mask = np.abs(ant_pos[:, 1] - y_val) < 1e-6
        x_vals = ant_pos[mask, 0]
        if len(x_vals) > 1:
            ax2.plot([x_vals.min(), x_vals.max()], [y_val, y_val],
                    'k--', alpha=0.3, linewidth=0.5)

    # Label each antenna with its index
    for i in range(len(ant_pos)):
        ax2.annotate(f'{i}', (ant_pos[i, 0], ant_pos[i, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=7, alpha=0.7)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'Top View: {Nx}x{Ny} Array Layout')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Add origin marker
    ax2.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Origin')
    ax2.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

# ========== Parameters ==========

# System parameters
num_users = 4
wavelength = 0.125  # meters
device = 'cpu'

# Antenna array (for digital beamformer)
Nx_antenna = 2
Ny_antenna = 2
num_antennas = Nx_antenna * Ny_antenna

# SIM parameters (for SIM beamformer)
sim_layers = 4
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
noise_power = 1e-10  # Watts
total_power = 1.0  # Watts

print("\n" + "="*80)
print("SYSTEM PARAMETERS")
print("="*80)
print(f"   Users (K): {num_users}")
print(f"   Wavelength: {wavelength} m")
print(f"   Digital antennas (M): {Nx_antenna}x{Ny_antenna} = {num_antennas}")
print(f"   SIM: {sim_layers} layers (L), {sim_metaatoms} meta-atoms per layer (N)")
print(f"   User distances: [{min_user_distance}, {max_user_distance}] m")

# ========== Case 1: SIM-Based Beamformer ==========
print("\n" + "="*80)
print("CASE 1: SIM-BASED BEAMFORMING (M=K Architecture)")
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
print(f"   Psi matrix shape: {sim_model.Psi.shape}")

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

print("\n3. Visualizing SIM structure and antenna positions...")
visualize_sim_with_antennas(sim_model, sim_beamformer.get_positions(),
                           save_path='sim_with_antennas.png')

print("\n4. Channel matrices (auto-generated during init):")
print(f"   A matrix (antennas → SIM layer 0): {sim_beamformer.A.shape}")
print(f"      Interpretation: {sim_beamformer.A.shape[1]} antennas → {sim_beamformer.A.shape[0]} meta-atoms")
print(f"   Psi matrix (SIM propagation):       {sim_model.Psi.shape}")
print(f"      Interpretation: Layer 0 → Layer {sim_layers-1} through {sim_layers} layers")
print(f"   H matrix (SIM → users):             {sim_beamformer.H.shape}")
print(f"      Interpretation: {sim_beamformer.H.shape[1]} meta-atoms → {sim_beamformer.H.shape[0]} users")
"""
SIM-based beamforming optimization here --------------------------------------------------
"""
print("\n" + "-"*80)
print("5. >>> OPTIMIZATION GOES HERE <<<")
print("-"*80)
# Dummy SIM phases - this is where optimization would go!
sim_phases_dummy = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi
print(f"   Phase matrix shape: {sim_phases_dummy.shape} = (L={sim_layers}, N={sim_metaatoms})")

#Uniform power allocation , this is where real power allocation strategy would go!
power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
print(f"   Power allocation:   {power_allocation.shape} = (K={num_users},)")
print("-"*80)
"""
---------------------------------------------------------------------------------------
"""
print("\n6. Computing SINR and Sum-Rate...")
sum_rate_sim = sim_beamformer.compute_sum_rate(phases=sim_phases_dummy, power_allocation=power_allocation)
print(f"   Sum-Rate:  {sum_rate_sim.item():.4f} bits/s/Hz")

print("\n7. Verifying channels used in SINR computation...")
print(f"   Pre-digital-weights channel shape: {sim_beamformer.pre_digital_weights_channel.shape}")
print(f"      = H @ Psi @ A (end-to-end through SIM)")
print(f"   Effective channel shape: {sim_beamformer.last_effective_channel.shape}")
print(f"      = Same as above (no digital weights in SIM-only case)")
print(f"   Interpretation: (K={sim_beamformer.last_effective_channel.shape[0]}, M={sim_beamformer.last_effective_channel.shape[1]})")
print(f"   Since M=K: H_eff[k,k] = signal, H_eff[k,j≠k] = interference")

# ========== Case 2: Digital Beamformer (No SIM) ==========
print("\n" + "="*80)
print("CASE 2: DIGITAL BEAMFORMING (Traditional M > K Architecture)")
print("="*80)

print("\n1. Creating Digital Beamformer ...")
digital_beamformer = Beamformer(
    # Transceiver params - Using M=64 antennas for K=4 users
    Nx=Nx_antenna,  # 8x8 = 64 antennas
    Ny=Ny_antenna,
    wavelength=wavelength,
    device=device,
    # Channel params (CLT mode - no user_positions)
    num_users=num_users,
    user_positions=None,  # Will use CLT mode
    reference_distance=reference_distance,
    path_loss_at_reference=path_loss_at_reference,
    min_user_distance=min_user_distance,
    max_user_distance=max_user_distance,
    # System params
    noise_power=noise_power,
    total_power=total_power,
    use_nearfield_user_channel=False  # Use CLT mode
)
print(f"   Created with M={num_antennas} antennas, K={num_users} users")

print("\n2. Visualizing antenna array positions...")
visualize_antenna_array(digital_beamformer.get_positions(),
                       array_config=(Nx_antenna, Ny_antenna),
                       save_path='antenna_array.png')


"""
Digital Only Beamformer Optimization ----------------------------------------------------------------
"""
print("\n" + "-"*80)
print("3. >>> OPTIMIZATION GOES HERE <<<")
print("-"*80)
# Digital beamforming weights
W_digital_dummy = torch.randn(num_antennas, num_users, dtype=torch.complex64, device=device)
# Normalize columns to unit norm (common practice)
for k in range(num_users):
    W_digital_dummy[:, k] /= torch.norm(W_digital_dummy[:, k])
print(f"   Digital beamforming weights W: {W_digital_dummy.shape}")
print(f"   Interpretation: (M={num_antennas}, K={num_users})")
print(f"   Column k = beamforming vector for user k")

power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
print(f"   Power allocation:              {power_allocation.shape} = (K={num_users},)")
print("-"*80)
"""
-------------------------------------------------------------------------------------
"""
print("\n4. Computing SINR and Sum-Rate...")
sum_rate_digital = digital_beamformer.compute_sum_rate(
    phases=None,  # No SIM
    power_allocation=power_allocation,
    digital_beamforming_weights=W_digital_dummy
)
print(f"   Sum-Rate:  {sum_rate_digital.item():.4f} bits/s/Hz")

print("\n5. Verifying channels used in SINR computation...")
print(f"   Pre-digital-weights channel shape: {digital_beamformer.pre_digital_weights_channel.shape}")
print(f"      = H (antennas → users)")
print(f"   Effective channel shape: {digital_beamformer.last_effective_channel.shape}")
print(f"      = H @ W (after digital beamforming weights)")
print(f"   Interpretation: (K={digital_beamformer.last_effective_channel.shape[0]}, K={digital_beamformer.last_effective_channel.shape[1]})")
print(f"   H_eff[k,k] = signal for user k, H_eff[k,j≠k] = interference from beam j")

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nSIM-based (M=K=4):    Sum-Rate = {sum_rate_sim.item():.4f} bits/s/Hz")
print(f"Digital (M=64, K=4):  Sum-Rate = {sum_rate_digital.item():.4f} bits/s/Hz")
print(f"\nNote: Random weights used - digital should perform better with optimized weights")
print("="*80)
