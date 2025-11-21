"""
Test script to demonstrate the flow of SIM-based vs Digital beamforming.
Shows where channel generation happens and where optimization would go.
"""

import torch
import numpy as np
from wireless.sim import sim
from Beamformer import Beamformer

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ========== Parameters ==========

# System parameters
num_users = 4
wavelength = 0.125  # meters
device = 'cpu'

# Antenna array (for digital beamformer)
Nx_antenna = 4
Ny_antenna = 1
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
sim_model = sim(
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

print("\n3. Channel matrices (auto-generated during init):")
print(f"   A matrix (antennas → SIM layer 0): {sim_beamformer.A.shape}")
print(f"      Interpretation: {sim_beamformer.A.shape[1]} antennas → {sim_beamformer.A.shape[0]} meta-atoms")
print(f"   Psi matrix (SIM propagation):       {sim_model.Psi.shape}")
print(f"      Interpretation: Layer 0 → Layer {sim_layers-1} through {sim_layers} layers")
print(f"   H matrix (SIM → users):             {sim_beamformer.H.shape}")
print(f"      Interpretation: {sim_beamformer.H.shape[1]} meta-atoms → {sim_beamformer.H.shape[0]} users")

print("\n" + "-"*80)
print("4. >>> OPTIMIZATION GOES HERE <<<")
print("-"*80)
# Dummy SIM phases - this is where optimization would go!
sim_phases_dummy = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi
print(f"   Phase matrix shape: {sim_phases_dummy.shape} = (L={sim_layers}, N={sim_metaatoms})")

#Uniform power allocation , this is where real power allocation strategy would go!
power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
print(f"   Power allocation:   {power_allocation.shape} = (K={num_users},)")
print("-"*80)

print("\n5. Computing end-to-end channel...")
# Compute end-to-end channel: H @ Psi @ A
H_eff_sim = sim_beamformer.compute_end_to_end_channel(sim_phases_dummy)
print(f"   H_eff = H @ Psi @ A")
print(f"   H_eff shape: {H_eff_sim.shape}")
print(f"   Interpretation: (K={H_eff_sim.shape[0]}, M={H_eff_sim.shape[1]}) effective channel")
print(f"   Since M=K: H_eff[k,k] = signal, H_eff[k,j≠k] = interference")

print("\n6. Computing SINR and Sum-Rate...")
sinr_sim = sim_beamformer.compute_sinr(phases=sim_phases_dummy, power_allocation=power_allocation)
sum_rate_sim = sim_beamformer.compute_sum_rate(phases=sim_phases_dummy, power_allocation=power_allocation)
print(f"   SINR shape: {sinr_sim.shape} = (K={num_users},)")
print(f"   SINR (dB): {10 * torch.log10(sinr_sim).cpu().numpy()}")
print(f"   Sum-Rate:  {sum_rate_sim.item():.4f} bits/s/Hz")

# ========== Case 2: Digital Beamformer (No SIM) ==========
print("\n" + "="*80)
print("CASE 2: DIGITAL BEAMFORMING (Traditional M > K Architecture)")
print("="*80)

print("\n1. Creating Digital Beamformer (M=64, K=4)...")
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

print("\n2. Generating channel (no SIM)...")
antenna_positions = digital_beamformer.get_positions()
H_digital = digital_beamformer.generate_channel(antenna_positions, time=0.0)
print(f"   H matrix (antennas → users): {H_digital.shape}")
print(f"   Interpretation: {H_digital.shape[1]} antennas → {H_digital.shape[0]} users")
print(f"   Note: All antennas can serve all users (traditional beamforming)")

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

print("\n4. Computing effective channel...")
# Compute effective channel with beamforming
H_effective = H_digital @ W_digital_dummy
print(f"   H_eff = H @ W")
print(f"   H_eff shape: {H_effective.shape}")
print(f"   Interpretation: (K={H_effective.shape[0]}, K={H_effective.shape[1]}) effective channel")
print(f"   H_eff[k,k] = signal for user k, H_eff[k,j≠k] = interference from beam j")

print("\n5. Computing SINR and Sum-Rate...")
sinr_digital = digital_beamformer.compute_sinr(
    phases=None,  # No SIM
    power_allocation=power_allocation,
    digital_beamforming_weights=W_digital_dummy
)
sum_rate_digital = digital_beamformer.compute_sum_rate(
    phases=None,  # No SIM
    power_allocation=power_allocation,
    digital_beamforming_weights=W_digital_dummy
)
print(f"   SINR shape: {sinr_digital.shape} = (K={num_users},)")
print(f"   SINR (dB): {10 * torch.log10(sinr_digital).cpu().numpy()}")
print(f"   Sum-Rate:  {sum_rate_digital.item():.4f} bits/s/Hz")

# ========== Summary ==========
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nSIM-based (M=K=4):    Sum-Rate = {sum_rate_sim.item():.4f} bits/s/Hz")
print(f"Digital (M=64, K=4):  Sum-Rate = {sum_rate_digital.item():.4f} bits/s/Hz")
print(f"\nNote: Random weights used - digital should perform better with optimized weights")
print("="*80)
