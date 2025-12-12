"""
Quick diagnostic to check channel norms with paper parameters.
"""

import torch
import numpy as np
from simpy.sim import Sim
from simpy.beamformer import Beamformer

np.random.seed(42)
torch.manual_seed(42)

# Parameters
num_users = 4
wavelength = 0.125
device = 'cpu'

# Paper's SIM parameters
sim_layers = 2
sim_metaatoms = 25
sim_layer_spacing = 0.1
sim_metaatom_spacing = 0.05
sim_metaatom_area_paper = 0.01 * 0.01  # 0.0001 m² (paper claims)
sim_metaatom_area = sim_metaatom_spacing ** 2  # 0.0025 m² (spacing squared - more reasonable)

print(f"Testing with CORRECTED parameters:")
print(f"  Meta-atom area (paper docs): {sim_metaatom_area_paper} m²")
print(f"  Meta-atom area (spacing²): {sim_metaatom_area} m²  << USING THIS")
print(f"  Layer spacing: {sim_layer_spacing} m")
print(f"  Meta-atom spacing: {sim_metaatom_spacing} m\n")

# Create SIM
sim_model = Sim(
    layers=sim_layers,
    metaAtoms=sim_metaatoms,
    layerSpacing=sim_layer_spacing,
    metaAtomSpacing=sim_metaatom_spacing,
    metaAtomArea=sim_metaatom_area,
    wavelength=wavelength,
    device=device
)

# Create beamformer with CLT
beamformer = Beamformer(
    Nx=2, Ny=2,
    wavelength=wavelength,
    device=device,
    num_users=num_users,
    user_positions=None,  # CLT
    min_user_distance=1.0,
    max_user_distance=5.5,
    reference_distance=1.0,
    path_loss_at_reference=-30.0,
    sim_model=sim_model,
    noise_power=10**(-80/10)/1000,
    total_power=10**(26/10)/1000,
)

print("Channel Norms:")
print(f"  A (Antenna→SIM): {torch.norm(beamformer.A).item():.6e}")
print(f"  H (SIM→Users):   {torch.norm(beamformer.H).item():.6e}")
print(f"  Combined scale:  {beamformer.channel_scale:.6e}")

print(f"\nA matrix statistics:")
print(f"  Shape: {beamformer.A.shape}")
print(f"  Max:   {torch.max(torch.abs(beamformer.A)).item():.6e}")
print(f"  Min:   {torch.min(torch.abs(beamformer.A)).item():.6e}")
print(f"  Mean:  {torch.mean(torch.abs(beamformer.A)).item():.6e}")

print(f"\nH matrix statistics:")
print(f"  Shape: {beamformer.H.shape}")
print(f"  Max:   {torch.max(torch.abs(beamformer.H)).item():.6e}")
print(f"  Min:   {torch.min(torch.abs(beamformer.H)).item():.6e}")
print(f"  Mean:  {torch.mean(torch.abs(beamformer.H)).item():.6e}")

# Compute H_eff with random phases
phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi
H_eff = beamformer.compute_end_to_end_channel(phases)

print(f"\nH_eff (H @ Psi @ A) statistics:")
print(f"  Shape: {H_eff.shape}")
print(f"  Norm:  {torch.norm(H_eff).item():.6e}")
print(f"  Diagonal: {torch.abs(torch.diag(H_eff)).cpu().numpy()}")
print(f"  Max:   {torch.max(torch.abs(H_eff)).item():.6e}")
print(f"  Min:   {torch.min(torch.abs(H_eff)).item():.6e}")

# Expected sum-rate at 26 dBm
power_w = 10**(26/10) / 1000
equal_power = torch.ones(num_users, device=device) * (power_w / num_users)
sumrate = beamformer.compute_sum_rate(phases, equal_power)
print(f"\nSum-rate with random phases at 26 dBm: {sumrate:.4f} bits/s/Hz")
print(f"Paper expects: ~15-20 bits/s/Hz with optimized (WF, PGA)")
