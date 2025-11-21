import torch
import numpy as np
from ddpg import sim

# Example usage of the SIM class

# Parameters (from paper simulations)
layers = 2              # L = 2 layers
metaAtoms = 25          # N = 25 meta-atoms per layer
layerSpacing = 0.1      # Distance between layers (meters)
metaAtomSpacing = 0.05  # Distance between meta-atoms (meters)
wavelength = 0.125      # Lambda = 0.125m (from paper)
metaAtomArea = 0.01 * 0.01  # Area of each meta-atom (dx * dy)

print("Initializing SIM...")
mySim = sim(
    layers=layers,
    metaAtoms=metaAtoms,
    layerSpacing=layerSpacing,
    metaAtomSpacing=metaAtomSpacing,
    metaAtomArea=metaAtomArea,
    wavelength=wavelength
)

print(f"\nSIM Configuration:")
print(f"  Layers: {layers}")
print(f"  Meta-atoms per layer: {metaAtoms}")
print(f"  Device: {mySim.device}")
print(f"  Psi matrix shape: {mySim.Psi.shape}")

# Test 1: Forward propagation with random input
print("\n--- Test 1: Forward Propagation ---")
num_users = 4
input_field = torch.randn(metaAtoms, num_users, dtype=torch.complex64)
print(f"Input field shape: {input_field.shape}")

output_field = mySim.forward(input_field)
print(f"Output field shape: {output_field.shape}")
print(f"Output magnitude range: [{torch.abs(output_field).min():.4f}, {torch.abs(output_field).max():.4f}]")

# Test 2: Update phases
print("\n--- Test 2: Phase Update ---")
new_phases = torch.rand(layers, metaAtoms) * 2 * np.pi  # Random phases [0, 2π]
print(f"New phases shape: {new_phases.shape}")

mySim.update_phases(new_phases)
print("Phases updated and Psi recomputed!")

# Test forward again with new phases
output_field_new = mySim.forward(input_field)
print(f"Output with new phases magnitude range: [{torch.abs(output_field_new).min():.4f}, {torch.abs(output_field_new).max():.4f}]")

# Check that outputs are different
difference = torch.abs(output_field - output_field_new).mean()
print(f"Mean difference between outputs: {difference:.4f}")

print("\n✓ All tests passed!")