import torch
from transceiver import Transceiver

# Example 1: Single antenna (Nx=1, Ny=1)
print("=" * 60)
print("Example 1: Single Antenna")
print("=" * 60)
tx_single = Transceiver(
    Nx=1,
    Ny=1,
    wavelength=0.125,  # 125mm
    max_scan_angle=0.0,
    device='cpu'
)
print(f"Position:\n{tx_single.get_positions()}\n")

# Example 2: 4×4 square array
print("=" * 60)
print("Example 2: 4×4 Square Array")
print("=" * 60)
tx_square = Transceiver(
    Nx=4,
    Ny=4,
    wavelength=0.125,
    max_scan_angle=30.0,  # 30° max scan
    device='cpu'
)
print(f"\nFirst 5 antenna positions:\n{tx_square.get_positions()[:5]}\n")

# Example 3: Linear array (8×1)
print("=" * 60)
print("Example 3: Linear Array (8×1)")
print("=" * 60)
tx_linear = Transceiver(
    Nx=8,
    Ny=1,
    wavelength=0.125,
    max_scan_angle=45.0,  # 45° scanning
    device='cpu'
)
print(f"\nPositions:\n{tx_linear.get_positions()}\n")

# Example 4: Broadside-only (larger spacing allowed)
print("=" * 60)
print("Example 4: Broadside Only (No Scanning)")
print("=" * 60)
tx_broadside = Transceiver(
    Nx=4,
    Ny=4,
    wavelength=0.125,
    max_scan_angle=0.0,  # No scanning → d < λ/2
    device='cpu'
)
print(f"Note: Broadside-only allows spacing up to λ/2\n")

# Visualize arrays
print("=" * 60)
print("Generating visualizations...")
print("=" * 60)

try:
    tx_square.visualize_array(save_path='array_4x4.png')
    tx_linear.visualize_array(save_path='array_8x1.png')
    print("✓ Visualizations saved!")
except Exception as e:
    print(f"Could not generate visualizations: {e}")