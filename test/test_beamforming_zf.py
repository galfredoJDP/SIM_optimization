import torch
import numpy as np
from transceiver import Transceiver

# Simulation parameters
wavelength = 0.125  # 125mm wavelength
K = 3  # Number of users
noise_power = 1e-10  # Noise power (σ²)

print("=" * 60)
print("Zero-Forcing Beamforming Test with 3 Users")
print("=" * 60)

# Create transceiver with 4x1 antenna array (4 antennas, 3 users)
print(f"\nCreating transceiver array...")
transceiver = Transceiver(
    Nx=4,
    Ny=1,
    wavelength=wavelength,
    max_scan_angle=30.0,
    device='cpu'
)

# Define user positions (randomly placed in far-field)
print(f"\nGenerating {K} user positions...")
np.random.seed(42)
user_positions = np.array([
    [10.0, 5.0, 0.0],   # User 1
    [10.0, 0.0, 0.0],   # User 2
    [10.0, -5.0, 0.0],  # User 3
])
print(f"User positions:\n{user_positions}")

# Get antenna positions
antenna_positions = transceiver.get_positions().cpu().numpy()
print(f"\nAntenna array: {transceiver.num_antennas} elements")
print(f"First antenna position: {antenna_positions[0]}")
print(f"Last antenna position: {antenna_positions[-1]}")

# Generate far-field channel (simple model with angle of arrival)
print(f"\nGenerating far-field channel...")
A = transceiver.num_antennas
H = torch.zeros((K, A), dtype=torch.complex64)

for k in range(K):
    # Direction from array center to user
    direction = user_positions[k] - antenna_positions.mean(axis=0)
    distance = np.linalg.norm(direction)
    direction = direction / distance

    # Path loss (simple model)
    path_loss = wavelength / (4 * np.pi * distance)

    # Phase shifts based on angle of arrival
    for a in range(A):
        relative_pos = antenna_positions[a] - antenna_positions.mean(axis=0)
        phase_shift = 2 * np.pi / wavelength * np.dot(relative_pos, direction)
        H[k, a] = path_loss * np.exp(1j * phase_shift)

    # Add small-scale fading (Rayleigh)
    fading = (np.random.randn(A) + 1j * np.random.randn(A)) / np.sqrt(2)
    H[k, :] *= torch.tensor(fading, dtype=torch.complex64) * 0.1 + 0.9

print(f"Channel matrix H shape: {H.shape}")
print(f"Channel condition number: {torch.linalg.cond(H).item():.2f}")

# Compute Zero-Forcing beamforming weights
print(f"\n" + "=" * 60)
print("Computing Zero-Forcing Beamforming Weights")
print("=" * 60)
W_zf = transceiver.compute_zf_weights(H)
print(f"ZF weights shape: {W_zf.shape}")
norms = [torch.norm(W_zf[:, k]).item() for k in range(K)]
print(f"ZF weights norm per user: {[f'{n:.4f}' for n in norms]}")

# Set beamforming weights
transceiver.set_beamforming_weights(W_zf)

# Verify zero-forcing property: H @ W should be approximately diagonal
print(f"\nVerifying Zero-Forcing property...")
effective_channel = H @ W_zf  # (K, K)
print(f"Effective channel H @ W (should be ~diagonal):")
for k in range(K):
    row_str = "  ["
    for j in range(K):
        val = effective_channel[k, j].item()
        if k == j:
            row_str += f" {abs(val):6.3f}*"  # Diagonal (signal)
        else:
            row_str += f" {abs(val):6.3f} "  # Off-diagonal (interference)
    row_str += "]"
    print(row_str)
print("  (* = diagonal elements, should be >> off-diagonal)")

# Power allocation (equal power)
total_power = 1.0  # Total transmit power (Watts)
power_allocation = torch.ones(K) * (total_power / K)
print(f"\nPower allocation: {power_allocation.numpy()}")

# Compute SINR
print(f"\n" + "=" * 60)
print("Computing SINR for Each User")
print("=" * 60)
sinr = transceiver.compute_sinr_downlink(H, power_allocation, noise_power)

print(f"\nResults:")
for k in range(K):
    sinr_db = 10 * np.log10(sinr[k].item())
    print(f"  User {k+1}: SINR = {sinr_db:.2f} dB  ({sinr[k].item():.2e} linear)")

# Compute achievable rates (Shannon capacity)
print(f"\nAchievable Rates (Shannon Capacity):")
sum_rate = 0.0
for k in range(K):
    rate = np.log2(1 + sinr[k].item())  # bits/s/Hz
    sum_rate += rate
    print(f"  User {k+1}: {rate:.3f} bits/s/Hz")
print(f"  Sum-rate: {sum_rate:.3f} bits/s/Hz")

# Test beamforming application
print(f"\n" + "=" * 60)
print("Testing Beamformed Transmission")
print("=" * 60)

# Generate test symbols (QPSK)
symbols = torch.tensor([1+1j, 1-1j, -1+1j], dtype=torch.complex64) / np.sqrt(2)
print(f"Transmit symbols: {symbols.numpy()}")

# Apply beamforming and propagate through channel
received = transceiver.apply_beamforming_downlink(H, symbols)
print(f"Received signals (before noise):")
for k in range(K):
    print(f"  User {k+1}: {received[k].numpy()}")

# Add noise
noise = torch.sqrt(torch.tensor(noise_power)) * (torch.randn(K) + 1j * torch.randn(K)) / np.sqrt(2)
received_noisy = received + noise

print(f"\nReceived signals (with noise):")
for k in range(K):
    val = received_noisy[k].numpy()
    print(f"  User {k+1}: {val.real:.4f} + {val.imag:.4f}j")

# Verify each user receives primarily their own symbol
print(f"\nSignal separation check:")
for k in range(K):
    intended_symbol = symbols[k].numpy()
    received_symbol = received[k].numpy()
    error = abs(received_symbol - intended_symbol)
    print(f"  User {k+1}: intended={intended_symbol:.3f}, received={received_symbol:.3f}, error={error:.4f}")

print(f"\n" + "=" * 60)
print("✓ Zero-Forcing Beamforming Test Complete!")
print("=" * 60)