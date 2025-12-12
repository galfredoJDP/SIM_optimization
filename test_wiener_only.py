'''
Simple test: Wiener filter ONLY (no alpha scaling) to debug basic precoding
'''
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/alfredogonzalez/Desktop/code/SIM_optimization')

from simpy.beamformer import Beamformer

np.random.seed(42)
torch.manual_seed(42)

def wiener_filter_paper(H, symbols, noise_power, total_power):
    """Paper Equation 6"""
    K, M = H.shape
    device = H.device
    dtype = H.dtype

    # Regularization: Kσ² (NO division by power)
    reg = K * noise_power

    A = H @ H.conj().T + reg * torch.eye(K, dtype=dtype, device=device)
    P = H.conj().T @ torch.linalg.inv(A)

    x = P @ symbols

    # Power normalize
    current_power = torch.sum(torch.abs(x) ** 2).real
    x = x * torch.sqrt(total_power / current_power)

    # Calculate received SNR: SNR_i = |y_i|² / σ² where y = Hx
    y = H @ x  # Received signal
    received_power = torch.abs(y) ** 2  # |y|² for each user
    snr_linear = received_power / noise_power
    snr_db = 10 * torch.log10(snr_linear)

    return x, torch.mean(snr_db).item()


device = 'cpu'
M, K = 64, 8

# Setup
noise_power_dbm = -80  # Realistic noise power
noise_power = 10**(noise_power_dbm/10) / 1000

power_dbm = noise_power_dbm + 10  # Mid-range power
total_power = 10**(power_dbm/10) / 1000

# Create beamformer
beamformer = Beamformer(
    Nx=8, Ny=8, wavelength=0.125, device=device,
    num_users=K, user_positions=None,
    reference_distance=1.0,
    path_loss_at_reference=-30.0,
    min_user_distance=20.1,
    max_user_distance=20.2,
    sim_model=None,
    noise_power=noise_power,
    total_power=total_power,
)

beamformer.update_user_channel(time=0.0)
H = beamformer.H

print(f"Channel H: {H.shape}")
print(f"Channel condition number: {torch.linalg.cond(H).item():.2f}\n")

# QPSK constellation
qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex64, device=device) / np.sqrt(2)

# Random symbols
symbol_indices = torch.randint(0, 4, (K,), device=device)
symbols = qpsk[symbol_indices]

print(f"Symbols: {symbols}\n")

snr_hold = []
ser_hold = []
# Sweep transmit power to overcome path loss (~50 dB) and reach positive SNR
# Need Tx >> -80 dBm to compensate for ~50 dB path loss
power_dbm = noise_power_dbm + np.arange(42, 68, 2)  # -30 to +10 dBm Tx power
power_linear = 10**(power_dbm/10) / 1000
for total_power_i in power_linear:
    # Wiener filter precoding
    print(f"\n=== Testing Tx Power: {10*np.log10(total_power_i*1000):.2f} dBm ===")
    x, snr = wiener_filter_paper(H, symbols, noise_power, total_power_i)
    snr_hold.append(snr)

    # Detection WITH noise (multiple trials to average)
    num_noise_trials = 10000
    total_errors = 0
    total_symbols = 0

    for trial in range(num_noise_trials):
        # Add complex Gaussian noise: CN(0, σ²)
        noise_real = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
        noise_imag = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
        noise = noise_real + 1j * noise_imag

        # Received signal with noise
        y = H @ x + noise

        # Detection (nearest neighbor)
        for k in range(K):
            distances = torch.abs(y[k] - qpsk)
            decoded_idx = torch.argmin(distances)
            true_idx = symbol_indices[k]

            if decoded_idx != true_idx:
                total_errors += 1

            total_symbols += 1

    ser = total_errors / total_symbols
    ser_hold.append(ser)
    print(f"  Mean SNR: {snr:.2f} dB → SER: {ser:.4f} ({ser*100:.2f}%)")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'SNR (dB)':<15} {'SER':<15} {'SER %':<15}")
print("-"*60)
for snr_val, ser_val in zip(snr_hold, ser_hold):
    print(f"{snr_val:<15.2f} {ser_val:<15.4f} {ser_val*100:<15.2f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.semilogy(snr_hold, ser_hold, 'b-o', linewidth=2.5,
            markersize=7, label='Wiener Filter (Continuous DAC)')
ax.set_xlabel('Received SNR (dB)', fontsize=13, fontweight='bold')
ax.set_ylabel('Symbol Error Rate (SER)', fontsize=13, fontweight='bold')
ax.set_title('Wiener Filter: SER vs SNR\n(M=64, K=8, QPSK)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both', linestyle='--')
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.set_ylim(10**-5, 0)
plt.tight_layout()
plt.savefig('test_wiener_ser_vs_snr.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: test_wiener_ser_vs_snr.png")
plt.show()
# %%
