'''
Author: Alfredo Gonzalez

Test: Verify 1-bit ADMM performance using Symbol Error Rate (SER)
      Paper: "One-Bit Downlink Precoding for Massive MIMO OFDM System" (Wen et al., 2023)

This test uses SER vs Transmit Power to validate the ~3dB gap, matching the paper's methodology.
'''
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.insert(0, '/Users/alfredogonzalez/Desktop/code/SIM_optimization')

from simpy.beamformer import Beamformer
from simpy.algorithm import CG_MC1bit

# Set plot style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['figure.figsize'] = (10, 7)

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

def compute_wiener_filter_precoding(H, symbols, noise_power, total_power):
    """
    Wiener filter precoding following the paper's formulation.

    Paper Equation 6: P_WF[k] = √γ H^H[k](H[k]H^H[k] + Uσ²I_U)^(-1)

    Args:
        H: (K, M) channel matrix
        symbols: (K,) symbols to transmit
        noise_power: σ² (noise power in Watts)
        total_power: total transmit power constraint

    Returns:
        x: (M,) precoded antenna signals (power normalized)
    """
    K, M = H.shape
    device = H.device
    dtype = H.dtype

    # Paper's regularization: Uσ² (U = K = number of users)
    # NOT SNR-aware (no division by total_power)
    regularization = K * noise_power

    # A = H H^H + Kσ²I_K
    A = H @ H.conj().T + regularization * torch.eye(K, dtype=dtype, device=device)

    # P = H^H A^{-1}
    P = H.conj().T @ torch.linalg.inv(A)  # (M, K)

    # Unnormalized transmit vector
    x = P @ symbols  # (M,)

    # Normalize to meet total transmit power
    current_power = torch.sum(torch.abs(x) ** 2).real
    x = x * torch.sqrt(total_power / current_power)

    return x


# def compute_wiener_filter_precoding(H, symbols, noise_power, total_power):
#     """
#     Compute Wiener filter (MMSE) precoding for continuous DAC.

#     Args:
#         H: (K, M) channel matrix
#         symbols: (K,) symbols to transmit
#         noise_power: scalar
#         total_power: total transmit power constraint

#     Returns:
#         precoded_signal: (M,) precoded antenna signals
#     """
#     K, M = H.shape

#     # Wiener filter: P = H^H (H H^H + σ²I)^(-1)
#     HH_hermitian = H @ H.conj().T
#     regularization = noise_power * torch.eye(K, dtype=torch.complex64, device=H.device)

#     inv_term = torch.linalg.inv(HH_hermitian + regularization)
#     P = H.conj().T @ inv_term  # (M, K)

#     precoded_signal = P @ symbols

#     # Normalize to satisfy power constraint
#     current_power = torch.sum(torch.abs(precoded_signal)**2)
#     precoded_signal = precoded_signal * torch.sqrt(total_power / current_power)

#     return precoded_signal


def compute_ser_for_precoding(H, precoded_signal, transmitted_symbols, qpsk_constellation,
                               noise_power, num_noise_trials, device):
    """
    Compute Symbol Error Rate with simple nearest-neighbor detection.

    Wiener filter does NOT need alpha scaling - it's already optimal!

    Args:
        H: (K, M) channel matrix
        precoded_signal: (M,) precoded antenna signals
        transmitted_symbols: (K,) QPSK symbols that were transmitted
        qpsk_constellation: QPSK constellation points
        noise_power: scalar noise power
        num_noise_trials: number of noise realizations to average over
        device: torch device

    Returns:
        ser: Symbol error rate (0 to 1)
        mean_snr_db: Average received SNR across all users (dB)
    """
    K = H.shape[0]
    errors = 0
    total_symbols = 0

    # Received signal (without noise)
    y_noiseless = H @ precoded_signal  # (K,)

    # Calculate received SNR for each user
    received_powers = torch.abs(y_noiseless) ** 2  # |y|² for each user
    snr_linear = received_powers / noise_power
    snr_db = 10 * torch.log10(snr_linear)
    mean_snr_db = torch.mean(snr_db).item()

    for _ in range(num_noise_trials):
        # Add complex Gaussian noise: CN(0, σ²)
        noise_real = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
        noise_imag = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
        noise = noise_real + 1j * noise_imag

        # Received signal with noise
        y = y_noiseless + noise

        # Decode each user's symbol (simple nearest-neighbor)
        for k in range(K):
            # Find closest QPSK symbol (NO alpha scaling for Wiener filter!)
            distances = torch.abs(y[k] - qpsk_constellation)
            decoded_idx = torch.argmin(distances)

            # Check if error occurred
            transmitted_idx = torch.argmin(torch.abs(transmitted_symbols[k] - qpsk_constellation))

            if decoded_idx != transmitted_idx:
                errors += 1

            total_symbols += 1

    ser = errors / total_symbols if total_symbols > 0 else 0
    return ser, mean_snr_db


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"TESTING 1-BIT ADMM vs CONTINUOUS USING SYMBOL ERROR RATE (SER)")
    print(f"Paper: Wen et al., 'One-Bit Downlink Precoding for Massive MIMO OFDM'")
    print(f"{'='*80}\n")

    device = 'cpu'

    # Paper configuration
    M = 64
    K = 8
    Nx = 8
    Ny = 8

    wavelength = 0.125
    noise_power_dbm = -80
    noise_power = 10**(noise_power_dbm/10) / 1000

    # Channel parameters
    min_user_distance = 10
    max_user_distance = 100
    path_loss_at_reference = -30.0
    reference_distance = 1.0

    # Transmit power sweep - extended to higher powers
    transmit_powers_dbm = np.arange(0, 10, 1)  # -20 to +10 dBm in 3dB steps

    # Monte Carlo parameters
    num_channel_trials = 10  # Different channel realizations (reduced for faster test)
    num_noise_trials = 50    # Noise realizations per channel (reduced for faster test)

    print(f"Configuration:")
    print(f"  Antennas: M={M}, Users: K={K}")
    print(f"  Transmit power range: {transmit_powers_dbm[0]} to {transmit_powers_dbm[-1]} dBm")
    print(f"  Noise power: {noise_power_dbm} dBm")
    print(f"  Channel trials: {num_channel_trials}")
    print(f"  Noise trials per channel: {num_noise_trials}")
    print(f"  Modulation: QPSK")
    print(f"  Lambda penalty: 1.0 (validated)\n")

    # QPSK constellation
    qpsk_constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                     dtype=torch.complex64, device=device) / np.sqrt(2)

    # Storage for results
    ser_1bit_all = []
    ser_continuous_all = []
    ser_1bit_std = []
    ser_continuous_std = []

    print("Running SER simulation...")

    # Sweep transmit power
    for power_idx, power_dbm in enumerate(transmit_powers_dbm):
        total_power = 10**(power_dbm/10) / 1000  # Convert to Watts

        print(f"\n  Tx Power: {power_dbm} dBm ({power_idx+1}/{len(transmit_powers_dbm)})")

        # Create beamformer
        beamformer = Beamformer(
            Nx=Nx, Ny=Ny, wavelength=wavelength, device=device,
            num_users=K, user_positions=None,
            reference_distance=reference_distance,
            path_loss_at_reference=path_loss_at_reference,
            min_user_distance=min_user_distance,
            max_user_distance=max_user_distance,
            sim_model=None,
            noise_power=noise_power,
            total_power=total_power,
        )

        ser_1bit_trials = []
        ser_continuous_trials = []

        # Multiple channel realizations
        for trial in range(num_channel_trials):
            # Generate new channel
            beamformer.update_user_channel(time=float(trial + power_idx * 1000))
            H = beamformer.H

            # Generate random QPSK symbols
            symbol_indices = torch.randint(0, 4, (K,), device=device)
            symbols = qpsk_constellation[symbol_indices]

            # ===== 1-bit ADMM =====
            admm = CG_MC1bit(beamformer, lambda_penalty=1.0, max_iterations=100,
                            verbose=False, device=device)
            result_1bit = admm.optimize(symbols, total_power)
            antenna_signals_1bit = result_1bit['antenna_signals']

            ser_1bit, snr_1bit = compute_ser_for_precoding(
                H, antenna_signals_1bit, symbols, qpsk_constellation,
                noise_power, num_noise_trials, device
            )
            ser_1bit_trials.append(ser_1bit)

            # ===== Continuous Wiener Filter =====
            antenna_signals_continuous = compute_wiener_filter_precoding(
                H, symbols, noise_power, total_power
            )

            ser_continuous, snr_continuous = compute_ser_for_precoding(
                H, antenna_signals_continuous, symbols, qpsk_constellation,
                noise_power, num_noise_trials, device
            )
            ser_continuous_trials.append(ser_continuous)

        # Compute statistics
        mean_ser_1bit = np.mean(ser_1bit_trials)
        std_ser_1bit = np.std(ser_1bit_trials)
        mean_ser_continuous = np.mean(ser_continuous_trials)
        std_ser_continuous = np.std(ser_continuous_trials)

        ser_1bit_all.append(mean_ser_1bit)
        ser_1bit_std.append(std_ser_1bit)
        ser_continuous_all.append(mean_ser_continuous)
        ser_continuous_std.append(std_ser_continuous)

        print(f"    1-bit SER: {mean_ser_1bit:.4f} ± {std_ser_1bit:.4f}")
        print(f"    Continuous SER: {mean_ser_continuous:.4f} ± {std_ser_continuous:.4f}")

    # Convert to arrays
    ser_1bit_all = np.array(ser_1bit_all)
    ser_continuous_all = np.array(ser_continuous_all)
    ser_1bit_std = np.array(ser_1bit_std)
    ser_continuous_std = np.array(ser_continuous_std)

    print(f"\n{'='*80}")
    print(f"PLOTTING RESULTS")
    print(f"{'='*80}\n")

    # ============================================================================
    # PLOT: SER vs Transmit Power (matches paper's methodology)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot on log scale
    ax.semilogy(transmit_powers_dbm, ser_continuous_all, 'b-o', linewidth=2.5,
                markersize=7, label='Continuous (WF-inf)', markevery=1)
    ax.fill_between(transmit_powers_dbm,
                     np.maximum(ser_continuous_all - ser_continuous_std, 1e-5),
                     ser_continuous_all + ser_continuous_std,
                     alpha=0.2, color='blue')

    ax.semilogy(transmit_powers_dbm, ser_1bit_all, 'r-s', linewidth=2.5,
                markersize=7, label='1-bit ADMM (λ=1.0)', markevery=1)
    ax.fill_between(transmit_powers_dbm,
                     np.maximum(ser_1bit_all - ser_1bit_std, 1e-5),
                     ser_1bit_all + ser_1bit_std,
                     alpha=0.2, color='red')

    # Find 3dB crossover points (where continuous achieves target SER)
    target_ser = 0.01  # 1% SER
    if np.min(ser_continuous_all) < target_ser < np.max(ser_continuous_all):
        # Find power where continuous achieves target SER
        idx_continuous = np.argmin(np.abs(ser_continuous_all - target_ser))
        power_continuous = transmit_powers_dbm[idx_continuous]

        # Find power where 1-bit achieves target SER
        if np.min(ser_1bit_all) < target_ser < np.max(ser_1bit_all):
            idx_1bit = np.argmin(np.abs(ser_1bit_all - target_ser))
            power_1bit = transmit_powers_dbm[idx_1bit]

            gap_db = power_1bit - power_continuous

            # Add annotation
            ax.axhline(y=target_ser, color='k', linestyle='--', alpha=0.3)
            ax.plot([power_continuous, power_1bit], [target_ser, target_ser],
                   'go-', linewidth=2, markersize=8)
            ax.annotate(f'{gap_db:.1f} dB gap\n(at SER={target_ser})',
                       xy=(power_continuous + gap_db/2, target_ser),
                       xytext=(power_continuous + gap_db/2, target_ser * 5),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=11, ha='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Transmit Power (dBm)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Symbol Error Rate (SER)', fontsize=13, fontweight='bold')
    ax.set_title('1-bit ADMM vs Continuous: Symbol Error Rate\n'
                 '(M=64, K=8, QPSK) - Validating Paper Results',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
    ax.set_ylim([1e-9, 1])

    plt.tight_layout()
    plt.savefig('test/admm_comparison/ser_vs_txpower.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: test/admm_comparison/ser_vs_txpower.png")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}\n")

    print(f"Symbol Error Rate Results:")
    print(f"{'Tx Power (dBm)':<18} {'1-bit SER':<15} {'Continuous SER':<15}")
    print(f"{'-'*50}")
    for i, power_dbm in enumerate(transmit_powers_dbm[::2]):  # Show every other
        idx = i * 2
        print(f"{power_dbm:<18.1f} {ser_1bit_all[idx]:<15.4f} {ser_continuous_all[idx]:<15.4f}")

    print(f"\n{'='*80}")
    print(f"KEY FINDING")
    print(f"{'='*80}")
    if 'gap_db' in locals():
        print(f"\n✓ At SER = {target_ser} (1% error rate):")
        print(f"  - Continuous requires: {power_continuous:.1f} dBm")
        print(f"  - 1-bit requires: {power_1bit:.1f} dBm")
        print(f"  - Gap: {gap_db:.1f} dB")
        print(f"\n  Expected from paper: ~3 dB")
        if abs(gap_db - 3.0) < 1.0:
            print(f"  ✓ VALIDATED: Gap matches paper within 1 dB!")
        else:
            print(f"  ⚠ Gap differs from paper - may need more trials or different target SER")
    else:
        print(f"\n⚠ Could not compute gap at SER={target_ser}")
        print(f"  SER range: {np.min(ser_1bit_all):.4f} to {np.max(ser_1bit_all):.4f}")
        print(f"  Consider adjusting power range or target SER")

    print(f"\n{'='*80}\n")

    plt.show()