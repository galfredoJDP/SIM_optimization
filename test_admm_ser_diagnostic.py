'''
Author: Alfredo Gonzalez

Diagnostic: Investigate why 1-bit ADMM has such high SER
'''

import torch
import numpy as np

import sys
sys.path.insert(0, '/Users/alfredogonzalez/Desktop/code/SIM_optimization')

from simpy.beamformer import Beamformer
from simpy.algorithm import CG_MC1bit

# Set random seed
np.random.seed(42)
torch.manual_seed(42)


def compute_wiener_filter_precoding(H, symbols, noise_power, total_power):
    """Compute Wiener filter (MMSE) precoding."""
    K, M = H.shape
    HH_hermitian = H @ H.conj().T
    regularization = noise_power * torch.eye(K, dtype=torch.complex64, device=H.device)
    inv_term = torch.linalg.inv(HH_hermitian + regularization)
    P = H.conj().T @ inv_term
    precoded_signal = P @ symbols
    current_power = torch.sum(torch.abs(precoded_signal)**2)
    precoded_signal = precoded_signal * torch.sqrt(total_power / current_power)
    return precoded_signal


def analyze_received_signals(H, antenna_signals, symbols, noise_power, method_name):
    """Analyze received signal properties."""
    K, M = H.shape

    # Compute received signal
    y = H @ antenna_signals

    # Compute signal and interference components
    print(f"\n{method_name} Analysis:")
    print(f"{'='*60}")

    # Check transmit power
    tx_power = torch.sum(torch.abs(antenna_signals)**2).item()
    print(f"Transmit power: {tx_power:.6f} W ({10*np.log10(tx_power*1000):.2f} dBm)")

    # For each user
    sinr_db_list = []
    for k in range(K):
        # Desired signal: h_k^H x (for user k's symbol)
        # But we need to isolate the component from user k's symbol

        # Received signal at user k
        y_k = y[k]

        # Compute signal power (assume symbols have unit power on average)
        signal_power = torch.abs(y_k)**2

        # Estimated SNR (simplified - treating everything as signal)
        snr_k = signal_power / noise_power
        snr_db_k = 10 * np.log10(snr_k.item())
        sinr_db_list.append(snr_db_k)

        print(f"  User {k}: Received power = {signal_power.item():.6e} W, "
              f"SNR = {snr_db_k:.2f} dB")

    print(f"  Mean SNR: {np.mean(sinr_db_list):.2f} dB")
    print(f"  Min SNR:  {np.min(sinr_db_list):.2f} dB")

    return y


def compute_ser_single_trial(H, precoded_signal, transmitted_symbols,
                             qpsk_constellation, noise_power, device):
    """Compute SER for a single noise realization."""
    K = H.shape[0]

    # Add noise
    noise_real = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
    noise_imag = torch.randn(K, device=device) * np.sqrt(noise_power / 2)
    noise = noise_real + 1j * noise_imag

    # Received signal
    y = H @ precoded_signal + noise

    # Decode and count errors
    errors = 0
    for k in range(K):
        distances = torch.abs(y[k] - qpsk_constellation)
        decoded_idx = torch.argmin(distances)
        transmitted_idx = torch.argmin(torch.abs(transmitted_symbols[k] - qpsk_constellation))

        if decoded_idx != transmitted_idx:
            errors += 1

    return errors, K


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: Why is 1-bit ADMM SER so high?")
    print(f"{'='*80}\n")

    device = 'cpu'

    # Configuration
    M = 64
    K = 8
    Nx = 8
    Ny = 8

    wavelength = 0.125
    noise_power_dbm = -80
    noise_power = 10**(-80/10) / 1000

    min_user_distance = 10
    max_user_distance = 100
    path_loss_at_reference = -30.0
    reference_distance = 1.0

    # Test at high power where continuous achieves 0% SER but 1-bit is still 31%
    power_dbm = 10
    total_power = 10**(power_dbm/10) / 1000

    print(f"Configuration:")
    print(f"  M={M}, K={K}, Noise={noise_power_dbm} dBm")
    print(f"  Transmit power: {power_dbm} dBm ({total_power*1000:.6f} mW)")
    print(f"  Target SNR: ~{power_dbm - noise_power_dbm} dB\n")

    # QPSK constellation
    qpsk_constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                     dtype=torch.complex64, device=device) / np.sqrt(2)

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

    # Generate channel and symbols
    beamformer.update_user_channel(time=0.0)
    H = beamformer.H

    symbol_indices = torch.randint(0, 4, (K,), device=device)
    symbols = qpsk_constellation[symbol_indices]

    print(f"Channel matrix H: {H.shape}")
    print(f"Channel condition number: {torch.linalg.cond(H).item():.2f}")
    print(f"Symbols: {symbols}\n")

    # ===== 1-bit ADMM with diagnostics =====
    print(f"\n{'='*80}")
    print(f"1-BIT ADMM OPTIMIZATION")
    print(f"{'='*80}")

    admm = CG_MC1bit(beamformer, lambda_penalty=1.0, max_iterations=100,
                    verbose=True, device=device)  # Enable verbose
    result_1bit = admm.optimize(symbols, total_power)

    antenna_signals_1bit = result_1bit['antenna_signals']

    print(f"\nADMM Result:")
    print(f"  Converged: {result_1bit.get('converged', 'Unknown')}")
    print(f"  Iterations: {result_1bit.get('iterations', 'Unknown')}")
    print(f"  Final residual: {result_1bit.get('residual', 'Unknown')}")

    # Check quantization
    unique_vals_real = torch.unique(antenna_signals_1bit.real).cpu().numpy()
    unique_vals_imag = torch.unique(antenna_signals_1bit.imag).cpu().numpy()
    print(f"\nQuantization check:")
    print(f"  Unique real values: {unique_vals_real}")
    print(f"  Unique imag values: {unique_vals_imag}")
    print(f"  Expected: scales of {{-1, +1}}")

    # Analyze received signals
    y_1bit = analyze_received_signals(H, antenna_signals_1bit, symbols, noise_power, "1-BIT")

    # ===== Continuous Wiener Filter =====
    print(f"\n{'='*80}")
    print(f"CONTINUOUS WIENER FILTER")
    print(f"{'='*80}")

    antenna_signals_continuous = compute_wiener_filter_precoding(H, symbols, noise_power, total_power)
    y_continuous = analyze_received_signals(H, antenna_signals_continuous, symbols,
                                           noise_power, "CONTINUOUS")

    # ===== Compare SER =====
    print(f"\n{'='*80}")
    print(f"SYMBOL ERROR RATE COMPARISON")
    print(f"{'='*80}")

    num_trials = 1000
    errors_1bit = 0
    errors_continuous = 0
    total_symbols = 0

    for trial in range(num_trials):
        err_1bit, n_symbols = compute_ser_single_trial(
            H, antenna_signals_1bit, symbols, qpsk_constellation,
            noise_power, device
        )
        err_cont, _ = compute_ser_single_trial(
            H, antenna_signals_continuous, symbols, qpsk_constellation,
            noise_power, device
        )

        errors_1bit += err_1bit
        errors_continuous += err_cont
        total_symbols += n_symbols

    ser_1bit = errors_1bit / total_symbols
    ser_continuous = errors_continuous / total_symbols

    print(f"\nAfter {num_trials} noise realizations:")
    print(f"  1-bit SER:      {ser_1bit:.4f} ({ser_1bit*100:.2f}%)")
    print(f"  Continuous SER: {ser_continuous:.4f} ({ser_continuous*100:.2f}%)")
    print(f"  Gap: {ser_1bit/ser_continuous:.1f}x worse" if ser_continuous > 0 else "  Continuous is perfect!")

    # ===== Analysis =====
    print(f"\n{'='*80}")
    print(f"ANALYSIS")
    print(f"{'='*80}\n")

    print("Expected: At 10 dBm, both should have low SER with ~3dB gap in required power")
    print("Observed: 1-bit has ~31% SER, continuous has 0% SER\n")

    print("Possible issues:")
    print("1. Lambda=1.0 might be wrong for SER (worked for sum rate)")
    print("2. ADMM might not be converging properly")
    print("3. Power normalization might be incorrect")
    print("4. Quantization constraint might be too strict")
    print("5. Different path losses per user might need alpha scaling")

    print(f"\n{'='*80}\n")
