'''
Diagnostic test to identify cause of gap discrepancy.

Tests:
1. More Monte Carlo trials (100 instead of 20)
2. Different lambda_penalty values
3. Actual experienced SNR vs target SNR
'''

import torch
import numpy as np
from datetime import datetime

from simpy.beamformer import Beamformer
from simpy.algorithm import CG_MC1bit, quantize_to_1bit

# Import from test_admm.py
import sys
sys.path.insert(0, '/Users/alfredogonzalez/Desktop/code/SIM_optimization')
from test_admm import compute_sum_rate_with_precoding, compute_wiener_filter_precoding

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

print(f"\n{'='*80}")
print(f"DIAGNOSTIC TEST: Investigating Gap Discrepancy")
print(f"{'='*80}\n")

device = 'cpu'

# Paper configuration
M = 64
K = 8
Nx = 8
Ny = 8

wavelength = 0.125
noise_power_dbm = -80
noise_power = 10**(-80/10) / 1000

# Test at SNR = 5dB (where paper shows ~3dB gap)
target_snr_db = 5
avg_channel_gain_db = -60
transmit_power_dbm = target_snr_db - avg_channel_gain_db + noise_power_dbm
total_power = 10**(transmit_power_dbm/10) / 1000

print(f"Configuration:")
print(f"  M={M}, K={K}")
print(f"  Target SNR: {target_snr_db} dB")
print(f"  Transmit power: {transmit_power_dbm:.1f} dBm")
print(f"  Noise power: {noise_power_dbm} dBm\n")

# Channel parameters
min_user_distance = 10
max_user_distance = 100
path_loss_at_reference = -30.0
reference_distance = 1.0

# =============================================================================
# TEST 1: More Monte Carlo Trials
# =============================================================================
print(f"{'='*80}")
print(f"TEST 1: Effect of Monte Carlo Sample Size")
print(f"{'='*80}\n")

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

for num_trials in [20, 50, 100]:
    print(f"Testing with {num_trials} trials...")

    results_1bit = []
    results_continuous = []
    actual_snrs = []

    for trial in range(num_trials):
        beamformer.update_user_channel(time=float(trial))

        # Compute actual experienced SNR
        H = beamformer.H
        avg_channel_gain = torch.mean(torch.sum(torch.abs(H)**2, dim=1))
        actual_snr = total_power * avg_channel_gain / noise_power
        actual_snr_db = 10 * np.log10(actual_snr.item())
        actual_snrs.append(actual_snr_db)

        # Generate QPSK symbols
        qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                           dtype=torch.complex64, device=device) / np.sqrt(2)
        symbols = qpsk[torch.randint(0, 4, (K,), device=device)]

        # 1-bit ADMM
        admm = CG_MC1bit(beamformer, lambda_penalty=0.01, max_iterations=100,
                        verbose=False, device=device)
        result = admm.optimize(symbols, total_power)
        sumrate_1bit = compute_sum_rate_with_precoding(
            H, result['antenna_signals'], symbols, noise_power, device
        )
        results_1bit.append(sumrate_1bit.item())

        # Continuous Wiener filter
        antenna_signals_continuous = compute_wiener_filter_precoding(
            H, symbols, noise_power, total_power
        )
        sumrate_continuous = compute_sum_rate_with_precoding(
            H, antenna_signals_continuous, symbols, noise_power, device
        )
        results_continuous.append(sumrate_continuous.item())

    mean_1bit = np.mean(results_1bit)
    mean_continuous = np.mean(results_continuous)
    loss_db = 10 * np.log10(mean_continuous / mean_1bit)
    mean_actual_snr = np.mean(actual_snrs)

    print(f"  Trials: {num_trials}")
    print(f"    Target SNR: {target_snr_db} dB")
    print(f"    Actual SNR: {mean_actual_snr:.2f} ± {np.std(actual_snrs):.2f} dB")
    print(f"    1-bit: {mean_1bit:.2f} bits/s/Hz")
    print(f"    Continuous: {mean_continuous:.2f} bits/s/Hz")
    print(f"    Loss: {loss_db:.2f} dB\n")

# =============================================================================
# TEST 2: Different Lambda Penalty Values
# =============================================================================
print(f"{'='*80}")
print(f"TEST 2: Effect of Lambda Penalty Parameter")
print(f"{'='*80}\n")

num_trials = 50
lambda_values = [0.001, 0.01, 0.1, 1.0]

print(f"Testing lambda values: {lambda_values}")
print(f"Number of trials: {num_trials}\n")

for lambda_penalty in lambda_values:
    print(f"Lambda = {lambda_penalty}...")

    results_1bit = []
    results_continuous = []

    for trial in range(num_trials):
        beamformer.update_user_channel(time=float(trial + 1000))  # Different seed
        H = beamformer.H

        # Generate symbols
        qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                           dtype=torch.complex64, device=device) / np.sqrt(2)
        symbols = qpsk[torch.randint(0, 4, (K,), device=device)]

        # 1-bit ADMM with this lambda
        admm = CG_MC1bit(beamformer, lambda_penalty=lambda_penalty,
                        max_iterations=100, verbose=False, device=device)
        result = admm.optimize(symbols, total_power)
        sumrate_1bit = compute_sum_rate_with_precoding(
            H, result['antenna_signals'], symbols, noise_power, device
        )
        results_1bit.append(sumrate_1bit.item())

        # Continuous (same for all lambda)
        antenna_signals_continuous = compute_wiener_filter_precoding(
            H, symbols, noise_power, total_power
        )
        sumrate_continuous = compute_sum_rate_with_precoding(
            H, antenna_signals_continuous, symbols, noise_power, device
        )
        results_continuous.append(sumrate_continuous.item())

    mean_1bit = np.mean(results_1bit)
    mean_continuous = np.mean(results_continuous)
    loss_db = 10 * np.log10(mean_continuous / mean_1bit)

    print(f"  Lambda = {lambda_penalty}:")
    print(f"    1-bit: {mean_1bit:.2f} bits/s/Hz")
    print(f"    Continuous: {mean_continuous:.2f} bits/s/Hz")
    print(f"    Loss: {loss_db:.2f} dB\n")

# =============================================================================
# TEST 3: SNR Definition Verification
# =============================================================================
print(f"{'='*80}")
print(f"TEST 3: SNR Definition - Target vs Actual")
print(f"{'='*80}\n")

print("Testing different target SNRs to verify actual experienced SNR...\n")

num_trials = 50

for target_snr_db in [0, 3, 5, 7, 10]:
    transmit_power_dbm = target_snr_db - avg_channel_gain_db + noise_power_dbm
    total_power = 10**(transmit_power_dbm/10) / 1000

    beamformer.total_power = total_power

    actual_snrs = []

    for trial in range(num_trials):
        beamformer.update_user_channel(time=float(trial + 2000))
        H = beamformer.H

        # Compute actual SNR
        avg_channel_gain = torch.mean(torch.sum(torch.abs(H)**2, dim=1))
        actual_snr = total_power * avg_channel_gain / noise_power
        actual_snr_db = 10 * np.log10(actual_snr.item())
        actual_snrs.append(actual_snr_db)

    mean_actual_snr = np.mean(actual_snrs)
    std_actual_snr = np.std(actual_snrs)
    snr_error = mean_actual_snr - target_snr_db

    print(f"Target SNR: {target_snr_db} dB")
    print(f"  Transmit power: {transmit_power_dbm:.1f} dBm")
    print(f"  Actual SNR: {mean_actual_snr:.2f} ± {std_actual_snr:.2f} dB")
    print(f"  Error: {snr_error:.2f} dB\n")

print(f"{'='*80}")
print(f"DIAGNOSTIC TEST COMPLETE")
print(f"{'='*80}\n")

print("Summary of Findings:")
print("  1. Monte Carlo sample size: Check if gap grows with more trials")
print("  2. Lambda penalty: Check if different λ affects gap")
print("  3. SNR definition: Check if target SNR matches actual SNR")
print("\nIf none of these explain the 1.5dB vs 3dB discrepancy,")
print("the most likely cause is OFDM vs single-carrier difference.")
print(f"{'='*80}\n")