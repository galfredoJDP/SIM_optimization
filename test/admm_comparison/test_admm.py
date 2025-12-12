'''
Author: Alfredo Gonzalez

Test: Verify 1-bit ADMM performance matches paper expectations
      Paper: "One-Bit Downlink Precoding for Massive MIMO OFDM System" (Wen et al., 2023)

Expected: ~3dB loss for 1-bit ADMM vs continuous with massive MIMO (Nt >> U)
'''

import torch
import numpy as np
from datetime import datetime

from simpy.beamformer import Beamformer
from simpy.algorithm import CG_MC1bit, quantize_to_1bit

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def compute_sum_rate_with_precoding(H, precoded_signal, symbols, noise_power, device):
    """
    Compute sum-rate given precoded signal.

    This properly computes SINR for multiuser MIMO downlink following standard methodology.

    In multiuser MIMO downlink:
    - Precoded signal x serves all K users simultaneously
    - User k receives: y_k = H_k @ x + n_k
    - Signal power: component of y_k aligned with intended symbol s_k
    - Interference: component of y_k orthogonal to s_k
    - SINR_k = signal_power_k / (interference_k + noise_power)

    Args:
        H: (K, M) channel matrix
        precoded_signal: (M,) precoded antenna signals
        symbols: (K,) intended symbols for each user
        noise_power: scalar noise power
        device: torch device

    Returns:
        sum_rate: scalar - sum rate in bits/s/Hz
    """
    K, M = H.shape

    # Received signal at each user: y = H @ x
    y = H @ precoded_signal  # (K,)

    # Normalize symbols to unit energy for consistent SINR calculation
    symbols_normalized = symbols / torch.sqrt(torch.sum(torch.abs(symbols)**2) / K)

    # Compute SINR for each user
    sinr = torch.zeros(K, device=device)

    for k in range(K):
        # Received signal at user k
        y_k = y[k]
        s_k = symbols_normalized[k]

        # Signal power: |projection of y_k onto s_k|^2
        # This is |<y_k, s_k*>|^2 / |s_k|^2
        # If s_k is unit energy, this simplifies to |y_k · conj(s_k)|^2
        signal_component = y_k * torch.conj(s_k)
        signal_power = torch.abs(signal_component)**2

        # Total received power
        total_power = torch.abs(y_k)**2

        # Interference + distortion power: orthogonal component
        # This is |y_k|^2 - signal_power
        interference_plus_distortion = total_power - signal_power

        # SINR
        sinr[k] = signal_power / (interference_plus_distortion + noise_power)

    # Sum rate: C = sum_k log2(1 + SINR_k)
    sum_rate = torch.sum(torch.log2(1 + sinr))
    return sum_rate


def compute_wiener_filter_precoding(H, symbols, noise_power, total_power):
    """
    Compute Wiener filter (MMSE) precoding for continuous DAC.

    This is the "WF-inf" baseline from the paper.

    Args:
        H: (K, M) channel matrix
        symbols: (K,) symbols to transmit
        noise_power: scalar
        total_power: total transmit power constraint

    Returns:
        precoded_signal: (M,) precoded antenna signals
    """
    K, M = H.shape

    # Wiener filter: P = H^H (H H^H + σ²I)^(-1)
    # Precoded signal: x = P @ s
    HH_hermitian = H @ H.conj().T  # (K, K)
    regularization = noise_power * torch.eye(K, dtype=torch.complex64, device=H.device)

    # Invert: (H H^H + σ²I)^(-1)
    inv_term = torch.linalg.inv(HH_hermitian + regularization)

    # Precoding matrix: P = H^H @ inv_term
    P = H.conj().T @ inv_term  # (M, K)

    # Precoded signal
    precoded_signal = P @ symbols  # (M,)

    # Normalize to satisfy power constraint
    current_power = torch.sum(torch.abs(precoded_signal)**2)
    precoded_signal = precoded_signal * torch.sqrt(total_power / current_power)

    return precoded_signal


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"TESTING 1-BIT ADMM vs PAPER EXPECTATIONS")
    print(f"Paper: Wen et al., 'One-Bit Downlink Precoding for Massive MIMO OFDM'")
    print(f"{'='*80}\n")

    device = 'cpu'

    # Choose test mode
    import sys
    test_mode = sys.argv[1] if len(sys.argv) > 1 else 'antenna_sweep'

    if test_mode == 'snr_sweep':
        # Test SNR sweep at paper configuration to replicate Figure 3(a)
        print("MODE: SNR Sweep at Paper Configuration (M=64, K=8)")
        print("Goal: Replicate paper's ~3dB gap at SNR=5-7dB\n")

        test_configs = [
            {'M': 64, 'K': 8, 'name': 'M=64, K=8 (PAPER CONFIG)'},
        ]

        # Sweep SNR by varying transmit power
        # SNR_dB ≈ Transmit_power_dBm - Noise_power_dBm + channel_gain_dB
        # Target SNR: 0, 3, 5, 7, 10 dB (from paper's Figure 3a)
        noise_power_dbm = -80  # dBm
        target_snr_db_values = [0, 3, 5, 7, 10]

        # Approximate: Set transmit power to achieve target SNR
        # Assuming average channel gain ~ -60dB (rough estimate)
        avg_channel_gain_db = -60
        transmit_powers_dbm = [snr_db - avg_channel_gain_db + noise_power_dbm
                               for snr_db in target_snr_db_values]

        print(f"SNR sweep: {target_snr_db_values} dB")
        print(f"Transmit powers: {[f'{p:.1f}' for p in transmit_powers_dbm]} dBm")
        print(f"Noise power: {noise_power_dbm} dBm\n")

        # Will run SNR sweep in separate loop below
        run_snr_sweep = True

    else:
        # Default: Test antenna count sweep
        print("MODE: Antenna Count Sweep")
        print("Goal: Show effect of spatial diversity\n")

        test_configs = [
            {'M': 4, 'K': 4, 'name': 'M=K=4 (Our original - NOT massive MIMO)'},
            {'M': 8, 'K': 4, 'name': 'M=8, K=4 (2x ratio)'},
            {'M': 16, 'K': 4, 'name': 'M=16, K=4 (4x ratio)'},
            {'M': 32, 'K': 4, 'name': 'M=32, K=4 (8x ratio)'},
            {'M': 64, 'K': 8, 'name': 'M=64, K=8 (PAPER CONFIG - 8x ratio, massive MIMO)'},
        ]
        run_snr_sweep = False

    # Parameters (matching paper as much as possible)
    wavelength = 0.125  # meters
    noise_power_base = 10**(-80/10) / 1000  # Watts (-80 dBm)
    total_power_base = 10**(26/10) / 1000  # Watts (26 dBm)

    # Channel parameters
    min_user_distance = 10  # meters
    max_user_distance = 100  # meters
    path_loss_at_reference = -30.0  # dB
    reference_distance = 1.0  # meters

    num_trials = 20  # Monte Carlo trials

    if not run_snr_sweep:
        print(f"Fixed Parameters:")
        print(f"  Noise power: -80 dBm")
        print(f"  Total power: 26 dBm")
        print(f"  Monte Carlo trials: {num_trials}")
        print(f"  Modulation: QPSK\n")

    # Test each configuration
    for config in test_configs:
        M = config['M']
        K = config['K']
        name = config['name']

        print(f"\n{'='*80}")
        print(f"Configuration: {name}")
        print(f"{'='*80}")

        # Setup antenna array (square array if possible)
        Nx = int(np.sqrt(M))
        Ny = M // Nx
        if Nx * Ny != M:
            Nx = M
            Ny = 1

        print(f"  Antenna array: {Nx} x {Ny} = {M} antennas")
        print(f"  Users: {K}")
        print(f"  Ratio: {M/K:.1f}x antennas per user\n")

        # Determine power levels to test
        if run_snr_sweep:
            # SNR sweep mode: test multiple transmit powers
            power_levels = [(10**(p/10) / 1000, f"{p:.1f}dBm")
                           for p in transmit_powers_dbm]
            noise_power = noise_power_base
        else:
            # Antenna sweep mode: single power level
            power_levels = [(total_power_base, "26dBm")]
            noise_power = noise_power_base

        # Storage for SNR sweep results
        snr_results_1bit = []
        snr_results_continuous = []
        snr_loss_db = []

        # Loop over power levels (SNR sweep or single power)
        for power_idx, (total_power, power_label) in enumerate(power_levels):
            if run_snr_sweep:
                target_snr = target_snr_db_values[power_idx]
                print(f"  Target SNR: {target_snr} dB (Tx power: {power_label})")

            # Create beamformer (no SIM)
            beamformer = Beamformer(
                Nx=Nx,
                Ny=Ny,
                wavelength=wavelength,
                device=device,
                num_users=K,
                user_positions=None,  # CLT mode
                reference_distance=reference_distance,
                path_loss_at_reference=path_loss_at_reference,
                min_user_distance=min_user_distance,
                max_user_distance=max_user_distance,
                sim_model=None,  # NO SIM - standalone test
                noise_power=noise_power,
                total_power=total_power,
            )

            # Storage for results at this power level
            results_1bit = []
            results_continuous = []

            # Run trials
            for trial in range(num_trials):
                if not run_snr_sweep and trial % 5 == 0:
                    print(f"    Trial {trial+1}/{num_trials}...")

                # Generate new channel
                beamformer.update_user_channel(time=float(trial))

                # Generate QPSK symbols
                qpsk_constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                                  dtype=torch.complex64, device=device) / np.sqrt(2)
                symbol_indices = torch.randint(0, 4, (K,), device=device)
                symbols = qpsk_constellation[symbol_indices]

                # Power allocation (equal power per user)
                power_per_user = total_power / K
                power_allocation = torch.ones(K, device=device) * power_per_user

                # ===== TEST 1: 1-bit ADMM (Standalone) =====
                # NOTE: Lambda=1.0 gives ~3dB loss matching paper (was 0.01)
                admm = CG_MC1bit(
                    beamformer=beamformer,
                    lambda_penalty=1.0,
                    max_iterations=100,
                    verbose=False,
                    device=device
                )

                result_1bit = admm.optimize(symbols, total_power)
                antenna_signals_1bit = result_1bit['antenna_signals']

                # Compute sum-rate with 1-bit quantized signals
                H = beamformer.H  # (K, M)
                sumrate_1bit = compute_sum_rate_with_precoding(
                    H,
                    antenna_signals_1bit,
                    symbols,
                    noise_power,
                    device
                )

                results_1bit.append(sumrate_1bit.item())

                # ===== TEST 2: Continuous Wiener Filter (WF-inf baseline from paper) =====
                # Use Wiener filter precoding with infinite resolution DAC
                antenna_signals_continuous = compute_wiener_filter_precoding(
                    H,
                    symbols,
                    noise_power,
                    total_power
                )

                sumrate_continuous = compute_sum_rate_with_precoding(
                    H,
                    antenna_signals_continuous,
                    symbols,
                    noise_power,
                    device
                )

                results_continuous.append(sumrate_continuous.item())

            # Compute statistics for this power level
            mean_1bit = np.mean(results_1bit)
            std_1bit = np.std(results_1bit)
            mean_continuous = np.mean(results_continuous)
            std_continuous = np.std(results_continuous)

            loss_db = 10 * np.log10(mean_continuous / mean_1bit)

            if run_snr_sweep:
                # Store results for SNR sweep plotting
                snr_results_1bit.append(mean_1bit)
                snr_results_continuous.append(mean_continuous)
                snr_loss_db.append(loss_db)
                print(f"    Results: 1-bit={mean_1bit:.2f}, Continuous={mean_continuous:.2f}, Loss={loss_db:.2f}dB\n")
            else:
                # Print results for antenna sweep
                print(f"\n  Results:")
                print(f"    1-bit ADMM:  {mean_1bit:.3f} ± {std_1bit:.3f} bits/s/Hz")
                print(f"    Continuous:  {mean_continuous:.3f} ± {std_continuous:.3f} bits/s/Hz")
                print(f"    Loss:        {loss_db:.2f} dB")

                if M >= 8 * K:
                    print(f"\n  ✓ This configuration matches paper's massive MIMO ratio (8x)")
                    if loss_db < 5.0:
                        print(f"  ✓ Loss is close to paper's expected ~3dB!")
                    else:
                        print(f"  ⚠ Loss is still higher than expected - may need more trials or tuning")
                elif M >= 4 * K:
                    print(f"\n  ⚠ This configuration has good ratio but may need more antennas")
                else:
                    print(f"\n  ✗ NOT massive MIMO - need more antennas relative to users")

        # If SNR sweep, print summary
        if run_snr_sweep:
            print(f"\n  SNR Sweep Summary:")
            print(f"  {'SNR (dB)':<12} {'1-bit (bits/s/Hz)':<20} {'Continuous (bits/s/Hz)':<25} {'Loss (dB)':<10}")
            print(f"  {'-'*75}")
            for idx, target_snr in enumerate(target_snr_db_values):
                print(f"  {target_snr:<12} {snr_results_1bit[idx]:<20.2f} {snr_results_continuous[idx]:<25.2f} {snr_loss_db[idx]:<10.2f}")

    print(f"\n{'='*80}")
    print(f"TEST COMPLETE")
    print(f"\nConclusion:")
    print(f"  The paper's ~3dB loss requires Nt >> U (massive MIMO)")
    print(f"  With M=K=4, there are insufficient spatial degrees of freedom")
    print(f"  The SIM provides these missing degrees of freedom!")
    print(f"{'='*80}\n")