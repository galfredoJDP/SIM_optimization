'''
Author: Alfredo Gonzalez

Test: Sweep lambda penalty to find optimal value for SER performance
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/Users/alfredogonzalez/Desktop/code/SIM_optimization')

from simpy.beamformer import Beamformer
from simpy.algorithm import CG_MC1bit

np.random.seed(42)
torch.manual_seed(42)


def compute_ser_quick(H, antenna_signals, symbols, qpsk_constellation, noise_power, num_trials, device):
    """Quick SER computation."""
    K = H.shape[0]
    errors = 0
    total = 0

    y_clean = H @ antenna_signals

    for _ in range(num_trials):
        noise = (torch.randn(K, device=device) + 1j * torch.randn(K, device=device)) * np.sqrt(noise_power / 2)
        y = y_clean + noise

        for k in range(K):
            decoded_idx = torch.argmin(torch.abs(y[k] - qpsk_constellation))
            true_idx = torch.argmin(torch.abs(symbols[k] - qpsk_constellation))
            if decoded_idx != true_idx:
                errors += 1
            total += 1

    return errors / total


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"LAMBDA PENALTY SWEEP FOR SER OPTIMIZATION")
    print(f"{'='*80}\n")

    device = 'cpu'

    # Configuration
    M = 64
    K = 8
    Nx = 8
    Ny = 8
    wavelength = 0.125
    noise_power = 10**(-80/10) / 1000

    min_user_distance = 10
    max_user_distance = 100
    path_loss_at_reference = -30.0
    reference_distance = 1.0

    # Test at high power
    power_dbm = 10
    total_power = 10**(power_dbm/10) / 1000

    # Lambda values to test
    lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"Testing lambda values: {lambda_values}")
    print(f"Power: {power_dbm} dBm, M={M}, K={K}\n")

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

    results = []

    for lambda_val in lambda_values:
        print(f"\nTesting lambda = {lambda_val}")

        ser_trials = []
        converged_count = 0
        mse_final_list = []

        # Multiple channel realizations
        for trial in range(5):
            beamformer.update_user_channel(time=float(trial))
            H = beamformer.H

            symbol_indices = torch.randint(0, 4, (K,), device=device)
            symbols = qpsk_constellation[symbol_indices]

            # Run ADMM
            admm = CG_MC1bit(beamformer, lambda_penalty=lambda_val,
                           max_iterations=200, verbose=False, device=device)
            result = admm.optimize(symbols, total_power)

            antenna_signals = result['antenna_signals']
            converged = result['converged']
            mse_final = result['mse_history'][-1]

            if converged:
                converged_count += 1
            mse_final_list.append(mse_final)

            # Compute SER
            ser = compute_ser_quick(H, antenna_signals, symbols,
                                   qpsk_constellation, noise_power, 100, device)
            ser_trials.append(ser)

        mean_ser = np.mean(ser_trials)
        std_ser = np.std(ser_trials)
        mean_mse = np.mean(mse_final_list)

        print(f"  SER: {mean_ser:.4f} ± {std_ser:.4f}")
        print(f"  Converged: {converged_count}/5")
        print(f"  Final MSE: {mean_mse:.4f}")

        results.append({
            'lambda': lambda_val,
            'ser_mean': mean_ser,
            'ser_std': std_ser,
            'converged': converged_count,
            'mse': mean_mse
        })

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lambda_vals = [r['lambda'] for r in results]
    ser_means = [r['ser_mean'] for r in results]
    ser_stds = [r['ser_std'] for r in results]
    mses = [r['mse'] for r in results]

    # SER vs lambda
    ax1.errorbar(lambda_vals, ser_means, yerr=ser_stds, marker='o',
                 linewidth=2, markersize=8, capsize=5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Lambda Penalty', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Symbol Error Rate', fontsize=12, fontweight='bold')
    ax1.set_title('SER vs Lambda Penalty', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # MSE vs lambda
    ax2.plot(lambda_vals, mses, 'b-o', linewidth=2, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda Penalty', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final MSE', fontsize=12, fontweight='bold')
    ax2.set_title('Optimization MSE vs Lambda', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test/admm_comparison/lambda_sweep.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: test/admm_comparison/lambda_sweep.png")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    best_idx = np.argmin(ser_means)
    best_lambda = results[best_idx]['lambda']
    best_ser = results[best_idx]['ser_mean']

    print(f"Best lambda: {best_lambda}")
    print(f"Best SER: {best_ser:.4f} ({best_ser*100:.2f}%)")
    print(f"\nNote: Even best lambda has very high SER!")
    print(f"This suggests a fundamental issue with the ADMM implementation.")

    print(f"\n{'='*80}\n")

    plt.show()