'''
Author: Alfredo Gonzalez

Project: 1-bit DAC beamforming with and without SIM
         Implements CG-MC1bit algorithm from paper:
         "One-Bit Downlink Precoding for Massive MIMO OFDM System" (Wen et al., 2023)
'''
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA, WaterFilling, CG_MC1bit, quantize_to_1bit

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ========== Modified Beamformer Methods for 1-bit DAC ==========

def compute_sum_rate_1bit_standalone(beamformer, antenna_signals_quantized, power_allocation):
    """
    Compute sum-rate with 1-bit DAC (no SIM, direct channel transmission).

    Args:
        beamformer: Beamformer object
        antenna_signals_quantized: (M,) tensor - quantized antenna signals
        power_allocation: (K,) tensor - power per user

    Returns:
        sum_rate: scalar - sum of user rates
    """
    H = beamformer.H  # (K, M) direct channel
    K, M = H.shape

    # In M=K architecture: antenna m primarily serves user m
    # But with 1-bit DAC, we can't control magnitude - only phase
    # So we scale by power_allocation
    a_scaled = antenna_signals_quantized * torch.sqrt(power_allocation)

    # Compute SINR for each user
    sinr = torch.zeros(K, device=beamformer.device)
    for k in range(K):
        # Signal power: |H[k,:] @ a_scaled|²
        signal_power = torch.abs(H[k, :] @ a_scaled)**2

        # Interference: sum of signals from other antennas
        # In ideal case: antenna k serves user k, others interfere
        # But with quantization, this is approximate
        interference = 0.0
        for j in range(M):
            if j != k:
                # Interference from antenna j at user k
                interference += power_allocation[j] * torch.abs(H[k, j])**2

        sinr[k] = signal_power / (interference + beamformer.noise_power)

    sum_rate = torch.sum(torch.log2(1 + sinr))
    return sum_rate


def compute_sum_rate_1bit_with_sim(beamformer, phases, antenna_signals_quantized, power_allocation):
    """
    Compute sum-rate with 1-bit DAC + SIM.

    Args:
        beamformer: Beamformer object with SIM
        phases: (L, N) tensor - SIM phases
        antenna_signals_quantized: (M,) tensor - quantized antenna signals
        power_allocation: (K,) tensor - power per user

    Returns:
        sum_rate: scalar
    """
    # Compute effective channel through SIM
    H_eff = beamformer.compute_end_to_end_channel(phases)  # (K, M)

    # Scale quantized signals by power
    a_scaled = antenna_signals_quantized * torch.sqrt(power_allocation)

    # Compute SINR with quantized signals
    K, M = H_eff.shape
    sinr = torch.zeros(K, device=beamformer.device)

    for k in range(K):
        signal_power = torch.abs(H_eff[k, :] @ a_scaled)**2

        interference = 0.0
        for j in range(M):
            if j != k:
                interference += power_allocation[j] * torch.abs(H_eff[k, j])**2

        sinr[k] = signal_power / (interference + beamformer.noise_power)

    sum_rate = torch.sum(torch.log2(1 + sinr))
    return sum_rate


# ========== Main Execution ==========

if __name__ == "__main__":
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/1bit_run_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"1-BIT DAC BEAMFORMING COMPARISON")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")

    # ========== Parameters (same as main.py) ==========
    device = 'cpu'
    num_users = 4
    wavelength = 0.125  # meters

    # Antenna array
    Nx_antenna = 2
    Ny_antenna = 2
    num_antennas = Nx_antenna * Ny_antenna  # M = K = 4

    # SIM parameters
    sim_layers = 2
    sim_metaatoms = 25
    sim_layer_spacing = wavelength * 2.4
    sim_metaatom_spacing = wavelength / 2
    sim_metaatom_area = sim_metaatom_spacing**2

    # Channel parameters (CLT mode)
    min_user_distance = 10  # meters
    max_user_distance = 100  # meters
    path_loss_at_reference = -30.0  # dB
    reference_distance = 1.0  # meters

    # Power parameters
    noise_power = 10**(-80/10) / 1000  # Watts
    total_power = 10**(26/10) / 1000  # Watts (26 dBm)

    print(f"System Configuration:")
    print(f"  Users: K = {num_users}")
    print(f"  Antennas: M = {num_antennas} (M=K architecture)")
    print(f"  Wavelength: λ = {wavelength} m")
    print(f"  Noise power: σ² = -80 dBm")
    print(f"  Total power: P_T = 26 dBm")
    print(f"  SIM: {sim_layers} layers × {sim_metaatoms} meta-atoms\n")

    # ========== Setup Beamformers ==========

    # Beamformer 1: Standalone (no SIM) for 1-bit ADMM baseline
    beamformer_standalone = Beamformer(
        Nx=Nx_antenna,
        Ny=Ny_antenna,
        wavelength=wavelength,
        device=device,
        num_users=num_users,
        user_positions=None,  # CLT mode
        reference_distance=reference_distance,
        path_loss_at_reference=path_loss_at_reference,
        min_user_distance=min_user_distance,
        max_user_distance=max_user_distance,
        sim_model=None,  # NO SIM
        noise_power=noise_power,
        total_power=total_power,
    )

    # Beamformer 2: With SIM for 1-bit ADMM + SIM
    sim_model = Sim(
        layers=sim_layers,
        metaAtoms=sim_metaatoms,
        layerSpacing=sim_layer_spacing,
        metaAtomSpacing=sim_metaatom_spacing,
        metaAtomArea=sim_metaatom_area,
        wavelength=wavelength,
        device=device
    )

    beamformer_with_sim = Beamformer(
        Nx=Nx_antenna,
        Ny=Ny_antenna,
        wavelength=wavelength,
        device=device,
        num_users=num_users,
        user_positions=None,  # CLT mode
        reference_distance=reference_distance,
        path_loss_at_reference=path_loss_at_reference,
        min_user_distance=min_user_distance,
        max_user_distance=max_user_distance,
        sim_model=sim_model,  # WITH SIM
        noise_power=noise_power,
        total_power=total_power,
    )

    print("✓ Beamformers initialized")
    print(f"  Standalone channel shape: {beamformer_standalone.H.shape}")
    print(f"  SIM-based A matrix shape: {beamformer_with_sim.A.shape}")
    print(f"  SIM-based H matrix shape: {beamformer_with_sim.H.shape}\n")

    # ========== Power Sweep Configuration ==========
    power_values_db = np.array([10, 15, 20, 25, 30])  # dBm - SPOT CHECK: single power level
    power_values_linear = 10**(power_values_db/10) / 1000  # Watts

    num_runs_per_power = 20  # Monte Carlo trials - SPOT CHECK: reduced trials

    # Storage for results
    all_results = {}

    print(f"Power sweep: {power_values_db} dBm")
    print(f"Monte Carlo runs per power: {num_runs_per_power}\n")

    # ========== Run Comparisons ==========

    for power_idx, power_w in enumerate(power_values_linear):
        power_db = power_values_db[power_idx]
        print(f"\n{'='*80}")
        print(f"POWER LEVEL: {power_db:.1f} dBm ({power_w:.4e} W)")
        print(f"{'='*80}")

        # Update power for both beamformers
        beamformer_standalone.total_power = power_w
        beamformer_with_sim.total_power = power_w

        # Initialize results storage
        all_results[f'{power_db:.1f}dBm'] = {
            'power_linear': power_w,
            'power_db': power_db,
            'results': {
                '1bit_standalone': [],  # 1-bit ADMM only
                '1bit_sim': [],         # 1-bit ADMM + SIM + PGA
                'continuous_ref': []    # Continuous reference (no 1-bit)
            }
        }

        # Run Monte Carlo trials
        for run in range(num_runs_per_power):
            if run % 10 == 0:
                print(f"\nRun {run+1}/{num_runs_per_power}")

            # Generate new channel realization
            beamformer_standalone.update_user_channel(time=float(run))
            beamformer_with_sim.update_user_channel(time=float(run))

            # Generate QPSK symbols: {1+j, 1-j, -1+j, -1-j} / sqrt(2)
            qpsk_constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                              dtype=torch.complex64, device=device) / np.sqrt(2)
            symbol_indices = torch.randint(0, 4, (num_users,), device=device)
            symbols = qpsk_constellation[symbol_indices]

            # Initial power allocation (equal power)
            initial_power = torch.ones(num_users, device=device) * (power_w / num_users)

            # ========== SCENARIO 1: 1-bit ADMM Standalone (No SIM) ==========
            if run % 10 == 0:
                print("  Running: 1-bit ADMM standalone...")

            # NOTE: Lambda=1.0 gives ~3dB loss matching paper (was 0.01)
            admm_standalone = CG_MC1bit(
                beamformer=beamformer_standalone,
                lambda_penalty=1.0,
                max_iterations=100,
                verbose=False,
                device=device
            )

            result_standalone = admm_standalone.optimize(symbols, power_w)
            antenna_signals_1bit = result_standalone['antenna_signals']

            # Compute sum-rate with 1-bit quantized signals
            sumrate_standalone = compute_sum_rate_1bit_standalone(
                beamformer_standalone,
                antenna_signals_1bit,
                initial_power
            )

            all_results[f'{power_db:.1f}dBm']['results']['1bit_standalone'].append({
                'sumrate': sumrate_standalone.item(),
                'converged': result_standalone['converged'],
                'iterations': result_standalone['iterations'],
                'mse_history': result_standalone['mse_history']
            })

            # ========== SCENARIO 2: 1-bit ADMM + SIM + PGA ==========
            if run % 10 == 0:
                print("  Running: 1-bit ADMM + SIM + PGA...")

            # Initialize SIM phases randomly
            sim_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

            # First: Compute end-to-end channel with initial phases
            H_eff_init = beamformer_with_sim.compute_end_to_end_channel(sim_phases)  # (K, M)

            # Create a temporary beamformer with this end-to-end channel for ADMM
            # This ensures antenna signals are size (M,) not (N,)
            beamformer_temp = type('obj', (object,), {
                'H': H_eff_init,
                'noise_power': beamformer_with_sim.noise_power
            })()

            # Optimize 1-bit precoding on end-to-end channel
            # NOTE: Lambda=1.0 gives ~3dB loss matching paper (was 0.01)
            admm_with_sim = CG_MC1bit(
                beamformer=beamformer_temp,
                lambda_penalty=1.0,
                max_iterations=100,
                verbose=False,
                device=device
            )

            result_with_sim = admm_with_sim.optimize(symbols, power_w)
            antenna_signals_1bit_sim = result_with_sim['antenna_signals']  # Now (M,) = (4,)

            # Second: Optimize SIM phases with PGA (accounting for 1-bit constraint)
            # Define objective that accounts for quantized antenna signals
            def objective_1bit_sim(phases):
                return compute_sum_rate_1bit_with_sim(
                    beamformer_with_sim,
                    phases,
                    antenna_signals_1bit_sim,
                    initial_power
                )

            pga_optimizer = PGA(
                beamformer=beamformer_with_sim,
                objective_fn=objective_1bit_sim,
                learning_rate=0.05,
                max_iterations=1000,
                verbose=False,
                use_backtracking=True
            )

            pga_result = pga_optimizer.optimize(sim_phases)
            optimized_phases = pga_result['optimal_params']

            # Final sum-rate with optimized phases
            sumrate_with_sim = compute_sum_rate_1bit_with_sim(
                beamformer_with_sim,
                optimized_phases,
                antenna_signals_1bit_sim,
                initial_power
            )

            all_results[f'{power_db:.1f}dBm']['results']['1bit_sim'].append({
                'sumrate': sumrate_with_sim.item(),
                'converged_admm': result_with_sim['converged'],
                'converged_pga': pga_result['converged'],
                'iterations_admm': result_with_sim['iterations'],
                'iterations_pga': pga_result['iterations']
            })

            # ========== REFERENCE: Continuous (No 1-bit DAC) ==========
            if run % 10 == 0:
                print("  Running: Continuous reference (no 1-bit)...")

            # Use PGA to optimize phases with continuous signals
            continuous_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

            def objective_continuous(phases):
                return beamformer_with_sim.compute_sum_rate(phases, initial_power)

            pga_continuous = PGA(
                beamformer=beamformer_with_sim,
                objective_fn=objective_continuous,
                learning_rate=0.1,
                max_iterations=2000,
                verbose=False,
                use_backtracking=True
            )

            continuous_result = pga_continuous.optimize(continuous_phases)
            sumrate_continuous = continuous_result['optimal_objective']

            all_results[f'{power_db:.1f}dBm']['results']['continuous_ref'].append({
                'sumrate': sumrate_continuous,
                'converged': continuous_result['converged'],
                'iterations': continuous_result['iterations']
            })

            if run % 10 == 0:
                print(f"  Results: 1-bit standalone={sumrate_standalone:.3f}, "
                      f"1-bit+SIM={sumrate_with_sim:.3f}, "
                      f"continuous={sumrate_continuous:.3f}")

        # Print summary for this power level
        standalone_rates = [r['sumrate'] for r in all_results[f'{power_db:.1f}dBm']['results']['1bit_standalone']]
        sim_rates = [r['sumrate'] for r in all_results[f'{power_db:.1f}dBm']['results']['1bit_sim']]
        continuous_rates = [r['sumrate'] for r in all_results[f'{power_db:.1f}dBm']['results']['continuous_ref']]

        print(f"\nSummary for {power_db:.1f} dBm:")
        print(f"  1-bit standalone: {np.mean(standalone_rates):.3f} ± {np.std(standalone_rates):.3f} bits/s/Hz")
        print(f"  1-bit + SIM:      {np.mean(sim_rates):.3f} ± {np.std(sim_rates):.3f} bits/s/Hz")
        print(f"  Continuous ref:   {np.mean(continuous_rates):.3f} ± {np.std(continuous_rates):.3f} bits/s/Hz")
        print(f"  Loss (1-bit standalone): {10*np.log10(np.mean(continuous_rates)/np.mean(standalone_rates)):.2f} dB")
        print(f"  Loss (1-bit + SIM):      {10*np.log10(np.mean(continuous_rates)/np.mean(sim_rates)):.2f} dB")

    # ========== Save Results ==========
    results_file = os.path.join(results_dir, 'all_results_1bit.pt')
    torch.save(all_results, results_file)
    print(f"\n✓ Results saved: {results_file}")

    # ========== Plot Results ==========

    # Plot 1: Sum-rate vs Power
    fig, ax = plt.subplots(figsize=(10, 6))

    standalone_means = []
    sim_means = []
    continuous_means = []

    standalone_stds = []
    sim_stds = []
    continuous_stds = []

    for power_db in power_values_db:
        key = f'{power_db:.1f}dBm'
        standalone_rates = [r['sumrate'] for r in all_results[key]['results']['1bit_standalone']]
        sim_rates = [r['sumrate'] for r in all_results[key]['results']['1bit_sim']]
        continuous_rates = [r['sumrate'] for r in all_results[key]['results']['continuous_ref']]

        standalone_means.append(np.mean(standalone_rates))
        sim_means.append(np.mean(sim_rates))
        continuous_means.append(np.mean(continuous_rates))

        standalone_stds.append(np.std(standalone_rates))
        sim_stds.append(np.std(sim_rates))
        continuous_stds.append(np.std(continuous_rates))

    ax.errorbar(power_values_db, standalone_means, yerr=standalone_stds,
                marker='s', linestyle='--', linewidth=2, capsize=5,
                label='1-bit ADMM Standalone (No SIM)', color='red', alpha=0.8)
    ax.errorbar(power_values_db, sim_means, yerr=sim_stds,
                marker='o', linestyle='-', linewidth=2, capsize=5,
                label='1-bit ADMM + SIM + PGA', color='blue', alpha=0.8)
    ax.errorbar(power_values_db, continuous_means, yerr=continuous_stds,
                marker='^', linestyle='-.', linewidth=2, capsize=5,
                label='Continuous (No 1-bit) Reference', color='green', alpha=0.8)

    ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=12)
    ax.set_title('1-Bit DAC Performance: Standalone vs SIM', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, '1bit_comparison_vs_power.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_file}")
    plt.close()

    # Plot 2: Performance Loss (dB)
    fig, ax = plt.subplots(figsize=(10, 6))

    loss_standalone = [10*np.log10(continuous_means[i]/standalone_means[i])
                      for i in range(len(power_values_db))]
    loss_sim = [10*np.log10(continuous_means[i]/sim_means[i])
               for i in range(len(power_values_db))]

    ax.plot(power_values_db, loss_standalone, 'rs--', linewidth=2, markersize=8,
            label='Loss: 1-bit Standalone', alpha=0.8)
    ax.plot(power_values_db, loss_sim, 'bo-', linewidth=2, markersize=8,
            label='Loss: 1-bit + SIM', alpha=0.8)
    ax.axhline(y=3, color='k', linestyle=':', alpha=0.5, label='3 dB reference')

    ax.set_xlabel('Transmit Power (dBm)', fontsize=12)
    ax.set_ylabel('Performance Loss (dB)', fontsize=12)
    ax.set_title('1-Bit DAC Performance Loss vs Continuous Baseline', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, '1bit_loss_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_file}")
    plt.close()

    # Plot 3: Box plot comparison at highest power
    fig, ax = plt.subplots(figsize=(10, 6))

    highest_power_key = f'{power_values_db[-1]:.1f}dBm'
    standalone_rates = [r['sumrate'] for r in all_results[highest_power_key]['results']['1bit_standalone']]
    sim_rates = [r['sumrate'] for r in all_results[highest_power_key]['results']['1bit_sim']]
    continuous_rates = [r['sumrate'] for r in all_results[highest_power_key]['results']['continuous_ref']]

    data = [standalone_rates, sim_rates, continuous_rates]
    labels = ['1-bit\nStandalone', '1-bit\n+ SIM', 'Continuous\n(Reference)']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    colors = ['red', 'blue', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Sum-Rate (bits/s/Hz)', fontsize=12)
    ax.set_title(f'Performance Distribution at {power_values_db[-1]:.0f} dBm ({num_runs_per_power} runs)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_file = os.path.join(results_dir, '1bit_boxplot_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_file}")
    plt.close()

    print(f"\n{'='*80}")
    print(f"SIMULATION COMPLETE")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")

# %%