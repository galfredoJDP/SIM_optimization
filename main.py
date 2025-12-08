'''
Author: Alfredo Gonzalez

Project: SIM Model with phase optimization and waterfilling algorihtm for power allocation
'''
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from simpy.sim import Sim
from simpy.beamformer import Beamformer
from simpy.algorithm import ProjectedGradientAscent as PGA, WaterFilling, DDPG, TD3
from datetime import datetime
import os, copy


# Set random seed for reproducibility
np.random.seed(35)
torch.manual_seed(35)




# ========== Visualization Functions ==========

def visualize_sim_with_antennas(sim_model, antenna_positions, save_path=None):
    """
    Visualize the SIM structure along with antenna positions.

    Args:
        sim_model: The SIM model object
        antenna_positions: Tensor of antenna positions (M, 3)
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 7))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot SIM layers
    colors = plt.cm.viridis(np.linspace(0, 1, sim_model.layers))

    for layer in range(sim_model.layers):
        positions = sim_model.metaAtomPositions[layer].cpu().numpy()
        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=[colors[layer]], s=50, alpha=0.7,
                   label=f'SIM Layer {layer}')

    # Plot antennas
    ant_pos = antenna_positions.cpu().numpy()
    ax1.scatter(ant_pos[:, 0], ant_pos[:, 1], ant_pos[:, 2],
               c='red', s=100, marker='^', alpha=0.9,
               label='Antennas (BS)', edgecolors='black', linewidths=1.5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title('SIM-Based Beamforming: Antennas + SIM Structure')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(122)

    for layer in range(sim_model.layers):
        positions = sim_model.metaAtomPositions[layer].cpu().numpy()
        marker = 'o' if layer == 0 else ('s' if layer == sim_model.layers-1 else '^')
        label_suffix = ' (BS side)' if layer == 0 else (' (User side)' if layer == sim_model.layers-1 else '')

        ax2.scatter(positions[:, 0], positions[:, 1],
                   c=[colors[layer]], s=50, alpha=0.7,
                   marker=marker, label=f'SIM Layer {layer}{label_suffix}')

    # Plot antennas on top view
    ax2.scatter(ant_pos[:, 0], ant_pos[:, 1],
               c='red', s=100, marker='^', alpha=0.9,
               label='Antennas (BS)', edgecolors='black', linewidths=1.5)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Top View (XY Plane)')
    ax2.set_aspect('equal')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add origin marker
    ax2.plot(0, 0, 'k+', markersize=15, markeredgewidth=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_antenna_array(antenna_positions, array_config, save_path=None):
    """
    Visualize antenna array positions for digital beamforming.

    Args:
        antenna_positions: Tensor of antenna positions (M, 3)
        array_config: Tuple (Nx, Ny) for array configuration
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(16, 7))

    ant_pos = antenna_positions.cpu().numpy()
    Nx, Ny = array_config

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.scatter(ant_pos[:, 0], ant_pos[:, 1], ant_pos[:, 2],
               c='blue', s=100, marker='o', alpha=0.9,
               edgecolors='black', linewidths=1.5)

    # Add lines connecting antennas to show array structure
    for i in range(len(ant_pos)):
        ax1.plot([ant_pos[i, 0], ant_pos[i, 0]],
                [ant_pos[i, 1], ant_pos[i, 1]],
                [ant_pos[i, 2], -0.01],
                'k--', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title(f'Digital Beamforming: Antenna Array ({Nx}x{Ny} = {len(ant_pos)} antennas)')
    ax1.grid(True, alpha=0.3)

    # Top-down view (XY plane)
    ax2 = fig.add_subplot(122)

    ax2.scatter(ant_pos[:, 0], ant_pos[:, 1],
               c='blue', s=100, marker='o', alpha=0.9,
               edgecolors='black', linewidths=1.5)

    # Add grid lines to show array structure
    # Vertical lines (constant x)
    unique_x = np.unique(np.round(ant_pos[:, 0], decimals=6))
    for x_val in unique_x:
        mask = np.abs(ant_pos[:, 0] - x_val) < 1e-6
        y_vals = ant_pos[mask, 1]
        if len(y_vals) > 1:
            ax2.plot([x_val, x_val], [y_vals.min(), y_vals.max()],
                    'k--', alpha=0.3, linewidth=0.5)

    # Horizontal lines (constant y)
    unique_y = np.unique(np.round(ant_pos[:, 1], decimals=6))
    for y_val in unique_y:
        mask = np.abs(ant_pos[:, 1] - y_val) < 1e-6
        x_vals = ant_pos[mask, 0]
        if len(x_vals) > 1:
            ax2.plot([x_vals.min(), x_vals.max()], [y_val, y_val],
                    'k--', alpha=0.3, linewidth=0.5)

    # Label each antenna with its index
    for i in range(len(ant_pos)):
        ax2.annotate(f'{i}', (ant_pos[i, 0], ant_pos[i, 1]),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=7, alpha=0.7)

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'Top View: {Nx}x{Ny} Array Layout')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Add origin marker
    ax2.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Origin')
    ax2.legend(loc='best', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()

# ========== Parameters ==========
if __name__ == "__main__":
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(timestamp)
    results_dir = f"results/run_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # System parameters
    num_users = 4
    wavelength = 0.125  # meters
    device = 'cpu'

    # Antenna array (for digital beamformer)
    Nx_antenna = 2
    Ny_antenna = 2
    num_antennas = Nx_antenna * Ny_antenna

    # SIM parameters (for SIM beamformer)
    sim_layers = 2
    sim_metaatoms = 25
    sim_layer_spacing =  wavelength*2.4  # meters
    sim_metaatom_spacing = wavelength/2  # meters
    sim_metaatom_area = sim_metaatom_spacing**2 # m^2 - Paper unclear, use spacing² (tightly packed)

    # Channel parameters (CLT mode - no user positions)
    min_user_distance = 0 # meters
    max_user_distance = 100 # meters
    path_loss_at_reference = -30.0  # dB
    reference_distance = 1.0  # meters

    # Power parameters
    noise_power = 10**(-80/10)/1000  # Watts
    total_power = 10**(26/10)/1000  # Watts

    # ========== Case 1: SIM-Based Beamformer ==========
    sim_model = Sim(
        layers=sim_layers,
        metaAtoms=sim_metaatoms,
        layerSpacing=sim_layer_spacing,
        metaAtomSpacing=sim_metaatom_spacing,
        metaAtomArea=sim_metaatom_area,
        wavelength=wavelength,
        device=device
    )
    # print(f"   SIM created: {sim_layers}L x {sim_metaatoms}N")
    # print(f"   Psi matrix shape: {sim_model.Psi.shape}")

    sim_beamformer = Beamformer(
        Nx=2,  # 2x2 = 4 antennas to match K=4 users
        Ny=2,
        wavelength=wavelength,
        device=device,
        # Channel params (CLT mode - no user_positions)
        num_users=num_users,
        user_positions=None,  # Will use CLT mode
        reference_distance=reference_distance,
        path_loss_at_reference=path_loss_at_reference,
        min_user_distance=min_user_distance,
        max_user_distance=max_user_distance,
        # SIM
        sim_model=sim_model,
        # System params
        noise_power=noise_power,
        total_power=total_power,
    )

    sim_beamformer_ddpg = copy.deepcopy(sim_beamformer)
    sim_beamformer_ddpg.device = 'mps'
    sim_beamformer_td3 = copy.deepcopy(sim_beamformer_ddpg)
    sim_beamformer_td3.device = 'mps'

    digital_beamformer = Beamformer(
        Nx=2,  # 2x2 = 4 antennas to match K=4 users
        Ny=2,
        wavelength=wavelength,
        device=device,
        # Channel params (CLT mode - no user_positions)
        num_users=num_users,
        user_positions=None,  # Will use CLT mode
        reference_distance=reference_distance,
        path_loss_at_reference=path_loss_at_reference,
        min_user_distance=min_user_distance,
        max_user_distance=max_user_distance,
        # System params
        noise_power=noise_power,
        total_power=total_power,
    )

    #===========   Optimzation ===========


    """
    Power sweep configuration
    """
    power_values_db = np.array([26])  # dBm values to sweep
    # power_values_db = np.array([ 7, 12, 17, 22, 27, 32])  # dBm values to sweep
    power_values_linear = 10**(power_values_db/10) / 1000  # Convert to Watts

    num_runs_per_power = 20
    alternative_iterations = 5

    # Dictionary to store all results
    all_results = {}
    sumrate_all_powers = {}

    torch.seed()

    #compute ZF weights, only needed once 



    for power_idx, power_w in enumerate(power_values_linear):
        print(timestamp)
        power_db = power_values_db[power_idx]
        print(f"\nPower: {power_db:.1f} dBm ({power_w:.4e} W)")
        print("-"*80)

        # Update beamformer with new power
        sim_beamformer.total_power = power_w
        sim_beamformer_ddpg.total_power = power_w
        sim_beamformer_td3.total_power = power_w
        digital_beamformer.total_power = power_w

        # Run optimization multiple times at this power level
        sumrate = []
        sinr_per_user = []
        
        all_results[f'{power_db:.1f}dBm'] = {
            'power_linear': power_w,
            'power_db': power_db,
            'results': [],
            'sumrate': [],
            'reference' : []
        }

        torch.seed()  # Reset seed for each power level

        for i in range(num_runs_per_power):
            print(timestamp)
            print("================")
            print("Run "+str(i))
            print("================")

            # Regenerate channels for this run (new channel realization)
            sim_beamformer.update_user_channel(time=float(i))
            sim_beamformer_td3.H = sim_beamformer.H.clone()
            sim_beamformer_ddpg.H = sim_beamformer.H.clone()

            power_allocation = torch.ones(num_users, device=device) * (power_w / num_users)
            sim_phases = torch.rand(sim_layers, sim_metaatoms, device=device) * 2 * np.pi

            #first do the ZF as it's deterministic 
            zf_weights = digital_beamformer.compute_zf_weights(sim_beamformer.H)
            H_eff_digital = sim_beamformer.H @ zf_weights

            waterfiling_digital =  WaterFilling(
                    H_eff=H_eff_digital,
                    noise_power=digital_beamformer.noise_power,
                    total_power=power_w,
                    max_iterations=200,
                    tolerance=1e-6,
                    verbose=False,
                    device=device
                )
            wf_results = waterfiling_digital.optimize()
            sumrate_after_waterfilling = wf_results['optimal_sum_rate']

            all_results[f'{power_db:.1f}dBm']['reference'].append(sumrate_after_waterfilling)
        
            for j in range(alternative_iterations):
                '''
                PGA Phase for SIM
                '''
                optimizer = PGA(
                    beamformer=sim_beamformer,
                    objective_fn=lambda phases: sim_beamformer.compute_sum_rate(phases=phases, power_allocation=power_allocation),
                    learning_rate=0.1,
                    max_iterations=5000,
                    verbose=False,
                    use_backtracking=True,
                    backtrack_max_iter=1000
                )       
                results = optimizer.optimize(sim_phases) 

                '''
                DDPG Phase for SIM 
                '''
                optimizer = DDPG(
                    beamformer=sim_beamformer,
                    state_dim=2 * num_users * sim_metaatoms + num_users,
                    action_dim= sim_layers * sim_metaatoms
                )
                results = optimizer.optimize(
                    num_episodes=100,
                    steps_per_episode=20,
                    power_allocation=power_allocation
                )


                optimal_phases = results['optimal_params']
                sumrate_after_phase_opt = results['optimal_objective']

                # Step 2: Apply waterfilling for power allocation with fixed optimal phases
                H_eff = sim_beamformer.compute_end_to_end_channel(optimal_phases)

                # #apply digital beamforming to SIM here
                # zf_weights_sim = sim_beamformer.compute_zf_weights(H_eff)
                # H_eff_sim_digital = H_eff @ zf_weights_sim

                waterfilling = WaterFilling(
                    H_eff=H_eff,
                    noise_power=sim_beamformer.noise_power,
                    total_power=power_w,
                    max_iterations=200,
                    tolerance=1e-6,
                    verbose=False,
                    device=device
                )

                wf_results = waterfilling.optimize(
                    initial_power = power_allocation
                )
                optimal_power = wf_results['optimal_power']
                sumrate_after_waterfilling = wf_results['optimal_sum_rate']

                # Store results with both phase-only and phase+waterfilling sum-rates
                sumrate.append(sumrate_after_waterfilling)

                # Detach and clear computational graph for next iteration
                power_allocation = optimal_power.detach().clone()
                sim_phases = optimal_phases.detach().clone()

                # Clear memory cache based on device
                if device == 'cuda':
                    torch.cuda.empty_cache()
                elif device == 'mps':
                    torch.mps.empty_cache()
                # For CPU, PyTorch automatically manages memory

                # Update results dict with power and sum-rates
                results['optimal_power'] = optimal_power
                results['sumrate_phase_only'] = sumrate_after_phase_opt
                results['sumrate_with_waterfilling'] = sumrate_after_waterfilling

                print(f"PGA : {sumrate_after_phase_opt} | WF : {sumrate_after_waterfilling}")
                
                # Add WF diagnostics and store
                all_results[f'{power_db:.1f}dBm']['results'].append({
                    **results,  # PGA results
                    'wf_sinr': wf_results['sinr_per_user'],
                    'wf_snr': wf_results['snr_per_user'],
                    'wf_rate_per_user': wf_results['rate_per_user'],
                    'wf_power_per_user': wf_results['power_per_user']
                })
                all_results[f'{power_db:.1f}dBm']['sumrate'].append(sumrate_after_waterfilling)
            

                

        # Save results for this power level
        sumrate_all_powers[f'{power_db:.1f}dBm'] = sumrate

    results_file = os.path.join(results_dir, 'all_results.pt')
    torch.save(all_results, results_file)


    #%%
    # Plot results across all power levels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatter plot for each power level
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sumrate_all_powers)))
    for (power_db, sumrates), color in zip(sumrate_all_powers.items(), colors):
        ax.scatter(range(len(sumrates)), sumrates, label=f'{power_db}', alpha=0.7, s=50, color=color)
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Sum-Rate (bits/s/Hz)')
    ax.set_title('Sum-Rate across Power Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean and std across power levels
    ax = axes[1]
    power_labels = list(sumrate_all_powers.keys())
    means = [np.mean(sumrate_all_powers[p]) for p in power_labels]
    stds = [np.std(sumrate_all_powers[p]) for p in power_labels]
    ax.errorbar(range(len(power_labels)), means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xticks(range(len(power_labels)))
    ax.set_xticklabels(power_labels, rotation=0)
    ax.set_ylabel('Mean Sum-Rate (bits/s/Hz)')
    ax.set_xlabel('Transmit Power')
    ax.set_title('Mean Performance vs Transmit Power')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, 'WF_PGA.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_file}")



    #%%
    #plot 3: something more similar to the paper 
    plt.figure()
    power_labels = list(sumrate_all_powers.keys())
    means = [np.mean(sumrate_all_powers[p]) for p in power_labels]
    plt.plot(power_values_db, means, 'bo-', markersize=8)

    reference_means = [np.mean(all_results[p]['reference']) for p in power_labels]
    plt.plot(power_values_db, reference_means, 'ro-', markersize=8)
    plt.legend(["WF : PGA", "WF : ZF"])
    plt.ylim(0, 25)
    plt.xlim(10, 35)
    plt.grid()
    plt.xlabel('Transmit Power (dBm)')
    plt.ylabel('Mean Sum-Rate (bits/s/Hz)')
    plt.title('Mean Sum-Rate vs Transmit Power')
    plot_file = os.path.join(results_dir, 'WF_PGA_mean.png')

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')







    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    for power_db, sumrates in sumrate_all_powers.items():
        print(f"\nPower: {power_db}")
        print(f"  Mean:   {np.mean(sumrates):.4f} bits/s/Hz")
        print(f"  Std:    {np.std(sumrates):.4f} bits/s/Hz")
        print(f"  Min:    {np.min(sumrates):.4f} bits/s/Hz")
        print(f"  Max:    {np.max(sumrates):.4f} bits/s/Hz")

    #%% Plot per-user diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SINR distribution across runs
    ax = axes[0, 0]
    for power_db in power_labels:
        sinr_data = [r['wf_sinr'] for r in all_results[power_db]['results']]
        sinr_mean = np.mean(sinr_data, axis=0)  # Mean per user
        ax.plot(range(num_users), 10*np.log10(sinr_mean), 'o-', label=power_db)
    ax.set_xlabel('User Index')
    ax.set_ylabel('Mean SINR (dB)')
    ax.set_title('SINR per User vs Power Level')
    ax.legend()
    ax.grid(True)

    # SNR vs SINR comparison
    ax = axes[0, 1]
    power_db = power_labels[-1]  # Highest power
    sinr_data = np.mean([r['wf_sinr'] for r in all_results[power_db]['results']], axis=0)
    snr_data = np.mean([r['wf_snr'] for r in all_results[power_db]['results']], axis=0)
    x = np.arange(num_users)
    ax.bar(x - 0.2, 10*np.log10(snr_data), 0.4, label='SNR (no interference)')
    ax.bar(x + 0.2, 10*np.log10(sinr_data), 0.4, label='SINR (with interference)')
    ax.set_xlabel('User Index')
    ax.set_ylabel('dB')
    ax.set_title(f'SNR vs SINR at {power_db}')
    ax.legend()
    ax.grid(True)

    # Rate per user
    ax = axes[1, 0]
    for power_db in power_labels:
        rate_data = [r['wf_rate_per_user'] for r in all_results[power_db]['results']]
        rate_mean = np.mean(rate_data, axis=0)
        ax.plot(range(num_users), rate_mean, 'o-', label=power_db)
    ax.set_xlabel('User Index')
    ax.set_ylabel('Rate (bits/s/Hz)')
    ax.set_title('Rate per User vs Power Level')
    ax.legend()
    ax.grid(True)

    # Power allocation per user
    ax = axes[1, 1]
    for power_db in power_labels:
        power_data = [r['wf_power_per_user'] for r in all_results[power_db]['results']]
        power_mean = np.mean(power_data, axis=0)
        ax.plot(range(num_users), power_mean, 'o-', label=power_db)
    ax.set_xlabel('User Index')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Allocation per User')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'per_user_diagnostics.png'), dpi=150)

    # print("\n6. Computing SINR and Sum-Rate...")
    # sum_rate_sim = sim_beamformer.compute_sum_rate(phases=results['optimal_params'], power_allocation=power_allocation)
    # print(f"   Sum-Rate:  {sum_rate_sim.item():.4f} bits/s/Hz")

    # print("\n7. Verifying channels used in SINR computation...")
    # print(f"   Pre-digital-weights channel shape: {sim_beamformer.pre_digital_weights_channel.shape}")
    # print(f"      = H @ Psi @ A (end-to-end through SIM)")
    # print(f"   Effective channel shape: {sim_beamformer.last_effective_channel.shape}")
    # print(f"      = Same as above (no digital weights in SIM-only case)")
    # print(f"   Interpretation: (K={sim_beamformer.last_effective_channel.shape[0]}, M={sim_beamformer.last_effective_channel.shape[1]})")
    # print(f"   Since M=K: H_eff[k,k] = signal, H_eff[k,j≠k] = interference")

    # # ========== Case 2: Digital Beamformer (No SIM) ==========
    # print("\n" + "="*80)
    # print("CASE 2: DIGITAL BEAMFORMING (Traditional M > K Architecture)")
    # print("="*80)

    # print("\n1. Creating Digital Beamformer ...")
    # digital_beamformer = Beamformer(
    #     # Transceiver params - Using M=64 antennas for K=4 users
    #     Nx=Nx_antenna,  # 8x8 = 64 antennas
    #     Ny=Ny_antenna,
    #     wavelength=wavelength,
    #     device=device,
    #     # Channel params (CLT mode - no user_positions)
    #     num_users=num_users,
    #     user_positions=None,  # Will use CLT mode
    #     reference_distance=reference_distance,
    #     path_loss_at_reference=path_loss_at_reference,
    #     min_user_distance=min_user_distance,
    #     max_user_distance=max_user_distance,
    #     # System params
    #     noise_power=noise_power,
    #     total_power=total_power,
    #     use_nearfield_user_channel=False  # Use CLT mode
    # )
    # print(f"   Created with M={num_antennas} antennas, K={num_users} users")

    # print("\n2. Visualizing antenna array positions...")
    # visualize_antenna_array(digital_beamformer.get_positions(),
    #                        array_config=(Nx_antenna, Ny_antenna),
    #                        save_path='antenna_array.png')


    # """
    # Digital Only Beamformer Optimization ----------------------------------------------------------------
    # """
    # print("\n" + "-"*80)
    # print("3. >>> OPTIMIZATION GOES HERE <<<")
    # print("-"*80)
    # # Digital beamforming weights
    # W_digital_dummy = torch.randn(num_antennas, num_users, dtype=torch.complex64, device=device)
    # # Normalize columns to unit norm (common practice)
    # for k in range(num_users):
    #     W_digital_dummy[:, k] /= torch.norm(W_digital_dummy[:, k])
    # print(f"   Digital beamforming weights W: {W_digital_dummy.shape}")
    # print(f"   Interpretation: (M={num_antennas}, K={num_users})")
    # print(f"   Column k = beamforming vector for user k")

    # power_allocation = torch.ones(num_users, device=device) * (total_power / num_users)
    # print(f"   Power allocation:              {power_allocation.shape} = (K={num_users},)")
    # print("-"*80)
    # """
    # ------------------------------------------------------------------------------------------------------
    # """
    # print("\n4. Computing SINR and Sum-Rate...")
    # sum_rate_digital = digital_beamformer.compute_sum_rate(
    #     phases=None,  # No SIM
    #     power_allocation=power_allocation,
    #     digital_beamforming_weights=W_digital_dummy
    # )
    # print(f"   Sum-Rate:  {sum_rate_digital.item():.4f} bits/s/Hz")

    # print("\n5. Verifying channels used in SINR computation...")
    # print(f"   Pre-digital-weights channel shape: {digital_beamformer.pre_digital_weights_channel.shape}")
    # print(f"      = H (antennas → users)")
    # print(f"   Effective channel shape: {digital_beamformer.last_effective_channel.shape}")
    # print(f"      = H @ W (after digital beamforming weights)")
    # print(f"   Interpretation: (K={digital_beamformer.last_effective_channel.shape[0]}, K={digital_beamformer.last_effective_channel.shape[1]})")
    # print(f"   H_eff[k,k] = signal for user k, H_eff[k,j≠k] = interference from beam j")

    # # ========== Summary ==========
    # print("\n" + "="*80)
    # print("SUMMARY")
    # print("="*80)
    # print(f"\nSIM-based (M=K=4):    Sum-Rate = {sum_rate_sim.item():.4f} bits/s/Hz")
    # print(f"Digital (M=64, K=4):  Sum-Rate = {sum_rate_digital.item():.4f} bits/s/Hz")
    # print(f"\nNote: Random weights used - digital should perform better with optimized weights")
    # print("="*80)

    # %%
