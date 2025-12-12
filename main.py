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
from train import load_weights

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

    ddpg_flag = False
    td3_flag = False

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
    sim_beamformer_td3 = copy.deepcopy(sim_beamformer_ddpg)

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
    power_values_db = np.array([10, 15, 20, 25, 30])  # dBm values to sweep
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
        if ddpg_flag:
            sim_beamformer_ddpg.total_power = power_w
        if td3_flag:
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
            if ddpg_flag: 
                sim_beamformer_ddpg.H = sim_beamformer.H.clone()
            if td3_flag: 
                sim_beamformer_td3.H = sim_beamformer.H.clone()

            # Move all beamformer tensors to MPS for DDPG/TD3
            if ddpg_flag: 
                sim_beamformer_ddpg.to_device('mps')
            if td3_flag: 
                sim_beamformer_td3.to_device('mps')

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

            # Initial conditions (shared starting point)
            initial_phases = sim_phases.clone()
            initial_power = torch.ones(num_users, device=device) * (power_w / num_users)

            # Independent state for each algorithm
            phases_pga = initial_phases.clone()
            if ddpg_flag: 
                phases_ddpg = initial_phases.clone().to('mps')
            if td3_flag: 
                phases_td3 = initial_phases.clone().to('mps')

            power_pga = initial_power.clone()
            if ddpg_flag: 
                power_ddpg = initial_power.clone().to('mps')
            if td3_flag: 
                power_td3 = initial_power.clone().to('mps')

            # Track results for each path
            results_pga = {'sumrates': [], 'final_phases': None, 'final_power': None}
            if ddpg_flag: 
                results_ddpg = {'sumrates': [], 'final_phases': None, 'final_power': None}
            if td3_flag: 
                results_td3 = {'sumrates': [], 'final_phases': None, 'final_power': None}
            
            if ddpg_flag:
                optimizer_ddpg = DDPG(
                        beamformer=sim_beamformer_ddpg,
                        state_dim=2 * num_users * sim_metaatoms + num_users + (sim_layers * sim_metaatoms),
                        action_dim=sim_layers * sim_metaatoms,
                        verbose=False,
                        device = 'mps',
                        noise_std=1,
                        tau=0.01,
                        actor_lr=0.01,
                        critic_lr=0.01
                    )
                load_weights(agent=optimizer_ddpg, checkpoint_path='weights/ddpg_latest.pth', device='mps' )
            if td3_flag:
                optimizer_td3 = TD3(
                        beamformer=sim_beamformer_td3,
                        state_dim=2 * num_users * sim_metaatoms + num_users + (sim_layers * sim_metaatoms),
                        action_dim=sim_layers * sim_metaatoms,
                        verbose=False,
                        device='mps'
                    )
                load_weights(agent = optimizer_td3,checkpoint_path='weights/td3_latest.pth', device='mps' )
            
            for j in range(alternative_iterations):
                '''
                PATH 1: PGA + Waterfilling
                '''
                print("Starting PGA")
                optimizer_pga = PGA(
                    beamformer=sim_beamformer,
                    objective_fn=lambda phases: sim_beamformer.compute_sum_rate(phases=phases, power_allocation=power_pga),
                    learning_rate=0.1,
                    max_iterations=5000,
                    verbose=False,
                    use_backtracking=True,
                    backtrack_max_iter=1000
                )
                pga_result = optimizer_pga.optimize(phases_pga)
                phases_pga = pga_result['optimal_params']

                H_eff_pga = sim_beamformer.compute_end_to_end_channel(phases_pga)
                wf_pga = WaterFilling(
                    H_eff=H_eff_pga,
                    noise_power=sim_beamformer.noise_power,
                    total_power=power_w,
                    max_iterations=200,
                    tolerance=1e-6,
                    verbose=False,
                    device=device
                )
                wf_pga_result = wf_pga.optimize(initial_power=power_pga)
                power_pga = wf_pga_result['optimal_power']
                sumrate_pga = wf_pga_result['optimal_sum_rate']
                results_pga['sumrates'].append(sumrate_pga)

                '''
                 PATH 2: DDPG + Waterfilling
               '''
                if ddpg_flag:
                    print("Starting DDPG")
                    ddpg_result = optimizer_ddpg.optimize_with_policy(
                        initial_phases=phases_ddpg,  # Start from current DDPG state
                        power_allocation=power_ddpg,
                        num_iterations=5000,
                        early_stopping_patience=100
                    )
                
                    phases_ddpg = ddpg_result['optimal_params']

                    H_eff_ddpg = sim_beamformer.compute_end_to_end_channel(phases_ddpg.to(device))
                    wf_ddpg = WaterFilling(
                        H_eff=H_eff_ddpg,
                        noise_power=sim_beamformer.noise_power,
                        total_power=power_w,
                        max_iterations=200,
                        tolerance=1e-6,
                        verbose=False,
                        device=device
                    )
                    wf_ddpg_result = wf_ddpg.optimize(initial_power=power_ddpg.to(device))
                    power_ddpg = wf_ddpg_result['optimal_power'].to('mps')
                    sumrate_ddpg = wf_ddpg_result['optimal_sum_rate']
                    results_ddpg['sumrates'].append(sumrate_ddpg)

                '''
                PATH 3: TD3 + Waterfilling
                '''
                if td3_flag:
                    print("Starting TD3")
                    td3_result = optimizer_td3.optimize_with_policy(
                        initial_phases=phases_td3,  # Start from current TD3 state
                        power_allocation=power_td3,
                        num_iterations=5000,
                        early_stopping_patience=100
                    )
                    phases_td3 = td3_result['optimal_params']

                    H_eff_td3 = sim_beamformer.compute_end_to_end_channel(phases_td3.to(device))
                    wf_td3 = WaterFilling(
                        H_eff=H_eff_td3,
                        noise_power=sim_beamformer.noise_power,
                        total_power=power_w,
                        max_iterations=200,
                        tolerance=1e-6,
                        verbose=False,
                        device=device
                    )
                    wf_td3_result = wf_td3.optimize(initial_power=power_td3.to(device))
                    power_td3 = wf_td3_result['optimal_power'].to('mps')
                    sumrate_td3 = wf_td3_result['optimal_sum_rate']
                    results_td3['sumrates'].append(sumrate_td3)

                # Clear memory cache
                if device == 'cuda':
                    torch.cuda.empty_cache()
                torch.mps.empty_cache()

                # Print results
                print_str = f"Alt iter {j} | PGA: {sumrate_pga:.4f}"
                if ddpg_flag:
                    print_str += f" | DDPG: {sumrate_ddpg:.4f}"
                if td3_flag:
                    print_str += f" | TD3: {sumrate_td3:.4f}"
                print(print_str)

            # Store final results for this channel realization
            results_pga['final_phases'] = phases_pga
            results_pga['final_power'] = power_pga
            if ddpg_flag:
                results_ddpg['final_phases'] = phases_ddpg
                results_ddpg['final_power'] = power_ddpg
            if td3_flag:
                results_td3['final_phases'] = phases_td3
                results_td3['final_power'] = power_td3

            # Store the final sum-rate after all alternating iterations
            sumrate_entry = {'pga': results_pga['sumrates'][-1]}
            if ddpg_flag:
                sumrate_entry['ddpg'] = results_ddpg['sumrates'][-1]
            if td3_flag:
                sumrate_entry['td3'] = results_td3['sumrates'][-1]
            sumrate.append(sumrate_entry)

            # Store detailed results
            detailed_entry = {
                'pga': {
                    'sumrates_over_iterations': results_pga['sumrates'],
                    'final_sumrate': results_pga['sumrates'][-1],
                    'wf_sinr': wf_pga_result['sinr_per_user'],
                    'wf_rate_per_user': wf_pga_result['rate_per_user'],
                }
            }
            if ddpg_flag:
                detailed_entry['ddpg'] = {
                    'sumrates_over_iterations': results_ddpg['sumrates'],
                    'final_sumrate': results_ddpg['sumrates'][-1],
                    'wf_sinr': wf_ddpg_result['sinr_per_user'],
                    'wf_rate_per_user': wf_ddpg_result['rate_per_user'],
                }
            if td3_flag:
                detailed_entry['td3'] = {
                    'sumrates_over_iterations': results_td3['sumrates'],
                    'final_sumrate': results_td3['sumrates'][-1],
                    'wf_sinr': wf_td3_result['sinr_per_user'],
                    'wf_rate_per_user': wf_td3_result['rate_per_user'],
                }
            all_results[f'{power_db:.1f}dBm']['results'].append(detailed_entry)

            sumrate_simple = {'pga': results_pga['sumrates'][-1]}
            if ddpg_flag:
                sumrate_simple['ddpg'] = results_ddpg['sumrates'][-1]
            if td3_flag:
                sumrate_simple['td3'] = results_td3['sumrates'][-1]
            all_results[f'{power_db:.1f}dBm']['sumrate'].append(sumrate_simple)
            

                

        # Save results for this power level
        sumrate_all_powers[f'{power_db:.1f}dBm'] = sumrate

    results_file = os.path.join(results_dir, 'all_results.pt')
    torch.save(all_results, results_file)


    #%%
    # Plot results comparing PGA, DDPG, TD3
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatter plot comparing all algorithms
    ax = axes[0]
    power_labels = list(sumrate_all_powers.keys())
    for power_db in power_labels:
        sumrates = sumrate_all_powers[power_db]
        pga_rates = [s['pga'] for s in sumrates]
        x = range(len(pga_rates))
        ax.scatter(x, pga_rates, label=f'PGA ({power_db})', alpha=0.7, s=50, marker='o')
        if ddpg_flag:
            ddpg_rates = [s['ddpg'] for s in sumrates]
            ax.scatter(x, ddpg_rates, label=f'DDPG ({power_db})', alpha=0.7, s=50, marker='s')
        if td3_flag:
            td3_rates = [s['td3'] for s in sumrates]
            ax.scatter(x, td3_rates, label=f'TD3 ({power_db})', alpha=0.7, s=50, marker='^')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Sum-Rate (bits/s/Hz)')
    # Dynamic title based on enabled algorithms
    title_parts = ['PGA']
    if ddpg_flag:
        title_parts.append('DDPG')
    if td3_flag:
        title_parts.append('TD3')
    ax.set_title(' vs '.join(title_parts) + ' Sum-Rate Comparison')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean comparison bar chart
    ax = axes[1]
    num_algorithms = 1 + (1 if ddpg_flag else 0) + (1 if td3_flag else 0) + 1  # PGA + DDPG? + TD3? + ZF
    x = np.arange(len(power_labels))
    width = 0.2

    for i, power_db in enumerate(power_labels):
        sumrates = sumrate_all_powers[power_db]
        pga_mean = np.mean([s['pga'] for s in sumrates])
        ref_mean = np.mean(all_results[power_db]['reference'])

        bar_position = 0
        ax.bar(i - (num_algorithms-1)*width/2 + bar_position*width, pga_mean, width,
               label='PGA' if i == 0 else '', color='blue', alpha=0.7)
        bar_position += 1

        if ddpg_flag:
            ddpg_mean = np.mean([s['ddpg'] for s in sumrates])
            ax.bar(i - (num_algorithms-1)*width/2 + bar_position*width, ddpg_mean, width,
                   label='DDPG' if i == 0 else '', color='green', alpha=0.7)
            bar_position += 1

        if td3_flag:
            td3_mean = np.mean([s['td3'] for s in sumrates])
            ax.bar(i - (num_algorithms-1)*width/2 + bar_position*width, td3_mean, width,
                   label='TD3' if i == 0 else '', color='orange', alpha=0.7)
            bar_position += 1

        ax.bar(i - (num_algorithms-1)*width/2 + bar_position*width, ref_mean, width,
               label='ZF (ref)' if i == 0 else '', color='red', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(power_labels)
    ax.set_ylabel('Mean Sum-Rate (bits/s/Hz)')
    ax.set_xlabel('Transmit Power')
    ax.set_title('Mean Performance: PGA vs DDPG vs TD3')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(results_dir, 'PGA_DDPG_TD3_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_file}")

    #%%
    # Plot paper-style comparison
    plt.figure(figsize=(10, 6))
    power_labels = list(sumrate_all_powers.keys())

    # Extract means for each algorithm
    pga_means = [np.mean([s['pga'] for s in sumrate_all_powers[p]]) for p in power_labels]
    ddpg_means = [np.mean([s['ddpg'] for s in sumrate_all_powers[p]]) for p in power_labels]
    td3_means = [np.mean([s['td3'] for s in sumrate_all_powers[p]]) for p in power_labels]
    reference_means = [np.mean(all_results[p]['reference']) for p in power_labels]

    plt.plot(power_values_db, pga_means, 'bo-', markersize=8, linewidth=2, label='WF + PGA')
    plt.plot(power_values_db, ddpg_means, 'gs-', markersize=8, linewidth=2, label='WF + DDPG')
    plt.plot(power_values_db, td3_means, 'm^-', markersize=8, linewidth=2, label='WF + TD3')
    plt.plot(power_values_db, reference_means, 'ro--', markersize=8, linewidth=2, label='WF + ZF (reference)')

    plt.legend(fontsize=10)
    plt.ylim(0, 25)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Transmit Power (dBm)', fontsize=12)
    plt.ylabel('Mean Sum-Rate (bits/s/Hz)', fontsize=12)
    plt.title('Sum-Rate vs Transmit Power: Algorithm Comparison', fontsize=14)
    plot_file = os.path.join(results_dir, 'algorithm_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_file}")







    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    for power_db, sumrates in sumrate_all_powers.items():
        pga_rates = [s['pga'] for s in sumrates]
        ddpg_rates = [s['ddpg'] for s in sumrates]
        td3_rates = [s['td3'] for s in sumrates]
        print(f"\nPower: {power_db}")
        print(f"  PGA:  Mean={np.mean(pga_rates):.4f}, Std={np.std(pga_rates):.4f}")
        print(f"  DDPG: Mean={np.mean(ddpg_rates):.4f}, Std={np.std(ddpg_rates):.4f}")
        print(f"  TD3:  Mean={np.mean(td3_rates):.4f}, Std={np.std(td3_rates):.4f}")

    #%% Plot per-user diagnostics (comparing algorithms)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SINR comparison for PGA vs DDPG vs TD3
    ax = axes[0, 0]
    for power_db in power_labels:
        pga_sinr = [r['pga']['wf_sinr'] for r in all_results[power_db]['results']]
        ddpg_sinr = [r['ddpg']['wf_sinr'] for r in all_results[power_db]['results']]
        td3_sinr = [r['td3']['wf_sinr'] for r in all_results[power_db]['results']]
        pga_mean = np.mean(pga_sinr, axis=0)
        ddpg_mean = np.mean(ddpg_sinr, axis=0)
        td3_mean = np.mean(td3_sinr, axis=0)
        ax.plot(range(num_users), 10*np.log10(pga_mean), 'o-', label=f'PGA ({power_db})')
        ax.plot(range(num_users), 10*np.log10(ddpg_mean), 's--', label=f'DDPG ({power_db})')
        ax.plot(range(num_users), 10*np.log10(td3_mean), '^:', label=f'TD3 ({power_db})')
    ax.set_xlabel('User Index')
    ax.set_ylabel('Mean SINR (dB)')
    ax.set_title('SINR per User: PGA vs DDPG vs TD3')
    ax.legend(fontsize=7)
    ax.grid(True)

    # Rate per user comparison
    ax = axes[0, 1]
    for power_db in power_labels:
        pga_rate = [r['pga']['wf_rate_per_user'] for r in all_results[power_db]['results']]
        ddpg_rate = [r['ddpg']['wf_rate_per_user'] for r in all_results[power_db]['results']]
        td3_rate = [r['td3']['wf_rate_per_user'] for r in all_results[power_db]['results']]
        pga_mean = np.mean(pga_rate, axis=0)
        ddpg_mean = np.mean(ddpg_rate, axis=0)
        td3_mean = np.mean(td3_rate, axis=0)
        ax.plot(range(num_users), pga_mean, 'o-', label=f'PGA ({power_db})')
        ax.plot(range(num_users), ddpg_mean, 's--', label=f'DDPG ({power_db})')
        ax.plot(range(num_users), td3_mean, '^:', label=f'TD3 ({power_db})')
    ax.set_xlabel('User Index')
    ax.set_ylabel('Rate (bits/s/Hz)')
    ax.set_title('Rate per User: PGA vs DDPG vs TD3')
    ax.legend(fontsize=7)
    ax.grid(True)

    # Sum-rate distribution boxplot
    ax = axes[1, 0]
    for i, power_db in enumerate(power_labels):
        sumrates = sumrate_all_powers[power_db]
        pga_rates = [s['pga'] for s in sumrates]
        ddpg_rates = [s['ddpg'] for s in sumrates]
        td3_rates = [s['td3'] for s in sumrates]
        positions = [i*4, i*4+1, i*4+2]
        bp = ax.boxplot([pga_rates, ddpg_rates, td3_rates], positions=positions, widths=0.6)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Sum-Rate (bits/s/Hz)')
    ax.set_title('Sum-Rate Distribution by Algorithm')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['PGA', 'DDPG', 'TD3'])
    ax.grid(True, alpha=0.3)

    # Improvement over PGA
    ax = axes[1, 1]
    for power_db in power_labels:
        sumrates = sumrate_all_powers[power_db]
        pga_rates = np.array([s['pga'] for s in sumrates])
        ddpg_rates = np.array([s['ddpg'] for s in sumrates])
        td3_rates = np.array([s['td3'] for s in sumrates])
        ddpg_improvement = (ddpg_rates - pga_rates) / pga_rates * 100
        td3_improvement = (td3_rates - pga_rates) / pga_rates * 100
        ax.bar(0, np.mean(ddpg_improvement), yerr=np.std(ddpg_improvement),
               capsize=5, color='green', alpha=0.7, label='DDPG' if power_db == power_labels[0] else '')
        ax.bar(1, np.mean(td3_improvement), yerr=np.std(td3_improvement),
               capsize=5, color='orange', alpha=0.7, label='TD3' if power_db == power_labels[0] else '')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Improvement over PGA (%)')
    ax.set_title('RL Improvement over PGA Baseline')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DDPG', 'TD3'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'per_user_diagnostics.png'), dpi=150)
    print(f"✓ Plot saved: {os.path.join(results_dir, 'per_user_diagnostics.png')}")

    

    # %%
