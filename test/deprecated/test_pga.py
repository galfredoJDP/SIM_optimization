import torch
import numpy as np
import matplotlib.pyplot as plt
from simpy.sim import Sim
from simpy.transceiver import Transceiver
from simpy.channel import UserChannel
from simpy.algorithm import ProjectedGradientAscent
from simpy.util.util import rayleighSommerfeld

print("=" * 70)
print("Projected Gradient Ascent Test: SIM Phase Optimization")
print("=" * 70)

# ========== Simulation Parameters ==========
wavelength = 0.125  # 125mm
K = 3  # Number of users
noise_power = 1e-10  # Noise power
total_power = 1.0  # Total transmit power

# ========== 1. Create Transceiver (Base Station) ==========
print("\n1. Creating Transceiver Array...")
transceiver = Transceiver(
    Nx=4,
    Ny=1,
    wavelength=wavelength,
    max_scan_angle=30.0,
    device='cpu'
)
antenna_positions = transceiver.get_positions()
print(f"   Transceiver: {transceiver.num_antennas} antennas")

# ========== 2. Create SIM ==========
print("\n2. Creating SIM...")
sim_model = Sim(
    layers=3,
    metaAtoms=8,
    layerSpacing=wavelength / 2,
    metaAtomSpacing=wavelength / 2,
    metaAtomArea=(wavelength / 4)**2,
    wavelength=wavelength,
    device='cpu'
)
print(f"   SIM: {sim_model.layers} layers, {sim_model.metaAtoms} meta-atoms per layer")

# ========== 3. Create User Channels ==========
print("\n3. Creating User Channels...")
user_channel = UserChannel(
    num_users=K,
    wavelength=wavelength,
    reference_distance=1.0,
    path_loss_at_reference=-30.0,
    device='cpu'
)

# Set user positions (distributed around SIM at 10m)
user_positions = np.array([
    [10.0, 5.0, 0.0],   # User 1: right side
    [10.0, 0.0, 0.0],   # User 2: front
    [10.0, -5.0, 0.0],  # User 3: left side
])
user_channel.set_user_positions(user_positions)

# Configure channel parameters
user_channel.set_path_loss_exponents([2.0, 2.5, 3.0], variance=4.0)
user_channel.set_rician_k_factor([10.0, 5.0, 0.0])  # User 3: Rayleigh, others: Rician

print(f"   Users: {K}")
print(f"   User distances: {user_channel.get_distances()}")
print(f"   User angles: {np.rad2deg(user_channel.get_angles())} degrees")

# ========== 4. Compute Static Channels ==========
print("\n4. Computing Channels...")

# Channel A: Antenna -> SIM first layer (Rayleigh-Sommerfeld)
print("   Computing A (antenna -> SIM)...")
sim_first_layer = sim_model.get_first_layer_positions()
A = rayleighSommerfeld(
    antenna_positions.cpu().numpy(),
    sim_first_layer.cpu().numpy(),
    wavelength,
    sim_model.metaAtomArea,
    'cpu'
)
print(f"   A shape: {A.shape} (SIM meta-atoms × antennas)")

# Channel H: SIM last layer -> Users (Far-field model)
print("   Computing H (SIM -> users)...")
sim_last_layer = sim_model.get_last_layer_positions()
H = user_channel.generate_channel(sim_last_layer, time=0.0)
print(f"   H shape: {H.shape} (users × SIM meta-atoms)")

# ========== 5. Define Objective Function (Sum-Rate) ==========
print("\n5. Defining Objective Function (Sum-Rate)...")

def sum_rate_objective(phases):
    """
    Compute sum-rate for given SIM phases.

    End-to-end channel: H_eff = H @ Psi^H @ A
    where Psi depends on phases.

    Args:
        phases: (L, N) SIM phase configuration

    Returns:
        sum_rate: Scalar tensor (to maximize)
    """
    # Update SIM with current phases
    sim_model.update_phases(phases)

    # Get SIM channel Psi
    Psi = sim_model.simChannel()  # (N, N)

    # Compute end-to-end channel: H @ Psi^H @ A
    # H: (K, N), Psi^H: (N, N), A: (N, A)
    H_eff = H @ torch.conj(Psi).T @ A  # (K, A)

    # Compute ZF beamforming weights
    W_zf = transceiver.compute_zf_weights(H_eff)  # (A, K)
    transceiver.set_beamforming_weights(W_zf)

    # Equal power allocation
    power_allocation = torch.ones(K, device='cpu') * (total_power / K)

    # Compute SINR for each user
    sinr = transceiver.compute_sinr_downlink(H_eff, power_allocation, noise_power)

    # Sum-rate (bits/s/Hz)
    rates = torch.log2(1 + sinr)
    sum_rate = torch.sum(rates)

    return sum_rate

# Test objective with initial random phases
print("   Testing objective function...")
initial_phases = torch.rand((sim_model.layers, sim_model.metaAtoms)) * 2 * np.pi
initial_objective = sum_rate_objective(initial_phases)
print(f"   Initial sum-rate: {initial_objective.item():.4f} bits/s/Hz")

# ========== 6. Run PGA Optimization ==========
print("\n" + "=" * 70)
print("6. Running Projected Gradient Ascent Optimization")
print("=" * 70)

optimizer = ProjectedGradientAscent(
    sim_model=sim_model,
    objective_fn=sum_rate_objective,
    learning_rate=0.05,
    max_iterations=100,
    tolerance=1e-5,
    projection='modulo',
    verbose=True
)

result = optimizer.optimize(initial_phases=initial_phases)

# ========== 7. Analyze Results ==========
print("\n" + "=" * 70)
print("7. Performance Analysis")
print("=" * 70)

# Compute performance with optimal phases
optimal_phases = result['optimal_phases']
sim_model.update_phases(optimal_phases)

# End-to-end channel with optimized SIM
Psi_opt = sim_model.simChannel()
H_eff_opt = H @ torch.conj(Psi_opt).T @ A

# Compute optimal beamforming and SINR
W_zf_opt = transceiver.compute_zf_weights(H_eff_opt)
transceiver.set_beamforming_weights(W_zf_opt)
power_allocation = torch.ones(K) * (total_power / K)
sinr_opt = transceiver.compute_sinr_downlink(H_eff_opt, power_allocation, noise_power)
rates_opt = torch.log2(1 + sinr_opt).cpu().numpy()

# Compare with initial (random) phases
sim_model.update_phases(initial_phases)
Psi_init = sim_model.simChannel()
H_eff_init = H @ torch.conj(Psi_init).T @ A
W_zf_init = transceiver.compute_zf_weights(H_eff_init)
transceiver.set_beamforming_weights(W_zf_init)
sinr_init = transceiver.compute_sinr_downlink(H_eff_init, power_allocation, noise_power)
rates_init = torch.log2(1 + sinr_init).cpu().numpy()

print("\nInitial (Random Phases):")
for k in range(K):
    sinr_db = 10 * np.log10(sinr_init[k].item())
    print(f"  User {k+1}: SINR = {sinr_db:6.2f} dB, Rate = {rates_init[k]:.4f} bits/s/Hz")
print(f"  Sum-rate: {rates_init.sum():.4f} bits/s/Hz")

print("\nOptimized (PGA):")
for k in range(K):
    sinr_db = 10 * np.log10(sinr_opt[k].item())
    print(f"  User {k+1}: SINR = {sinr_db:6.2f} dB, Rate = {rates_opt[k]:.4f} bits/s/Hz")
print(f"  Sum-rate: {rates_opt.sum():.4f} bits/s/Hz")

improvement = (rates_opt.sum() - rates_init.sum()) / rates_init.sum() * 100
print(f"\nImprovement: {improvement:.2f}%")

# ========== 8. Visualizations ==========
print("\n" + "=" * 70)
print("8. Generating Visualizations...")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Convergence curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(result['history']['objective'], 'b-', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Sum-Rate (bits/s/Hz)')
ax1.set_title('Objective Function (Sum-Rate)')
ax1.grid(True, alpha=0.3)

# Plot 2: Gradient norm
ax2 = plt.subplot(2, 3, 2)
ax2.semilogy(result['history']['gradient_norm'], 'r-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Gradient Norm (log scale)')
ax2.set_title('Gradient Magnitude')
ax2.grid(True, alpha=0.3)

# Plot 3: Phase change
ax3 = plt.subplot(2, 3, 3)
ax3.semilogy(result['history']['phase_change'], 'g-', linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Phase Change (log scale)')
ax3.set_title('Phase Update Magnitude')
ax3.grid(True, alpha=0.3)

# Plot 4: Per-user rate comparison
ax4 = plt.subplot(2, 3, 4)
x = np.arange(K)
width = 0.35
ax4.bar(x - width/2, rates_init, width, label='Initial (Random)', alpha=0.8)
ax4.bar(x + width/2, rates_opt, width, label='Optimized (PGA)', alpha=0.8)
ax4.set_xlabel('User')
ax4.set_ylabel('Rate (bits/s/Hz)')
ax4.set_title('Per-User Rates')
ax4.set_xticks(x)
ax4.set_xticklabels([f'User {k+1}' for k in range(K)])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Phase configuration (initial vs optimal)
ax5 = plt.subplot(2, 3, 5)
im5 = ax5.imshow(initial_phases.cpu().numpy(), cmap='twilight', aspect='auto', vmin=0, vmax=2*np.pi)
ax5.set_xlabel('Meta-Atom Index')
ax5.set_ylabel('Layer')
ax5.set_title('Initial Phases (Random)')
plt.colorbar(im5, ax=ax5, label='Phase (rad)')

ax6 = plt.subplot(2, 3, 6)
im6 = ax6.imshow(optimal_phases.cpu().numpy(), cmap='twilight', aspect='auto', vmin=0, vmax=2*np.pi)
ax6.set_xlabel('Meta-Atom Index')
ax6.set_ylabel('Layer')
ax6.set_title('Optimized Phases (PGA)')
plt.colorbar(im6, ax=ax6, label='Phase (rad)')

plt.tight_layout()
plt.savefig('pga_optimization_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'pga_optimization_results.png'")

# ========== Summary ==========
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"✓ Optimized {sim_model.layers} layers with {sim_model.metaAtoms} meta-atoms each")
print(f"✓ Served {K} users with {transceiver.num_antennas} antennas")
print(f"✓ Converged in {result['iterations']} iterations")
print(f"✓ Sum-rate improvement: {improvement:.2f}%")
print(f"✓ Initial sum-rate: {rates_init.sum():.4f} bits/s/Hz")
print(f"✓ Optimized sum-rate: {rates_opt.sum():.4f} bits/s/Hz")
print("\n✓ PGA optimization test complete!")