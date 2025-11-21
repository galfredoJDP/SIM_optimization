import torch
import numpy as np
import matplotlib.pyplot as plt
from channel import UserChannel
from transceiver import Transceiver

print("=" * 70)
print("UserChannel Test: Multi-User Channel with Mobility and Fading")
print("=" * 70)

# Simulation parameters
wavelength = 0.125  # 125mm
K = 1  # Number of users

# Create transceiver array
print("\n1. Creating Antenna Array...")
transceiver = Transceiver(
    Nx=4,
    Ny=1,
    wavelength=wavelength,
    max_scan_angle=30.0,
    device='cpu'
)
antenna_positions = transceiver.get_positions()
print(f"   Antenna array: {transceiver.num_antennas} elements")

# Create user channel model
print("\n2. Creating User Channel Model...")
user_channel = UserChannel(
    num_users=K,
    wavelength=wavelength,
    reference_distance=1.0,
    path_loss_at_reference=-30.0,  # dB
    device='cpu'
)

# Set user positions (distributed around array at 10m distance)
print("\n3. Setting User Positions...")
user_positions = np.array([
    # [10.0, 5.0, 0.0],    # User 1: right side
    [10.0, 0.0, 0.0],    # User 2: front
    # [10.0, -5.0, 0.0],   # User 3: left side
])
user_channel.set_user_positions(user_positions)

distances = user_channel.get_distances()
angles_deg = np.rad2deg(user_channel.get_angles())
print(f"   User distances: {distances}")
print(f"   User angles: {angles_deg} degrees")

# Configure path loss (different environments per user)
print("\n4. Configuring Path Loss Parameters...")
# Example configurations based on K
if K == 1:
    path_loss_exp = [2.0]  # Free space
    print("   User 1: Free space (alpha=2.0)")
elif K == 2:
    path_loss_exp = [2.0, 3.5]  # Free space, Urban
    print("   User 1: Free space (alpha=2.0)")
    print("   User 2: Urban environment (alpha=3.5)")
else:  # K >= 3
    path_loss_exp = [2.0, 3.5, 2.5] + [2.5] * (K - 3)  # Add more suburban for extra users
    print("   User 1: Free space (alpha=2.0)")
    print("   User 2: Urban environment (alpha=3.5)")
    print("   User 3: Suburban (alpha=2.5)")
    if K > 3:
        print(f"   Users 4-{K}: Suburban (alpha=2.5)")

user_channel.set_path_loss_exponents(
    exponents=path_loss_exp,
    variance=4.0  # 4 dB log-normal shadowing
)

# Configure mobility
print("\n5. Configuring Mobility...")
if K == 1:
    velocities = [0.0]  # Static
    print("   User 1: Static (0 rad/s)")
elif K == 2:
    velocities = [0.1, 0.05]
    print("   User 1: 0.1 rad/s angular velocity (~5.7 deg/s)")
    print("   User 2: 0.05 rad/s (~2.9 deg/s)")
else:  # K >= 3
    velocities = [0.1, 0.05, 0.0] + [0.02] * (K - 3)
    print("   User 1: 0.1 rad/s angular velocity (~5.7 deg/s)")
    print("   User 2: 0.05 rad/s (~2.9 deg/s)")
    print("   User 3: Static (0 rad/s)")
    if K > 3:
        print(f"   Users 4-{K}: 0.02 rad/s (~1.1 deg/s)")

user_channel.set_angular_velocity(velocities)

print("\n   Adding angular noise:")
if K == 1:
    angle_noise = [0.005]
    print("   User 1: 0.005 rad std (~0.29 deg)")
elif K == 2:
    angle_noise = [0.01, 0.02]
    print("   User 1: 0.01 rad std (~0.57 deg)")
    print("   User 2: 0.02 rad std (~1.1 deg)")
else:  # K >= 3
    angle_noise = [0.01, 0.02, 0.005] + [0.01] * (K - 3)
    print("   User 1: 0.01 rad std (~0.57 deg)")
    print("   User 2: 0.02 rad std (~1.1 deg)")
    print("   User 3: 0.005 rad std (~0.29 deg)")
    if K > 3:
        print(f"   Users 4-{K}: 0.01 rad std (~0.57 deg)")

user_channel.set_angle_noise(angle_noise)

# Configure fading
print("\n6. Configuring Small-Scale Fading...")
if K == 1:
    fading = [0.9]
    print("   User 1: Mostly deterministic (factor=0.9)")
elif K == 2:
    fading = [0.9, 0.5]
    print("   User 1: Mostly deterministic (factor=0.9)")
    print("   User 2: Moderate fading (factor=0.5)")
else:  # K >= 3
    fading = [0.9, 0.5, 0.1] + [0.5] * (K - 3)
    print("   User 1: Mostly deterministic (factor=0.9)")
    print("   User 2: Moderate fading (factor=0.5)")
    print("   User 3: Heavy Rayleigh fading (factor=0.1)")
    if K > 3:
        print(f"   Users 4-{K}: Moderate fading (factor=0.5)")

user_channel.set_fading_factor(fading)

# Generate channels at different times
print("\n" + "=" * 70)
print("7. Generating Channels Over Time")
print("=" * 70)

times = [0.0, 1.0, 2.0, 3.0, 4.0, 5, 6, 7]  # seconds
channels = []

for t in times:
    H = user_channel.generate_channel(antenna_positions, time=t)
    channels.append(H)

    print(f"\nTime t={t:.1f}s:")
    for k in range(K):
        channel_power = torch.mean(torch.abs(H[k, :])**2).item()
        channel_power_db = 10 * np.log10(channel_power)
        print(f"  User {k+1}: Power={channel_power_db:.2f} dB, "
              f"Mean phase={torch.angle(H[k, :]).mean().item():.3f} rad")

# Analyze channel variation over time
print("\n" + "=" * 70)
print("8. Channel Time Variation Analysis")
print("=" * 70)

H_ref = channels[0]  # Reference channel at t=0
print("\nChannel change from t=0 (normalized difference):")
for i, t in enumerate(times[1:], 1):
    H_current = channels[i]
    diff = torch.abs(H_current - H_ref)
    mean_diff = diff.mean(dim=1)  # Average over antennas

    print(f"\nTime t={t:.1f}s:")
    for k in range(K):
        print(f"  User {k+1}: Δ={mean_diff[k].item():.4f}")

# Compute correlation between time snapshots
print("\n" + "=" * 70)
print("9. Temporal Channel Correlation")
print("=" * 70)

print("\nCorrelation with initial channel (t=0):")
for i, t in enumerate(times):
    H_current = channels[i]

    # Compute correlation per user
    correlations = []
    for k in range(K):
        h_ref = H_ref[k, :].flatten()
        h_curr = H_current[k, :].flatten()

        # Normalized correlation
        corr = torch.abs(torch.dot(h_curr.conj(), h_ref)) / (
            torch.norm(h_curr) * torch.norm(h_ref)
        )
        correlations.append(corr.item())

    # Print correlations dynamically based on K
    corr_str = f"  t={t:.1f}s: " + ", ".join([f"User {k+1}: {correlations[k]:.3f}" for k in range(K)])
    print(corr_str)

# Test beamforming with time-varying channel
print("\n" + "=" * 70)
print("10. Beamforming with Time-Varying Channel")
print("=" * 70)

# Compute ZF weights at t=0
H_t0 = channels[0]
W_zf = transceiver.compute_zf_weights(H_t0)
transceiver.set_beamforming_weights(W_zf)

print("\nZF beamforming designed at t=0")
print("Testing performance at different times:")

power_allocation = torch.ones(K) * (1.0 / K)
noise_power = 1e-10

for i, t in enumerate(times):
    H_current = channels[i]
    sinr = transceiver.compute_sinr_downlink(H_current, power_allocation, noise_power)

    print(f"\nt={t:.1f}s:")
    for k in range(K):
        sinr_db = 10 * np.log10(sinr[k].item())
        rate = np.log2(1 + sinr[k].item())
        print(f"  User {k+1}: SINR={sinr_db:.2f} dB, Rate={rate:.3f} bits/s/Hz")

# Visualize channel magnitude over antennas
print("\n" + "=" * 70)
print("11. Generating Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Channel magnitude vs antenna for each user at t=0
ax = axes[0, 0]
H_t0_np = H_t0.cpu().numpy()
for k in range(K):
    channel_mag = 10*np.log10(np.abs(H_t0_np[k, :]))
    ax.plot(range(transceiver.num_antennas), channel_mag, 'o-',
            label=f'User {k+1}', linewidth=2, markersize=6)
ax.set_xlabel('Antenna Index')
ax.set_ylabel('Channel Magnitude')
ax.set_title('Channel Magnitude Across Antennas (t=0)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Channel phase vs antenna
ax = axes[0, 1]
for k in range(K):
    channel_phase = np.rad2deg(np.angle(H_t0_np[k, :]))
    ax.plot(range(transceiver.num_antennas), channel_phase, 's-',
            label=f'User {k+1}', linewidth=2, markersize=6)
ax.set_xlabel('Antenna Index')
ax.set_ylabel('Channel Phase (deg)')
ax.set_title('Channel Phase Across Antennas (t=0)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Channel power evolution over time
ax = axes[1, 0]
for k in range(K):
    powers = []
    for H in channels:
        power = torch.mean(torch.abs(H[k, :])**2).item()
        power_db = 10 * np.log10(power)
        powers.append(power_db)
    ax.plot(times, powers, 'o-', label=f'User {k+1}', linewidth=2, markersize=6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Channel Power (dB)')
ax.set_title('Channel Power Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Temporal correlation
ax = axes[1, 1]
for k in range(K):
    correlations = []
    for H_current in channels:
        h_ref = H_ref[k, :].flatten()
        h_curr = H_current[k, :].flatten()
        corr = torch.abs(torch.dot(h_curr.conj(), h_ref)) / (
            torch.norm(h_curr) * torch.norm(h_ref)
        )
        correlations.append(corr.item())
    ax.plot(times, correlations, 'o-', label=f'User {k+1}', linewidth=2, markersize=6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Correlation with t=0')
ax.set_title('Temporal Channel Correlation')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('channel_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'channel_analysis.png'")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\n✓ Tested {K} user{'s' if K > 1 else ''} with {transceiver.num_antennas} antennas")
print(f"✓ Simulated {len(times)} time instances over {times[-1]:.1f} seconds")
print(f"✓ Demonstrated path loss, mobility, and fading effects")
if K >= 1:
    print(f"✓ User 1: Strong channel with low fading")
if K >= 2:
    print(f"✓ User 2: Urban environment with moderate fading")
if K >= 3:
    print(f"✓ User 3: Heavy Rayleigh fading")
if K > 3:
    print(f"✓ Users 4-{K}: Moderate fading")
print(f"\n✓ All tests complete!")