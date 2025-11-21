import numpy as np
import torch
from typing import Optional

class Transceiver:
    """
    Rectangular antenna array transceiver with grating lobe suppression.
    """

    def __init__(self,
                 Nx: int,
                 Ny: int,
                 wavelength: float,
                 max_scan_angle: float = 0.0,
                 device: str = 'cpu'):
        """
        Initialize transceiver with rectangular antenna array.

        Args:
            Nx: Number of antenna elements in x-direction (can be 1)
            Ny: Number of antenna elements in y-direction (can be 1)
            wavelength: Operating wavelength (meters)
            max_scan_angle: Maximum beam scan angle (degrees) for grating lobe calculation
                           0° = broadside only
            device: Torch device ('cpu', 'cuda', 'mps')
        """
        self.Nx = Nx
        self.Ny = Ny
        self.num_antennas = Nx * Ny
        self.wavelength = wavelength
        self.max_scan_angle = np.deg2rad(max_scan_angle)
        self.device = device
        # Compute maximum spacing to avoid grating lobes
        self.spacing_x, self.spacing_y = self._compute_grating_lobe_spacing()

        # Generate antenna positions in 3D (x, y, z)
        self.antenna_positions = self._generate_rectangular_positions()

        # Beamforming weights (initialized to None)
        self.beamforming_weights = None
        
    def _compute_grating_lobe_spacing(self) -> tuple[float, float]:
        """
        Compute maximum element spacing to avoid grating lobes.

        Grating lobe condition: d sin(θ) = mλ (m ≠ 0)
        To avoid grating lobes in visible space:
            d < λ / (1 + sin(θ_max))

        For broadside (θ_max = 0): d < λ/2
        For scanning: tighter constraint

        Returns:
            (dx, dy): Maximum spacing in x and y directions (meters)
        """
        # Grating lobe criterion (same for both directions in rectangular grid)
        d_max = self.wavelength / (1 + np.sin(self.max_scan_angle))

        return (d_max, d_max)

    def _generate_rectangular_positions(self) -> torch.Tensor:
        """
        Generate 3D antenna positions for rectangular grid.

        Creates Nx × Ny array on z=0 plane, centered at origin.

        Returns:
            Tensor of shape (Nx*Ny, 3) with (x, y, z) positions
        """
        positions = []

        for ny in range(self.Ny):
            for nx in range(self.Nx):
                x = nx * self.spacing_x
                y = ny * self.spacing_y
                z = 0.0
                positions.append([x, y, z])

        # Convert to array and center
        positions = np.array(positions)
        positions = self._center_array(positions)

        return torch.tensor(positions, dtype=torch.float32, device=self.device)

    def _center_array(self, positions: np.ndarray) -> np.ndarray:
        """
        Center the array at the origin (mean position = [0, 0, 0]).

        Args:
            positions: (N, 3) array of positions

        Returns:
            Centered positions
        """
        centroid = np.mean(positions, axis=0)
        return positions - centroid

    def _compute_aperture(self) -> float:
        """
        Compute effective aperture area of the array.

        Returns:
            Aperture area (m²)
        """
        positions = self.antenna_positions.cpu().numpy()
        x_span = positions[:, 0].max() - positions[:, 0].min()
        y_span = positions[:, 1].max() - positions[:, 1].min()
        return x_span * y_span

    def get_positions(self) -> torch.Tensor:
        """
        Get antenna positions.

        Returns:
            (num_antennas, 3) tensor of (x, y, z) positions
        """
        return self.antenna_positions
    
    def visualize_array(self, save_path: Optional[str] = None):
        """
        Visualize the antenna array layout.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        positions = self.antenna_positions.cpu().numpy()

        fig = plt.figure(figsize=(10, 5))

        # 2D view (top-down)
        ax1 = fig.add_subplot(121)
        ax1.scatter(positions[:, 0], positions[:, 1], s=50, alpha=0.6)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title(f'Array Layout (Top View)\n{self.Nx} × {self.Ny} = {self.num_antennas} antennas')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # 3D view
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=50, alpha=0.6)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_zlabel('z (m)')
        ax2.set_title('3D View')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Array visualization saved to {save_path}")
        else:
            plt.show()

    # ========== Beamforming Methods ==========

    def set_beamforming_weights(self, weights: torch.Tensor) -> None:
        """
        Set beamforming weights for the antenna array.

        Args:
            weights: (num_antennas, num_users) complex tensor of beamforming weights
                     For each user k, weights[:, k] defines the beamforming vector
        """
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.complex64, device=self.device)

        if weights.shape[0] != self.num_antennas:
            raise ValueError(f"Weights must have {self.num_antennas} rows (one per antenna), got {weights.shape[0]}")

        self.beamforming_weights = weights.to(self.device)

    def compute_sinr_downlink(self,
                              channel: torch.Tensor,
                              power_allocation: torch.Tensor,
                              noise_power: float,
                              beamforming_weights : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SINR for each user in downlink with beamforming.

        SINR_k = P_k |h_k^H w_k|^2 / (sum_{j≠k} P_j |h_k^H w_j|^2 + σ^2)

        Args:
            channel: (K, A) complex tensor - channel from antennas to users
            power_allocation: (K,) tensor - transmit power for each user
            noise_power: scalar - noise power σ^2

        Returns:
            sinr: (K,) tensor - SINR for each user
        """
        if beamforming_weights is not None: #assume only SIM
            effective_channel = channel@beamforming_weights
        else:
            effective_channel = channel

        # Save the effective channel used for this computation
        self.last_effective_channel = effective_channel

        K = channel.shape[0]  # Number of users

        # Compute signal and interference powers
        sinr = torch.zeros(K, dtype=torch.float32, device=self.device)

        for k in range(K):
            # Signal power: P_k * |h_k^H w_k|^2
            signal_power = power_allocation[k] * torch.abs(effective_channel[k, k])**2

            # Interference power: sum_{j≠k} P_j * |h_k^H w_j|^2
            interference_power = 0.0
            for j in range(K):
                if j != k:
                    interference_power += power_allocation[j] * torch.abs(effective_channel[k, j])**2

            # SINR
            sinr[k] = signal_power / (interference_power + noise_power)

        return sinr

    def compute_mrt_weights(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Ratio Transmission (MRT) beamforming weights.
        MRT: w_k = h_k^* / ||h_k||

        Args:
            channel: (K, A) complex tensor - channel from A antennas to K users

        Returns:
            weights: (A, K) complex tensor - MRT beamforming weights
        """
        K, A = channel.shape
        weights = torch.zeros((A, K), dtype=torch.complex64, device=self.device)

        for k in range(K):
            h_k = channel[k, :].conj()  # (A,) conjugate of user k's channel
            weights[:, k] = h_k / torch.norm(h_k)  # Normalize

        return weights

    def compute_zf_weights(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Compute Zero-Forcing (ZF) beamforming weights.
        ZF: W = H^H (H H^H)^{-1}

        Args:
            channel: (K, A) complex tensor - channel from A antennas to K users
                     Requires A >= K (more antennas than users)

        Returns:
            weights: (A, K) complex tensor - ZF beamforming weights
        """
        K, A = channel.shape

        if A < K:
            raise ValueError(f"ZF requires A >= K antennas. Got A={A}, K={K}")

        # W = H^H (H H^H)^{-1}
        H = channel  # (K, A)
        HHH = H @ H.conj().T  # (K, K)

        # Add small regularization for numerical stability
        HHH_inv = torch.linalg.inv(HHH + 1e-8 * torch.eye(K, dtype=H.dtype, device=self.device))

        weights = H.conj().T @ HHH_inv  # (A, K)

        return weights