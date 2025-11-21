import torch
import os
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from typing import List, Union
from util.util import findDistance


class sim(object):
    def __init__(self, layers: int, metaAtoms: int, layerSpacing: float,
                 metaAtomSpacing: float, metaAtomArea: float, wavelength: float, device : str) -> None:
        """
        Initialize Stacked Intelligent Metasurface (SIM).

        Args:
            layers: Number of metasurface layers (L)
            metaAtoms: Number of meta-atoms per layer (N)
            layerSpacing: Distance between layers (meters)
            metaAtomSpacing: Distance between meta-atoms in same layer (meters)
            metaAtomArea: Area of each meta-atom (dx * dy) (meters^2)
            wavelength: Operating wavelength (lambda) (meters)
        """
        self.layers = layers
        self.metaAtoms = metaAtoms
        self.metaAtomArea = metaAtomArea
        self.layerSpacing = layerSpacing
        self.metaAtomsSpacing = metaAtomSpacing
        self.wavelength = wavelength

        self.device = device 
        print(f"Using device: {self.device}")

        # Create meta-atom positions for each layer
        # For simplicity, assume 1D array of meta-atoms (can extend to 2D grid)
        self.metaAtomPositions = self._create_meta_atom_positions()

        # Initialize phases to zero
        self.metaAtomPhase = torch.zeros((layers, metaAtoms), device=self.device)

        # Pre-compute W matrices 
        """
        Note: the RS scalar never changes and is only geometry and frequency dependent 

        Rayleigh-Sommerfeld Scalar Equation 

        W[ℓ]_(n,n') = (dx·dy · cos(ξ)) / d · (1/(2πd) - j/λ) · exp(j·2π/λ·d)
         
         where:
            dx, dy = meta-atom dimensions
            cos(ξ) = obliquity factor (angle between propagation and normal)
            d = distance between source and target
        """
        self.W = self._compute_RS_matrices()

        # Compute Theta matrices (phase-dependent)
        self.Theta = self._compute_Theta_matrices()

        # Compute full propagation matrix Psi Note this will be downlink Psi
        self.Psi = self._compute_Psi()

    def _create_meta_atom_positions(self) -> List[torch.Tensor]:
        """
        Create 3D positions for all meta-atoms in all layers.

        Creates a 2D grid (Nx × Ny) for each layer, centered at origin.
        First layer at z=0, subsequent layers in +z direction.

        Returns:
            List of L tensors, each of shape (N, 3) representing (x, y, z) positions
        """
        # Compute 2D grid dimensions (try square grid)
        Nx = int(np.sqrt(self.metaAtoms))
        Ny = int(np.ceil(self.metaAtoms / Nx))

        positions = []
        for layer in range(self.layers):
            z = layer * self.layerSpacing  # Stack in +z direction
            layer_positions = []

            # Create 2D grid
            for ny in range(Ny):
                for nx in range(Nx):
                    if len(layer_positions) >= self.metaAtoms:
                        break  # Stop if we've created enough meta-atoms
                    x = nx * self.metaAtomsSpacing
                    y = ny * self.metaAtomsSpacing
                    layer_positions.append([x, y, z])

            # Convert to array and center at origin
            layer_positions = np.array(layer_positions)
            centroid = layer_positions.mean(axis=0)
            layer_positions = layer_positions - centroid  # Center at origin
            layer_positions[:, 2] = z  # Restore z coordinate

            positions.append(torch.tensor(layer_positions, dtype=torch.float32, device=self.device))

        return positions

    def _rayleighSommerfeld(self, source_positions: Union[torch.Tensor, np.ndarray],
                           target_positions: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Compute Rayleigh-Sommerfeld diffraction between source and target points.

        Args:
            source_positions: (M, 3) array of source coordinates
            target_positions: (N, 3) array of target coordinates

        Returns:
            W: (N, M) complex transmission matrix from M sources to N targets
        """
        # Convert to numpy for distance calculation
        source_np = source_positions.cpu().numpy() if isinstance(source_positions, torch.Tensor) else source_positions
        target_np = target_positions.cpu().numpy() if isinstance(target_positions, torch.Tensor) else target_positions

        # Compute pairwise distances (N, M)
        distances = findDistance(target_np, source_np)  # Note: target first for correct dimensions

        # Compute propagation direction vectors
        # source_exp: (1, M, 3), target_exp: (N, 1, 3)
        source_exp = source_np[np.newaxis, :, :]  # (1, M, 3)
        target_exp = target_np[:, np.newaxis, :]  # (N, 1, 3)

        # Direction vectors: target - source (N, M, 3)
        directions = target_exp - source_exp

        # Normal vector (assume perpendicular to layer, pointing in +z)
        normal = np.array([0, 0, 1])

        # Compute obliquity factor: cos(angle) = dot(direction, normal) / distance
        # directions: (N, M, 3), normal: (3,)
        cos_xi = np.abs(np.dot(directions, normal)) / (distances + 1e-10)  # (N, M)

        k = 2 * np.pi / self.wavelength  # Wave number k = 2π/λ

        # Complex amplitude: (A · cos(ξ) / d) · (1/(2πd) - j/λ)
        amplitude = (self.metaAtomArea * cos_xi / distances) * (1 / (2 * np.pi * distances) - 1j / self.wavelength)

        # Phase term: exp(j·k·d)
        phase = np.exp(1j * k * distances)

        # Full transmission matrix W = amplitude × phase
        W = amplitude * phase  # (N, M)

        # Convert to torch tensor
        W = torch.tensor(W, dtype=torch.complex64, device=self.device)

        return W

    def _compute_RS_matrices(self) -> List[torch.Tensor]:
        """
        Pre-compute all layer-to-layer propagation matrices W^[l].

        Returns:
            List of (L-1) complex tensors, each of shape (N, N)
        """
        W_matrices = []

        for layer in range(1, self.layers):
            # Propagation from layer (l-1) to layer (l)
            source_pos = self.metaAtomPositions[layer - 1]  # Layer l-1
            target_pos = self.metaAtomPositions[layer]      # Layer l

            W_l = self._rayleighSommerfeld(source_pos, target_pos)  # (N, N)
            W_matrices.append(W_l)

        return W_matrices

    def _compute_Theta_matrices(self) -> List[torch.Tensor]:
        """
        Create phase shift vectors for each layer.

        Returns:
            List of L complex tensors, each of shape (N,) containing phase shifts
        """
        Theta_vectors = []

        for layer in range(self.layers):
            # Store only the phase values e^(j*theta), not full diagonal matrix
            phases = torch.exp(1j * self.metaAtomPhase[layer])  # (N,)
            Theta_vectors.append(phases)

        return Theta_vectors

    def _compute_Psi(self) -> torch.Tensor:
        """
        Compute full SIM propagation matrix.
        Psi = Θ^[L] W^[L] Θ^[L-1] ... Θ^[2] W^[2] Θ^[1]
        Note
        Antennas → [A] → SIM Layer 0 → [W[0]] → SIM Layer 1 → [W[1]] → ... → SIM Layer (L-1) → [H] → Users
        So always compute in downlink direction

        Returns:
            Complex tensor of shape (N, N) - full SIM propagation matrix
        """
        Psi = self.W[0] * self.Theta[0][None, :]  # (N, N) * (1, N) = (N, N)

        # Continue through remaining layers
        for layer in range(1, self.layers):
            Psi = self.Theta[layer][:, None] * Psi  # (N, 1) * (N, N) = (N, N)

            # If not last layer, multiply by next W matrix
            if layer < self.layers - 1:
                Psi = self.W[layer] @ Psi  # (N, N) @ (N, N) = (N, N)

        return Psi  # (N, N)

    def update_phases(self, new_phases: torch.Tensor, device=None) -> None:
        """
        Update meta-atom phases and recompute Psi.

        Args:
            new_phases: (L, N) tensor of new phase values in radians [0, 2π]
        """
        if device is None:
            this_device = self.device
        else:
            this_device = device
        self.metaAtomPhase = new_phases.to(this_device)
        self.Theta = self._compute_Theta_matrices()
        self.Psi = self._compute_Psi()

    def uplink(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Propagate signal through SIM in uplink direction (Layer L-1 -> Layer 0)

        Args:
            input_field: (N, K) complex tensor where N is meta-atoms, K is users/signals

        Returns:
            output_field: (N, K) complex tensor after propagation through all layers
        """
        input_field = input_field.to(self.device)
        # For uplink (reverse direction), use Hermitian of Psi due to reciprocity
        output_field = torch.conj(self.Psi).T @ input_field  # (N, N) @ (N, K) = (N, K)
        return output_field
    
    def downlink(self, input_field: torch.Tensor) -> torch.Tensor:
        """
        Propagate signal through SIM in downlink direction (Layer 0 -> Layer L-1)

        Args:
            input_field: (N, K) complex tensor where N is meta-atoms, K is antenna radiators

        Returns:
            output_field: (N, K) complex tensor after propagation through all layers
        """
        input_field = input_field.to(self.device)
        # Psi already represents downlink direction (BS side -> User side)
        output_field = self.Psi @ input_field  # (N, N) @ (N, K) = (N, K)
        return output_field

    def simChannel(self) -> torch.Tensor:
        """
        Returns Psi, the SIM channel 
        """
        return self.Psi
    
    def values(self) -> torch.Tensor:
        """
        Return current phase values.

        Returns:
            (L, N) tensor of phase values in radians
        """
        return self.metaAtomPhase

    def get_first_layer_positions(self) -> torch.Tensor:
        """
        Get positions of meta-atoms on first layer (Layer 1, facing base station).

        Use this to compute antenna → SIM channel A via Rayleigh-Sommerfeld:
            A = rayleighSommerfeld(antenna_positions, first_layer_positions)

        Returns:
            (N, 3) tensor of (x, y, z) positions for first layer
        """
        return self.metaAtomPositions[0]

    def get_last_layer_positions(self) -> torch.Tensor:
        """
        Get positions of meta-atoms on last layer (Layer L, facing users).

        Use this to compute user → SIM channel H via Rayleigh-Sommerfeld:
            H = rayleighSommerfeld(user_positions, last_layer_positions)

        Returns:
            (N, 3) tensor of (x, y, z) positions for last layer
        """
        return self.metaAtomPositions[-1]

    def visualize_structure(self, save_path: str = None, show_connections: bool = False):
        """
        Visualize the 3D structure of the SIM with all meta-atom positions.

        Args:
            save_path: Optional path to save figure
            show_connections: If True, draw lines connecting adjacent layers
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        fig = plt.figure(figsize=(14, 6))

        # 3D view
        ax1 = fig.add_subplot(121, projection='3d')

        colors = plt.cm.viridis(np.linspace(0, 1, self.layers))

        for layer in range(self.layers):
            positions = self.metaAtomPositions[layer].cpu().numpy()
            ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       c=[colors[layer]], s=50, alpha=0.7,
                       label=f'Layer {layer+1}')

            # Draw connections between layers
            if show_connections and layer > 0:
                prev_positions = self.metaAtomPositions[layer-1].cpu().numpy()
                for i in range(min(len(positions), len(prev_positions))):
                    ax1.plot([prev_positions[i, 0], positions[i, 0]],
                            [prev_positions[i, 1], positions[i, 1]],
                            [prev_positions[i, 2], positions[i, 2]],
                            'k-', alpha=0.1, linewidth=0.5)

        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('z (m)')
        ax1.set_title(f'SIM Structure: {self.layers} Layers × {self.metaAtoms} Meta-atoms')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Top-down view (XY plane)
        ax2 = fig.add_subplot(122)

        for layer in range(self.layers):
            positions = self.metaAtomPositions[layer].cpu().numpy()
            marker = 'o' if layer == 0 else ('s' if layer == self.layers-1 else '^')
            label = f'Layer {layer+1}'
            if layer == 0:
                label += ' (Base station side)'
            elif layer == self.layers - 1:
                label += ' (User side)'

            ax2.scatter(positions[:, 0], positions[:, 1],
                       c=[colors[layer]], s=50, alpha=0.7,
                       marker=marker, label=label)

        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_title('Top View (XY Plane)')
        ax2.set_aspect('equal')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Add origin marker
        ax2.plot(0, 0, 'r+', markersize=15, markeredgewidth=2, label='Origin')
        ax2.legend(loc='best', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"SIM structure saved to {save_path}")
        else:
            plt.show()

    def print_structure_info(self):
        """Print detailed information about the SIM structure."""
        print("=" * 60)
        print("SIM Structure Information")
        print("=" * 60)
        print(f"Number of layers: {self.layers}")
        print(f"Meta-atoms per layer: {self.metaAtoms}")
        print(f"Layer spacing: {self.layerSpacing*1000:.2f} mm")
        print(f"Meta-atom spacing: {self.metaAtomsSpacing*1000:.2f} mm")
        print(f"Wavelength: {self.wavelength*1000:.2f} mm")
        print()

        for layer in range(self.layers):
            positions = self.metaAtomPositions[layer].cpu().numpy()
            z_pos = positions[0, 2]
            x_range = (positions[:, 0].min(), positions[:, 0].max())
            y_range = (positions[:, 1].min(), positions[:, 1].max())

            print(f"Layer {layer+1}:")
            print(f"  z position: {z_pos*1000:.2f} mm")
            print(f"  x range: [{x_range[0]*1000:.2f}, {x_range[1]*1000:.2f}] mm")
            print(f"  y range: [{y_range[0]*1000:.2f}, {y_range[1]*1000:.2f}] mm")
            print(f"  Centered: ({positions[:, 0].mean():.2e}, {positions[:, 1].mean():.2e}) m")

        print("=" * 60)
