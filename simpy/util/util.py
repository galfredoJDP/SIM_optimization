import numpy as np
import torch 
from typing import List, Union

def findDistance(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances between all points in d1 and d2.

    Args:
        d1: array of shape (M, 3) - source coordinates
        d2: array of shape (N, 3) - target coordinates

    Returns:
        Array of shape (M, N) containing all pairwise distances
    """
    d1 = np.atleast_2d(d1)
    d2 = np.atleast_2d(d2)

    # Reshape for broadcasting: d1 -> (M, 1, 3), d2 -> (1, N, 3)
    d1_expanded = d1[:, np.newaxis, :]  # (M, 1, 3)
    d2_expanded = d2[np.newaxis, :, :]  # (1, N, 3)

    # Compute pairwise distances: (M, N)
    distances = np.sqrt(np.sum((d1_expanded - d2_expanded)**2, axis=2))

    return distances

def rayleighSommerfeld(source_positions: Union[torch.Tensor, np.ndarray],
                           target_positions: Union[torch.Tensor, np.ndarray],
                           wavelength : float,
                           aperature_area : float, 
                           device) -> torch.Tensor:
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
        # Add small epsilon to prevent division by zero
        distances_safe = distances + 1e-10
        cos_xi = np.abs(np.dot(directions, normal)) / distances_safe  # (N, M)

        k = 2 * np.pi / wavelength  # Wave number k = 2π/λ

        # Complex amplitude: (A · cos(ξ) / d) · (1/(2πd) - j/λ)
        amplitude = (aperature_area * cos_xi / distances_safe) * (1 / (2 * np.pi * distances_safe) - 1j / wavelength)

        # Phase term: exp(j·k·d)
        phase = np.exp(1j * k * distances)

        # Full transmission matrix W = amplitude × phase
        W = amplitude * phase  # (N, M)

        # Convert to torch tensor
        W = torch.tensor(W, dtype=torch.complex64, device=device)

        return W