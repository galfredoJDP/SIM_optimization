import numpy as np
import torch
from typing import Optional, Union


class UserChannel:
    """
    Multi-user channel model with configurable mobility, path loss, and fading.

    Features:
    - Per-user path loss exponents with variance
    - Angular velocity (user rotation/movement)
    - Angle noise (random angular variations)
    - Configurable small-scale fading (Rayleigh/Rician)

    Available Attributes (via self):
        .num_users (int): Number of users K
        .wavelength (float): Operating wavelength in meters
        .reference_distance (float): Reference distance for path loss (meters)
        .path_loss_at_reference (float): Path loss at reference distance (dB)
        .min_user_distance (float): Minimum user distance for CLT mode (meters)
        .max_user_distance (float): Maximum user distance for CLT mode (meters)
        .device (str): Computation device ('cpu', 'cuda', 'mps')
        .user_positions (np.ndarray or None): User positions (K, 3) [x, y, z] if set
        .path_loss_exponents (np.ndarray): Path loss exponent per user (K,), default=2.0
        .ricean_k_factors (np.ndarray): Rician K-factor per user (K,) in linear scale

    Available Methods:
        .generate_channel(positions, time=0.0) -> torch.Tensor:
            Generates channel matrix H (K, M) using CLT or specified positions
            Args: positions (torch.Tensor or np.ndarray) (M, 3), time (float)
            Returns: Complex channel matrix (K, M)

        .set_user_positions(positions):
            Sets fixed user positions
            Args: positions (np.ndarray) (K, 3)

        .compute_path_loss(distance) -> np.ndarray:
            Computes path loss for given distances
            Args: distance (np.ndarray) distances in meters
            Returns: Path loss in linear scale (not dB)
    """

    def __init__(self,
                 num_users: int = None,
                 wavelength: float = None,
                 reference_distance: float = 1.0,
                 path_loss_at_reference: float = -30.0,  # dB
                 min_user_distance: float = 10.0,  # meters
                 max_user_distance: float = 100.0,  # meters
                 ricean_k_factors: Optional[Union[np.ndarray, float]] = None,
                 device: str = 'cpu'):
        """
        Initialize multi-user channel model.

        Args:
            num_users: Number of users (K)
            wavelength: Operating wavelength (meters)
            reference_distance: Reference distance for path loss model (meters)
            path_loss_at_reference: Path loss at reference distance (dB)
            min_user_distance: Minimum user distance for CLT mode (meters)
            max_user_distance: Maximum user distance for CLT mode (meters)
            device: Torch device ('cpu', 'cuda', 'mps')
        """
        self.num_users = num_users
        self.wavelength = wavelength
        self.reference_distance = reference_distance
        self.path_loss_at_reference = path_loss_at_reference  # dB
        self.min_user_distance = min_user_distance
        self.max_user_distance = max_user_distance
        self.device = device

        # User positions (K, 3) - to be set by user
        self.user_positions = None
        self.initial_positions = None  # Store initial positions for mobility

        # Per-user path loss parameters
        self.path_loss_exponents = np.ones(num_users) * 2.0  # alpha (free space default)
        self.path_loss_variance = 0.0  # Log-normal shadowing variance (dB)

        # Per-user mobility parameters
        self.angular_velocities = np.zeros(num_users)  # rad/s (rotation rate)
        self.angle_noise_std = np.zeros(num_users)  # rad (angular jitter std)

        # Per-user fading parameters
        if ricean_k_factors is not None:
            if isinstance(ricean_k_factors, (int, float)): #same K-factor for all users
                self.rician_k_factors = np.ones(num_users) * ricean_k_factors
            else: #user specific K-factors
                self.rician_k_factors = np.array(ricean_k_factors)
        else:
            self.rician_k_factors = np.zeros(num_users)  # Default to Rayleigh fading (K=0)

    def set_user_positions(self, positions: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Set user positions.

        Args:
            positions: (K, 3) array of user positions [x, y, z]
        """
        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()

        if positions.shape[0] != self.num_users:
            raise ValueError(f"Expected {self.num_users} users, got {positions.shape[0]}")

        self.user_positions = positions.copy()
        self.initial_positions = positions.copy()  # Store for mobility reference
        print(f"User positions set: {positions.shape}")

    def set_path_loss_exponents(self, exponents: Union[np.ndarray, float],
                                variance: float = 0.0) -> None:
        """
        Set path loss exponent alpha for each user.

        Path loss model: PL(d) = PL(d0) + 10*alpha*log10(d/d0) + X_sigma
        where X_sigma ~ N(0, variance^2) is log-normal shadowing

        Args:
            exponents: Path loss exponent alpha per user (K,) or scalar for all
            variance: Log-normal shadowing standard deviation (dB)
        """
        if isinstance(exponents, (int, float)):
            exponents = np.ones(self.num_users) * exponents

        if len(exponents) != self.num_users:
            raise ValueError(f"Expected {self.num_users} exponents, got {len(exponents)}")

        self.path_loss_exponents = np.array(exponents)
        self.path_loss_variance = variance

    def set_angular_velocity(self, velocities: Union[np.ndarray, float]) -> None:
        """
        Set angular velocity for each user (rotation rate around array).

        Args:
            velocities: Angular velocity per user (rad/s) or scalar for all
                       Example: 0.1 rad/s ~ 5.7 deg/s (slow rotation)
                                1.0 rad/s ~ 57 deg/s (fast rotation)
        """
        if isinstance(velocities, (int, float)):
            velocities = np.ones(self.num_users) * velocities

        if len(velocities) != self.num_users:
            raise ValueError(f"Expected {self.num_users} velocities, got {len(velocities)}")

        self.angular_velocities = np.array(velocities)

    def set_angle_noise(self, noise_std: Union[np.ndarray, float]) -> None:
        """
        Set angular noise standard deviation for each user.

        Models random angular jitter (e.g., from small-scale movements, scattering).

        Args:
            noise_std: Angular noise std dev per user (rad) or scalar for all
                      Example: 0.01 rad ~ 0.57 deg (small jitter)
                               0.1 rad ~ 5.7 deg (large jitter)
        """
        if isinstance(noise_std, (int, float)):
            noise_std = np.ones(self.num_users) * noise_std

        if len(noise_std) != self.num_users:
            raise ValueError(f"Expected {self.num_users} noise values, got {len(noise_std)}")

        self.angle_noise_std = np.array(noise_std)

    def set_rician_k_factor(self, k_factors_db: Union[np.ndarray, float]) -> None:
        """
        Set Rician K-factor for each user (alternative to fading_factor).

        Rician fading models LOS + scattered components.
        K = power_LOS / power_scattered

        Args:
            k_factors_db: Rician K-factor per user (dB) or scalar for all
        """
        if isinstance(k_factors_db, (int, float)):
            k_factors_db = np.ones(self.num_users) * k_factors_db

        if len(k_factors_db) != self.num_users:
            raise ValueError(f"Expected {self.num_users} K-factors, got {len(k_factors_db)}")

        self.rician_k_factors = np.array(k_factors_db)

    def _apply_mobility(self, time: float) -> np.ndarray:
        """
        Apply user mobility (angular rotation + noise).

        Args:
            time: Current time (seconds)

        Returns:
            Updated user positions (K, 3)
        """
        if self.initial_positions is None:
            raise ValueError("User positions not set. Call set_user_positions() first.")

        positions = self.initial_positions.copy()

        for k in range(self.num_users):
            # Angular rotation around z-axis (assumes users rotate around array)
            angle = self.angular_velocities[k] * time

            # Add angular noise
            if self.angle_noise_std[k] > 0:
                angle += np.random.normal(0, self.angle_noise_std[k])

            # Apply rotation (around origin)
            x0, y0, z0 = positions[k]
            r = np.sqrt(x0**2 + y0**2)  # Radial distance
            theta0 = np.arctan2(y0, x0)  # Initial angle

            # New angle
            theta_new = theta0 + angle

            # Update position
            positions[k, 0] = r * np.cos(theta_new)
            positions[k, 1] = r * np.sin(theta_new)
            # z remains the same

        return positions

    def generate_channel(self, antenna_positions: Union[np.ndarray, torch.Tensor],
                        time: float = 0.0) -> torch.Tensor:
        """
        Generate channel matrix H from antennas to users.

        H[k, a] = channel from antenna a to user k

        Args:
            antenna_positions: (A, 3) antenna positions, could also be positions of meta-atoms on the last layer 
            time: Current time for mobility modeling (seconds) #leave None if assuming iid 

        Returns:
            H: (K, A) complex channel matrix

        Notes:
            - If user_positions is None, assumes Central Limit Theorem (CLT):
              Channel coefficients are i.i.d. CN(0, σ²) with average path loss
        """
        # Convert to numpy
        if isinstance(antenna_positions, torch.Tensor):
            antenna_positions = antenna_positions.cpu().numpy()

        A = antenna_positions.shape[0]
        K = self.num_users

        # Case 1: No user positions - use Central Limit Theorem assumption
        if self.user_positions is None:
            # Sample user distances uniformly from [min_user_distance, max_user_distance]
            user_distances = np.random.uniform(self.min_user_distance,
                                              self.max_user_distance,
                                              K)
            
            #this is just to match to the reference paper 
            x = np.random.uniform(self.min_user_distance,
                                              self.max_user_distance,
                                              K)
            y = np.random.uniform(self.min_user_distance,
                                              self.max_user_distance,
                                              K)
            user_distances = np.sqrt(x**2 + y**2)

            
            

            # Reference path loss (C_0 in paper)
            path_loss_ref_linear = 10**(self.path_loss_at_reference / 10)

            # Initialize channel
            H = np.zeros((K, A), dtype=np.complex64)

            # Generate channel for each user with per-user distance-dependent path loss
            for k in range(K):
                # Compute per-user path loss: β_k = C_0 * (d_k / d_0)^(-α)
                distance_factor = (user_distances[k] / self.reference_distance) ** (-self.path_loss_exponents[k])
                beta_k = path_loss_ref_linear * distance_factor

                # Apply per-user log-normal shadowing
                if self.path_loss_variance > 0:
                    shadowing_db = np.random.normal(0, self.path_loss_variance)
                    shadowing_linear = 10**(shadowing_db / 10)
                else:
                    shadowing_linear = 1.0

                # Total path loss for user k: sqrt(β_k * shadowing)
                user_path_loss = np.sqrt(beta_k * shadowing_linear)

                # Generate LOS component (deterministic with random phase per user)
                # Use a random but consistent phase for this user
                random_phase = np.random.uniform(0, 2 * np.pi, A)
                H_LOS = user_path_loss * np.exp(1j * random_phase)

                # Generate scattered component (Rayleigh fading)
                fading_real = np.random.randn(A)
                fading_imag = np.random.randn(A)
                H_scattered = ((fading_real + 1j * fading_imag) / np.sqrt(2)) * user_path_loss

                # Mix LOS and scattered components based on Rician K-factor
                K_dB = self.rician_k_factors[k]
                K_linear = 10**(K_dB / 10)

                # Rician fading: weight components by their power ratio
                # Total power is normalized: K/(K+1) for LOS, 1/(K+1) for scattered
                H[k, :] = np.sqrt(K_linear / (K_linear + 1)) * H_LOS + np.sqrt(1 / (K_linear + 1)) * H_scattered

            # Convert to torch
            H_torch = torch.tensor(H, dtype=torch.complex64, device=self.device)
            return H_torch

        """
        Not 100% correct yet, need to do some cluster delay line stuff here.
        Stopped half way in developement for efficiency sake and made the above just assuming i.i.d CLT
        But will need to do the below if we want to center on FR3 frequencies 
        """
        # Case 2: User positions available - use geometric model, not 100% right
        # Apply mobility
        current_positions = self._apply_mobility(time)

        # Initialize channel
        H = np.zeros((K, A), dtype=np.complex64)

        # Compute array center for phase reference
        array_center = antenna_positions.mean(axis=0)

        # Generate channel for each user
        for k in range(K):
            # 1. Compute distance and direction
            direction = current_positions[k] - array_center
            distance = np.linalg.norm(direction)
            direction = direction / distance

            # 2. Path loss (Friis + distance-dependent)
            # PL(d) = PL(d0) + 10 log10(d/d0) + X
            path_loss_ref_linear = 10**(self.path_loss_at_reference / 10)
            distance_factor = (distance / self.reference_distance) ** (-self.path_loss_exponents[k])

            # Log-normal shadowing
            if self.path_loss_variance > 0:
                shadowing_db = np.random.normal(0, self.path_loss_variance)
                shadowing_linear = 10**(shadowing_db / 10)
            else:
                shadowing_linear = 1.0

            path_loss = np.sqrt(path_loss_ref_linear * distance_factor * shadowing_linear)

            # 3. Array phase response (angle of arrival)
            for a in range(A):
                relative_pos = antenna_positions[a] - array_center
                phase_shift = 2 * np.pi / self.wavelength * np.dot(relative_pos, direction)
                H[k, a] = path_loss * np.exp(1j * phase_shift)

            # 4. Small-scale fading (Rician model)
            # H currently contains LOS component (path loss + phase)
            H_LOS = H[k, :].copy()

            # Generate scattered component (Rayleigh fading)
            fading_real = np.random.randn(A)
            fading_imag = np.random.randn(A)
            H_scattered = ((fading_real + 1j * fading_imag) / np.sqrt(2) ) * path_loss #path_loss ensures that sigma is good (eq. 2.54 Tse's 2005)

            # Mix LOS and scattered components based on Rician K-factor
            # K = power_LOS / power_scattered (in dB)
            K_dB = self.rician_k_factors[k]
            K_linear = 10**(K_dB / 10)

            # Rician fading: weight components by their power ratio
            # Total power is normalized: K/(K+1) for LOS, 1/(K+1) for scattered (eq 2.54 of Tse 2005 MIMO book )
            H[k, :] = np.sqrt(K_linear / (K_linear + 1)) * H_LOS + np.sqrt(1 / (K_linear + 1)) * H_scattered

        # Convert to torch
        H_torch = torch.tensor(H, dtype=torch.complex64, device=self.device)

        return H_torch

    def get_distances(self) -> np.ndarray:
        """
        Get distances from array center to each user.

        Returns:
            distances: (K,) array of distances (meters)
        """
        if self.user_positions is None:
            raise ValueError("User positions not set.")

        distances = np.linalg.norm(self.user_positions, axis=1)
        return distances

    def get_angles(self) -> np.ndarray:
        """
        Get angles (azimuth) from array to each user.

        Returns:
            angles: (K,) array of angles (radians)
        """
        if self.user_positions is None:
            raise ValueError("User positions not set.")

        angles = np.arctan2(self.user_positions[:, 1], self.user_positions[:, 0])
        return angles