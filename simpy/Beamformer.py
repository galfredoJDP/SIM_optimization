import torch
import numpy as np
from typing import Optional, Dict, Union
from simpy.sim import Sim
from simpy.transceiver import Transceiver
from simpy.channel import UserChannel
from simpy.util.util import rayleighSommerfeld, complex_matmul


class Beamformer(Transceiver, UserChannel):
    """
    Complete digital beamforming system with SIM.

    Inherits from both Transceiver and UserChannel, combining all their
    functionality into a unified interface.

    Architecture:
        Antennas → [A] → SIM [Ψ(phases)] → [H] → Users

    Available Attributes (via self):
        # From Transceiver:
        .Nx (int): Number of antennas in x-direction
        .Ny (int): Number of antennas in y-direction
        .num_antennas (int): Total antennas M = Nx * Ny
        .wavelength (float): Operating wavelength in meters
        .device (str): Computation device ('cpu', 'cuda', 'mps')
        .get_positions() -> torch.Tensor: Returns antenna positions (M, 3)
        .compute_zf_weights(H) -> torch.Tensor: Zero-forcing weights (M, K)
        .compute_mrt_weights(H) -> torch.Tensor: MRT weights (M, K)
        .compute_sinr_downlink(phases, power, digital_weights) -> torch.Tensor: SINR per user (K,)

        # From UserChannel:
        .num_users (int): Number of users K
        .user_positions (np.ndarray): User positions (K, 3) if provided
        .reference_distance (float): Reference distance for path loss
        .path_loss_at_reference (float): Path loss at reference in dB
        .generate_channel(positions, time) -> torch.Tensor: Channel matrix (K, M)

        # Beamformer-specific:
        .sim_model (Sim): SIM metasurface object (if provided)
        .noise_power (float): Noise power σ² in Watts
        .total_power (float): Total transmit power budget in Watts
        .H (torch.Tensor): Channel from SIM last layer to users (K, N) or antennas to users (K, M)
        .A (torch.Tensor): Channel from antennas to SIM first layer (N, M) [only if sim_model provided]

    Available Methods:
        .compute_end_to_end_channel(phases, direction='down') -> torch.Tensor:
            Computes H_eff = H @ Ψ @ A (K, M) for downlink
            Args: phases (L, N), direction ('down' or 'up')
            Returns: Effective channel (K, M) for downlink or (M, K) for uplink

        .compute_sum_rate(phases, power_allocation, digital_beamforming_weights=None) -> torch.Tensor:
            Computes sum-rate Σ log₂(1 + SINR_k) in bits/s/Hz
            Args: phases (L, N) or None, power (K,), weights (M, K) or None
            Returns: Scalar sum-rate

        .compute_sinr(phases, power_allocation, digital_beamforming_weights=None) -> torch.Tensor:
            Computes SINR per user
            Args: phases (L, N) or None, power (K,), weights (M, K) or None
            Returns: SINR per user (K,)

        .update_user_channel(time=0.0):
            Regenerates user channel H for time-varying scenarios
    """

    def __init__(self,
                 # Transceiver parameters
                 Nx: int,
                 Ny: int,
                 wavelength: float,
                 max_scan_angle: float = 0.0,
                 device: str = 'cpu',
                 # UserChannel parameters
                 num_users: int = 1,
                 user_positions: Optional[np.ndarray] = None,
                 reference_distance: float = 1.0,
                 path_loss_at_reference: float = -30.0,
                 min_user_distance: float = 10.0,
                 max_user_distance: float = 100.0,
                 # SIM
                 sim_model: Optional[Sim] = None,
                 # System parameters
                 noise_power: float = 1e-10,
                 total_power: float = 1.0,
                 beamforming_method: Optional[str] = None,
                 use_nearfield_user_channel: bool = False):
        """
        Initialize Digital Beamformer system.

        Args:
            Nx: Number of antenna elements in x-direction
            Ny: Number of antenna elements in y-direction
            wavelength: Operating wavelength (meters)
            max_scan_angle: Maximum beam scan angle (degrees)
            device: Torch device ('cpu', 'cuda', 'mps')
            num_users: Number of users (K)
            user_positions: (K, 3) array of user positions [x, y, z]
            reference_distance: Reference distance for path loss model (meters)
            path_loss_at_reference: Path loss at reference distance (dB)
            min_user_distance: Minimum user distance for CLT mode (meters)
            max_user_distance: Maximum user distance for CLT mode (meters)
            sim_model: SIM object (metasurface)
            noise_power: Noise power σ² (Watts)
            total_power: Total transmit power (Watts)
            beamforming_method: Beamforming method ('zf', 'mrt', or None for SIM-only)
                               None: SIM phases do all beamforming (no digital weights)
                               'zf': Zero-forcing digital beamforming after SIM
                               'mrt': Maximum ratio transmission after SIM
            use_nearfield_user_channel: Use Rayleigh-Sommerfeld for user channel
        """
        # Initialize parent classes
        Transceiver.__init__(self, Nx, Ny, wavelength, max_scan_angle, device)
        UserChannel.__init__(self, num_users, wavelength, reference_distance,
                            path_loss_at_reference, min_user_distance,
                            max_user_distance, ricean_k_factors=None, device=device)

        # Set user positions if provided
        if user_positions is not None:
            UserChannel.set_user_positions(self, user_positions)

        # DigitalBeamformer-specific attributes
        self.sim_model = sim_model
        self.noise_power = noise_power
        self.total_power = total_power
        self.beamforming_method = beamforming_method
        self.use_nearfield_user_channel = use_nearfield_user_channel

        # Precompute static channels if SIM is provided
        if sim_model is not None:
            self._compute_static_channels()
        else:
            self.H = self.generate_channel(self.get_positions(), time=0.0)

    def _compute_static_channels(self):
        """
        Precompute static channel matrices A and H with normalization.

        A: Antenna → SIM first layer (doesn't change)
        H: SIM last layer → Users (can be time-varying)

        Stores both original and normalized versions for numerical stability.
        """
        # Channel A: Antenna → SIM first layer (Rayleigh-Sommerfeld)
        antenna_positions = self.get_positions().cpu().numpy()  # From Transceiver
        sim_first_layer = self.sim_model.get_first_layer_positions().cpu().numpy()

        self.A = rayleighSommerfeld(
            antenna_positions,
            sim_first_layer,
            self.wavelength,
            self.sim_model.metaAtomArea,
            self.device
        )

        # Channel H: SIM last layer → Users
        sim_last_layer = self.sim_model.get_last_layer_positions()

        if self.use_nearfield_user_channel:
            # Near-field: Use Rayleigh-Sommerfeld
            user_positions = self.user_positions  # From UserChannel
            self.H = rayleighSommerfeld(
                sim_last_layer.cpu().numpy(),
                user_positions,
                self.wavelength,
                self.sim_model.metaAtomArea,
                self.device
            )
        else:
            # Far-field: Use statistical channel model
            self.H = self.generate_channel(sim_last_layer, time=0.0)  # From UserChannel

    def update_user_channel(self, time: float = 0.0):
        """Update user channel H for time-varying channels with mobility."""
        if not self.use_nearfield_user_channel:
            sim_last_layer = self.sim_model.get_last_layer_positions()
            self.H = self.generate_channel(sim_last_layer, time=time)  # From UserChannel

    def compute_end_to_end_channel(self, phases: torch.Tensor = None, direction : str = 'down') -> torch.Tensor:
        """
        Compute end-to-end channel: H_eff = H @ Ψ @ A

        Args:
            phases: (L, N) SIM phase configuration

        Returns:
            H_eff: (K, A) end-to-end channel from antennas to users
        """
        if phases is not None:
            self.sim_model.update_phases(phases)
            Psi = self.sim_model.simChannel()
            # Use complex_matmul for MPS compatibility
            H_eff = complex_matmul(complex_matmul(self.H, Psi), self.A)
        else:
            H_eff = self.H
        if direction == 'up':
            H_eff = torch.conj(H_eff).T #Hermittian for uplink, assume reciprocity
        return H_eff

    def compute_sinr(self,
                     phases: torch.Tensor = None, #the SIM weights
                     power_allocation: Optional[torch.Tensor] = None,
                     digital_beamforming_weights : Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Actually computes the channel first then computes the SINR. Could handle SIM-only, digital-only, or hybrid.

        Args:
            phases: (L, N) SIM phase configuration. If None, assumes digital beamforming only .
            power_allocation: (K,) power allocation. If None, uses equal power.
            digital_beamforming_weights: (M, K) digital beamforming weights. If None, assumes SIM-only.

        Returns:
            sinr: (K,) SINR for each user
        """

        H_eff = self.compute_end_to_end_channel(phases) #here you get Psi if needed , else return H directly

        # Save the channel before digital weights are applied
        self.pre_digital_weights_channel = H_eff

        # Default: equal power allocation
        if power_allocation is None:
            power_allocation = torch.ones(self.num_users, device=self.device)
            power_allocation *= (self.total_power / self.num_users)

        if digital_beamforming_weights is not None: #assume only SIM
            # Use complex_matmul for MPS compatibility
            effective_channel = complex_matmul(H_eff, digital_beamforming_weights)
        else:
            effective_channel = H_eff

        # Compute SINR (this will save last_effective_channel in compute_sinr_downlink)
        sinr = self.compute_sinr_downlink(effective_channel, power_allocation, self.noise_power)  # From Transceiver
        return sinr

    def compute_sum_rate(self, phases: torch.Tensor,
                        power_allocation: Optional[torch.Tensor] = None,
                        digital_beamforming_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute sum-rate: Σ log₂(1 + SINR_k)

        Args:
            phases: (L, N) SIM phase configuration
            power_allocation: (K,) power allocation

        Returns:
            sum_rate: Scalar tensor (bits/s/Hz)
        """
        sinr = self.compute_sinr(phases, power_allocation, digital_beamforming_weights)
        rates = torch.log2(1 + sinr)
        return torch.sum(rates)

    def compute_per_user_rates(self, phases: torch.Tensor,
                               power_allocation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute per-user rates."""
        sinr = self.compute_sinr(phases, power_allocation)
        return torch.log2(1 + sinr)

    def evaluate_performance(self, phases: torch.Tensor,
                            power_allocation: Optional[torch.Tensor] = None) -> Dict:
        """
        Comprehensive performance evaluation.

        Returns:
            Dictionary with 'sinr', 'sinr_db', 'rates', 'sum_rate'
        """
        sinr = self.compute_sinr(phases, power_allocation)
        rates = torch.log2(1 + sinr)

        return {
            'sinr': sinr,
            'sinr_db': 10 * torch.log10(sinr),
            'rates': rates,
            'sum_rate': torch.sum(rates)
        }

    #================= Objective Functions for Optimization =================#
    def compute_sum_rate_given_power(self, power_allocation: torch.Tensor) -> torch.Tensor:
        """
        SIM-only: Compute sum-rate given fixed SIM phases and power allocation.

        Args:
            power_allocation: (K,) power allocation

        """
        pass
    def get_sim_phase_objective(self, power_allocation: Optional[torch.Tensor] = None):
        """
        SIM-only: Optimize phases, power fixed, no digital weights.

        Args:
            power_allocation: Fixed power (K,). If None, uses equal power.

        Returns:
            Callable: f(phases) -> sum_rate
        """
        if power_allocation is None:
            power_allocation = torch.ones(self.num_users, device=self.device) * (self.total_power / self.num_users)

        def objective_fn(phases):
            return self.compute_sum_rate(phases, power_allocation, digital_beamforming_weights=None)

        return objective_fn

    def get_sim_power_objective(self, phases: torch.Tensor):
        """
        SIM-only: Optimize power, phases fixed, no digital weights.

        Args:
            phases: Fixed SIM phases (L, N)

        Returns:
            Callable: f(power) -> sum_rate
        """
        def objective_fn(power):
            return self.compute_sum_rate(phases, power, digital_beamforming_weights=None)

        return objective_fn

    def get_digital_weight_objective(self, power_allocation: Optional[torch.Tensor] = None):
        """
        Digital-only: Optimize weights, power fixed, no SIM phases.

        Args:
            power_allocation: Fixed power (K,). If None, uses equal power.

        Returns:
            Callable: f(weights) -> sum_rate
        """
        if power_allocation is None:
            power_allocation = torch.ones(self.num_users, device=self.device) * (self.total_power / self.num_users)

        def objective_fn(weights):
            return self.compute_sum_rate(phases=None, power_allocation=power_allocation, digital_beamforming_weights=weights)

        return objective_fn

    def get_digital_power_objective(self, weights: torch.Tensor):
        """
        Digital-only: Optimize power, weights fixed, no SIM phases.

        Args:
            weights: Fixed digital beamforming weights (M, K)

        Returns:
            Callable: f(power) -> sum_rate
        """
        def objective_fn(power):
            return self.compute_sum_rate(phases=None, power_allocation=power, digital_beamforming_weights=weights)

        return objective_fn

    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'num_antennas': self.num_antennas,
            'num_users': self.num_users,
            'wavelength': self.wavelength,
            'noise_power': self.noise_power,
            'total_power': self.total_power,
            'beamforming_method': self.beamforming_method,
        }
        if self.sim_model is not None:
            info.update({
                'sim_layers': self.sim_model.layers,
                'sim_metaatoms': self.sim_model.metaAtoms,
                'channel_A_shape': tuple(self.A.shape),
                'channel_H_shape': tuple(self.H.shape)
            })
        return info