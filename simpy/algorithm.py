import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Union
from collections import deque
import random
from simpy.sim import Sim
from simpy.beamformer import Beamformer


class ProjectedGradientAscent:
    """
    Generic Projected Gradient Ascent (PGA) optimizer.

    Optimizes any differentiable parameters to maximize an objective function
    subject to constraints via projection.

    Supports:
        - SIM phases: [0, 2π] constraints
        - Digital beamforming weights: Normalization constraints
        - Power allocation: Sum and non-negativity constraints
        - Any custom parameters with custom projections

    Algorithm:
        1. Compute gradient of objective w.r.t. parameters
        2. Update: params_new = params_old + step_size * gradient
        3. Project: params_new = projection_fn(params_new)
        4. Repeat until convergence or max iterations
    """

    def __init__(self,
                 beamformer: Beamformer,
                 objective_fn: Callable,
                 projection_fn: Optional[Callable] = None,
                 learning_rate: float = 0.1,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = True,
                 use_backtracking: bool = False,
                 backtrack_beta: float = 0.5,
                 backtrack_c: float = 0.5,
                 backtrack_max_iter: int = 100):
        """
        Initialize Projected Gradient Ascent optimizer.

        Args:
            beamformer: Beamformer object (contains SIM, channels, and all components)
            objective_fn: Function to maximize, signature: objective_fn(params) -> scalar
                         Should return a scalar tensor to maximize
                         Examples:
                           - Phases: lambda phases: beamformer.compute_sum_rate(phases, power)
                           - Weights: lambda W: beamformer.compute_sum_rate(None, power, W)
            projection_fn: Function to project parameters onto constraint set
                          Signature: projection_fn(params) -> projected_params
                          If None, defaults to phase projection [0, 2π]
                          Use static methods like:
                            - PGA.project_phases (default)
                            - PGA.project_weights_normalize
                            - PGA.project_power
                            - Your custom projection
            learning_rate: Step size for gradient ascent (used if use_backtracking=False)
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold (stop if |objective_change| < tolerance)
            verbose: Print optimization progress
            use_backtracking: If True, use backtracking line search for adaptive step size
            backtrack_beta: Backtracking shrinkage factor (typically 0.5)
            backtrack_c: Armijo condition parameter (typically 0.5)
            backtrack_max_iter: Maximum backtracking iterations (default 100)
        """
        self.beamformer = beamformer
        self.sim_model = beamformer.sim_model
        self.objective_fn = objective_fn
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        # Backtracking line search parameters
        self.use_backtracking = use_backtracking
        self.backtrack_beta = backtrack_beta
        self.backtrack_c = backtrack_c
        self.backtrack_max_iter = backtrack_max_iter

        # Default projection: phases [0, 2π]
        if projection_fn is None:
            self.projection_fn = self.project_phases
        else:
            self.projection_fn = projection_fn

        # Get device from beamformer
        self.device = beamformer.device

        # Optimization history
        self.history = {
            'objective': [],
            'gradient_norm': [],
            'param_change': [],
            'step_size': []
        }

    # ========== Projection Functions (Static Methods) ==========

    @staticmethod
    def project_phases(phases: torch.Tensor) -> torch.Tensor:
        """
        Project phases onto [0, 2π] constraint set using modulo.

        Args:
            phases: (L, N) tensor of phase values

        Returns:
            Projected phases in [0, 2π]
        """
        return torch.remainder(phases, 2 * np.pi)

    @staticmethod
    def project_phases_clip(phases: torch.Tensor) -> torch.Tensor:
        """
        Project phases onto [0, 2π] by clipping.

        Args:
            phases: (L, N) tensor of phase values

        Returns:
            Clipped phases in [0, 2π]
        """
        return torch.clamp(phases, 0, 2 * np.pi)

    @staticmethod
    def project_phases_lore(phases: torch.Tensor) -> torch.Tensor:
        """
        A placeholder for finding the optimal phases using the lorenztian constraint method.
        """
        pass
    
    @staticmethod
    def project_weights_normalize(weights: torch.Tensor) -> torch.Tensor:
        """
        Project digital beamforming weights by normalizing each column.

        Args:
            weights: (M, K) complex tensor - beamforming weights

        Returns:
            Normalized weights where ||w_k|| = 1 for each user k
        """
        weights_normalized = weights.clone()
        K = weights.shape[1]
        for k in range(K):
            norm = torch.norm(weights_normalized[:, k])
            if norm > 1e-10:  # Avoid division by zero
                weights_normalized[:, k] = weights_normalized[:, k] / norm
        return weights_normalized

    @staticmethod
    def project_power(power: torch.Tensor, total_power: float = 1.0) -> torch.Tensor:
        """
        Project power allocation onto feasible set: sum(power) = total_power, power >= 0.

        Args:
            power: (K,) tensor of power values
            total_power: Total power constraint

        Returns:
            Projected power allocation
        """
        # Clip negative values to zero
        power_positive = torch.clamp(power, min=0.0)

        # Normalize to sum to total_power
        power_sum = power_positive.sum()
        if power_sum > 1e-10:
            power_normalized = power_positive * (total_power / power_sum)
        else:
            # If all zeros, use uniform allocation
            power_normalized = torch.ones_like(power) * (total_power / len(power))

        return power_normalized


    # ========== Backtracking Line Search ==========

    def _backtracking_line_search(self, params: torch.Tensor, gradient: torch.Tensor,
                                   current_objective: float) -> float:
        """
        Perform backtracking line search to find adaptive step size.

        Implements Armijo condition: f(x + μ∇f) >= f(x) + c·μ·||∇f||²

        Args:
            params: Current parameters
            gradient: Gradient at current parameters
            current_objective: Current objective value f(params)

        Returns:
            Step size μ that satisfies Armijo condition
        """
        mu = 1.0
        grad_norm_sq = (torch.norm(gradient)**2).item()

        for _ in range(self.backtrack_max_iter):
            # Test new parameters with current step size
            with torch.no_grad():
                params_test = params + mu * gradient
                # Project onto constraint set
                params_test = self.projection_fn(params_test)

            # Evaluate objective at test point
            new_objective = self.objective_fn(params_test).item()

            # Check Armijo condition
            if new_objective >= current_objective + self.backtrack_c * mu * grad_norm_sq:
                # Accept this step size
                break

            # Shrink step size
            mu = self.backtrack_beta * mu

        return mu

    # ========== Main Optimization ==========

    def optimize(self, initial_params: torch.Tensor) -> Dict:
        """
        Run Projected Gradient Ascent optimization.

        Args:
            initial_params: Initial parameter values (any shape).
                           Examples:
                             - Phases: torch.rand(L, N) * 2 * np.pi
                             - Weights: torch.randn(M, K, dtype=torch.complex64)
                             - Power: torch.ones(K) * (total_power / K)

        Returns:
            Dictionary with:
                - 'optimal_params': Optimized parameters
                - 'optimal_objective': Final objective value
                - 'iterations': Number of iterations performed
                - 'converged': Whether algorithm converged
                - 'history': Optimization history
        """
        # Initialize parameters (clone to avoid modifying input)
        params = initial_params.clone().detach()

        params = params.to(self.device)
        params.requires_grad = True

        # Reset history
        self.history = {
            'objective': [],
            'gradient_norm': [],
            'param_change': [],
            'step_size': []
        }

        if self.verbose:
            print("=" * 70)
            print("Projected Gradient Ascent Optimization")
            print("=" * 70)
            print(f"Parameter shape: {params.shape}")
            if self.use_backtracking:
                print(f"Step size: Adaptive (backtracking line search)")
                print(f"  Beta: {self.backtrack_beta}, c: {self.backtrack_c}")
            else:
                print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"Projection: {self.projection_fn.__name__}")
            print()

        converged = False
        prev_objective = None

        for iteration in range(self.max_iterations):
            # Zero gradients
            if params.grad is not None:
                params.grad.zero_()

            # Compute objective (forward pass)
            # The objective_fn receives params and computes the objective
            objective = self.objective_fn(params)

            # Compute gradient (backward pass)
            objective.backward()

            # Store gradient and objective value
            gradient = params.grad.detach().clone()
            grad_norm = torch.norm(gradient).item()
            objective_value = objective.item()

            # Determine step size
            if self.use_backtracking:
                step_size = self._backtracking_line_search(params, gradient, objective_value)
            else:
                step_size = self.learning_rate

            # Gradient ascent step
            with torch.no_grad():
                params_old = params.clone()

                # Update parameters with adaptive or fixed step size
                params.data = params.data + step_size * gradient

                # Project onto constraint set
                params.data = self.projection_fn(params.data)

                # Compute parameter change
                param_change = torch.norm(params - params_old).item()

            # Detach and reattach for next iteration
            params = params.detach()
            params.requires_grad = True

            # Store history
            self.history['objective'].append(objective_value)
            self.history['gradient_norm'].append(grad_norm)
            self.history['param_change'].append(param_change)
            self.history['step_size'].append(step_size)

            # Check convergence
            if prev_objective is not None:
                objective_change = abs(objective_value - prev_objective)
                if objective_change < self.tolerance:
                    converged = True
                    if self.verbose:
                        print(f"\nConverged at iteration {iteration+1}")
                        print(f"Objective change: {objective_change:.2e} < {self.tolerance:.2e}")
                    break

            prev_objective = objective_value

            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iter {iteration:4d} | Obj: {objective_value:12.6f} | "
                      f"Grad norm: {grad_norm:10.6f} | Param change: {param_change:10.6f}")

        # Final update and evaluation
        optimal_params = params.detach()
        final_objective = self.objective_fn(optimal_params).item()

        if self.verbose:
            print()
            print("=" * 70)
            print("Optimization Complete")
            print("=" * 70)
            print(f"Final objective: {final_objective:.6f}")
            print(f"Iterations: {iteration+1}")
            print(f"Converged: {converged}")
            print()

        return {
            'optimal_params': optimal_params,
            'optimal_objective': final_objective,
            'iterations': iteration + 1,
            'converged': converged,
            'history': self.history
        }

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot optimization convergence curves.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot objective
        ax = axes[0]
        ax.plot(self.history['objective'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Objective Function')
        ax.grid(True, alpha=0.3)

        # Plot gradient norm
        ax = axes[1]
        ax.semilogy(self.history['gradient_norm'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm (log scale)')
        ax.set_title('Gradient Magnitude')
        ax.grid(True, alpha=0.3)

        # Plot parameter change
        ax = axes[2]
        ax.semilogy(self.history['param_change'], 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Change (log scale)')
        ax.set_title('Parameter Update Magnitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()


class WaterFilling:
    """
    Iterative Water-Filling for Multi-User Power Allocation.

    Pure power allocation algorithm that only needs effective channel and noise.
    Works with any beamforming scheme (SIM, digital, hybrid).

    Algorithm:
        1. Given fixed effective channel H_eff (K, K)
        2. Iteratively optimize power for each user treating interference as noise
        3. Apply water-filling: allocate more power to users with better channels
        4. Repeat until convergence

    Reference: "Water-Filling: A Simple Concept" - various papers on multi-user systems

    Attributes (accessible via self):
        H_eff (torch.Tensor): Effective channel matrix (K, K)
                              H_eff[k,k] = signal gain for user k
                              H_eff[k,j] = interference from beam j to user k
        noise_power (float): Noise power in Watts
        total_power (float): Total power budget P_T in Watts
        max_iterations (int): Maximum iterations for convergence
        tolerance (float): Convergence threshold for power change
        verbose (bool): Whether to print optimization progress
        device (str): Computation device ('cpu', 'cuda', 'mps')
        K (int): Number of users
        channel_gains (torch.Tensor): Diagonal channel gains |H_eff[k,k]|^2 (K,)
        history (dict): Optimization history with keys:
                        - 'sum_rate': List of sum-rate values per iteration
                        - 'power_change': List of power changes per iteration
    """

    def __init__(self,
                 H_eff: torch.Tensor,
                 noise_power: float,
                 total_power: float = 1.0,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6,
                 verbose: bool = True,
                 device: str = 'cpu',
                 num_bits: Optional[int] = None,
                 antenna_signals_quantized: Optional[torch.Tensor] = None,
                 beamformer = None):
        """
        Initialize Water-Filling optimizer.

        Args:
            H_eff: Effective channel matrix (K, K) after beamforming
                   H_eff[k,k] = signal gain for user k
                   H_eff[k,j] = interference from beam j to user k
            noise_power: Noise power in Watts
            total_power: Total power budget P_T in Watts
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence threshold for power change
            verbose: Print optimization progress
            device: Computation device ('cpu', 'cuda', 'mps')
            num_bits: Number of DAC bits. None for continuous (default), 1 for 1-bit quantization
            antenna_signals_quantized: (M,) tensor - quantized antenna signals (required if num_bits=1)
            beamformer: Beamformer object (required if num_bits=1 for channel access)
        """
        self.H_eff = H_eff.detach() if isinstance(H_eff, torch.Tensor) else H_eff
        self.noise_power = noise_power
        self.total_power = total_power
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.device = device
        self.K = H_eff.shape[0]
        self.sinr = []

        # 1-bit DAC support
        self.num_bits = num_bits
        self.antenna_signals_quantized = antenna_signals_quantized
        self.beamformer = beamformer

        # Validate 1-bit parameters
        if self.num_bits == 1:
            if self.antenna_signals_quantized is None or self.beamformer is None:
                raise ValueError("For num_bits=1, must provide antenna_signals_quantized and beamformer")

        # Channel gains |H_eff[k,k]|^2 for each user
        self.channel_gains = torch.abs(torch.diag(self.H_eff))**2

        # Optimization history
        self.history = {
            'sum_rate': [],
            'power_change': []
        }

    def _compute_interference_plus_noise(self, power: torch.Tensor, user_k: int) -> float:
        """
        Compute interference plus noise for user k.

        Args:
            power: (K,) current power allocation
            user_k: Index of user to compute interference for

        Returns:
            interference + noise for user k
        """
        interference = 0.0
        for j in range(self.K):
            if j != user_k:
                # Interference from user j to user k
                interference += power[j] * (torch.abs(self.H_eff[user_k, j])**2).item()

        return interference + self.noise_power

    def _compute_sum_rate(self, power: torch.Tensor, return_diagnostics: bool = False) -> Union[float, Dict]:
        """
        Compute sum-rate given power allocation.

        Args:
            power: (K,) power allocation
            return_diagnostics: If True, return detailed per-user metrics

        Returns:
            If return_diagnostics=False: sum_rate (float)
            If return_diagnostics=True: dict with:
                - 'sum_rate': Total sum-rate in bits/s/Hz
                - 'sinr_per_user': SINR for each user (list)
                - 'snr_per_user': SNR for each user (signal/noise, no interference)
                - 'rate_per_user': Individual rate for each user in bits/s/Hz
                - 'power_per_user': Power allocation per user in Watts
        """
        sum_rate = 0.0
        sinr_per_user = []
        snr_per_user = []
        rate_per_user = []
        
        for k in range(self.K):
            # Signal power
            signal_power = power[k] * self.channel_gains[k]
            
            # SNR (no interference)
            snr = signal_power / self.noise_power
            
            # Interference power
            interference = 0.0
            for j in range(self.K):
                if j != k:
                    interference += power[j] * (torch.abs(self.H_eff[k, j])**2)
            
            # SINR (with interference)
            sinr = signal_power / (interference + self.noise_power)
            
            # Rate for user k
            rate_k = torch.log2(1 + sinr).item()
            sum_rate += rate_k
            
            if return_diagnostics:
                sinr_per_user.append(sinr.item() if isinstance(sinr, torch.Tensor) else sinr)
                snr_per_user.append(snr.item() if isinstance(snr, torch.Tensor) else snr)
                rate_per_user.append(rate_k)

        if return_diagnostics:
            return {
                'sum_rate': sum_rate,
                'sinr_per_user': sinr_per_user,
                'snr_per_user': snr_per_user,
                'rate_per_user': rate_per_user,
                'power_per_user': power.cpu().numpy().tolist(),
                'noise_power': self.noise_power
            }
        return sum_rate

    def _water_fill_single_user(self, user_k: int, power: torch.Tensor, remaining_power: float) -> float:
        """
        Water-filling for a single user given interference from others.

        Args:
            user_k: Index of user to optimize power for
            power: Current power allocation for all users
            remaining_power: Available power for this user

        Returns:
            Optimal power for user k
        """
        # Interference + noise seen by user k
        interference_noise = self._compute_interference_plus_noise(power, user_k)

        # Channel gain for user k
        gain_k = self.channel_gains[user_k].item()

        # This is now not used in the new waterfilling approach
        # We'll compute proper waterfilling in the optimize method

        # For now, return a simple allocation based on channel quality
        if gain_k > 1e-10:
            # Allocate proportional to channel quality relative to noise+interference
            quality_ratio = gain_k / (interference_noise + 1e-10)
            return remaining_power * min(1.0, quality_ratio / 10.0)  # Scale factor to prevent overshooting
        else:
            return 0.0

    def optimize(self, initial_power: Optional[torch.Tensor] = None) -> Dict:
        """
        Run iterative water-filling optimization.

        Args:
            initial_power: Initial power allocation (K,). If None, uses equal allocation.

        Returns:
            Dictionary with:
                - 'optimal_power': Optimized power allocation (K,)
                - 'optimal_sum_rate': Final sum-rate value
                - 'iterations': Number of iterations performed
                - 'converged': Whether algorithm converged
                - 'history': Optimization history
        """
        # Initialize power
        if initial_power is None:
            power = torch.ones(self.K, device=self.device) * (self.total_power / self.K)
        else:
            power = initial_power.clone().detach()

        # Reset history
        self.history = {
            'sum_rate': [],
            'power_change': []
        }

        if self.verbose:
            print("=" * 70)
            print("Water-Filling Power Optimization")
            print("=" * 70)
            print(f"Number of users: {self.K}")
            print(f"Total power: {self.total_power:.4f} W")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print()

        converged = False

        for iteration in range(self.max_iterations):
            power_old = power.clone()

            # Compute interference for each user using previous iteration's power
            interference = torch.zeros(self.K, device=self.device)
            for k in range(self.K):
                for j in range(self.K):
                    if j != k:
                        interference[k] += power_old[j] * torch.abs(self.H_eff[k, j])**2

            # Add noise to interference
            noise_plus_interference = interference + self.noise_power

            # Compute effective channel gains divided by noise+interference
            # This is the "depth" for waterfilling
            effective_gains = self.channel_gains / noise_plus_interference

            mu_min = 0.0
            mu_max = self.total_power + torch.max(1.0 / effective_gains).item()

            for _ in range(50):  # Binary search iterations
                mu = (mu_min + mu_max) / 2

                # Compute power with this water level
                # Waterfilling formula: p_k = [μ - 1/effective_gain_k]⁺
                powers_test = torch.clamp(mu - 1.0 / effective_gains, min=0.0)

                total_test = powers_test.sum().item()

                if abs(total_test - self.total_power) < 1e-6 * self.total_power:
                    break
                elif total_test < self.total_power:
                    mu_min = mu
                else:
                    mu_max = mu

            # Final power allocation with found water level
            new_power = torch.clamp(mu - 1.0 / effective_gains, min=0.0)

            # Normalize to ensure exact power constraint
            power_sum = new_power.sum()
            if power_sum > 0:
                new_power = new_power * (self.total_power / power_sum)
            else:
                # If all zeros, use uniform allocation
                new_power = torch.ones(self.K, device=self.device) * (self.total_power / self.K)

            power = new_power

            # Compute sum-rate
            sum_rate = self._compute_sum_rate(power)

            # Compute power change
            power_change = torch.norm(power - power_old).item()

            # Store history
            self.history['sum_rate'].append(sum_rate)
            self.history['power_change'].append(power_change)

            # Check convergence
            if power_change < self.tolerance:
                converged = True
                if self.verbose:
                    print(f"\nConverged at iteration {iteration+1}")
                    print(f"Power change: {power_change:.2e} < {self.tolerance:.2e}")
                break

            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iter {iteration:4d} | Sum-Rate: {sum_rate:12.6f} | "
                      f"Power change: {power_change:10.6f}")

        # Final sum-rate and diagnostics
        diagnostics = self._compute_sum_rate(power=power, return_diagnostics=True)

        if self.verbose:
            print()
            print("=" * 70)
            print("Optimization Complete")
            print("=" * 70)
            print(f"Final sum-rate: {diagnostics['sum_rate']:.6f} bits/s/Hz")
            print(f"SINR per user (linear): {[f'{s:.2f}' for s in diagnostics['sinr_per_user']]}")
            print(f"SINR per user (dB): {[f'{10*np.log10(s):.2f}' for s in diagnostics['sinr_per_user']]}")
            print(f"SNR per user (dB): {[f'{10*np.log10(s):.2f}' for s in diagnostics['snr_per_user']]}")
            print(f"Rate per user: {[f'{r:.2f}' for r in diagnostics['rate_per_user']]} bits/s/Hz")
            print(f"Iterations: {iteration+1}")
            print(f"Converged: {converged}")
            print(f"Power allocation: {power.cpu().numpy()}")
            print(f"Total power used: {power.sum().item():.6f} W")
            print()

        return {
            'optimal_power': power,
            'optimal_sum_rate': diagnostics['sum_rate'],
            'sinr_per_user': diagnostics['sinr_per_user'],
            'snr_per_user': diagnostics['snr_per_user'],
            'rate_per_user': diagnostics['rate_per_user'],
            'power_per_user': diagnostics['power_per_user'],
            'noise_power': diagnostics['noise_power'],
            'iterations': iteration + 1,
            'converged': converged,
            'history': self.history
        }

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot water-filling convergence.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot sum-rate
        ax = axes[0]
        ax.plot(self.history['sum_rate'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Sum-Rate (bits/s/Hz)')
        ax.set_title('Water-Filling: Sum-Rate Evolution')
        ax.grid(True, alpha=0.3)

        # Plot power change
        ax = axes[1]
        ax.semilogy(self.history['power_change'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Power Change (log scale)')
        ax.set_title('Power Allocation Convergence')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()

# ========== Reinforcement Learning Algorithms ==========

class ReplayBuffer:
    """Experience replay buffer for RL algorithms."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network with configurable output type."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, output_type: str = 'phases'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            output_type: Type of output - 'phases', 'power', or 'both'
        """
        super(Actor, self).__init__()
        self.output_type = output_type
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        if self.output_type == 'phases':
            # Phases: [0, 2π]
            return torch.sigmoid(output) * 2 * np.pi
        elif self.output_type == 'power':
            # Power: use softmax for sum constraint (ensures sum=1)
            return F.softmax(output, dim=-1)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class Critic(nn.Module):
    """Critic Q(s,a) network."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TwinCritic(nn.Module):
    """Twin critics for TD3."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TwinCritic, self).__init__()
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_1 = nn.Linear(hidden_dim, 1)
        self.fc1_2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc2_1(q1))
        q1 = self.fc3_1(q1)
        q2 = F.relu(self.fc1_2(x))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.fc3_2(q2)
        return q1, q2
    
    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc2_1(q1))
        return self.fc3_1(q1)


class DDPG:
    """Deep Deterministic Policy Gradient for SIM optimization (phases, power, or both)."""
    def __init__(self, beamformer: Beamformer, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005, buffer_size: int = 100000,
                 batch_size: int = 64, noise_std: float = 0.1,
                 optimize_target: str = 'phases', verbose: bool = True,
                 device: Optional[str] = None):
        """
        Initialize DDPG agent for beamforming optimization.

        Args:
            beamformer (Beamformer): The beamformer system to optimize
            state_dim (int): Dimension of state space. Typically 2*K*N + K for SIM (K users, N metaatoms)
                             State = [channel_H.real, channel_H.imag, power_allocation]
            action_dim (int): Dimension of action space. Typically L*N for phase optimization (L layers, N metaatoms)
                              Action = SIM phases to apply

            hidden_dim (int): Number of hidden units in actor/critic networks. Default 256
            actor_lr (float): Learning rate for actor network. Default 1e-4
            critic_lr (float): Learning rate for critic network. Default 1e-3

            gamma (float): Discount factor for future rewards [0, 1]. Default 0.99
                          How much the agent values future rewards vs immediate rewards
            tau (float): Soft update coefficient for target networks [0, 1]. Default 0.005
                        target_weight = tau * online_weight + (1-tau) * target_weight
                        Smaller tau = slower, more stable updates

            buffer_size (int): Maximum size of replay buffer. Default 100000
                              Stores (state, action, reward, next_state, done) transitions
            batch_size (int): Mini-batch size for training. Default 64
                             Number of samples to use per gradient update
            noise_std (float): Standard deviation of exploration noise added to actions. Default 0.1
                              Helps agent explore action space during training

            optimize_target (str): What to optimize - 'phases' or 'power'. Default 'phases'
                - 'phases': optimize SIM phases (action_dim = L*N)
                - 'power': optimize power allocation (action_dim = K)
            verbose (bool): Print training progress. Default True

            device (str): Device for all operations ('cpu', 'cuda', 'mps'). Default None (uses beamformer's device)
                         All neural networks and tensors will be on this device
        """
        self.beamformer = beamformer
        self.device = device if device is not None else beamformer.device

        # Move beamformer tensors to device if needed
        if self.device != beamformer.device:
            self.beamformer.to_device(self.device)
        self.action_dim = action_dim
        self.optimize_target = optimize_target
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.verbose = verbose

        # Determine output type for actor
        output_type = optimize_target

        self.actor = Actor(state_dim, action_dim, hidden_dim, output_type).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, output_type).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.history = {'episode_rewards': [], 'actor_loss': [], 'critic_loss': []}
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if add_noise:
            if self.optimize_target == 'phases':
                # Add noise for phases and clip to [0, 2π]
                action = np.clip(action + np.random.normal(0, self.noise_std, action.shape), 0, 2 * np.pi)
            elif self.optimize_target == 'power':
                # For power, actor already outputs valid distribution via softmax
                # Add small noise and renormalize
                noise = np.random.normal(0, self.noise_std * 0.1, action.shape)
                action = np.maximum(action + noise, 0)  # Ensure non-negative
                action = action / action.sum()  # Renormalize to sum to 1

        return action
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
    
    def optimize(self, num_episodes: int = 500, steps_per_episode: int = 50,
                 power_allocation: Optional[torch.Tensor] = None,
                 initial_phases: Optional[torch.Tensor] = None,
                 warmup_episodes: int = 50) -> Dict:
        """
        Optimize SIM phases using DDPG.

        Args:
            power_allocation: Fixed power allocation for all users
            initial_phases: Starting phases for warm-start (shape: L x N)
            warmup_episodes: Number of episodes to explore around initial phases (default 50)
        """
        if power_allocation is None:
            power_allocation = torch.ones(self.beamformer.num_users, device=self.device) * (self.beamformer.total_power / self.beamformer.num_users)
        power_allocation_fixed = power_allocation

        if self.verbose:
            print("=" * 70)
            print(f"DDPG Training (optimizing phases)")
            print("=" * 70)

        best_reward = -float('inf')
        best_params = None
        initial_action = None

        # Initialize with starting point if provided
        if initial_phases is not None:
            best_params = initial_phases.clone().detach()
            best_reward = self.beamformer.compute_sum_rate(initial_phases, power_allocation_fixed).item()
            initial_action = initial_phases.flatten().cpu().numpy()
            if self.verbose:
                print(f"Warm-start from initial_phases with sum-rate: {best_reward:.6f}")
                print(f"Using initial phases as base for first {warmup_episodes} episodes")

            # Pre-seed replay buffer with experiences around initial point
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed]).cpu().numpy()

            for _ in range(self.batch_size):
                noise = np.random.normal(0, self.noise_std * 0.3, initial_action.shape)
                action_var = np.clip(initial_action + noise, 0, 2 * np.pi)
                phases_var = torch.FloatTensor(action_var).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)
                reward = self.beamformer.compute_sum_rate(phases_var, power_allocation_fixed).item()
                self.replay_buffer.push(state, action_var, reward, state, False)

            if self.verbose:
                print(f"Pre-seeded replay buffer with {self.batch_size} experiences around initial point")

        for episode in range(num_episodes):
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed]).cpu().numpy()

            episode_reward = 0

            for step in range(steps_per_episode):
                # Warm-start: explore around initial phases during warmup period
                if initial_action is not None and episode < warmup_episodes:
                    warmup_progress = episode / warmup_episodes
                    noise_scale = 0.3 + 0.7 * warmup_progress
                    noise = np.random.normal(0, self.noise_std * noise_scale, initial_action.shape)
                    action = np.clip(initial_action + noise, 0, 2 * np.pi)
                else:
                    action = self.select_action(state, add_noise=True)

                phases_current = torch.FloatTensor(action).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)
                reward = self.beamformer.compute_sum_rate(phases_current, power_allocation_fixed).item()

                episode_reward += reward

                self.replay_buffer.push(state, action, reward, state, step == steps_per_episode - 1)
                actor_loss, critic_loss = self.update()

                if actor_loss is not None:
                    self.history['actor_loss'].append(actor_loss)
                    self.history['critic_loss'].append(critic_loss)

            avg_reward = episode_reward / steps_per_episode
            self.history['episode_rewards'].append(avg_reward)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = self.select_action(state, add_noise=False)
                best_params = torch.FloatTensor(best_action).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)

            if self.verbose and episode % 10 == 0:
                print(f"Episode {episode:4d} | Reward: {avg_reward:10.4f} | Best: {best_reward:10.4f}")

        if self.verbose:
            print(f"\nBest sum-rate: {best_reward:.6f} bits/s/Hz\n")

        return {'optimal_params': best_params, 'optimal_objective': best_reward, 'history': self.history}

    def save_weights(self, path: str):
        """Save agent weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'history': self.history,
        }, path)
        if self.verbose:
            print(f"DDPG weights saved to {path}")

    def load_weights(self, path: str):
        """Load agent weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if self.verbose:
            print(f"DDPG weights loaded from {path}")

    def optimize_with_policy(self, initial_phases: torch.Tensor,
                            power_allocation: torch.Tensor,
                            num_iterations: int = 5000,
                            early_stopping_patience: int = 100) -> Dict:
        """
        Use trained policy to optimize phases via iterative policy rollout.
        This is for inference/deployment, not training. No exploration noise.

        Args:
            initial_phases: Starting phases (L, N)
            power_allocation: Fixed power allocation (K,)
            num_iterations: Number of optimization iterations
            early_stopping_patience: Stop if no improvement for this many iterations

        Returns:
            Dictionary with 'optimal_params', 'optimal_objective', 'history'
        """
        power_allocation_fixed = power_allocation.to(self.device)

        # Start from initial phases
        current_phases = initial_phases.clone().to(self.device)
        best_phases = current_phases.clone()

        # Evaluate initial
        best_reward = self.beamformer.compute_sum_rate(current_phases, power_allocation_fixed).item()

        no_improvement_count = 0
        iteration_rewards = []

        for iteration in range(num_iterations):
            # Get state from current channel, power, and current phases
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed, current_phases.flatten()]).cpu().numpy()

            # Use policy to get next action (NO NOISE - pure exploitation)
            action = self.select_action(state, add_noise=False)

            # Convert action to phases
            phases = torch.FloatTensor(action).reshape(
                self.beamformer.sim_model.layers,
                self.beamformer.sim_model.metaAtoms
            ).to(self.device)

            # Evaluate
            reward = self.beamformer.compute_sum_rate(phases, power_allocation_fixed).item()
            iteration_rewards.append(reward)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_phases = phases.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Update current phases for next iteration
            current_phases = phases

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {early_stopping_patience} iterations)")
                break

            if self.verbose and iteration % 500 == 0:
                print(f"Iteration {iteration:5d} | Current: {reward:10.4f} | Best: {best_reward:10.4f}")

        if self.verbose:
            print(f"\nPolicy optimization complete. Best sum-rate: {best_reward:.6f} bits/s/Hz\n")

        return {
            'optimal_params': best_phases,
            'optimal_objective': best_reward,
            'history': {'iteration_rewards': iteration_rewards}
        }


class TD3:
    """Twin Delayed DDPG for SIM phase optimization."""
    def __init__(self, beamformer: Beamformer, state_dim: int, action_dim: int,
                 hidden_dim: int = 256, actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005, policy_noise: float = 0.2,
                 noise_clip: float = 0.5, policy_delay: int = 2, buffer_size: int = 100000,
                 batch_size: int = 64, noise_std: float = 0.1, verbose: bool = True,
                 device: Optional[str] = None):
        """
        Initialize TD3 (Twin Delayed DDPG) agent for beamforming optimization.
        TD3 uses two critic networks and delayed policy updates for improved stability.

        Args:
            beamformer (Beamformer): The beamformer system to optimize
            state_dim (int): Dimension of state space. Typically 2*K*N + K for SIM (K users, N metaatoms)
                             State = [channel_H.real, channel_H.imag, power_allocation]
            action_dim (int): Dimension of action space. Typically L*N for phase optimization (L layers, N metaatoms)
                              Action = SIM phases to apply

            hidden_dim (int): Number of hidden units in actor/critic networks. Default 256
            actor_lr (float): Learning rate for actor network. Default 1e-4
            critic_lr (float): Learning rate for critic network. Default 1e-3

            gamma (float): Discount factor for future rewards [0, 1]. Default 0.99
                          How much the agent values future rewards vs immediate rewards
            tau (float): Soft update coefficient for target networks [0, 1]. Default 0.005
                        target_weight = tau * online_weight + (1-tau) * target_weight
                        Smaller tau = slower, more stable updates

            policy_noise (float): Std dev of target policy smoothing noise. Default 0.2
                                 Noise added to target actions to prevent exploiting Q-value errors
            noise_clip (float): Range for clipping target policy noise. Default 0.5
                               Noise clipped to [-noise_clip, noise_clip]

            policy_delay (int): Update actor every N critic updates. Default 2
                              TD3 delays actor updates to let critic stabilize
                              Example: policy_delay=2 means update actor every 2 critic updates

            buffer_size (int): Maximum size of replay buffer. Default 100000
                              Stores (state, action, reward, next_state, done) transitions
            batch_size (int): Mini-batch size for training. Default 64
                             Number of samples to use per gradient update
            noise_std (float): Standard deviation of exploration noise added to actions. Default 0.1
                              Helps agent explore action space during training

            verbose (bool): Print training progress. Default True

            device (str): Device for all operations ('cpu', 'cuda', 'mps'). Default None (uses beamformer's device)
                         All neural networks and tensors will be on this device
        """
        self.beamformer = beamformer
        self.device = device if device is not None else beamformer.device

        # Move beamformer tensors to device if needed
        if self.device != beamformer.device:
            self.beamformer.to_device(self.device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.verbose = verbose
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim, hidden_dim, 'phases').to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, 'phases').to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.history = {'episode_rewards': [], 'actor_loss': [], 'critic_loss': []}

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if add_noise:
            action = np.clip(action + np.random.normal(0, self.noise_std, action.shape), 0, 2 * np.pi)

        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(0, 2 * np.pi)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = actor_loss.item()

        return actor_loss, critic_loss.item()

    def optimize(self, num_episodes: int = 500, steps_per_episode: int = 50,
                 power_allocation: Optional[torch.Tensor] = None,
                 initial_phases: Optional[torch.Tensor] = None,
                 warmup_episodes: int = 50) -> Dict:
        """
        Optimize SIM phases using TD3.

        Args:
            power_allocation: Fixed power allocation for all users
            initial_phases: Starting phases for warm-start (shape: L x N)
            warmup_episodes: Number of episodes to explore around initial phases (default 50)
        """
        if power_allocation is None:
            power_allocation = torch.ones(self.beamformer.num_users, device=self.device) * (self.beamformer.total_power / self.beamformer.num_users)
        power_allocation_fixed = power_allocation

        if self.verbose:
            print("=" * 70)
            print(f"TD3 Training (optimizing phases)")
            print("=" * 70)

        best_reward = -float('inf')
        best_params = None
        initial_action = None

        # Initialize with starting point if provided
        if initial_phases is not None:
            best_params = initial_phases.clone().detach()
            best_reward = self.beamformer.compute_sum_rate(initial_phases, power_allocation_fixed).item()
            initial_action = initial_phases.flatten().cpu().numpy()
            if self.verbose:
                print(f"Warm-start from initial_phases with sum-rate: {best_reward:.6f}")
                print(f"Using initial phases as base for first {warmup_episodes} episodes")

            # Pre-seed replay buffer with experiences around initial point
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed]).cpu().numpy()

            for _ in range(self.batch_size):
                noise = np.random.normal(0, self.noise_std * 0.3, initial_action.shape)
                action_var = np.clip(initial_action + noise, 0, 2 * np.pi)
                phases_var = torch.FloatTensor(action_var).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)
                reward = self.beamformer.compute_sum_rate(phases_var, power_allocation_fixed).item()
                self.replay_buffer.push(state, action_var, reward, state, False)

            if self.verbose:
                print(f"Pre-seeded replay buffer with {self.batch_size} experiences around initial point")

        for episode in range(num_episodes):
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed]).cpu().numpy()

            episode_reward = 0

            for step in range(steps_per_episode):
                # Warm-start: explore around initial phases during warmup period
                if initial_action is not None and episode < warmup_episodes:
                    warmup_progress = episode / warmup_episodes
                    noise_scale = 0.3 + 0.7 * warmup_progress
                    noise = np.random.normal(0, self.noise_std * noise_scale, initial_action.shape)
                    action = np.clip(initial_action + noise, 0, 2 * np.pi)
                else:
                    action = self.select_action(state, add_noise=True)

                phases_current = torch.FloatTensor(action).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)
                reward = self.beamformer.compute_sum_rate(phases_current, power_allocation_fixed).item()

                episode_reward += reward

                self.replay_buffer.push(state, action, reward, state, step == steps_per_episode - 1)
                actor_loss, critic_loss = self.update()

                if actor_loss is not None:
                    self.history['actor_loss'].append(actor_loss)
                if critic_loss is not None:
                    self.history['critic_loss'].append(critic_loss)

            avg_reward = episode_reward / steps_per_episode
            self.history['episode_rewards'].append(avg_reward)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = self.select_action(state, add_noise=False)
                best_params = torch.FloatTensor(best_action).reshape(
                    self.beamformer.sim_model.layers, self.beamformer.sim_model.metaAtoms
                ).to(self.device)

            if self.verbose and episode % 10 == 0:
                print(f"Episode {episode:4d} | Reward: {avg_reward:10.4f} | Best: {best_reward:10.4f}")

        if self.verbose:
            print(f"\nBest sum-rate: {best_reward:.6f} bits/s/Hz\n")

        return {'optimal_params': best_params, 'optimal_objective': best_reward, 'history': self.history}

    def save_weights(self, path: str):
        """Save agent weights to disk."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'history': self.history,
        }, path)
        if self.verbose:
            print(f"TD3 weights saved to {path}")

    def load_weights(self, path: str):
        """Load agent weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if self.verbose:
            print(f"TD3 weights loaded from {path}")

    def optimize_with_policy(self, initial_phases: torch.Tensor,
                            power_allocation: torch.Tensor,
                            num_iterations: int = 5000,
                            early_stopping_patience: int = 100) -> Dict:
        """
        Use trained policy to optimize phases via iterative policy rollout.
        This is for inference/deployment, not training. No exploration noise.

        Args:
            initial_phases: Starting phases (L, N)
            power_allocation: Fixed power allocation (K,)
            num_iterations: Number of optimization iterations
            early_stopping_patience: Stop if no improvement for this many iterations

        Returns:
            Dictionary with 'optimal_params', 'optimal_objective', 'history'
        """
        power_allocation_fixed = power_allocation.to(self.device)

        # Start from initial phases
        current_phases = initial_phases.clone().to(self.device)
        best_phases = current_phases.clone()

        # Evaluate initial
        best_reward = self.beamformer.compute_sum_rate(current_phases, power_allocation_fixed).item()

        no_improvement_count = 0
        iteration_rewards = []

        for iteration in range(num_iterations):
            # Get state from current channel, power, and current phases
            H_eff = self.beamformer.H.flatten()
            state = torch.cat([H_eff.real, H_eff.imag, power_allocation_fixed, current_phases.flatten()]).cpu().numpy()

            # Use policy to get next action (NO NOISE - pure exploitation)
            action = self.select_action(state, add_noise=False)

            # Convert action to phases
            phases = torch.FloatTensor(action).reshape(
                self.beamformer.sim_model.layers,
                self.beamformer.sim_model.metaAtoms
            ).to(self.device)

            # Evaluate
            reward = self.beamformer.compute_sum_rate(phases, power_allocation_fixed).item()
            iteration_rewards.append(reward)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_phases = phases.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Update current phases for next iteration
            current_phases = phases

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {early_stopping_patience} iterations)")
                break

            if self.verbose and iteration % 500 == 0:
                print(f"Iteration {iteration:5d} | Current: {reward:10.4f} | Best: {best_reward:10.4f}")

        if self.verbose:
            print(f"\nPolicy optimization complete. Best sum-rate: {best_reward:.6f} bits/s/Hz\n")

        return {
            'optimal_params': best_phases,
            'optimal_objective': best_reward,
            'history': {'iteration_rewards': iteration_rewards}
        }


# ========== 1-Bit DAC Precoding ==========

def quantize_to_1bit(signal):
    """
    Quantize complex signal to 1-bit discrete set: {±1±j}

    Args:
        signal: (N,) complex tensor - continuous signal

    Returns:
        (N,) complex tensor - quantized to {-1-j, -1+j, 1-j, 1+j}
    """
    real_part = torch.sign(torch.real(signal))
    imag_part = torch.sign(torch.imag(signal))

    # Handle zeros (map to +1 by default)
    real_part = torch.where(real_part == 0, torch.ones_like(real_part), real_part)
    imag_part = torch.where(imag_part == 0, torch.ones_like(imag_part), imag_part)

    return real_part + 1j * imag_part


class CG_MC1bit:
    """
    Convergence-Guaranteed Multi-Carrier 1-bit Precoding (CG-MC1bit)

    Based on Algorithm 1 from:
    Wen et al., "One-Bit Downlink Precoding for Massive MIMO OFDM System"
    IEEE Trans. Wireless Commun., Vol. 22, No. 9, September 2023

    Solves the non-convex optimization problem:
        minimize: MSE = Σ_k ||s[k] - A·H[k]·x[k]||²₂ + Nσ²Σα_i²
        subject to: x[k] ∈ {±1±j} for all k (1-bit DAC constraint)

    Uses modified ADMM (Alternating Direction Method of Multipliers) with:
        - Convergence guarantee when λ > √(2·Lϕ)
        - Nonlinear precoding that accounts for quantization
        - Per-user adjustment factors for different path losses

    Key Features:
        - Handles OFDM multi-carrier systems
        - Accounts for 1-bit DAC quantization during optimization (not after)
        - Supports users with different channel conditions
        - Guaranteed convergence to stationary point
    """

    def __init__(self, beamformer, lambda_penalty=0.01, max_iterations=100,
                 tolerance=1e-6, verbose=False, device='cpu'):
        """
        Initialize CG-MC1bit optimizer.

        Args:
            beamformer: Beamformer object with channel H
            lambda_penalty: ADMM penalty factor (must be > √(2·Lϕ) for convergence)
                          Typical value: 0.01 - 0.1
            max_iterations: Maximum ADMM iterations (typically 50-200)
            tolerance: Convergence tolerance for MSE change
            verbose: Print iteration details
            device: 'cpu', 'cuda', or 'mps'
        """
        self.beamformer = beamformer
        self.lambda_penalty = lambda_penalty
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.device = device

        self.H = beamformer.H.to(device)  # (K, M) channel matrix
        self.K, self.M = self.H.shape
        self.noise_power = beamformer.noise_power

    def optimize(self, symbols, total_power):
        """
        Run CG-MC1bit optimization to find 1-bit precoded signals.

        Args:
            symbols: (K,) tensor - input symbols per user (e.g., QPSK, QAM)
            total_power: Scalar - total transmit power constraint

        Returns:
            dict: {
                'antenna_signals': (M,) complex tensor - quantized to {±1±j},
                'alpha': (K,) tensor - per-user adjustment factors,
                'mse_history': list - MSE value at each iteration,
                'converged': bool - whether algorithm converged,
                'iterations': int - number of iterations used
            }
        """
        symbols = symbols.to(self.device)

        # ========== Initialize ADMM Variables ==========
        # X: continuous antenna signals (primal variable, frequency domain)
        # R: auxiliary variable (will be quantized to enforce 1-bit constraint)
        # V: dual variable (Lagrange multiplier for constraint X = R)
        # alpha: per-user adjustment factors (accounts for different path losses)

        X = torch.ones(self.M, dtype=torch.complex64, device=self.device)
        R = X.clone()
        V = torch.zeros(self.M, dtype=torch.complex64, device=self.device)
        alpha = torch.ones(self.K, device=self.device) * 0.01  # Small initial value

        mse_history = []

        # ========== ADMM Iterations ==========
        for t in range(self.max_iterations):
            X_old = X.clone()

            # ===== Step 1: Update X (Continuous Precoding Variable) =====
            # Minimize: ||s - A·H·x||² + (λ/2)||x - R + V/λ||²
            # Solution: x = (2H^H·H + λI)^(-1) · (2H^H·s + λR - V)
            #
            # H_tilde = diag(α) @ H accounts for per-user scaling

            # Convert alpha to complex for matrix multiplication with complex H
            H_tilde = torch.diag(alpha.to(torch.complex64)) @ self.H  # (K, M) weighted channel

            # Build linear system: (2H^H·H + λI) x = (2H^H·s + λR - V)
            term1 = 2 * H_tilde.conj().T @ H_tilde + self.lambda_penalty * torch.eye(
                self.M, dtype=torch.complex64, device=self.device
            )
            term2 = (2 * H_tilde.conj().T @ symbols) + (self.lambda_penalty * R) - V

            # Solve for X (closed-form solution, equation 17 from paper)
            X = torch.linalg.solve(term1, term2)

            # ===== Step 2: Update R (1-Bit Quantization/Projection) =====
            # Project: R = P_A(X + V/λ) where A = {±1±j}
            # This is the CRITICAL step that enforces the 1-bit DAC constraint
            #
            # Projection simply: sign(real) + j·sign(imag)

            temp = X + V / self.lambda_penalty
            R = quantize_to_1bit(temp)

            # ===== Step 3: Update Dual Variable V =====
            # Dual ascent: V^(t+1) = V^t + λ(X^(t+1) - R^(t+1))
            # Enforces constraint X = R through penalty

            V = V + (self.lambda_penalty * (X - R))

            # ===== Step 4: Update Per-User Adjustment Factors α =====
            # α_i = Re(R^H · h_i · s_i) / (||h_i · R||² + σ²)
            # Accounts for users with different channel conditions/path losses
            #
            # From equation (22) in paper

            for i in range(self.K):
                h_i = self.H[i, :]  # (M,) channel for user i
                # FIX: Symbol should be conjugated in alpha formula
                # α_i = Re(s_i · R^H · h_i) / (|h_i · R|² + σ²)
                numerator = torch.real(torch.conj(R).T * h_i * symbols[i])
                denominator = (torch.conj(R).T * h_i * torch.conj(h_i).T * R) + self.noise_power
                alpha[i] = torch.sum(numerator) / torch.sum(denominator)

            # ===== Compute MSE for Monitoring =====
            # MSE = ||s - A·H·R||² + σ²·Σα_i²

            H_tilde = torch.diag(alpha.to(torch.complex64)) @ self.H
            mse = torch.sum(torch.norm(symbols - H_tilde @ R)**2) + self.noise_power * torch.sum(alpha**2)
            mse_history.append(mse.item())

            # ===== Check Convergence =====
            # Stop if X doesn't change significantly

            change = torch.norm(X - X_old).item()
            if change < self.tolerance:
                if self.verbose:
                    print(f"   CG-MC1bit converged at iteration {t+1}")
                break

            if self.verbose and t % 20 == 0:
                print(f"   Iter {t}: MSE={mse.item():.6f}, change={change:.6e}")

        # ========== Post-Processing: Power Normalization ==========
        # Normalize R to satisfy total power constraint
        # γ = P_total / ||R||²

        gamma = total_power / torch.sum(torch.abs(R)**2)
        R_normalized = torch.sqrt(gamma) * R

        converged = (t < self.max_iterations - 1)

        if self.verbose:
            print(f"   CG-MC1bit finished: {t+1} iterations, "
                  f"converged={converged}, final MSE={mse_history[-1]:.6f}")

        return {
            'antenna_signals': R_normalized,
            'alpha': alpha,
            'mse_history': mse_history,
            'converged': converged,
            'iterations': t + 1
        }
