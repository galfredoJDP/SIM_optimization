import torch
import numpy as np
from typing import Callable, Optional, Dict, List
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
                 verbose: bool = True):
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
            learning_rate: Step size for gradient ascent
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold (stop if |objective_change| < tolerance)
            verbose: Print optimization progress
        """
        self.beamformer = beamformer
        self.sim_model = beamformer.sim_model
        self.objective_fn = objective_fn
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

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
            'param_change': []
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
        return torch.fmod(phases, 2 * np.pi) % (2 * np.pi)

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
            'param_change': []
        }

        if self.verbose:
            print("=" * 70)
            print("Projected Gradient Ascent Optimization")
            print("=" * 70)
            print(f"Parameter shape: {params.shape}")
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

            # Store gradient norm
            grad_norm = torch.norm(params.grad).item()

            # Gradient ascent step
            with torch.no_grad():
                params_old = params.clone()

                # Update parameters
                params.data = params.data + self.learning_rate * params.grad

                # Project onto constraint set
                params.data = self.projection_fn(params.data)

                # Compute parameter change
                param_change = torch.norm(params - params_old).item()

            # Detach and reattach for next iteration
            params = params.detach()
            params.requires_grad = True

            # Store history
            objective_value = objective.item()
            self.history['objective'].append(objective_value)
            self.history['gradient_norm'].append(grad_norm)
            self.history['param_change'].append(param_change)

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