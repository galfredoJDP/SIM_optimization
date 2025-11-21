import torch
import numpy as np
from typing import Callable, Optional, Dict, List
from sim import sim


class ProjectedGradientAscent:
    """
    Projected Gradient Ascent (PGA) for SIM phase optimization.

    Optimizes SIM meta-atom phases to maximize an objective function
    (e.g., sum-rate, SINR) subject to phase constraints [0, 2À].

    Algorithm:
        1. Compute gradient of objective w.r.t. phases
        2. Update: phases_new = phases_old + step_size * gradient
        3. Project: phases_new = mod(phases_new, 2À)
        4. Repeat until convergence or max iterations
    """

    def __init__(self,
                 sim_model: sim,
                 objective_fn: Callable,
                 learning_rate: float = 0.1,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 projection: str = 'modulo',
                 verbose: bool = True):
        """
        Initialize Projected Gradient Ascent optimizer.

        Args:
            sim_model: SIM object to optimize
            objective_fn: Function to maximize, signature: objective_fn(phases) -> scalar
                         Should return a scalar tensor (loss/reward to maximize)
            learning_rate: Step size for gradient ascent
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold (stop if |objective_change| < tolerance)
            projection: Projection method ('modulo' or 'clip')
                       'modulo': phases mod 2À (wraps around)
                       'clip': clip to [0, 2À] (hard boundaries)
            verbose: Print optimization progress
        """
        self.sim_model = sim_model
        self.objective_fn = objective_fn
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.projection = projection
        self.verbose = verbose

        # Get device from SIM
        self.device = sim_model.device

        # Optimization history
        self.history = {
            'objective': [],
            'gradient_norm': [],
            'phase_change': []
        }

    def _project_phases(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Project phases onto [0, 2À] constraint set.

        Args:
            phases: (L, N) tensor of phase values

        Returns:
            Projected phases in [0, 2À]
        """
        if self.projection == 'modulo':
            # Wrap phases to [0, 2À] using modulo
            return torch.fmod(phases, 2 * np.pi) % (2 * np.pi)
        elif self.projection == 'clip':
            # Clip phases to [0, 2À]
            return torch.clamp(phases, 0, 2 * np.pi)
        else:
            raise ValueError(f"Unknown projection method: {self.projection}")

    def optimize(self, initial_phases: Optional[torch.Tensor] = None) -> Dict:
        """
        Run Projected Gradient Ascent optimization.

        Args:
            initial_phases: (L, N) initial phase values. If None, uses current SIM phases.

        Returns:
            Dictionary with:
                - 'optimal_phases': (L, N) optimized phases
                - 'optimal_objective': final objective value
                - 'iterations': number of iterations performed
                - 'converged': whether algorithm converged
                - 'history': optimization history
        """
        # Initialize phases
        if initial_phases is None:
            phases = self.sim_model.values().clone().detach()
        else:
            phases = initial_phases.clone().detach()

        phases = phases.to(self.device)
        phases.requires_grad = True

        # Reset history
        self.history = {
            'objective': [],
            'gradient_norm': [],
            'phase_change': []
        }

        if self.verbose:
            print("=" * 70)
            print("Projected Gradient Ascent Optimization")
            print("=" * 70)
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Tolerance: {self.tolerance}")
            print(f"Projection: {self.projection}")
            print()

        converged = False
        prev_objective = None

        for iteration in range(self.max_iterations):
            # Zero gradients
            if phases.grad is not None:
                phases.grad.zero_()

            # Update SIM with current phases
            self.sim_model.update_phases(phases.detach())

            # Compute objective (forward pass)
            objective = self.objective_fn(phases)

            # Compute gradient (backward pass)
            objective.backward()

            # Store gradient norm
            grad_norm = torch.norm(phases.grad).item()

            # Gradient ascent step
            with torch.no_grad():
                phases_old = phases.clone()
                phases.data = phases.data + self.learning_rate * phases.grad

                # Project onto constraint set
                phases.data = self._project_phases(phases.data)

                # Compute phase change
                phase_change = torch.norm(phases - phases_old).item()

            # Detach and reattach for next iteration
            phases = phases.detach()
            phases.requires_grad = True

            # Store history
            objective_value = objective.item()
            self.history['objective'].append(objective_value)
            self.history['gradient_norm'].append(grad_norm)
            self.history['phase_change'].append(phase_change)

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
                      f"Grad norm: {grad_norm:10.6f} | Phase change: {phase_change:10.6f}")

        # Final update
        optimal_phases = phases.detach()
        self.sim_model.update_phases(optimal_phases)
        final_objective = self.objective_fn(optimal_phases).item()

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
            'optimal_phases': optimal_phases,
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

        # Plot phase change
        ax = axes[2]
        ax.semilogy(self.history['phase_change'], 'g-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Phase Change (log scale)')
        ax.set_title('Phase Update Magnitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()


class AdaptiveProjectedGradientAscent(ProjectedGradientAscent):
    """
    PGA with adaptive learning rate (line search or momentum).

    Inherits from ProjectedGradientAscent and adds:
    - Backtracking line search for adaptive step size
    - Momentum for faster convergence
    """

    def __init__(self,
                 sim_model: sim,
                 objective_fn: Callable,
                 initial_learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 backtracking: bool = True,
                 backtrack_factor: float = 0.5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 projection: str = 'modulo',
                 verbose: bool = True):
        """
        Initialize Adaptive PGA optimizer.

        Args:
            sim_model: SIM object to optimize
            objective_fn: Function to maximize
            initial_learning_rate: Initial step size
            momentum: Momentum coefficient [0, 1] (0 = no momentum)
            backtracking: Use backtracking line search for step size
            backtrack_factor: Factor to reduce step size in backtracking
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold
            projection: Projection method
            verbose: Print progress
        """
        super().__init__(sim_model, objective_fn, initial_learning_rate,
                        max_iterations, tolerance, projection, verbose)

        self.momentum = momentum
        self.backtracking = backtracking
        self.backtrack_factor = backtrack_factor
        self.velocity = None  # For momentum

    # TODO: Implement adaptive optimization in future version
    # For now, inherits base PGA behavior