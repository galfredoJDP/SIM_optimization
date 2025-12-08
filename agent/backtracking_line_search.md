# Backtracking Line Search

## Purpose
Adaptively find a step size that guarantees improvement at each iteration.

## Standard Gradient Ascent
```
θ_{t+1} = θ_t + α · ∇f(θ_t)
```
**Problem**: Fixed `α` may overshoot (too large) or converge slowly (too small).

## Backtracking Line Search
```
Start with large step μ = 1
While Armijo condition fails:
    μ = β · μ  (shrink step, typically β = 0.5)
θ_{t+1} = θ_t + μ · ∇f(θ_t)
```

## Armijo Condition
Accept step if:
```
f(θ + μ·∇f) ≥ f(θ) + c · μ · ||∇f||²
```
Where `c ∈ (0, 1)`, typically `c = 0.5`.

**Meaning**: New objective must improve by at least a fraction of what gradient predicts.

## PyTorch Implementation
```python
def gradient_with_backtracking(params, objective_fn, _):
    # Compute gradient via autograd
    params_temp = params.clone().requires_grad_(True)
    obj = objective_fn(params_temp)
    obj.backward()
    gradient = params_temp.grad.detach()
    
    # Backtracking
    mu = 1.0
    beta = 0.5
    c = 0.5
    R_ini = obj.item()
    grad_norm_sq = (torch.norm(gradient)**2).item()
    
    for _ in range(100):
        with torch.no_grad():
            params_test = params + mu * gradient
        R_new = objective_fn(params_test).item()
        
        if R_new >= R_ini + c * mu * grad_norm_sq:
            break  # Accept
        mu = beta * mu  # Shrink
    
    return gradient, mu
```

## Usage with PGA
```python
pga = ProjectedGradientAscent(
    beamformer=bf,
    objective_fn=my_obj,
    gradient_func=gradient_with_backtracking
)
```

## Benefits
- **No tuning**: No need to manually set learning rate
- **Adaptive**: Large steps when possible, small when needed
- **Guaranteed improvement**: Each iteration makes progress

## Trade-off
Each backtracking evaluation calls `objective_fn()`, adding cost per iteration.
