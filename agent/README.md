# Agent Memory Files

## Purpose
These files contain context and knowledge about the SIM_optimization project. Future Claude sessions should read these files at the start of conversations to understand the project state and past decisions.

## Files in This Directory

### 1. `project_context.md` - READ FIRST
- Project overview and architecture
- Core classes and system flow
- Key design decisions
- Channel generation strategy
- Parameter conventions from paper
- Common operations and usage patterns

### 2. `recent_modifications.md`
- Log of recent changes with dates
- Problems solved and solutions implemented
- Implementation rationale
- Testing status
- User's specific notes and comments

### 3. `technical_details.md`
- Mathematical formulas and equations
- Signal model details
- Channel model mathematics
- Optimization problem formulation
- Numerical stability considerations
- Code patterns for device handling

## Quick Reference

### Project Type
Beamforming optimization for Stacked Intelligent Metasurfaces (SIM) in holographic MIMO.

### Key Files to Know
- `wireless/channel.py`: Channel models (recently modified)
- `Beamformer.py`: Main beamformer class
- `wireless/sim.py`: Metasurface model
- `files/Sum-Rate_Maximization_*.pdf`: Reference paper

### Recent Changes (2025-12-01)

**Major Session Updates:**
1. **DDPG/TD3 Enhancement** - Now support flexible optimization targets:
   - Optimize phases with fixed power
   - Optimize power with fixed phases
   - State/action dimensions: 204→250 corrected

2. **Power Sweep in main.py** - Parametric optimization across power levels:
   - Automatic result saving (individual + combined files)
   - Auto-generated plots and statistics
   - Supports any number of power levels

3. **Device Optimization** - CPU for PGA, MPS for RL recommended

4. **Quantization Support** - Discussion and implementation for quantized CSI

See `recent_modifications.md` for detailed changes and `technical_details.md` for implementation patterns.

### Important User Preferences
- Channels should NOT regenerate during optimization (per paper)
- CLT mode preferred for efficiency
- Geometric mode noted as incomplete (needs cluster delay line for FR3)
- User wants fast, practical implementations
- RL algorithms should be flexible (optimize phases OR power, not just phases)
- Results should be saved automatically and frequently

## For Future Claude Sessions

When starting a new conversation:
1. Read `project_context.md` for overall understanding
2. Check `recent_modifications.md` for latest changes
3. Refer to `technical_details.md` for equations/formulas as needed
4. These files represent what "I know" about the project

## Maintenance
Update these files when:
- Making significant code changes
- User provides important context or decisions
- Discovering new project requirements
- Solving problems that might recur

Keep files concise but complete - focus on what future sessions need to know.