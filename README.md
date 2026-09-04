# ICOS co-design: locomotion and symmetry on an icosahedral robot

Figures, data and visualization code for **"Locomotion Selects for Symmetry: Evolutionary Co-Design of
Legged Robots on an Icosahedral Body"** (IROS 2026 Workshop on Learning-based Robot Co-design).

Project page: https://jl6017.github.io/icos-codesign/

- `figures/` — paper figures (symmetry-plane metric, evolved bodies in the abstract and realistic simulators,
  symmetry-drift regressions, mixed-competition dynamics).
- `vis/` — plotting and rendering scripts (matplotlib / MuJoCo). Training and evolution code is being
  released incrementally; see the project page for updates.

Simulation stack: MuJoCo XLA (MJX) via MuJoCo Playground, Brax PPO, JAX. Hardware line: Dynamixel XM430-W350,
Raspberry Pi 5, RealSense T265.
