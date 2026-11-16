# Bimodal Tactile Dexterous Hand (`bimodal-tactile-dexhand`)

A complete engineering reference, multi-rate Sim2Real simulation toolkit, GPU kernel prototype, and 1,000 Hz anti-slip admittance control library for **heterogeneous tactile dexterous hands** combining **6D Hall-Effect Magnetoelectric Arrays** (e.g., PaXini ITPU / PX-6AX architecture) and **Vision-Based Tactile Sensors (VBT)** (e.g., GelSight / DIGIT).

---

## Project Architecture

```text
bimodal-tactile-dexhand/
├── pyproject.toml                  # uv / Hatchling project configuration
├── README.md                       # Quickstart & architecture overview
├── assets/
│   └── mjcf/
│       └── bimodal_tactile_finger.xml  # Single-source-of-truth MuJoCo MJCF model
├── docs/                           # Complete technical architecture & math derivations
│   ├── 01_sensor_comparison_and_physics.md
│   ├── 02_heterogeneous_fusion_architecture.md
│   ├── 03_mujoco_warp_sim2real_pipeline.md
│   ├── 04_distributed_8gpu_training.md
│   └── 05_fast_slow_slip_admittance_control.md
├── src/bimodal_tactile/            # Modular Python library
│   ├── sensors/
│   │   ├── hall_6d.py              # Elastic kernel scatter, magnetic dipole coupling & stick-slip injection
│   │   └── vbt_optical.py          # Orthographic depth clipping, 2D-FFT Poisson draping, Taxim LUT & Boussinesq flow
│   ├── sim/
│   │   ├── warp_kernels.py         # NVIDIA Warp fused GPU kernels + NumPy fallback
│   │   └── multi_rate_env.py       # 1000 Hz Hall ring buffer + 60 Hz VBT exposure-midpoint sync
│   ├── control/
│   │   ├── slip_detector.py        # 1000 Hz Biquad BPF (50-300 Hz), ellipsoidal friction cone & common-mode rejection
│   │   └── admittance.py           # Decoupled discrete admittance controller & bumpless fast-slow handoff
│   └── distributed/
│       ├── ring_buffer.py          # Continuous frame arena + strided offset indexing (45% VRAM savings)
│       └── ddp_trainer.py          # Reverse-order NCCL bucketing, BF16 comm hook & contact-aware grad scaling
├── scripts/                        # Executable end-to-end demonstrations
│   ├── run_sim2real_demo.py
│   ├── run_slip_control_sim.py
│   └── benchmark_distributed.py
└── notebooks/                      # Interactive design & tutorial Jupyter notebooks
    ├── 01_sensor_physics_and_simulation.ipynb
    ├── 02_fast_slow_slip_admittance_tutorial.ipynb
    └── 03_gpu_kernels_and_distributed_scaling.ipynb
```

---

## Quickstart with `uv`

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment and install core dependencies**:
   ```bash
   cd bimodal-tactile-dexhand
   uv sync
   ```
   *(Optional)* Install GPU simulation (`warp-lang`, `mujoco`, `torch`) and JupyterLab extras:
   ```bash
   uv sync --extra notebooks
   # On Linux/CUDA workstations with NVIDIA GPUs:
   uv sync --extra gpu --extra notebooks
   ```

3. **Run the simulation & control scripts**:
   ```bash
   # 1. Multi-rate 6D Hall (1000 Hz) + VBT (60 Hz) forward physics demo
   uv run python scripts/run_sim2real_demo.py

   # 2. Closed-loop 1000 Hz stick-slip detection & discrete admittance control
   uv run python scripts/run_slip_control_sim.py

   # 3. 8-GPU (16,384 envs) memory & NCCL communication scaling benchmark
   uv run python scripts/benchmark_distributed.py
   ```

4. **Launch interactive Jupyter tutorials**:
   ```bash
   uv run jupyter lab notebooks/
   ```
