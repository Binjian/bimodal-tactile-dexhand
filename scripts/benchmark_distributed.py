"""8-GPU (16,384 Parallel Envs) Memory & Continuous Arena Benchmark."""

import numpy as np
from bimodal_tactile.distributed import ContinuousHallArena, DistributedMemoryBudget


def main() -> None:
    print("=== 8-GPU Distributed Bimodal Tactile Training Memory & Arena Benchmark ===")
    budget = DistributedMemoryBudget(
        num_gpus=8,
        envs_per_gpu=2048,
        n_taxels=1140,
        hall_window=32,
        substeps_per_rl_step=17,
        rollout_horizon=32,
        model_params_billions=1.5,
    )
    stats = budget.compute_breakdown_gib()
    for k, v in stats.items():
        print(f"  {k:<28}: {v:8.2f}")

    # Verify ContinuousHallArena strided indexing accuracy on a scaled test instance
    arena = ContinuousHallArena(
        rollout_horizon=8,
        substeps_per_step=17,
        window_len=32,
        batch_size=16,
        n_taxels=10,
    )
    for sub in range(arena.total_frames):
        arena.write_substep(sub, np.full((16, 10, 6), fill_value=float(sub), dtype=np.float16))

    for rl_step in range(8):
        win = arena.gather_rl_step_window(rl_step)
        expected_start = rl_step * 17
        assert int(win[0, 0, 0, 0]) == expected_start
        assert int(win[-1, 0, 0, 0]) == expected_start + 31
    print("ContinuousHallArena strided window integrity check: PASSED")


if __name__ == "__main__":
    main()
