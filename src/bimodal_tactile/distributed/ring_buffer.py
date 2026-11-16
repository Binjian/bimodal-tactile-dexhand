"""Continuous Frame Arena & Strided Offset Indexing for Multi-Rate Trajectory Buffers.

Eliminates the 45% VRAM overlap redundancy caused by copying overlapping 32 ms
1,000 Hz Hall windows at every 60 Hz (~17 ms) RL rollout step.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DistributedMemoryBudget:
    """Analytical VRAM & NVLink bandwidth calculator for multi-GPU bimodal tactile RL."""

    num_gpus: int = 8
    envs_per_gpu: int = 2048
    n_taxels: int = 1140
    hall_window: int = 32
    substeps_per_rl_step: int = 17
    rollout_horizon: int = 32
    vbt_tips: int = 2
    vbt_res: int = 64
    model_params_billions: float = 1.5

    def compute_breakdown_gib(self) -> dict[str, float]:
        b = self.envs_per_gpu
        gib = 1024 ** 3

        # 1. Naive overlapping Hall rollout buffer: [T, W, B, N_taxels, 6] fp16 (2 bytes)
        naive_hall_bytes = (
            self.rollout_horizon * self.hall_window * b * self.n_taxels * 6 * 2
        )
        # 2. Continuous frame arena: [W + T * stride, B, N_taxels, 6] fp16
        arena_len = self.hall_window + self.rollout_horizon * self.substeps_per_rl_step
        arena_hall_bytes = arena_len * b * self.n_taxels * 6 * 2

        # 3. VBT trajectory buffer: uint8 RGB (3ch) + int8 flow (2ch) = 5 bytes/px
        naive_vbt_fp32_bytes = (
            self.rollout_horizon * b * self.vbt_tips * self.vbt_res * self.vbt_res * 5 * 4
        )
        compact_vbt_bytes = (
            self.rollout_horizon * b * self.vbt_tips * self.vbt_res * self.vbt_res * 5 * 1
        )

        # 4. ZeRO-2 optimizer & parameter memory (bf16 weights replicated + fp32 master/Adam sharded)
        params = self.model_params_billions * 1e9
        naive_model_bytes = params * (2 + 4 + 4 + 4 + 2)  # 16 bytes/param
        zero2_model_bytes = params * 2 + (params * 14) / self.num_gpus

        return {
            "total_parallel_envs": float(self.num_gpus * self.envs_per_gpu),
            "naive_hall_rollout_gib": naive_hall_bytes / gib,
            "arena_hall_rollout_gib": arena_hall_bytes / gib,
            "hall_vram_saving_pct": 100.0 * (1.0 - arena_hall_bytes / naive_hall_bytes),
            "naive_vbt_fp32_gib": naive_vbt_fp32_bytes / gib,
            "compact_vbt_int8_gib": compact_vbt_bytes / gib,
            "naive_model_opt_gib": naive_model_bytes / gib,
            "zero2_model_opt_gib": zero2_model_bytes / gib,
        }


class ContinuousHallArena:
    """Zero-redundancy rollout storage using strided integer window indexing."""

    def __init__(
        self,
        rollout_horizon: int,
        substeps_per_step: int,
        window_len: int,
        batch_size: int,
        n_taxels: int,
    ) -> None:
        self.T = rollout_horizon
        self.stride = substeps_per_step
        self.W = window_len
        self.total_frames = self.W + self.T * self.stride
        self.arena = np.zeros((self.total_frames, batch_size, n_taxels, 6), dtype=np.float16)

    def write_substep(self, global_substep: int, data_fp16: np.ndarray) -> None:
        """Write a 1,000 Hz substep directly into the continuous arena."""
        idx = global_substep % self.total_frames
        self.arena[idx] = data_fp16

    def gather_rl_step_window(self, rl_step: int) -> np.ndarray:
        """Retrieve [W, B, N_taxels, 6] chronological view for RL step `rl_step`."""
        start = rl_step * self.stride
        indices = np.arange(start, start + self.W) % self.total_frames
        return self.arena[indices]
