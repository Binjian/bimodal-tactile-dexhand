"""Multi-Rate Hardware/Sim Synchronization Bus (1,000 Hz Hall Ring Buffer + 60 Hz VBT)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SyncConfig:
    """Timing configuration matching FPGA hardware exposure-midpoint synchronization."""

    fast_hz: int = 1000
    slow_hz: int = 60
    ring_window_steps: int = 32   # 32 ms high-frequency history window
    n_taxels: int = 1140          # DexH13 full-hand ITPU taxel count
    vbt_shape: tuple[int, int, int] = (2, 64, 64)  # (num_tips, H, W)

    @property
    def decimation(self) -> int:
        return max(1, int(round(self.fast_hz / self.slow_hz)))


class MultiRateTactileBus:
    """Zero-shift circular buffer aligning 1,000 Hz 6D Hall samples with 60 Hz VBT frames."""

    def __init__(self, cfg: SyncConfig | None = None) -> None:
        self.cfg = cfg or SyncConfig()
        self.ring_buffer = np.zeros(
            (self.cfg.ring_window_steps, self.cfg.n_taxels, 6),
            dtype=np.float16,
        )
        self.latest_vbt_rgb = np.zeros((*self.cfg.vbt_shape, 3), dtype=np.uint8)
        self.latest_vbt_flow = np.zeros((*self.cfg.vbt_shape, 2), dtype=np.float32)

        self.step_count = 0
        self.last_vbt_sync_step = -1

    def push_hall_sample(self, hall_wrench_6d: np.ndarray) -> None:
        """Write one 1,000 Hz tactile frame into the circular buffer via modulo pointer."""
        head = self.step_count % self.cfg.ring_window_steps
        self.ring_buffer[head] = hall_wrench_6d.astype(np.float16)
        self.step_count += 1

    def should_trigger_vbt(self) -> bool:
        """Check if current fast tick aligns with the VBT camera exposure midpoint."""
        return (self.step_count % self.cfg.decimation) == 0

    def push_vbt_frame(self, rgb: np.ndarray, flow: np.ndarray) -> None:
        """Latch synchronized VBT RGB + optical flow frame."""
        self.latest_vbt_rgb[:] = rgb
        self.latest_vbt_flow[:] = flow
        self.last_vbt_sync_step = self.step_count

    def get_chronological_hall_window(self) -> np.ndarray:
        """Return chronological [32, N_taxels, 6] window ending at the most recent sample."""
        head = self.step_count % self.cfg.ring_window_steps
        return np.roll(self.ring_buffer, shift=-head, axis=0)
