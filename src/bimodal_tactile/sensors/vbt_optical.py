"""Vision-Based Tactile (VBT / GelSight-style) Optical & Deformation Simulator.

Implements:
1. Orthographic depth buffer penetration clipping: H_raw = max(0, z_far - Z).
2. 2D-FFT Screened Poisson PDE solver for continuous silicone membrane draping.
3. Surface normal computation and Taxim-style polynomial photometric stereo RGB shading.
4. Boussinesq-Cerruti Green's function tangential & torsional marker optical flow field.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VBTConfig:
    """Configuration for a GelSight / DIGIT-style vision-based tactile sensor."""

    resolution: tuple[int, int] = (64, 64)   # (H, W) pixels
    pixel_pitch_m: float = 0.0003125         # 0.3125 mm/px -> 20 mm x 20 mm sensing pad
    z_far_m: float = 0.010                   # Undeformed gel outer surface distance (m)
    poisson_lambda: float = 18.0             # Silicone membrane lateral tension smoothing factor
    shear_compliance: float = 1.8e-4         # Tangential gel displacement per Newton (m/N)
    torsion_compliance: float = 0.025        # Torsional twist angle per N*m (rad / (N*m))
    vignette_strength: float = 0.22          # Optical chamber edge vignetting


class VBTOpticalSimulator:
    """Fast GPU-friendly photometric stereo & marker flow simulator for VBT sensors."""

    def __init__(self, config: VBTConfig | None = None, seed: int = 123) -> None:
        self.cfg = config or VBTConfig()
        self.H, self.W = self.cfg.resolution
        self.rng = np.random.default_rng(seed)

        # Precompute 2D frequency grid for Screened Poisson membrane PDE
        fy = np.fft.fftfreq(self.H, d=1.0)
        fx = np.fft.fftfreq(self.W, d=1.0)
        wx, wy = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy)
        self.poisson_denom = 1.0 + self.cfg.poisson_lambda * (wx ** 2 + wy ** 2)

        # Normalized spatial coordinates [-1, 1] for Taxim polynomial LUT & vignetting
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, self.H),
            np.linspace(-1.0, 1.0, self.W),
            indexing="ij",
        )
        self.grid_y = yy
        self.grid_x = xx
        self.vignette = 1.0 - self.cfg.vignette_strength * (xx ** 2 + yy ** 2)

    def solve_membrane_height(self, depth_buffer_m: np.ndarray) -> np.ndarray:
        """Compute smooth silicone surface indentation H(u,v) from orthographic depth Z(u,v)."""
        h_raw = np.maximum(0.0, self.cfg.z_far_m - np.asarray(depth_buffer_m, dtype=np.float64))
        if np.max(h_raw) <= 1e-9:
            return np.zeros_like(h_raw)
        h_fft = np.fft.fft2(h_raw)
        h_smooth = np.real(np.fft.ifft2(h_fft / self.poisson_denom))
        return np.maximum(0.0, h_smooth)

    def compute_surface_normals(self, height_map_m: np.ndarray) -> np.ndarray:
        """Compute unit surface normals n = normalize(-dH/dx, -dH/dy, 1) from height map."""
        dh_dy, dh_dx = np.gradient(height_map_m, self.cfg.pixel_pitch_m)
        nx = -dh_dx
        ny = -dh_dy
        nz = np.ones_like(height_map_m)
        norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
        return np.stack([nx / norm, ny / norm, nz / norm], axis=-1)

    def taxim_photometric_shade(self, normals: np.ndarray) -> np.ndarray:
        """Render RGB tactile image via calibrated 3rd-order polynomial Taxim LUT model.

        Models oblique Red (left), Green (top-right), and Blue (bottom-right) LED illumination
        plus spatial chamber reflection gradients and quadratic cross-terms.
        """
        nx = normals[..., 0]
        ny = normals[..., 1]
        nz = normals[..., 2]
        u = self.grid_x
        v = self.grid_y

        # Polynomial basis mimicking calibrated multi-LED internal chamber reflection
        r_chan = (
            0.46
            + 0.42 * nx
            + 0.08 * ny
            - 0.12 * (nx ** 2)
            + 0.06 * nx * u
            - 0.05 * u
        )
        g_chan = (
            0.48
            - 0.21 * nx
            + 0.38 * ny
            - 0.10 * (ny ** 2)
            + 0.05 * ny * v
            + 0.04 * v
        )
        b_chan = (
            0.50
            - 0.22 * nx
            - 0.34 * ny
            + 0.08 * (nz - 1.0)
            - 0.05 * nx * ny
            + 0.03 * (u - v)
        )

        rgb = np.stack([r_chan, g_chan, b_chan], axis=-1) * self.vignette[..., None]
        rgb += self.rng.normal(0.0, 0.006, size=rgb.shape)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def compute_marker_optical_flow(
        self,
        height_map_m: np.ndarray,
        tangential_force_xy: tuple[float, float],
        torsional_torque_z: float,
    ) -> np.ndarray:
        """Compute 2D tangential marker displacement field [H, W, 2] (in pixels).

        Combines Boussinesq-Cerruti elastic shear spreading around the contact patch
        with rigid rotational curl v_rot = omega x r for detecting pure torsional slip.
        """
        if np.max(height_map_m) <= 1e-7:
            return np.zeros((self.H, self.W, 2), dtype=np.float32)

        # Contact weight envelope normalized to [0, 1]
        contact_weight = height_map_m / (np.max(height_map_m) + 1e-8)
        # Smooth spatial decay outside direct contact via Gaussian envelope
        r_sq = self.grid_x ** 2 + self.grid_y ** 2
        elastic_kernel = 0.7 * contact_weight + 0.3 * np.exp(-r_sq / 0.35) * np.mean(contact_weight)

        fx, fy = tangential_force_xy
        disp_x_m = self.cfg.shear_compliance * fx * elastic_kernel
        disp_y_m = self.cfg.shear_compliance * fy * elastic_kernel

        # Rotational slip curl component: u_rot = theta_z * [-y, x]
        theta_z = self.cfg.torsion_compliance * torsional_torque_z
        pad_half_width = 0.5 * self.W * self.cfg.pixel_pitch_m
        x_m = self.grid_x * pad_half_width
        y_m = self.grid_y * pad_half_width
        disp_x_m += -theta_z * y_m * elastic_kernel
        disp_y_m += theta_z * x_m * elastic_kernel

        flow_px = np.stack(
            [disp_x_m / self.cfg.pixel_pitch_m, disp_y_m / self.cfg.pixel_pitch_m],
            axis=-1,
        )
        return flow_px.astype(np.float32)

    def render(
        self,
        depth_buffer_m: np.ndarray,
        tangential_force_xy: tuple[float, float] = (0.0, 0.0),
        torsional_torque_z: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Full VBT forward pass returning height map, normals, uint8 RGB, and 2D optical flow."""
        height = self.solve_membrane_height(depth_buffer_m)
        normals = self.compute_surface_normals(height)
        rgb = self.taxim_photometric_shade(normals)
        flow = self.compute_marker_optical_flow(height, tangential_force_xy, torsional_torque_z)
        return {
            "height_m": height,
            "normals": normals,
            "rgb": rgb,
            "flow_px": flow,
        }
