"""6D Hall-Effect Magnetoelectric Tactile Array Physics Simulator.

Implements:
1. Force- and torque-conserving elastic half-space Gaussian kernel scatter per body link.
2. Compliance displacement -> 3D magnetic dipole flux -> stray motor field -> inverse calibration matrix.
3. Critical-friction stick-slip micro-vibration (50-300 Hz) stochastic injection.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Hall6DConfig:
    """Configuration for a distributed 6D Hall tactile array (e.g., PaXini ITPU grid)."""

    sigma_elast: float = 0.0045          # Elastic spatial spread radius (m)
    compliance_xyz: tuple[float, float, float] = (1.2e-4, 1.2e-4, 0.8e-4)  # m/N
    r0_dipole: tuple[float, float, float] = (0.0, 0.0, -0.002)  # Initial magnet-to-Hall offset (m)
    dipole_moment: tuple[float, float, float] = (0.0, 0.0, 0.015)  # A*m^2
    cross_axis_error: float = 0.025      # 2.5% residual cross-axis calibration error
    noise_force_std: float = 0.003       # 3 mN sensor noise floor (0.005-0.01 N resolution)
    noise_torque_std: float = 1.5e-5     # N*m torque noise floor
    mu_static: float = 0.65              # Static friction coefficient
    slip_vib_freq_hz: float = 165.0      # Dominant stick-slip micro-vibration frequency
    slip_vib_gain: float = 0.045         # Micro-vibration amplitude scaling


class Hall6DArraySimulator:
    """Forward physical model for a multi-taxel 6D Hall magnetoelectric tactile array."""

    def __init__(
        self,
        taxel_positions: np.ndarray,
        link_ids: np.ndarray,
        config: Hall6DConfig | None = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            taxel_positions: [N_taxels, 3] taxel positions in hand/local frame (m).
            link_ids: [N_taxels] integer body link ID for each taxel (enables link-bucketing).
            config: Physical and calibration parameters.
            seed: Random seed for noise and stick-slip phase jitter.
        """
        self.taxel_positions = np.asarray(taxel_positions, dtype=np.float64)
        self.link_ids = np.asarray(link_ids, dtype=np.int32)
        self.n_taxels = self.taxel_positions.shape[0]
        self.cfg = config or Hall6DConfig()
        self.rng = np.random.default_rng(seed)

        self.compliance = np.diag(self.cfg.compliance_xyz)
        self.r0 = np.asarray(self.cfg.r0_dipole, dtype=np.float64)
        self.m_vec = np.asarray(self.cfg.dipole_moment, dtype=np.float64)

        # Precompute baseline magnetic field B(r0) and linearized Jacobian M_cal
        self.b0 = self._dipole_field(self.r0)
        self.m_cal_inv = self._build_calibration_inverse()
        self._vib_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=self.n_taxels)

    def _dipole_field(self, r: np.ndarray) -> np.ndarray:
        """Exact magnetic dipole flux density B(r) up to mu_0 / (4*pi) scale factor."""
        r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
        r_norm = np.maximum(r_norm, 1e-6)
        m_dot_r = np.sum(r * self.m_vec, axis=-1, keepdims=True)
        return (3.0 * m_dot_r * r / (r_norm ** 5)) - (self.m_vec / (r_norm ** 3))

    def _build_calibration_inverse(self) -> np.ndarray:
        """Build linear inverse calibration matrix M_cal^-1 with realistic cross-axis perturbation."""
        eps = 1e-7
        jac_b_r = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            dr = np.zeros(3)
            dr[axis] = eps
            jac_b_r[:, axis] = (self._dipole_field(self.r0 + dr) - self.b0) / eps

        # B = J_br @ C @ F  =>  M_cal_true = J_br @ C
        m_cal_true = jac_b_r @ self.compliance
        # Inject slight calibration matrix mismatch to preserve physical cross-coupling
        perturbation = np.eye(3) + self.rng.normal(0.0, self.cfg.cross_axis_error, size=(3, 3))
        np.fill_diagonal(perturbation, 1.0)
        return np.linalg.inv(m_cal_true @ perturbation)

    def scatter_contacts(
        self,
        contact_positions: np.ndarray,
        contact_wrenches: np.ndarray,
        contact_link_ids: np.ndarray,
        taxel_positions_world: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map sparse contact 6D wrenches onto taxel array with strict wrench conservation.

        Args:
            contact_positions: [N_con, 3] active contact positions in world frame.
            contact_wrenches: [N_con, 6] contact wrenches [fx, fy, fz, tx, ty, tz].
            contact_link_ids: [N_con] link ID hit by each contact (-1 for non-sensor bodies).
            taxel_positions_world: Optional [N_taxels, 3] updated world coordinates.

        Returns:
            taxel_wrenches: [N_taxels, 6] conserved distributed 6D forces and torques.
        """
        pos_w = (
            self.taxel_positions
            if taxel_positions_world is None
            else np.asarray(taxel_positions_world, dtype=np.float64)
        )
        out_wrenches = np.zeros((self.n_taxels, 6), dtype=np.float64)
        if len(contact_positions) == 0:
            return out_wrenches

        cutoff = 3.0 * self.cfg.sigma_elast
        two_sig_sq = 2.0 * (self.cfg.sigma_elast ** 2)

        for c_pos, c_wrench, c_link in zip(contact_positions, contact_wrenches, contact_link_ids):
            if c_link < 0:
                continue
            mask = self.link_ids == c_link
            if not np.any(mask):
                continue

            local_pos = pos_w[mask]
            dists = np.linalg.norm(local_pos - c_pos[None, :], axis=1)
            in_range = dists <= cutoff
            if not np.any(in_range):
                # Fallback to nearest taxel on the contacted link to strictly conserve force
                nearest = np.argmin(dists)
                weights = np.zeros_like(dists)
                weights[nearest] = 1.0
            else:
                raw_w = np.where(in_range, np.exp(-(dists ** 2) / two_sig_sq), 0.0)
                weights = raw_w / np.sum(raw_w)

            out_wrenches[mask] += weights[:, None] * c_wrench[None, :]

        return out_wrenches

    def step(
        self,
        true_taxel_wrenches: np.ndarray,
        dt: float = 0.001,
        tangential_slip_speed: np.ndarray | None = None,
        motor_stray_field: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate magnetoelectric transduction, stick-slip micro-vibrations, and calibration.

        Args:
            true_taxel_wrenches: [N_taxels, 6] physical forces [fx, fy, fz] (fz >= 0 normal) and torques.
            dt: Physics timestep (default 0.001 s for 1,000 Hz).
            tangential_slip_speed: Optional [N_taxels] relative sliding speed (m/s).
            motor_stray_field: Optional [N_taxels, 3] residual uncompensated motor stray B-field.

        Returns:
            measured_wrenches: [N_taxels, 6] noisy, nonlinearly coupled 6D Hall sensor output.
        """
        f_true = true_taxel_wrenches[:, :3].copy()
        tau_true = true_taxel_wrenches[:, 3:].copy()

        # 1. Inject 50-300 Hz stick-slip micro-vibrations when approaching static friction cone
        f_tan = np.linalg.norm(f_true[:, :2], axis=1)
        f_norm = np.maximum(f_true[:, 2], 1e-4)
        fric_ratio = f_tan / (self.cfg.mu_static * f_norm)

        critical_mask = (f_true[:, 2] > 0.02) & (fric_ratio >= 0.82)
        if np.any(critical_mask):
            speed = (
                np.ones(self.n_taxels) * 0.015
                if tangential_slip_speed is None
                else np.asarray(tangential_slip_speed, dtype=np.float64)
            )
            freq_mod = self.cfg.slip_vib_freq_hz * (0.85 + 0.3 * np.clip(fric_ratio, 0.8, 1.2))
            self._vib_phase += 2.0 * np.pi * freq_mod * dt
            # Envelope scales smoothly with friction margin and micro-slip speed
            severity = np.clip((fric_ratio - 0.82) / 0.18, 0.0, 1.5)
            amp = self.cfg.slip_vib_gain * f_norm * severity * np.tanh(speed / 0.01 + 0.35)
            vib_x = amp * np.sin(self._vib_phase)
            vib_y = amp * np.cos(1.13 * self._vib_phase + 0.4)
            f_true[critical_mask, 0] += vib_x[critical_mask]
            f_true[critical_mask, 1] += vib_y[critical_mask]
            tau_true[critical_mask, 2] += 0.003 * vib_x[critical_mask]

        # 2. Nonlinear magnetic dipole displacement & flux computation
        delta_r = f_true @ self.compliance.T
        b_field = self._dipole_field(self.r0[None, :] + delta_r) - self.b0[None, :]
        if motor_stray_field is not None:
            b_field = b_field + motor_stray_field

        # 3. Inverse calibration matrix + sensor thermal/electronic noise
        f_meas = b_field @ self.m_cal_inv.T
        f_meas += self.rng.normal(0.0, self.cfg.noise_force_std, size=f_meas.shape)
        tau_meas = tau_true + self.rng.normal(0.0, self.cfg.noise_torque_std, size=tau_true.shape)

        return np.hstack([f_meas, tau_meas])
