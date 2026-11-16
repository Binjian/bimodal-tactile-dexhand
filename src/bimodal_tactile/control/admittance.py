"""1,000 Hz Decoupled Discrete-Time Admittance Controller & Bumpless Slow-Loop Handoff."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdmittanceConfig:
    """Parameters for normal/tangential decoupled discrete admittance control."""

    dt: float = 0.001                  # 1,000 Hz control period
    m_t: float = 0.025                 # Virtual tangential mass (kg)
    m_n: float = 0.025                 # Virtual normal mass (kg)
    k_t0: float = 180.0                # Nominal tangential stiffness (N/m)
    k_n0: float = 450.0                # Nominal normal stiffness (N/m)
    zeta_t: float = 1.0                # Tangential damping ratio
    zeta_n: float = 1.15               # Normal overdamped ratio (>= 1.0 prevents bounce)
    beta_k_stiffen: float = 1.2        # Normal stiffness boost under slip risk
    safety_factor: float = 1.30        # Friction cone recovery safety margin
    mu_static_est: float = 0.65        # Estimated static friction coefficient
    kp_slip: float = 1.8               # Proportional anti-slip force gain (N)
    ki_slip: float = 45.0              # Integral anti-slip force gain (N/s)
    i_max: float = 0.12                # Anti-windup integrator ceiling


class DiscreteAdmittanceController:
    """Fast-loop 1,000 Hz anti-slip force modulator & backward-Euler admittance law."""

    def __init__(self, cfg: AdmittanceConfig | None = None) -> None:
        self.cfg = cfg or AdmittanceConfig()
        self.integral_slip = 0.0
        self.dx_prev1 = np.zeros(3, dtype=np.float64)  # Delta x[k-1] in contact frame {C}
        self.dx_prev2 = np.zeros(3, dtype=np.float64)  # Delta x[k-2] in contact frame {C}
        self.last_fn_desired = 0.0
        self.last_fn_ref = 0.0

    def bumpless_slow_loop_handoff(self, new_fn_ref: float) -> None:
        """Complementary integrator reset when a new 60 Hz slow-loop reference arrives.

        Prevents step drops in total desired normal force when responsibility for
        sustaining the increased grip force transfers from fast integrator to slow policy.
        """
        delta_ref = new_fn_ref - self.last_fn_ref
        if self.cfg.ki_slip > 1e-6:
            self.integral_slip = max(0.0, self.integral_slip - delta_ref / self.cfg.ki_slip)
        self.last_fn_ref = new_fn_ref

    def step(
        self,
        f_ref_contact: np.ndarray,
        f_meas_contact: np.ndarray,
        slip_info: dict[str, float],
        fn_max_limit: float = 15.0,
    ) -> dict[str, np.ndarray | float]:
        """Compute one 1,000 Hz anti-slip normal force & compliant displacement correction.

        Args:
            f_ref_contact: [ft1_ref, ft2_ref, fn_ref] nominal force from slow loop (N).
            f_meas_contact: [ft1, ft2, fn] measured contact force in frame {C} (N).
            slip_info: Output dictionary from `IncipientSlipDetector.step()`.
            fn_max_limit: Fragility-bounded maximum normal grip force (N).

        Returns:
            Dictionary with `f_desired_contact`, `delta_x_contact`, `fn_req`, `k_n`, `d_n`.
        """
        sigma = slip_info["sigma_slip"]
        equiv_shear = slip_info["equiv_shear"]
        fn_ref = float(f_ref_contact[2])
        self.last_fn_ref = fn_ref

        # 1. Minimum normal force required to pull load back inside safety friction cone
        fn_req = (self.cfg.safety_factor / max(self.cfg.mu_static_est, 0.1)) * equiv_shear

        # 2. Anti-windup PI + friction-deficit compensation
        if sigma > 0.0:
            self.integral_slip = float(
                np.clip(self.integral_slip + sigma * self.cfg.dt, 0.0, self.cfg.i_max)
            )
        else:
            # Gentle decay when completely stable
            self.integral_slip = max(0.0, self.integral_slip - 0.25 * self.cfg.dt)

        deficit = max(0.0, fn_req - fn_ref) if sigma > 0.05 else 0.0
        delta_fn_slip = (
            deficit
            + self.cfg.kp_slip * sigma
            + self.cfg.ki_slip * self.integral_slip
        )
        fn_desired = float(np.clip(fn_ref + delta_fn_slip, 0.0, fn_max_limit))
        self.last_fn_desired = fn_desired

        f_desired = np.array([f_ref_contact[0], f_ref_contact[1], fn_desired], dtype=np.float64)

        # 3. Normal/tangential decoupled variable impedance modulation
        k_t = self.cfg.k_t0 * (1.0 - 0.5 * sigma)
        k_n = self.cfg.k_n0 * (1.0 + self.cfg.beta_k_stiffen * sigma)
        d_t = 2.0 * self.cfg.zeta_t * np.sqrt(self.cfg.m_t * k_t)
        d_n = 2.0 * self.cfg.zeta_n * np.sqrt(self.cfg.m_n * k_n)

        m_vec = np.array([self.cfg.m_t, self.cfg.m_t, self.cfg.m_n])
        d_vec = np.array([d_t, d_t, d_n])
        k_vec = np.array([k_t, k_t, k_n])

        # 4. Backward-Euler scalar recurrence per axis (< 5 us on MCU)
        dt = self.cfg.dt
        dt2 = dt * dt
        e_f = f_desired - np.asarray(f_meas_contact, dtype=np.float64)

        denom = (m_vec / dt2) + (d_vec / dt) + k_vec
        numer = (
            e_f
            + ((2.0 * m_vec / dt2) + (d_vec / dt)) * self.dx_prev1
            - (m_vec / dt2) * self.dx_prev2
        )
        dx_curr = numer / denom

        self.dx_prev2 = self.dx_prev1.copy()
        self.dx_prev1 = dx_curr.copy()

        return {
            "f_desired_contact": f_desired,
            "delta_x_contact": dx_curr,
            "fn_req": float(fn_req),
            "k_n": float(k_n),
            "d_n": float(d_n),
        }
