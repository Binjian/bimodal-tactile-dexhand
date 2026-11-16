"""1,000 Hz Zero-Window-Delay Incipient Slip Detector for 6D Hall Tactile Sensors.

Implements:
1. 2nd-order Butterworth IIR Biquad bandpass filter (50-300 Hz, < 1.2 ms group delay)
   for stick-slip tangential force and torsional moment micro-vibrations.
2. 1st-order LPF (20 Hz) for quasi-static contact load extraction.
3. Generalized vibration power & EWMA band energy (tau_E = 4 ms).
4. Howe-Cutkosky ellipsoidal friction cone margin + unloaded-finger common-mode rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter


@dataclass
class SlipDetectorConfig:
    """Parameters for 1,000 Hz real-time incipient slip risk estimation."""

    fs_hz: float = 1000.0
    bpf_low_hz: float = 50.0
    bpf_high_hz: float = 300.0
    lpf_cutoff_hz: float = 20.0
    ewma_tau_s: float = 0.004          # 4 ms vibration energy smoothing time constant
    mu_static_est: float = 0.65        # Estimated static friction coefficient
    r_eff_m: float = 0.006             # Effective contact patch radius (6 mm)
    c_r_hertz: float = 0.589           # 3*pi/16 Hertzian torsional pressure distribution constant
    eta_thresh: float = 0.72           # Friction cone warning threshold
    i_crit: float = 0.0018             # Critical normalized micro-vibration intensity
    w_fric: float = 0.40               # Weight of static friction margin
    w_vib: float = 0.60                # Weight of dynamic micro-vibration intensity
    gamma_cm: float = 1.0              # Common-mode inertial vibration rejection gain
    noise_floor: float = 2e-5          # Sensor electronic noise floor (N^2)


class BiquadFilterVector:
    """Direct Form II Transposed 2nd-order IIR filter for multi-channel vector signals."""

    def __init__(self, b: np.ndarray, a: np.ndarray, n_channels: int) -> None:
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.z1 = np.zeros(n_channels, dtype=np.float64)
        self.z2 = np.zeros(n_channels, dtype=np.float64)

    def step(self, x: np.ndarray) -> np.ndarray:
        y = self.b[0] * x + self.z1
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2
        self.z2 = self.b[2] * x - self.a[2] * y
        return y


class IncipientSlipDetector:
    """Real-time 1,000 Hz slip risk detector operating in local contact coordinates."""

    def __init__(self, cfg: SlipDetectorConfig | None = None) -> None:
        self.cfg = cfg or SlipDetectorConfig()
        nyq = 0.5 * self.cfg.fs_hz
        b_bp, a_bp = butter(
            N=1,
            Wn=[self.cfg.bpf_low_hz / nyq, self.cfg.bpf_high_hz / nyq],
            btype="bandpass",
        )
        # Channels: [f_t1, f_t2, tau_n]
        self.bpf = BiquadFilterVector(b_bp, a_bp, n_channels=3)

        dt = 1.0 / self.cfg.fs_hz
        rc_lpf = 1.0 / (2.0 * np.pi * self.cfg.lpf_cutoff_hz)
        self.alpha_lpf = dt / (rc_lpf + dt)
        self.alpha_ewma = dt / (self.cfg.ewma_tau_s + dt)

        self.quasi_wrench = np.zeros(6, dtype=np.float64)  # [ft1, ft2, fn, tt1, tt2, tn]
        self.e_vib = 0.0

    def step(
        self,
        wrench_contact_6d: np.ndarray,
        e_bg_unloaded: float = 0.0,
    ) -> dict[str, float]:
        """Process one 1,000 Hz sample in contact frame {C}: [ft1, ft2, fn, tt1, tt2, tn].

        Args:
            wrench_contact_6d: 6D force/torque in local contact frame (fn > 0 compression).
            e_bg_unloaded: Average vibration energy across unloaded fingers for inertial rejection.

        Returns:
            Dictionary containing `sigma_slip` in [0, 1], `eta_fric`, `i_slip`, `e_vib`, and
            quasi-static loads `fn_bar`, `ft_norm_bar`, `tn_bar`.
        """
        w = np.asarray(wrench_contact_6d, dtype=np.float64)
        self.quasi_wrench = (1.0 - self.alpha_lpf) * self.quasi_wrench + self.alpha_lpf * w

        ft1_bar, ft2_bar, fn_bar, _, _, tn_bar = self.quasi_wrench
        ft_norm_bar = float(np.hypot(ft1_bar, ft2_bar))

        # 1. Zero-window-delay Biquad bandpass filtering for stick-slip micro-vibrations
        vib_vec = self.bpf.step(np.array([w[0], w[1], w[5]], dtype=np.float64))
        p_vib = float(
            vib_vec[0] ** 2
            + vib_vec[1] ** 2
            + (vib_vec[2] / max(self.cfg.r_eff_m, 1e-4)) ** 2
        )
        self.e_vib = (1.0 - self.alpha_ewma) * self.e_vib + self.alpha_ewma * p_vib

        # 2. Howe-Cutkosky ellipsoidal friction cone utilization
        equiv_shear = float(
            np.hypot(
                ft_norm_bar,
                tn_bar / (self.cfg.c_r_hertz * max(self.cfg.r_eff_m, 1e-4)),
            )
        )
        eta_fric = equiv_shear / (self.cfg.mu_static_est * max(fn_bar, 0.05))

        # 3. Inertial common-mode rejection & normal-load normalization
        net_vib = max(0.0, self.e_vib - self.cfg.gamma_cm * e_bg_unloaded - self.cfg.noise_floor)
        i_slip = net_vib / (fn_bar + 0.25)

        # 4. Continuous slip risk index sigma_slip in [0, 1]
        term_fric = max(0.0, (eta_fric - self.cfg.eta_thresh) / (1.0 - self.cfg.eta_thresh))
        term_vib = i_slip / self.cfg.i_crit
        sigma_slip = float(np.clip(self.cfg.w_fric * term_fric + self.cfg.w_vib * term_vib, 0.0, 1.0))

        return {
            "sigma_slip": sigma_slip,
            "eta_fric": float(eta_fric),
            "i_slip": float(i_slip),
            "e_vib": float(self.e_vib),
            "equiv_shear": equiv_shear,
            "fn_bar": float(fn_bar),
            "ft_norm_bar": ft_norm_bar,
            "tn_bar": float(tn_bar),
        }
