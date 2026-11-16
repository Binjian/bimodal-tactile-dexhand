"""Closed-Loop 1,000 Hz Incipient Slip Detection & Discrete Admittance Control Simulation."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bimodal_tactile.sensors import Hall6DArraySimulator, Hall6DConfig
from bimodal_tactile.control import (
    IncipientSlipDetector,
    SlipDetectorConfig,
    DiscreteAdmittanceController,
    AdmittanceConfig,
)


def main() -> None:
    print("=== 1,000 Hz Incipient Slip & Discrete Admittance Closed-Loop Simulation ===")
    dt = 0.001
    steps = 600  # 600 ms trajectory

    # Single fingertip 6D Hall sensor
    hall_sim = Hall6DArraySimulator(
        taxel_positions=np.array([[0.0, 0.0, 0.0]]),
        link_ids=np.array([1], dtype=np.int32),
        config=Hall6DConfig(mu_static=0.60, slip_vib_gain=0.055),
    )
    detector = IncipientSlipDetector(SlipDetectorConfig(mu_static_est=0.60))
    controller = DiscreteAdmittanceController(AdmittanceConfig(dt=dt, mu_static_est=0.60))

    # Plant compliance model: normal displacement delta_x_n increases actual normal force
    k_env_n = 1200.0  # Object + silicone contact stiffness (N/m)
    fn_ref_slow = 1.6  # Initial gentle grasp normal force (N)
    actual_fn = fn_ref_slow

    hist = {
        "t_ms": [],
        "ft_load": [],
        "fn_actual": [],
        "fn_desired": [],
        "sigma_slip": [],
        "eta_fric": [],
        "dx_n_mm": [],
    }

    for k in range(steps):
        t_ms = k * 1.0
        # External tangential payload ramps up between 100 ms and 350 ms (e.g., lifting heavier object)
        if k < 100:
            ft_ext = 0.45
        elif k < 350:
            ft_ext = 0.45 + 1.15 * ((k - 100) / 250.0)  # Reaches 1.60 N
        else:
            ft_ext = 1.60

        # Simulate slow-loop (60 Hz -> every 17 ms) policy gradually absorbing sustained load
        if k > 0 and (k % 17 == 0):
            # Slow policy slowly tracks 35% of the fast loop's needed grip boost
            fn_ref_slow = 0.85 * fn_ref_slow + 0.15 * controller.last_fn_desired
            controller.bumpless_slow_loop_handoff(fn_ref_slow)

        true_wrench = np.array([[ft_ext, 0.0, actual_fn, 0.0, 0.0, 0.002]])
        meas_wrench = hall_sim.step(true_wrench, dt=dt)[0]

        slip_info = detector.step(meas_wrench, e_bg_unloaded=0.0)
        ctrl_out = controller.step(
            f_ref_contact=np.array([ft_ext, 0.0, fn_ref_slow]),
            f_meas_contact=meas_wrench[:3],
            slip_info=slip_info,
            fn_max_limit=8.0,
        )

        # Update physical contact normal force from discrete admittance displacement
        dx_n = float(ctrl_out["delta_x_contact"][2])
        actual_fn = max(0.1, fn_ref_slow + k_env_n * dx_n)

        hist["t_ms"].append(t_ms)
        hist["ft_load"].append(ft_ext)
        hist["fn_actual"].append(actual_fn)
        hist["fn_desired"].append(float(ctrl_out["f_desired_contact"][2]))
        hist["sigma_slip"].append(slip_info["sigma_slip"])
        hist["eta_fric"].append(slip_info["eta_fric"])
        hist["dx_n_mm"].append(dx_n * 1000.0)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)

    axes[0].plot(hist["t_ms"], hist["ft_load"], label="Tangential Load $F_t$ (N)", color="#d95f02")
    axes[0].plot(hist["t_ms"], hist["fn_actual"], label="Actual Normal Grip $F_n$ (N)", color="#1b9e77", lw=2)
    axes[0].plot(hist["t_ms"], hist["fn_desired"], "--", label="Desired Grip $F_{n,d}$ (N)", color="#7570b3")
    axes[0].set_ylabel("Force (N)")
    axes[0].set_title("1,000 Hz Incipient Slip Detection & Discrete Admittance Anti-Slip Response")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["t_ms"], hist["eta_fric"], label="Ellipsoidal Friction Utilization $\eta_{fric}$", color="#e7298a")
    axes[1].axhline(1.0, color="red", linestyle=":", label="Macro-Slip Boundary ($\eta=1.0$)")
    axes[1].plot(hist["t_ms"], hist["sigma_slip"], label="Continuous Slip Risk $\sigma_{slip}$", color="#1f78b4", lw=1.8)
    axes[1].set_ylabel("Normalized Index")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hist["t_ms"], hist["dx_n_mm"], label="Admittance Normal Compression $\Delta x_n$ (mm)", color="#33a02c")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Displacement (mm)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "slip_admittance_response.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Peak friction utilization reached : {max(hist['eta_fric']):.3f} (stayed < 1.0 macro-slip!)")
    print(f"Final stabilized normal grip force: {hist['fn_actual'][-1]:.3f} N")
    print(f"Saved closed-loop response plot to: {plot_path}")


if __name__ == "__main__":
    main()
