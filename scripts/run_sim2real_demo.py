"""End-to-End Multi-Rate Bimodal Tactile Sim2Real Demo (1,000 Hz Hall + 60 Hz VBT)."""

from pathlib import Path
import numpy as np
from bimodal_tactile.sensors import Hall6DArraySimulator, VBTOpticalSimulator
from bimodal_tactile.sim import MultiRateTactileBus, SyncConfig, HAS_WARP, batched_hall_forward_reference


def main() -> None:
    print("=== Bimodal Tactile Sim2Real Multi-Rate Pipeline Demo ===")
    print(f"NVIDIA Warp GPU Backend Available: {HAS_WARP}")

    # 1. Build a 4x4 fingerpad 6D Hall array on link_id=1
    gx, gy = np.meshgrid(np.linspace(-0.008, 0.008, 4), np.linspace(-0.008, 0.008, 4))
    taxel_pos = np.stack([gx.ravel(), gy.ravel(), np.zeros(16)], axis=-1)
    link_ids = np.ones(16, dtype=np.int32)

    hall_sim = Hall6DArraySimulator(taxel_pos, link_ids)
    vbt_sim = VBTOpticalSimulator()
    bus = MultiRateTactileBus(SyncConfig(fast_hz=1000, slow_hz=60, ring_window_steps=32, n_taxels=16))

    # 2. Construct synthetic depth buffer with a diagonal ridge indentation
    yy, xx = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="ij")
    ridge_indent = 0.0014 * np.exp(-((xx - 0.35 * yy) ** 2) / 0.04) * np.exp(-(yy ** 2) / 0.45)
    depth_buffer_m = vbt_sim.cfg.z_far_m - ridge_indent

    # 3. Step 100 ms (100 fast ticks at 1,000 Hz, triggering 6 VBT frames at ~60 Hz)
    vbt_frames_captured = 0
    for step in range(100):
        # Ramp tangential shear force toward static friction limit to trigger stick-slip
        fn = 2.0
        ft_x = 0.4 + 0.0085 * step  # Reaches 1.25 N (> 0.82 * 0.65 * 2.0 = 1.066 N)
        contact_pos = np.array([[0.001, -0.001, 0.0]])
        contact_wrench = np.array([[ft_x, 0.05, fn, 0.0, 0.0, 0.004]])
        distributed = hall_sim.scatter_contacts(contact_pos, contact_wrench, np.array([1]))
        measured = hall_sim.step(distributed, dt=0.001)
        bus.push_hall_sample(measured)

        if bus.should_trigger_vbt():
            vbt_out = vbt_sim.render(
                depth_buffer_m,
                tangential_force_xy=(ft_x, 0.05),
                torsional_torque_z=0.004,
            )
            rgb_dual = np.stack([vbt_out["rgb"], np.zeros_like(vbt_out["rgb"])], axis=0)
            flow_dual = np.stack([vbt_out["flow_px"], np.zeros_like(vbt_out["flow_px"])], axis=0)
            bus.push_vbt_frame(rgb_dual, flow_dual)
            vbt_frames_captured += 1

    # 4. Verify batched kernel reference force conservation
    batch_out = batched_hall_forward_reference(
        contact_dist=np.array([[-0.001]]),
        contact_pos=np.array([[[0.001, -0.001, 0.0]]]),
        contact_wrench=np.array([[[1.1, 0.2, 2.5, 0.0, 0.0, 0.003]]]),
        contact_link_id=np.array([[1]]),
        taxel_pos_world=taxel_pos[None, :, :],
        taxel_link_id=link_ids,
    )
    total_conserved_fn = float(np.sum(batch_out[0, :, 2]))

    window = bus.get_chronological_hall_window()
    print(f"Total 1,000 Hz Hall steps pushed     : {bus.step_count}")
    print(f"Synchronized 60 Hz VBT frames latched: {vbt_frames_captured}")
    print(f"Chronological Hall window shape      : {window.shape} ({window.dtype})")
    print(f"Conserved total normal force check   : {total_conserved_fn:.6f} N (expected 2.500000 N)")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "sample_vbt_rgb.npy", bus.latest_vbt_rgb[0])
    print(f"Saved sample VBT RGB frame to        : {out_dir / 'sample_vbt_rgb.npy'}")


if __name__ == "__main__":
    main()
