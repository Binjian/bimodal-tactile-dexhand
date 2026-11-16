"""NVIDIA Warp Fused GPU Kernels for Parallel Bimodal Tactile Simulation.

Provides:
1. Link-bucketed contact-to-taxel shared-memory normalized scatter kernel (`scatter_contact_forces_kernel`).
2. Register-fused magnetic dipole nonlinearity + inverse calibration kernel (`fused_hall_dipole_decouple_kernel`).
3. Vectorized NumPy/CPU reference implementation for verification on non-CUDA machines.
"""

from __future__ import annotations

import numpy as np

try:
    import warp as wp

    wp.init()
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False


if HAS_WARP:

    @wp.kernel
    def scatter_contact_forces_kernel(
        contact_dist: wp.array2d(dtype=float),
        contact_pos: wp.array2d(dtype=wp.vec3),
        contact_wrench: wp.array2d(dtype=wp.spatial_vector),
        contact_link_id: wp.array2d(dtype=int),
        link_taxel_offset: wp.array(dtype=int),
        link_taxel_count: wp.array(dtype=int),
        taxel_pos_world: wp.array2d(dtype=wp.vec3),
        sigma_elast: wp.array(dtype=float),
        out_taxel_wrench: wp.array2d(dtype=wp.spatial_vector),
    ):
        """Grid: (B, nconmax), Block: (64,) -> Zero intermediate global VRAM force scatter."""
        env_id, con_id = wp.tid()
        local_tid = wp.thread_idx()

        if contact_dist[env_id, con_id] > 0.0:
            return
        link_id = contact_link_id[env_id, con_id]
        if link_id < 0:
            return

        count = link_taxel_count[link_id]
        offset = link_taxel_offset[link_id]

        shared_weights = wp.zeros(shape=(64,), dtype=float, storage="shared")
        w = float(0.0)
        if local_tid < count:
            taxel_idx = offset + local_tid
            d = wp.length(taxel_pos_world[env_id, taxel_idx] - contact_pos[env_id, con_id])
            sig = sigma_elast[env_id]
            w = wp.exp(-(d * d) / (2.0 * sig * sig))

        shared_weights[local_tid] = w
        wp.syncthreads()

        if local_tid < count and w > 1e-5:
            w_sum = float(0.0)
            for i in range(count):
                w_sum += shared_weights[i]
            norm_w = w / (w_sum + 1e-8)
            wrench = contact_wrench[env_id, con_id] * norm_w
            wp.atomic_add(out_taxel_wrench, env_id, offset + local_tid, wrench)

    @wp.kernel
    def fused_hall_dipole_decouple_kernel(
        in_taxel_wrench: wp.array2d(dtype=wp.spatial_vector),
        compliance_mat: wp.array(dtype=wp.mat33),
        cal_inv_mat: wp.array(dtype=wp.mat33),
        motor_stray_field: wp.array2d(dtype=wp.vec3),
        ring_buffer: wp.array4d(dtype=wp.float16),
        ring_head_idx: int,
    ):
        """Register-fused dipole physics + inverse calibration + fp16 ring buffer write."""
        env_id, taxel_id = wp.tid()
        w_true = in_taxel_wrench[env_id, taxel_id]
        f_true = wp.spatial_top(w_true)
        tau_true = wp.spatial_bottom(w_true)

        delta_r = compliance_mat[env_id] * f_true
        r0 = wp.vec3(0.0, 0.0, -0.002)
        r = r0 + delta_r
        r_norm = wp.length(r)
        m_vec = wp.vec3(0.0, 0.0, 0.015)

        b_field = (3.0 * wp.dot(m_vec, r) * r / (r_norm ** 5.0)) - (m_vec / (r_norm ** 3.0))
        b_total = b_field + motor_stray_field[env_id, taxel_id]
        f_measured = cal_inv_mat[env_id] * b_total

        ring_buffer[ring_head_idx, env_id, taxel_id, 0] = wp.float16(f_measured[0])
        ring_buffer[ring_head_idx, env_id, taxel_id, 1] = wp.float16(f_measured[1])
        ring_buffer[ring_head_idx, env_id, taxel_id, 2] = wp.float16(f_measured[2])
        ring_buffer[ring_head_idx, env_id, taxel_id, 3] = wp.float16(tau_true[0])
        ring_buffer[ring_head_idx, env_id, taxel_id, 4] = wp.float16(tau_true[1])
        ring_buffer[ring_head_idx, env_id, taxel_id, 5] = wp.float16(tau_true[2])


def batched_hall_forward_reference(
    contact_dist: np.ndarray,
    contact_pos: np.ndarray,
    contact_wrench: np.ndarray,
    contact_link_id: np.ndarray,
    taxel_pos_world: np.ndarray,
    taxel_link_id: np.ndarray,
    sigma_elast: float = 0.0045,
) -> np.ndarray:
    """Batched NumPy reference matching `scatter_contact_forces_kernel` semantics.

    Args:
        contact_dist: [B, N_con] penetration distances (<= 0 means active contact).
        contact_pos: [B, N_con, 3] contact world positions.
        contact_wrench: [B, N_con, 6] contact 6D wrenches.
        contact_link_id: [B, N_con] link IDs (-1 for non-sensor bodies).
        taxel_pos_world: [B, N_taxels, 3] taxel world positions.
        taxel_link_id: [N_taxels] link ID per taxel.
        sigma_elast: Elastic spread radius (m).

    Returns:
        out_wrenches: [B, N_taxels, 6] conserved distributed wrenches.
    """
    B, N_con = contact_dist.shape
    N_taxels = taxel_pos_world.shape[1]
    out_wrenches = np.zeros((B, N_taxels, 6), dtype=np.float32)
    two_sig_sq = 2.0 * (sigma_elast ** 2)

    for b in range(B):
        for c in range(N_con):
            if contact_dist[b, c] > 0.0:
                continue
            lid = contact_link_id[b, c]
            if lid < 0:
                continue
            mask = taxel_link_id == lid
            if not np.any(mask):
                continue
            diff = taxel_pos_world[b, mask] - contact_pos[b, c][None, :]
            d_sq = np.sum(diff * diff, axis=-1)
            w = np.exp(-d_sq / two_sig_sq)
            w_sum = np.sum(w)
            if w_sum > 1e-8:
                out_wrenches[b, mask] += (w / w_sum)[:, None] * contact_wrench[b, c][None, :]

    return out_wrenches
