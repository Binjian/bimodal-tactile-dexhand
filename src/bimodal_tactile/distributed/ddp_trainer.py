"""8-GPU Distributed Training Utilities: Reverse-Order Bucketing & Contact-Aware Grad Scaling."""

from __future__ import annotations

try:
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    HAS_TORCH = True
except Exception:
    torch = None
    dist = None
    default_hooks = None
    HAS_TORCH = False


def setup_multimodal_distributed_training(policy_model, local_rank: int):
    """Wrap bimodal policy in DDP with 8 MB priority buckets & BF16/FP32 communication hook."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for distributed DDP training setup.")

    torch.cuda.set_device(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        policy_model,
        device_ids=[local_rank],
        bucket_cap_mb=8,
        gradient_as_bucket_view=True,
    )
    ddp_model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
    return ddp_model


def step_with_contact_aware_grad_scaling(
    ddp_model,
    optimizer,
    loss,
    local_contact_count: int,
    batch_size_local: int,
    tactile_params,
    vision_params,
) -> float:
    """Execute backward pass with 8-byte global contact ratio compensation & decoupled clipping."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for distributed training step.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    contact_tensor = torch.tensor([local_contact_count], dtype=torch.float32, device=loss.device)
    dist.all_reduce(contact_tensor, op=dist.ReduceOp.SUM)
    total_batch = batch_size_local * dist.get_world_size()
    contact_ratio = float(torch.clamp(contact_tensor.item() / total_batch, min=0.1, max=1.0))

    for p in tactile_params:
        if p.grad is not None:
            p.grad.mul_(1.0 / contact_ratio)

    torch.nn.utils.clip_grad_norm_(tactile_params, max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(vision_params, max_norm=0.5)
    optimizer.step()
    return contact_ratio
