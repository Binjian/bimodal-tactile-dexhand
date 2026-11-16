# 4. GPU Kernel Optimization & 8-GPU Distributed Training (16,384 Parallel Envs)

## 4.1 Single-GPU Memory & Kernel Architecture

* **Contact-Centric Scatter with Link-Bucketing**: Taxels are sorted by `body_id` ($M_{\text{link}} \le 64$). Each Warp block `(B, nconmax)` processes one contact across 64 threads in `__shared__` memory, eliminating a 9.56 GB dense `[B, 1140, 512]` distance tensor down to **0 MB** intermediate allocation.
* **VBT Local-Frame Raycast & Contact-Gated Early Exit**: Only a single read-only mesh/BVH asset is stored in global VRAM; rays are transformed into object local space via $\mathbf{T}_{b,\text{obj}}^{-1}$. Non-contacting fingertips exit immediately in thread 0.
* **Zero-Copy DLPack**: Fixed `fp16` Hall ring buffers and `uint8`/`int8` VBT buffers are exposed directly to PyTorch via `wp.to_torch()`.

## 4.2 8-GPU Distributed Scaling ($B_{\text{total}} = 16,384, B_{\text{local}} = 2,048$)

1. **Zero Cross-GPU Raw Sensory Traffic**: Each GPU runs 2,048 environments locally; only model gradients traverse NVLink/NVSwitch.
2. **Continuous Frame Arena + Strided Indexing**: Instead of copying overlapping 32 ms windows at every 60 Hz RL step (27.36 GiB/GPU), physical substeps are written sequentially into a `[576, 2048, 1140, 6]` arena and sliced via integer stride offsets (`15.03 GiB/GPU`, **45% VRAM reduction**).
3. **ZeRO-2 (`SHARD_GRAD_OP`)**: Preserves full `bf16` weights on each rank for communication-free rollouts while sharding optimizer states and gradients $8\times$.
4. **Reverse-Order Two-Tier NCCL Bucketing & Contact-Aware Grad Scaling**: Small 4–8 MB buckets flush tactile/fusion gradients first over NVLink while ViT backbones finish backprop; an 8-byte `all_reduce` counts active contacts across all 8 GPUs to prevent zero-contact episodes from diluting tactile gradients.
