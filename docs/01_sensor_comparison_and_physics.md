# 1. Sensor Physics & Route Comparison: 6D Hall Array vs. Vision-Based Tactile (VBT)

## 1.1 Overview of PaXini Tech (帕西尼感知科技) & 6D Hall Array Technology

Founded in 2021 in Shenzhen with roots in Waseda University's **Sugano Lab** (led by Dr. Jincheng Xu), PaXini Tech pioneered high-density **6D Hall-effect magnetoelectric tactile arrays** (GEN4 FUSE chip / ITPU tactile processing units) integrated across full-stack dexterous hands (**DexH13**, >1,140 ITPU units) and humanoid platforms (**TORA-ONE**).

### Physical Transduction Mechanism
* **6D Hall Array (Magnetoelectric Direct Measurement)**: A three-layer sandwich comprising a wear-resistant elastomer skin, a flexible magnetic layer embedded with micro-magnets/magnetic particles, and a 3D Hall-effect sensor chip array. External forces induce micro-scale normal/shear deformations, shifting the magnetic flux density vector $\mathbf{B} = (B_x, B_y, B_z)$. On-chip calibration matrices and neural decoupling models resolve **15 tactile dimensions**—including 3-axis forces $(F_x, F_y, F_z)$, 3-axis torques $(M_x, M_y, M_z)$, texture, rebound elasticity, and incipient slip—with micro-force resolution down to **0.005–0.01 N** at **1,000 Hz**.
* **GelSight / DIGIT (Photometric Stereo & Optical Inversion)**: A miniature dark-chamber camera observes an internal reflective or marker-embedded transparent silicone gel illuminated by multi-color oblique LEDs. Surface indentations alter surface normals $\mathbf{n}(u,v)$, enabling micron-scale 3D height-field reconstruction via **Photometric Stereo** and tangential displacement tracking via **marker optical flow**.

---

## 1.2 Engineering Trade-Off Matrix

| Dimension | 6D Hall Array (Magnetoelectric) | GelSight / DIGIT (Vision-Based Tactile) |
| :--- | :--- | :--- |
| **Transduction Chain** | Deformation $\rightarrow$ Magnetic flux vector $(B_x, B_y, B_z)$ $\rightarrow$ Direct 6D decoupling | Deformation $\rightarrow$ Optical reflection/marker shift $\rightarrow$ Inverse image reconstruction |
| **Spatial Resolution** | Millimeter-scale taxel pitch (area force distribution) | **10–50 $\mu$m pixel-equivalent** (resolves threads, pins, micro-scratches) |
| **Sampling Rate & Latency** | **1,000 Hz (< 1 ms)** — ideal for high-speed reactive force loops | **30–60 Hz (16–33 ms)** — bounded by CMOS exposure & USB/MIPI bandwidth |
| **Force Linearity & Drift** | Native decoupled 6D wrench; high linearity & low hysteresis | 3D geometry is native; forces suffer from **silicone viscoelastic creep** |
| **Form Factor & Curvature** | **2–3 mm thin flexible skin** — conforms to fingertips, phalanges, and palm | **10–25 mm rigid optical chamber** — constrained by lens focal distance |
| **Durability & Interference** | **IP68 waterproof/oil-proof**; sensitive to strong external ferromagnetic fields | **Immune to EMI/magnetic fields**; reflective coating wears under abrasion |
| **Compute & AI Alignment** | Ultra-low bandwidth; runs on edge MCU; requires tactile sequence encoder | High video bandwidth; natively compatible with **ViT / VLA** backbones |
