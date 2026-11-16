# 2. Heterogeneous Fusion Hardware & Multi-Modal Algorithm Architecture

## 2.1 Biological Foveated Spatial Topology

Rather than naively stacking a VBT silicone pad directly on top of a Hall magnetic film (which introduces mechanical bulk and viscoelastic distortion), we adopt a **foveated anatomical layout**:

| Hand Region | Sensor Configuration | Thickness | Primary Functional Role |
| :--- | :--- | :--- | :--- |
| **Thumb & Index Fingertips** | **Series Dual-Mode**: VBT optical chamber mounted atop a **6D Hall F/T base** | 12–15 mm (VBT) + 2.5 mm (Hall base) | VBT captures micron-level 3D contact geometry & pose angle; series 6D Hall base measures drift-free net 6D wrench. |
| **Middle, Ring, Pinky Tips** | High-density **6D Hall Array** fingertip modules | 3–4 mm curved | 1,000 Hz multi-axis force/torque feedback for assistive stabilization with low distal inertia. |
| **Phalanges & Palm** | Large-area flexible **6D Hall e-skin (ITPU)** | 2–3 mm flexible | Full-hand contact localization during power grasps, collision detection, and load sharing. |

---

## 2.2 Triple Physical Isolation & Zero-Jitter Hardware Sync

1. **Magnetic Shielding & Differential Reference**: High-permeability **Mu-metal (坡莫合金)** foil backs the Hall PCB to attenuate stray fields from joint BLDC motors, paired with reference (magnet-free) Hall taxels for real-time common-mode background subtraction $\mathbf{B}_{\text{net}} = \mathbf{B}_{\text{raw}} - \mathbf{B}_{\text{env}}$.
2. **Thermal Isolation**: High-brightness VBT RGB LEDs induce $5\text{--}15^\circ\text{C}$ local heating. A **PEEK / aerogel thermal barrier** separates the optical chamber from the Hall die, supplemented by on-chip 2nd-order polynomial temperature drift compensation.
3. **Intra-Finger SerDes Multiplexing**: Micro-serializers merge MIPI CSI-2 camera streams and SPI Hall packets over ultra-fine coaxial cables across high-flexion finger joints.
4. **Hardware Exposure-Midpoint Synchronization**: A palm FPGA distributes a master **1,000 Hz clock**. Every $N$-th pulse triggers the VBT camera such that the **camera exposure midpoint** aligns within $< 50\ \mu\text{s}$ of the $k$-th Hall sample rising edge.

---

## 2.3 Layered Multi-Modal Software Architecture

1. **Online Viscoelastic Creep Calibration (RLS)**: Recursive Least Squares uses the series 6D Hall base force $\mathbf{F}_{\text{Hall}}(t)$ as an unbiased physical anchor to continuously update the VBT elastomer modulus parameter, yielding *micron-scale 3D geometry + drift-free absolute force*.
2. **Dual-Modal Incipient Slip Detection**:
   * **6D Hall ($< 2\text{ ms}$)**: Biquad bandpass filter ($50\text{--}300\text{ Hz}$) captures stick-slip micro-vibrations.
   * **VBT ($16\text{--}33\text{ ms}$)**: Marker optical flow curl $\nabla \times \mathbf{v}_{\text{opt}}$ detects pure rotational slip where normal force barely changes.
3. **3D Canonical Hand Mesh Alignment & Asymmetric Dual-Stream Encoder**: Forward kinematics projects VBT local depth point clouds and all 6D Hall taxel vectors into a unified 3D hand frame. A lightweight ViT encodes spatial geometry $\mathbf{z}_{\text{geo}}$ (Query) to attend over a `1D-CNN + Temporal Transformer` encoding the 32 ms high-frequency Hall window $\mathbf{z}_{\text{dyn}}$ (Key/Value).
