# 3. MuJoCo / NVIDIA Warp Multi-Rate Sim2Real Pipeline

## 3.1 Single-Source-of-Truth Physics Coupling

Both simulated sensor modalities are driven from the exact same MuJoCo contact state (`dt = 1 ms`, `condim="6"` elliptic friction cone) to guarantee physical consistency:

### A. 6D Hall Forward Model (1,000 Hz)
1. **Force-Conserving Elastic Kernel Scatter**: For each active contact point $c$ with 6D wrench $\mathbf{w}_c$, weights over neighboring taxels $i \in \mathcal{N}(c)$ on the same body link are normalized:
   $$w_{ic} = \frac{\exp\left(-\|\mathbf{p}_i - \mathbf{p}_c\|^2 / 2\sigma_{\text{elast}}^2\right)}{\sum_{j \in \mathcal{N}(c)} \exp\left(-\|\mathbf{p}_j - \mathbf{p}_c\|^2 / 2\sigma_{\text{elast}}^2\right)}$$
2. **Magnetic Dipole Nonlinearity & Inverse Calibration**: Elastic compliance $\Delta\mathbf{r}_i = \mathbf{C}\mathbf{f}_i$ shifts the micro-magnet relative to the Hall sensor, generating dipole flux $\mathbf{B}(\mathbf{r}_0 + \Delta\mathbf{r}_i)$, which is corrupted by stray motor fields and inverted via $\mathbf{M}_{\text{cal}}^{-1}$ to reproduce realistic cross-axis coupling.
3. **Critical Friction Stick-Slip Injection**: When $0.85\mu_s \le \|\mathbf{f}_{t,i}\| / f_{n,i} \le \mu_s$, damped sinusoidal micro-vibrations ($50\text{--}300\text{ Hz}$) are injected into the tangential channels.

### B. VBT Optical & Deformation Model (60 Hz)
1. **Orthographic Depth Clipping**: Masking out the silicone gel geom (`group="3"`), raw penetration is $H_{\text{raw}}(u,v) = \max(0, z_{\text{far}} - Z(u,v))$.
2. **2D-FFT Screened Poisson Membrane Draping**: Continuous elastomer tension is solved in frequency space:
   $$H(u,v) = \mathcal{F}^{-1}\left( \frac{\mathcal{F}(H_{\text{raw}})}{1 + \lambda(\omega_u^2 + \omega_v^2)} \right)$$
3. **Taxim Photometric LUT & Boussinesq Marker Flow**: Surface normals $\mathbf{n}(u,v)$ query a calibrated polynomial/MLP lookup table for RGB shading, while tangential forces and torsional moments drive Boussinesq-Cerruti Green's function displacement fields for internal markers.
