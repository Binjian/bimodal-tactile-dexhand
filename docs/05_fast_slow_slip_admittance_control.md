# 5. 1,000 Hz Incipient Slip Detection & Discrete-Time Admittance Control

## 5.1 Zero-Window-Delay Micro-Vibration Extraction

In local contact frame $\{C\} = [\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2, \hat{\mathbf{n}}]^\top$, tangential forces $\mathbf{f}_t[k]$ and torsional moment $\tau_n[k]$ pass through a 2nd-order Butterworth IIR Biquad bandpass filter ($50\text{--}300\text{ Hz}$, group delay $< 1.2\text{ ms}$) to isolate stick-slip oscillations $\tilde{\mathbf{f}}_t[k], \tilde{\tau}_n[k]$ alongside a 20 Hz LPF for quasi-static loads $\bar{f}_n[k], \bar{\mathbf{f}}_t[k], \bar{\tau}_n[k]$.

Instantaneous generalized vibration power and EWMA band energy ($\tau_E \approx 4\text{ ms}$):
$$p_{\text{vib}}[k] = \|\tilde{\mathbf{f}}_t[k]\|_2^2 + \left(\frac{\tilde{\tau}_n[k]}{r_{\text{eff}}}\right)^2, \qquad E_{\text{vib}}[k] = (1 - \alpha) E_{\text{vib}}[k-1] + \alpha p_{\text{vib}}[k]$$

## 5.2 Ellipsoidal Friction Margin & Common-Mode Inertial Rejection

1. **Howe-Cutkosky Ellipsoidal Friction Margin**:
   $$\eta_{\text{fric}}[k] = \frac{1}{\hat{\mu}_s \max(\bar{f}_n[k], f_{\min})} \sqrt{ \|\bar{\mathbf{f}}_t[k]\|_2^2 + \left(\frac{\bar{\tau}_n[k]}{c_r r_{\text{eff}}}\right)^2 }, \quad c_r = \frac{3\pi}{16} \approx 0.589$$
2. **Inertial Common-Mode Rejection**: Subtract average vibration $E_{\text{bg}}[k]$ from unloaded fingers $j \in \mathcal{U}$:
   $$I_{\text{slip}}[k] = \frac{\max(0, E_{\text{vib}}[k] - \gamma_{\text{cm}} E_{\text{bg}}[k] - E_{\text{noise}})}{\bar{f}_n[k] + f_0}$$
3. **Continuous Slip Risk Index** $\sigma_{\text{slip}}[k] \in [0, 1]$ drives normal force reconstruction $f_{n,\text{req}}[k]$, anti-windup PI increment $\Delta f_{n,\text{slip}}[k]$, normal stiffness stiffening ($\zeta_n \ge 1.0$), tangential compliance softening, and backward-Euler discrete admittance updates with **bumpless integrator handoff** across the 60 Hz slow-loop boundary.
