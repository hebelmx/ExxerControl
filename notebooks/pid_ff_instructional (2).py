# PID + Feedforward Control Instructional Notebook

...(previous sections unchanged)...

## Section 14: Manim Animation – Step Response Flow (Wood & Berry Model)

## Section 15: DOE on Physical Processes

### Overview
In this section, we shift focus to physical plant processes (Heating, Pressure, Speed) and explore their control behavior using Design of Experiments (DOE). These systems are more typical in real-world automation environments and help build intuition for controller tuning and response analysis.

---

### 🔧 Heating Process (SISO)
**Inputs**: Heater power, ambient temperature  
**Output**: Tank temperature  
**Model**:
```python
G1 = tf([0.8], [120, 1])  # τ=120s, Gain=0.8
```
DOE suggestion: 2² (Power low/high, Ambient low/high)

---

### 💨 Pressure Tank (SISO)
**Inputs**: Inlet valve %, outlet valve %  
**Output**: Tank pressure  
**Model**:
```python
G2_1 = tf([1.0], [60, 1])   # Inlet to pressure
G2_2 = tf([-0.8], [80, 1])  # Outlet to pressure (negative gain)
```
DOE: 2² factorial to analyze inlet/outlet effect

---

### ⚙️ Motor Speed System (SISO)
**Inputs**: Voltage (u), Load torque (disturbance)  
**Output**: RPM (y)  
**Model**:
```python
G3 = tf([5], [5, 1, 0])  # 2nd order, inertia dominant
```
DOE: Test voltage level + load torque change

---

### 🔁 MIMO Process – Heating System with Two Heaters and Two Sensors
**Inputs**: Heater1, Heater2  
**Outputs**: Temp1, Temp2  
**Model**:
```python
Gm = [[tf([0.9], [100, 1]), tf([0.3], [120, 1])],
      [tf([0.4], [150, 1]), tf([1.1], [90, 1])]]
```
- Interactions are present (off-diagonal terms)
- Good test case for DOE and MPC comparison

---

### 🧪 DOE Walkthrough: Heating Tank (2² Factorial)

**Objective**: Study how heater power and ambient temperature affect final tank temperature.

**Factors:**
- Heater Power: Low (50%), High (100%)
- Ambient Temp: Low (15°C), High (25°C)

| Run | Heater Power | Ambient Temp | Final Temp (hypothetical) | Time to 90% |
|-----|---------------|----------------|----------------------------|---------------|
| 1   | Low           | Low            | T₁                        | t₁            |
| 2   | High          | Low            | T₂                        | t₂            |
| 3   | Low           | High           | T₃                        | t₃            |
| 4   | High          | High           | T₄                        | t₄            |

→ Compute effects, interactions, and significance graphically or statistically (contrast, ANOVA, etc).

---

### 📈 Model Identification: Regression for Dynamic Models

For DOE-based experiments, if system parameters are unknown, we can identify dynamic models via regression on input/output data.

Example (Heating Tank):
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# First-order model response function
def first_order(t, K, tau):
    return K * (1 - np.exp(-t / tau))

# Hypothetical data
u = 1.0  # unit step input
t = np.linspace(0, 300, 100)
y_meas = first_order(t, 0.8, 120) + np.random.normal(0, 0.01, t.shape)

# Fit model
params, _ = curve_fit(first_order, t, y_meas)
K_est, tau_est = params

# Plot
plt.plot(t, y_meas, label='Measured')
plt.plot(t, first_order(t, K_est, tau_est), label=f'Fit: K={K_est:.2f}, τ={tau_est:.1f}')
plt.legend(); plt.title("Heating Tank – Model Fit"); plt.xlabel("Time"); plt.ylabel("Temp")
plt.grid(True); plt.show()
```
Use this approach to fit empirical data and derive usable transfer functions.

---

### Next Steps
- Run DOE walkthrough for Heating Tank
- Simulate DOE conditions for remaining systems
- Compare open-loop vs closed-loop response
- For MIMO system: demonstrate MPC coordination advantages

---
