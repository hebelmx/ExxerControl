# PID + Feedforward Control Instructional Notebook

## Section 1: Introduction

"""
This notebook is an instructional guide based on real-world experience tuning PID + Feedforward (FF) controllers
for industrial processes. The methodology balances theoretical models with empirical adjustments.
"""

## Section 2: Process Description

# Define key variables
process_description = {
    "controlled_variable": "Temperature (TY201)",
    "manipulated_variable": "Control Valve Opening (VFD206)",
    "disturbance_variable": "Feed flow rate (TY203)",
    "setpoint_variable": "Setpoint from operator interface"
}

# Display process setup
for k, v in process_description.items():4
    print(f"{k}: {v}")

## Section 3: Control Strategy Overview

"""
Control Strategy:
- Use PID control to manage the primary loop (TY201).
- Add Feedforward compensation from TY203 to anticipate disturbances.
- Model TY201 response as a function of TY203 using regression or experimental data.
"""

## Section 4: First-Order Process Model and PID Step Response

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step
from IPython.display import Image, display

# Define first-order process G(s) = K / (tau*s + 1)
K = 1.0       # Gain
tau = 5.0     # Time constant
num = [K]
den = [tau, 1]
G = TransferFunction(num, den)

# Time vector and step response
time = np.linspace(0, 40, 500)
t, y = step(G, T=time)

plt.plot(t, y)
plt.title("First-Order Process Step Response")
plt.xlabel("Time (s)")
plt.ylabel("Process Output")
plt.grid(True)
plt.show()

## Section 5: Heat Exchanger Model with Nonlinear Regulator

"""
Model: Copper inductor submerged in water (electric heat exchanger) with nonlinear actuator (e.g., TRIAC).
Assumptions:
- Volume = 0.4 m³, Flow rate = 50 L/s (50 kg/s), Residence time ≈ 8s
- Regulator output is inversely proportional to firing angle (α)
- Power input ∝ (180 - α)/180
- Heat balance: Q_in = m * Cp * dT/dt - Q_loss
- Estimated heat transfer area ≈ 0.2 m² (based on coil geometry)
- Include losses and efficiency: Q_loss = U*A*(T - T_ambient), efficiency < 1
"""

# Display image of heat exchanger
display(Image(filename="heaterinductor.png"))

# Parameters
Cp = 4.18       # specific heat (kJ/kg.K)
m = 50.0        # mass flow rate (kg/s)
Pmax = 600.0    # max heater power (kW)
T_ambient = 25  # °C
U = 100         # estimated heat transfer coefficient (W/m²·K)
A = 0.2         # heat exchange area (m²)
efficiency = 0.95

# Time vector
t = np.linspace(0, 600, 600)
alpha = np.linspace(180, 0, len(t))  # simulate ramp-down of firing angle
power_input = Pmax * (180 - alpha) / 180  # non-linear power mapping

# Simulate water temperature in exchanger
T = np.zeros_like(t)
for i in range(1, len(t)):
    Q_loss = U * A * (T[i-1] - T_ambient) / 1000  # convert W to kW
    effective_power = power_input[i] * efficiency
    dTdt = (effective_power - Q_loss) / (m * Cp)
    T[i] = T[i-1] + dTdt

plt.plot(t, T)
plt.title("Heat Exchanger Temperature with Nonlinear Firing Control")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C")
plt.grid(True)
plt.show()

## Section 6: Power Rectifier Unit (Industrial Source)

"""
This section displays the industrial power rectifier unit that drives the inductor.
It provides regulated high-power AC/DC output controlled by firing angle (e.g., SCRs or IGBTs).
"""

display(Image(filename="rectifier.png"))

## Section 7: Feedforward Modeling Placeholder

"""
Placeholder for regression-based or data-derived model of feedforward gain.
Use experimental data to fit TY201 = a*TY203 + b
Then include in control algorithm.
"""

## Section 8: Next Steps

"""
- Add symbolic model using sympy.
- Integrate real data (step test or historical logs).
- Convert control law to PLC-implementable function block.
"""
