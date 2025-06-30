# PID + Feedforward Control Instructional Notebook

## Section 1: Introduction

"""
This notebook is an instructional guide based on real-world experience tuning PID + Feedforward (FF) controllers
for industrial processes. The methodology balances theoretical models with empirical adjustments.
"""

## Section 2: Process Description

# Define key variables
process_description = {
    "controlled_variable": "Product Concentration (CV201)",
    "manipulated_variable": "Cooling Water Flow (MV203)",
    "disturbance_variable": "Inlet Reactant Temperature (DV205)",
    "secondary_variable": "Reactor Jacket Temperature (CV204)",
    "setpoint_variable": "Desired Product Concentration"
}

# Display process setup
for k, v in process_description.items():
    print(f"{k}: {v}")

## Section 3: Control Strategy Overview

"""
Control Strategy:
- Inner loop: Reactor Jacket Temperature (CV204) controlled by MV203 using a fast-acting PID.
- Outer loop: Product Concentration (CV201) controlled by adjusting the setpoint of the inner loop.
- Feedforward signal DV205 compensates for changes in inlet reactant temperature.
- The system has interacting variables since both CV201 and CV204 depend on MV203.
- Nonlinearity is introduced in CV201 due to saturation in reaction rate as temperature increases.
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

plt.figure(figsize=(10, 6))
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

# Display image of heat exchanger (with error handling)
try:
    display(Image(filename="heaterinductor.png"))
except FileNotFoundError:
    print("Heat exchanger image (heaterinductor.png) not found in current directory")

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

plt.figure(figsize=(10, 6))
plt.plot(t, T)
plt.title("Heat Exchanger Temperature with Nonlinear Firing Control")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()

## Section 6: Nonlinear Output Response (Saturation Model)

"""
We simulate the product concentration (CV201) as a nonlinear function of jacket temperature (CV204).
This mimics a saturating reaction rate: CV201 = a * T / (b + T)
"""

a = 10.0
b = 60.0
CV201 = a * T / (b + T)  # Saturation-type nonlinearity

plt.figure(figsize=(10, 6))
plt.plot(t, CV201)
plt.title("Nonlinear Product Concentration Response")
plt.xlabel("Time (s)")
plt.ylabel("CV201: Product Concentration")
plt.grid(True)
plt.show()

## Section 7: External Process Disturbance with Time Delay

"""
We simulate a separate non-interacting process startup that indirectly affects plant utilities or shared resources.
This includes a time delay and slow ramp response, such as activating a second heat exchanger or compressor.
"""

delay = 100  # seconds
duration = 100
total_time = len(t)
ext_process = np.zeros_like(t)

for i in range(delay, min(delay + duration, total_time)):
    ext_input = (i - delay) / duration
    ext_process[i] = ext_input ** 2  # example nonlinear ramp-up

plt.figure(figsize=(10, 6))
plt.plot(t, ext_process)
plt.title("External Machine Ramp-Up with Delay")
plt.xlabel("Time (s)")
plt.ylabel("Load (normalized)")
plt.grid(True)
plt.show()

## Section 8: Power Rectifier Unit (Industrial Source)

"""
This section displays the industrial power rectifier unit that drives the inductor.
It provides regulated high-power AC/DC output controlled by firing angle (e.g., SCRs or IGBTs).
"""

# Display rectifier image (with error handling)
try:
    display(Image(filename="rectifier.png"))
except FileNotFoundError:
    print("Rectifier image (rectifier.png) not found in current directory")

## Section 9: Feedforward Modeling Placeholder

"""
Placeholder for regression-based or data-derived model of feedforward gain.
Use experimental data to fit CV204 = a*DV205 + b
Then include in control algorithm.
"""

# Placeholder for feedforward gain calculation
def calculate_feedforward_gain(dv205_data, cv204_data):
    """
    Calculate feedforward gain using linear regression
    CV204 = a * DV205 + b
    Returns the slope 'a' as the feedforward gain
    """
    # Implementation would go here with real data
    # For now, return a placeholder value
    return 0.75

print("Feedforward gain calculation placeholder added.")
print("Example feedforward gain:", calculate_feedforward_gain(None, None))

## Section 10: Next Steps

"""
Next Steps:
- Add symbolic model using sympy for advanced mathematical analysis.
- Simulate disturbance injection and feedforward cancellation.
- Implement cascade and interaction analysis for multi-loop systems.
- Convert control law to PLC-implementable function block.
- Add PID tuning algorithms (Ziegler-Nichols, Cohen-Coon, etc.).
- Validate feedforward compensation effectiveness under various operating conditions.
- Create real-time simulation environment with operator interface.

Implementation Notes:
- Consider practical constraints like actuator saturation and measurement noise.
- Document tuning procedures for operators and maintenance personnel.
- Include safety interlocks and emergency shutdown procedures.
"""
