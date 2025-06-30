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
plt.ylabel("Temperature (°C)")  # Fixed missing closing parenthesis
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

## Section 8: MIMO System Example (3x3 Transfer Function Matrix)

"""
Using a 3x3 MIMO system inspired by Skogestad's textbook, we simulate the open-loop response.
Each element G_ij(s) represents a process path from input j to output i.
We'll use scipy.signal for transfer function modeling since it's more widely available.
"""

# Define individual transfer functions using scipy.signal
# G_ij represents transfer from input j to output i
from scipy.signal import lti

# Define the 3x3 transfer function matrix elements
# G11, G12, G13 (first row - output 1)
G11 = lti([12.8], [16.7, 1])      # Strong effect
G12 = lti([-18.9], [21.0, 1])     # Inverse response
G13 = lti([6.6], [10.9, 1])       # Moderate effect

# G21, G22, G23 (second row - output 2)  
G21 = lti([-19.4], [14.4, 1])     # Inverse response
G22 = lti([46.2], [14.4, 1])      # Strong positive effect
G23 = lti([-0.5], [7.9, 1])       # Weak inverse effect

# G31, G32, G33 (third row - output 3)
G31 = lti([0.1], [20.0, 1])       # Very weak coupling
G32 = lti([0.1], [20.0, 1])       # Very weak coupling  
G33 = lti([41.0], [8.15, 1])      # Strong diagonal effect

# Store transfer functions in matrix form
G_matrix = [[G11, G12, G13], 
            [G21, G22, G23], 
            [G31, G32, G33]]

# Simulate step response for each input
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
t_sim = np.linspace(0, 100, 500)
input_labels = ['Input 1 (u₁)', 'Input 2 (u₂)', 'Input 3 (u₃)']
output_labels = ['Output 1 (y₁)', 'Output 2 (y₂)', 'Output 3 (y₃)']
colors = ['blue', 'red', 'green']

for input_idx in range(3):
    for output_idx in range(3):
        # Get step response for this transfer function
        _, y_response = step(G_matrix[output_idx][input_idx], T=t_sim)
        
        # Plot on the appropriate subplot
        axes[output_idx].plot(t_sim, y_response, 
                            color=colors[input_idx], 
                            linewidth=2, 
                            label=input_labels[input_idx])
        
        axes[output_idx].set_ylabel(output_labels[output_idx])
        axes[output_idx].grid(True, alpha=0.7)
        axes[output_idx].legend()

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Open-Loop Step Response of 3×3 MIMO System", fontsize=14)
plt.tight_layout()
plt.show()

# Display interaction analysis
print("\nMIMO SYSTEM INTERACTION ANALYSIS")
print("=" * 50)
print("System characteristics:")
print("• Output 1: Shows strong response to inputs 1&2, moderate to input 3")
print("• Output 2: Responds strongly to input 2, with inverse responses to inputs 1&3") 
print("• Output 3: Primarily controlled by input 3, minimal coupling from inputs 1&2")
print("\nControl implications:")
print("• Diagonal dominance in G33 suggests output 3 is easiest to control")
print("• Strong interactions between outputs 1&2 require careful tuning")
print("• Inverse responses (negative gains) may cause control challenges")

## Section 9: Power Rectifier Unit (Industrial Source)

"""
This section displays the industrial power rectifier unit that drives the inductor.
It provides regulated high-power AC/DC output controlled by firing angle (e.g., SCRs or IGBTs).
"""

# Display rectifier image (with error handling)
try:
    display(Image(filename="rectifier.png"))
except FileNotFoundError:
    print("Rectifier image (rectifier.png) not found in current directory")

## Section 10: Enhanced Feedforward Modeling

"""
Enhanced feedforward modeling with MIMO considerations.
For MIMO systems, feedforward compensation becomes more complex due to interactions.
"""

# Enhanced feedforward calculation for MIMO systems
def calculate_mimo_feedforward_gains(disturbance_data, output_data):
    """
    Calculate feedforward gains for MIMO system
    Returns matrix of feedforward gains
    """
    # Placeholder for advanced MIMO feedforward calculation
    # In practice, this would use system identification techniques
    ff_gains = np.array([[0.75, 0.20, 0.10],   # FF gains for output 1
                        [0.30, 0.85, 0.15],    # FF gains for output 2  
                        [0.05, 0.10, 0.90]])   # FF gains for output 3
    return ff_gains

# Display feedforward gain matrix
ff_matrix = calculate_mimo_feedforward_gains(None, None)
print("MIMO FEEDFORWARD GAIN MATRIX")
print("=" * 40)
print("Feedforward gains (rows=outputs, cols=disturbances):")
print(f"FF₁₁={ff_matrix[0,0]:.2f}  FF₁₂={ff_matrix[0,1]:.2f}  FF₁₃={ff_matrix[0,2]:.2f}")
print(f"FF₂₁={ff_matrix[1,0]:.2f}  FF₂₂={ff_matrix[1,1]:.2f}  FF₂₃={ff_matrix[1,2]:.2f}")
print(f"FF₃₁={ff_matrix[2,0]:.2f}  FF₃₂={ff_matrix[2,1]:.2f}  FF₃₃={ff_matrix[2,2]:.2f}")

# Visualize feedforward gain matrix
plt.figure(figsize=(8, 6))
plt.imshow(ff_matrix, cmap='RdBu_r', aspect='auto')
plt.colorbar(label='Feedforward Gain')
plt.title('MIMO Feedforward Gain Matrix')
plt.xlabel('Disturbance Input')
plt.ylabel('Controlled Output')
plt.xticks([0, 1, 2], ['Dist 1', 'Dist 2', 'Dist 3'])
plt.yticks([0, 1, 2], ['Output 1', 'Output 2', 'Output 3'])

# Add text annotations
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{ff_matrix[i,j]:.2f}', 
                ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

## Section 11: Advanced MIMO Control Strategies

"""
Advanced next steps for MIMO PID + Feedforward control implementation:

IMMEDIATE IMPLEMENTATIONS:
- Independent PID controllers for each output (diagonal control)
- Relative Gain Array (RGA) analysis for interaction assessment
- Feedforward decoupling for disturbance rejection
- Performance monitoring and loop interaction metrics

ADVANCED CONTROL STRUCTURES:
- Full MIMO controller with cross-coupling compensation
- Model Predictive Control (MPC) for constrained optimization
- Decoupling networks to reduce loop interactions
- Adaptive feedforward tuning based on operating conditions

INDUSTRIAL CONSIDERATIONS:
- Control valve sizing and rangeability analysis
- Sensor placement optimization for minimal interaction
- Safety interlocks and constraint handling
- Real-time performance monitoring and diagnostics

VALIDATION AND TUNING:
- Closed-loop system identification
- Robustness analysis under model uncertainty
- Economic optimization of control performance
- Operator training and control strategy documentation
"""

# Example RGA calculation for interaction analysis
def calculate_rga_matrix(G_matrix, frequency=0):
    """
    Calculate Relative Gain Array (RGA) for interaction analysis
    RGA = G(jω) ⊗ [G(jω)]^(-T)
    For steady-state analysis, use ω = 0
    """
    # Extract steady-state gains from transfer functions
    K_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Get DC gain (numerator/denominator at s=0)
            num_coeffs = G_matrix[i][j].num
            den_coeffs = G_matrix[i][j].den
            K_matrix[i, j] = num_coeffs[-1] / den_coeffs[-1]
    
    # Calculate RGA = K ⊗ (K^-1)^T
    try:
        K_inv_T = np.linalg.inv(K_matrix).T
        RGA = K_matrix * K_inv_T  # Element-wise multiplication
        return RGA, K_matrix
    except np.linalg.LinAlgError:
        print("Warning: Singular gain matrix, RGA calculation failed")
        return None, K_matrix

# Calculate and display RGA
RGA, K_gains = calculate_rga_matrix(G_matrix)

if RGA is not None:
    print("\nRELATIVE GAIN ARRAY (RGA) ANALYSIS")
    print("=" * 50)
    print("Steady-state gain matrix K:")
    print(K_gains)
    print(f"\nRGA matrix (λᵢⱼ):")
    print(RGA)
    print(f"\nRGA interpretation:")
    print(f"• λ₁₁ = {RGA[0,0]:.2f}: Input 1 → Output 1 pairing")
    print(f"• λ₂₂ = {RGA[1,1]:.2f}: Input 2 → Output 2 pairing") 
    print(f"• λ₃₃ = {RGA[2,2]:.2f}: Input 3 → Output 3 pairing")
    print(f"\nRecommendations:")
    if all(RGA[i,i] > 0.5 and RGA[i,i] < 2.0 for i in range(3)):
        print("✓ Diagonal pairing is recommended (good for independent PID control)")
    else:
        print("⚠ Consider alternative pairing or advanced MIMO control structure")
else:
    print("RGA analysis could not be completed due to singular gain matrix")


# PID + Feedforward Control Instructional Notebook

#...(previous sections unchanged)...

## Section 11: Closed-Loop Simulation with Independent PID Controllers

#"""
#Implementing independent PID controllers on the 3x3 MIMO system (no decoupling).
#This simulates basic decentralized control, where each loop is tuned independently.
#"""

from control import feedback

# Define simple proportional controllers (PID tuning omitted for clarity)
Kp1, Kp2, Kp3 = 0.5, 0.5, 0.5
C11 = tf([Kp1], [1])
C22 = tf([Kp2], [1])
C33 = tf([Kp3], [1])

# Closed-loop systems (independent controllers)
T11 = feedback(C11 * G11, 1)
T22 = feedback(C22 * G22, 1)
T33 = feedback(C33 * G33, 1)

# Simulate step responses of closed-loop system
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
t_cl = np.linspace(0, 100, 500)

for i, T_sys in enumerate([T11, T22, T33]):
    _, y_cl = step_response(T_sys, T=t_cl)
    axes[i].plot(t_cl, y_cl, label=f'T{i+1}{i+1}')
    axes[i].set_ylabel(f'y{i+1}')
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Closed-Loop Step Responses (Independent Controllers)")
plt.tight_layout()
plt.show()

## Section 12: Manim Animation Script for 3x3 MIMO System

#"""
#This block outlines a Manim scene visualizing the 3x3 control system structure.
#It depicts the matrix of transfer functions and shows how input disturbances flow through the system.
#"""

from manim import *

class MimoSystemScene(Scene):
    def construct(self):
        title = Title("3x3 MIMO Control Structure", include_underline=True)
        self.play(Write(title))

        inputs = [MathTex(f"u_{i+1}") for i in range(3)]
        outputs = [MathTex(f"y_{i+1}") for i in range(3)]
        blocks = [[MathTex(f"G_{{{i+1}{j+1}}}(s)") for j in range(3)] for i in range(3)]

        # Positioning
        for i, inp in enumerate(inputs):
            inp.to_edge(LEFT).shift(DOWN * (i - 1))
        for j, out in enumerate(outputs):
            out.to_edge(RIGHT).shift(DOWN * (j - 1))
        for i in range(3):
            for j in range(3):
                blocks[i][j].move_to((j - 1) * RIGHT + (1 - i) * UP)

        # Display
        for obj in inputs + outputs:
            self.play(FadeIn(obj))
        for row in blocks:
            for blk in row:
                self.play(FadeIn(blk, shift=UP))

        # Arrows
        for i in range(3):
            for j in range(3):
                arrow = Arrow(inputs[j].get_right(), blocks[i][j].get_left(), buff=0.1)
                self.play(GrowArrow(arrow), run_time=0.2)
                arrow2 = Arrow(blocks[i][j].get_right(), outputs[i].get_left(), buff=0.1)
                self.play(GrowArrow(arrow2), run_time=0.2)

        self.wait(2)

## Section 13: Feedforward Modeling Placeholder