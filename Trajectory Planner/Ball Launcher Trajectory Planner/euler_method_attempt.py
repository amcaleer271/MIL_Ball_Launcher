import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
S = 2
rho = 1.18
A = 0.00255
R = 0.0285
Cd = 0.4
m = 40 / 1000
G = 0.001
theta = 45 * math.pi / 180  # measured in rads
theta_dot = 0

# Initial conditions
pos = np.array([0, 0, 0])
vel = np.array([10, 10, 10])  # give some initial velocity
acc = np.array([0, 0, 0])

# Simulation parameters
step_size = 0.01
num_iter = 1500  # Increased iterations for longer trajectory

# Arrays to store trajectory
positions = []

# Simulation loop
for t in range(num_iter):
    v = np.linalg.norm(vel)

    acc = (1 / m) * np.array([
        S * rho * R * v * theta_dot * math.sin(theta) - 0.5 * Cd * rho * A * (v ** 2) * math.cos(theta),
        -S * rho * R * v * theta_dot * math.cos(theta) - 0.5 * Cd * rho * A * (v ** 2) * math.sin(theta),
        0])

    vel = vel + step_size * acc
    pos = pos + step_size * vel

    # Store position
    positions.append(pos.copy())

    if pos[0] <= 0:
        break

# Convert positions to numpy array for easy indexing
positions = np.array(positions)

# Plotting the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates
x_vals = positions[:, 0]
y_vals = positions[:, 1]
z_vals = positions[:, 2]

# Create 3D plot
ax.plot(x_vals, y_vals, z_vals, label="Trajectory")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Trajectory of the Object")
ax.legend()

# Show the plot
plt.show()
