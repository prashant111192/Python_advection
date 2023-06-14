import numpy as np
import matplotlib.pyplot as plt

# Constants
num_points = 100  # Number of points in the wave
time_steps = 100  # Number of time steps for the simulation
c = 0.1  # Wave speed
dt = 0.1  # Time step size
dx = 1.0  # Spatial step size
mean = 50  # Mean of the normal distribution
std_dev = 10  # Standard deviation of the normal distribution

# Initialize arrays
x = np.arange(num_points) * dx
u = np.zeros(num_points)
u_next = np.zeros(num_points)

# Set initial conditions
u = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)  # Gaussian distribution
u /= np.max(u)  # Normalize the amplitude to 1

plt.plot(x, u, label='initial')
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('1D Wave Propagation')
# Perform simulation
for t in range(time_steps):
    for i in range(1, num_points - 1):
        u_next[i] = u[i] - c * (dt / dx) * (u[i] - u[i - 1])
    u = u_next.copy()

# Plot the final state of the wave
plt.plot(x, u, label='final')
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('1D Wave Propagation')
plt.show()
