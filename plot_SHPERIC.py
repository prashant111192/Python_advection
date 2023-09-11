import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Read data
data = np.loadtxt('save.csv', delimiter=',', skiprows=1)
# Read header
with open('save.csv', 'r') as f:
    header = f.readline().split(',')
# Get column names
col_names = [h.strip() for h in header]

# Plot for Non Normalized data
plt.clf()
fig, ax1 = plt.subplots()
# Concentration plots
ax1.plot(data[:,0], data[:,3], 'r', markevery= 1,label="Initial", fillstyle='none')
ax1.plot(data[:,0], data[:,5], 'k*', markevery = 5,label="Central_diff", fillstyle='none')
ax1.plot(data[:,0], data[:,8], 'g^', markevery = 6,label="Upwind", fillstyle='none')
ax1.plot(data[:,0], data[:,5], 'b:', markevery = 7,label="SPH")
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel("x position (m)")
ax1.set_ylabel("Concentration")
ax1.legend(loc="lower right")
ax1.grid()
# Velocity plot
# ax2 = ax1.twinx()
# ax2.plot(data[:,0], data[:,1], 'k.', markevery=80, label="Velocity", fillstyle='none')
# ax2.set_ylabel("Velocity (m/s)")
# ax2.legend(loc="upper right")
# ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax2.yaxis.offsetText.set_fontsize(10)
plt.title("Concentration along the x-axis")
plt.tight_layout()
plt.savefig("adv2d_lineNN_"+".png", dpi=300)

# Plot for Normalized data
plt.clf()
fig, ax1 = plt.subplots()
# Concentration plots
ax1.plot(data[:,0], data[:,3], 'r', markevery=10, label="Initial", fillstyle='none')
ax1.plot(data[:,0], data[:,6], 'k*', markevery = 20,label="Central_diff", fillstyle='none')
ax1.plot(data[:,0], data[:,9], 'g^', markevery = 15,label="Upwind", fillstyle='none')
ax1.plot(data[:,0], data[:,7], 'b-.', markevery = 1,label="SPH", fillstyle='none')
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel("x position (m)")
ax1.set_ylabel("Concentration")
ax1.legend(loc="upper right")
ax1.grid()
# Velocity plot
# ax2 = ax1.twinx()
# ax2.plot(data[:,0], data[:,1], 'k.', markevery=80, label="Velocity", fillstyle='none')
# ax2.set_ylabel("Velocity (m/s)")
# ax2.legend(loc="upper right")
# ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax2.yaxis.offsetText.set_fontsize(10)
plt.title("Concentration along the x-axis")
plt.tight_layout()
plt.savefig("adv2d_lineN_"+".png", dpi=300)
