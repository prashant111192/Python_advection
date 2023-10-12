
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import pointsToVTK

loc = './data/'
path_to_plot = './plots_SPH/'
methods = ['Linear', 'Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

v_mag_interpolated_avg = np.load(loc +'v_mag_interpolated_avg.npy')
vel_avg = np.load(loc+'vel_avg.npy')
# print(data_last.flags)
# print(mag.flags)
error = vel_avg- v_mag_interpolated_avg
error_rel = error/vel_avg
mean_error = np.mean(error, axis=1)
mean_error = np.reshape(mean_error, (len(mean_error), 1))
std_div = np.std(error, axis=1)
std_div = np.reshape(std_div, (len(std_div), 1))
plot_error = (error - mean_error)
# plot_error = error_rel
plot_error = vel_avg
# plot_error = plot_error/std_div
print(methods)
print(std_div.T)
print(mean_error.T)


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
x_min = min(min(stuff) for stuff in plot_error)
x_max = max(max(stuff) for stuff in plot_error)
# Plot each histogram in a separate subplot
for j, ax in enumerate(axes.flatten()):
    if j < len(plot_error):
        ax.hist(plot_error[j], bins=100)
        ax.set_title(f'{methods[j]}')
        ax.set_xlabel('Error')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_ylim(0, 250)
        # ax.set_xlim(-5, 5)

# Adjust layout and save the plot
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust the title position
plt.savefig(path_to_plot+'hist_subplots1.png')
plot_error = v_mag_interpolated_avg
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
x_min = min(min(stuff) for stuff in plot_error)
x_max = max(max(stuff) for stuff in plot_error)
# Plot each histogram in a separate subplot
for j, ax in enumerate(axes.flatten()):
    if j < len(plot_error):
        ax.hist(plot_error[j], bins=100)
        ax.set_title(f'{methods[j]}')
        ax.set_xlabel('Error')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_ylim(0, 250)
        # ax.set_xlim(-5, 5)

# Adjust layout and save the plot
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust the title position
plt.savefig(path_to_plot+'hist_subplots.png')

# error = np.abs(error)/vel_avg
# print(np.shape(data_last))
# print(data_last.flags)