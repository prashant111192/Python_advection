
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import pointsToVTK

loc = './output_data/'
path_to_plot = loc+'plots_SPH/'
methods = ['Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']
# methods = ['Linear', 'Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

v_mag_interpolated_avg = np.load(loc +'v_mag_interpolated_avg.npy')
vel_avg = np.load(loc+'vel_avg.npy')
pos = np.load(loc+'positions.npy')
error = vel_avg- v_mag_interpolated_avg
# Adjust layout and save the plot
mean_vel = vel_avg.mean(axis=1)
error_temp = error/mean_vel[:, None]

pointsToVTK(path_to_plot+"paraview", pos[:, 0], pos[:, 1], pos[:, 2], data={
                                                                        "Nearest_error": error_temp[0],
                                                                        "Shepards_error": error_temp[1],
                                                                        "Shepards_GMS_error": error_temp[2],
                                                                        "Cubic_Spline_error": error_temp[3],
                                                                        "Gaussian_error": error_temp[4],
                                                                        "Quintic_Spline_error": error_temp[5],
                                                                        "Nearest_component" : v_mag_interpolated_avg[0],
                                                                        "Shepards_component" : v_mag_interpolated_avg[1],
                                                                        "Shepards_GMS_component" : v_mag_interpolated_avg[2],
                                                                        "Cubic_Spline_component" : v_mag_interpolated_avg[3],
                                                                        "Gaussian_component" : v_mag_interpolated_avg[4],
                                                                        "Quintic_Spline_component" : v_mag_interpolated_avg[5],
                                                                        "Nearest_mag" : vel_avg[0],
                                                                        "Shepards_mag" : vel_avg[1],
                                                                        "Shepards_GMS_mag" : vel_avg[2],
                                                                        "Cubic_Spline_mag" : vel_avg[3],
                                                                        "Gaussian_mag" : vel_avg[4],
                                                                        "Quintic_Spline_mag" : vel_avg[5]
                                                                        })

error_rel = error/vel_avg
mean_error = np.mean(error, axis=1)
mean_error = np.reshape(mean_error, (len(mean_error), 1))
std_div = np.std(error, axis=1)
std_div = np.reshape(std_div, (len(std_div), 1))
plot_error = (error - mean_error)
# plot_error = error_rel
# plot_error = error
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
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # adjust the title position
plt.savefig(path_to_plot+'hist_subplots_h25.png')

# error = np.abs(error)/vel_avg
# print(np.shape(data_last))
# print(data_last.flags)