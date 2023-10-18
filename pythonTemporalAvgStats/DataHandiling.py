from pyevtk.hl import pointsToVTK
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import scipy.spatial as spatial
import sklearn.neighbors as NN
import math

#initialiation parameters
data_src = 'data'
path_to_data = './'+data_src+'/'
path_to_output = './output_'+data_src+'/'
path_to_plot = path_to_output+'/plots_SPH/'
if not os.path.exists(path_to_output):
    os.mkdir(path_to_output)
    os.mkdir(path_to_plot)
methods = ['Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

# methods = ['Linear', 'Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

def shepards_method_all_parts(x, y, z, s, u, v, w, NN_idx, h):
    shepards = np.zeros(len(u))
    shepards_gms = np.zeros(len(u))
    cs = np.zeros(len(u))
    gaussian = np.zeros(len(u))
    qs = np.zeros(len(u))
    p = 3
    for i in range(len(u)):
        shepards[i],shepards_gms[i], cs[i], gaussian[i], qs[i] = kernel_methods(x, y, z, s, u[i], v[i], w[i], p, NN_idx[i], h)
    return shepards, shepards_gms, cs, gaussian, qs

def kernel_methods(x, y, z, s, u, v, w, p, N_idx, hh):
    h = hh*2
    # x, y, z are the points we know
    # s is the scalar value at each point
    # u, v, w are the points we want to interpolate
    # p is the power
    # returns the interpolated scalar value at (u, v, w)
    d = np.sqrt((x[N_idx]-u)**2 + (y[N_idx]-v)**2 + (z[N_idx]-w)**2)
    
    w_basic_shepards = 1/d**p
    w_basic_shepards = w_basic_shepards/np.sum(w_basic_shepards)
    shepards = np.sum(w_basic_shepards*s[N_idx])/np.sum(w_basic_shepards)
    max_d = np.max(d)
    w_gms_shepards = ((max_d-d)/(max_d*d))**2
    w_gms_shepards = w_gms_shepards/np.sum(w_gms_shepards)
    shepards_gms = np.sum(w_gms_shepards*s[N_idx])
    
    q = d/h
    sigma_cs = 1/(math.pi*h**3)
    sigma_gaussian = 1/(math.pi**(3/2)*h**3)
    sigma_qs = 1/(120*math.pi*h**3)

    w_kernel_cs = np.zeros(len(N_idx))
    w_kernel_gaussian = np.zeros(len(N_idx))
    w_kernel_qs = np.zeros(len(N_idx))
    for i in range(len(N_idx)):

        if q[i] <= 1:
            w_kernel_cs[i] = sigma_cs*(1-(3/2)*(q[i]**2)*(1-(q[i]/2)))
            w_kernel_gaussian[i] = sigma_gaussian*(math.exp(-(q[i]**2)))
            w_kernel_qs[i] = sigma_qs*((3-q[i])**5 - 6*(2-q[i])**5 + 15*(1-q[i])**5)
        elif q[i] > 1 and q[i] <= 2:
            w_kernel_cs[i] = sigma_cs*(1/4)*(2-q[i])**3
            w_kernel_gaussian[i] = sigma_gaussian*(math.exp(-(q[i]**2)))
            w_kernel_qs[i] = sigma_qs*((3-q[i])**5 - 6*(2-q[i])**5)
        elif q[i] > 2 and q[i] <= 3:
            w_kernel_gaussian[i] = sigma_gaussian*(math.exp(-(q[i]**2)))
            w_kernel_cs[i] = 0
            w_kernel_qs[i] = sigma_qs*((3-q[i])**5)
        else:
            w_kernel_cs[i] = 0
            w_kernel_gaussian[i] = 0
            w_kernel_qs[i] = 0

    w_kernel_cs = w_kernel_cs/np.sum(w_kernel_cs)
    cs = np.sum(w_kernel_cs*s[N_idx])
    w_kernel_gaussian = w_kernel_gaussian/np.sum(w_kernel_gaussian)
    gaussian = np.sum(w_kernel_gaussian*s[N_idx])
    w_kernel_qs = w_kernel_qs/np.sum(w_kernel_qs)
    qs = np.sum(w_kernel_qs*s[N_idx])

    return shepards, shepards_gms, cs, gaussian, qs

def show_neighbours(NN_Idx, random_points, points, idx):
    print('show_nn')
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    ui = np.zeros((len(NN_Idx[idx]),3))
    print(np.shape(random_points))
    for i in range(len(NN_Idx[idx])):
        temp = NN_Idx[idx][i]
        ui[i] = points[temp]
    sctt = ax.scatter3D(random_points[idx,0], random_points[idx,1], random_points[idx,2], c= "r", alpha=1)
    sctt2 = ax.scatter3D(ui[:,0], ui[:,1], ui[:,2], alpha=.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Nearest Neighbours')
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    # ax.set_zlim(0,1)
    plt.savefig(path_to_plot+'NN_area.png')
    plt.clf()

def func(x, y, z):
    temp = np.abs(x+y+z) + np.sin(np.pi*x*y*z) + 1
    temp2 = np.abs(x+y+z) / (np.sin(np.pi*x*y*z) +1)
    # temp = np.abs(np.sin(np.pi*x)*np.cos(np.pi*y)*z)
    # temp = np.abs(np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*z)
    # temp = np.sin(x+y)+np.cos(z+10)
    # temp = (np.sin(2*np.pi*x*y*z))
    # temp = x**2+y**2+z**2
    # temp = temp/np.max(temp)
    return temp,temp2

def create_domain(num, box_max, seed):
    np.random.seed(seed)
    x = np.linspace(0, box_max, num)
    x = np.sort(x)
    y = np.linspace(0, box_max, num)
    y = np.sort(y)
    z = np.linspace(0, box_max, num)
    z = np.sort(z)
    X, Y, Z = np.meshgrid(x, y, z)
    return (X,Y,Z)

def readDSPHdata(filename):
    # Pos.x[m];Pos.y[m];Pos.z[m];Idp;Vel.x[m/s];Vel.y[m/s];Vel.z[m/s];Rhop[kg/m^3];Type;Mk;

    data = np.loadtxt(path_to_data+filename, delimiter=';', skiprows=4, usecols=(0,1,2,3,4,5,6,7,8,9))
    mask = data[:,8]==3
    data = data[mask]
    print(np.shape(data))
    
    pos = data[:,0:3]
    idp = data[:,3]
    vel = data[:,4:7]
    density = data[:,7]
    typeMk = data[:,8:10]

    return data


def list_file_names():
    # list all files in the directory
    files = os.listdir(path_to_data)
    files.sort()
    return files

def main():

    files = list_file_names()
    # print(files)

    data_last = readDSPHdata(files[len(files)-1])
    # print(data_last)
    density_last = data_last[:,7]
    mag = np.sqrt(data_last[:,4]**2+data_last[:,5]**2+data_last[:,6]**2)/len(files)
    vel_avg =   mag
    vel_avg =  np.tile(mag, (len(methods),1))
    v_x = data_last[:,4]
    v_x_avg = np.tile(v_x/len(files), (len(methods),1))
    v_y = data_last[:,5]
    v_y_avg = np.tile(v_y/len(files), (len(methods),1))
    v_z = data_last[:,6]
    v_z_avg = np.tile(v_z/len(files), (len(methods),1))

    # for i in range(1):
    for i in range(len(files)-1):
        file = files[i]
        data = readDSPHdata(file)
        tree = spatial.KDTree(data[:,0:3])
        # h =5.656854e-05
        h = 0.005196
        nbrs = NN.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data[:,0:3])
        NN_idx = nbrs.radius_neighbors(data_last[:,0:3], radius=2*h, return_distance=False)

        show_neighbours(NN_idx, data_last[:,0:3], data[:,0:3], 100)

        vel_mag = np.sqrt(data[:,4]**2+data[:,5]**2+data[:,6]**2)
        current_density = data[:,7]
        current_v_x = data[:,4]
        current_v_y = data[:,5]
        current_v_z = data[:,6]

        print('Nearest Interpolation')
        interpolated_fx_nearest = interp.NearestNDInterpolator((data[:,0], data[:,1], data[:,2]), vel_mag)
        nearest_vel_mag = interpolated_fx_nearest((data_last[:,0], data_last[:,1], data_last[:,2]))
        del(interpolated_fx_nearest)

        interpolated_fx_nearest = interp.NearestNDInterpolator((data[:,0], data[:,1], data[:,2]), v_x)
        nearest_v_x = interpolated_fx_nearest((data_last[:,0], data_last[:,1], data_last[:,2]))
        del(interpolated_fx_nearest)

        interpolated_fx_nearest = interp.NearestNDInterpolator((data[:,0], data[:,1], data[:,2]), v_y)
        nearest_v_y = interpolated_fx_nearest((data_last[:,0], data_last[:,1], data_last[:,2]))
        del(interpolated_fx_nearest)

        interpolated_fx_nearest = interp.NearestNDInterpolator((data[:,0], data[:,1], data[:,2]), v_z)
        nearest_v_z = interpolated_fx_nearest((data_last[:,0], data_last[:,1], data_last[:,2]))
        del(interpolated_fx_nearest)

        print('Shepards Interpolation')
        shepards_vel_mag, shepards_vel_mag_gms, cs_vel_mag, gaussian_vel_mag, qs_vel_mag = shepards_method_all_parts(data[:,0], data[:,1], data[:,2], vel_mag, data_last[:,0], data_last[:,1], data_last[:,2], NN_idx, h)
        vel_avg = vel_avg + np.vstack((nearest_vel_mag, shepards_vel_mag, shepards_vel_mag_gms, cs_vel_mag, gaussian_vel_mag, qs_vel_mag))/len(files)

        shepards_v_x, shepards_v_x_gms, cs_v_x, gaussian_v_x, qs_v_x = shepards_method_all_parts(data[:,0], data[:,1], data[:,2], current_v_x, data_last[:,0], data_last[:,1], data_last[:,2], NN_idx, h)
        v_x_avg = v_x_avg+ np.vstack((nearest_v_x, shepards_v_x, shepards_v_x_gms, cs_v_x, gaussian_v_x, qs_v_x))/len(files)

        shepards_v_y, shepards_v_y_gms, cs_v_y, gaussian_v_y, qs_v_y = shepards_method_all_parts(data[:,0], data[:,1], data[:,2], current_v_y, data_last[:,0], data_last[:,1], data_last[:,2], NN_idx, h)
        v_y_avg = v_y_avg+ np.vstack((nearest_v_y, shepards_v_y, shepards_v_y_gms, cs_v_y, gaussian_v_y, qs_v_y))/len(files)

        shepards_v_z, shepards_v_z_gms, cs_v_z, gaussian_v_z, qs_v_z = shepards_method_all_parts(data[:,0], data[:,1], data[:,2], current_v_z, data_last[:,0], data_last[:,1], data_last[:,2], NN_idx, h)
        v_z_avg = v_z_avg+ np.vstack((nearest_v_z, shepards_v_z, shepards_v_z_gms, cs_v_z, gaussian_v_z, qs_v_z))/len(files)
        
    
    print("Processing error and other stupid things...")

    v_mag_interpolated_avg = np.sqrt(v_x_avg**2+v_y_avg**2+v_z_avg**2)
    # print(data_last.flags)
    # print(mag.flags)
    data_last = np.asfortranarray(data_last)
    mag = np.asfortranarray(mag)
    print("saving...")
    np.save(path_to_output+'vel_avg.npy', vel_avg)
    np.save(path_to_output+'v_mag_interpolated_avg.npy', v_mag_interpolated_avg)
    np.save(path_to_output+'positions.npy', data_last)

    error = vel_avg- v_mag_interpolated_avg
    error_rel = error/vel_avg
    mean_error = np.mean(error, axis=0)
    std_div = np.std(error, axis=0)
    plot_error = (error - mean_error)
    plot_error = error
    # plot_error = plot_error/std_div

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
    x_min = min(min(stuff) for stuff in plot_error)
    x_max = max(max(stuff) for stuff in plot_error)
    # Plot each histogram in a separate subplot
    for j, ax in enumerate(axes.flatten()):
        if j < len(plot_error):
            ax.hist(plot_error[j], bins=100)
            ax.set_title(f'{methods[j]}')
            # ax.set_ylim(0, 250)
            ax.set_xlim(x_min, x_max)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the title position

    pointsToVTK('./VTK/vel_avg', data_last[:,0], data_last[:,1], data_last[:,2], data = {'vel_avg': mag*len(files),
    # })
                                                                                         'nearest_vel_mag': nearest_vel_mag,
                                                                                         'shepards_vel_mag': shepards_vel_mag,
                                                                                         'shepards_vel_mag_gms': shepards_vel_mag_gms,
                                                                                         'cs_vel_mag': cs_vel_mag,
                                                                                         'gaussian_vel_mag': gaussian_vel_mag,
                                                                                         'qs_vel_mag': qs_vel_mag,
                                                                                         'nearest_vel_mag_arr': vel_avg[0],
                                                                                         'shepards_vel_mag_arr': vel_avg[1],
                                                                                         'shepards_vel_mag_gms_arr': vel_avg[2],
                                                                                         'cs_vel_mag_arr': vel_avg[3],
                                                                                         'gaussian_vel_mag_arr': vel_avg[4],
                                                                                         'qs_vel_mag_arr': vel_avg[5],
                                                                                         'v_mag_int_nearest': v_mag_interpolated_avg[0],
                                                                                         'v_mag_int_shepards': v_mag_interpolated_avg[1],
                                                                                         'v_mag_int_shepards_gms': v_mag_interpolated_avg[2],
                                                                                         'v_mag_int_cs': v_mag_interpolated_avg[3],
                                                                                         'v_mag_int_gaussian': v_mag_interpolated_avg[4],
                                                                                         'v_mag_int_qs': v_mag_interpolated_avg[5],
                                                                                         'error_nearest': error_rel[0],
                                                                                         'error_shepards': error_rel[1],
                                                                                         'error_shepards_gms': error_rel[2],
                                                                                         'error_cs': error_rel[3],
                                                                                         'error_gaussian': error_rel[4],
                                                                                         'error_qs': error_rel[5]})
    # exit()
    # plt.clf()
    # fig = plt.figure(figsize = (16, 9))
    # ax = plt.axes(projection ="3d")
    # sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=liner_vel_mag, alpha=0.5, s=5, cmap='viridis')
    # plt.colorbar(sctt, ax=ax)
    # plt.title('Linear Interpolation')
    # plt.savefig(path_to_plot+'linear.png')
    # plt.show()

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=nearest_vel_mag, alpha=0.5, s = 5,cmap='viridis')   
    # sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=nearest_vel_mag, alpha=0.5, s = 5,cmap='viridis', vmin=min(mag), vmax=max(mag))   
    plt.gca().set_box_aspect([1,1,1])
    plt.colorbar(sctt, ax=ax)
    plt.title('Nearest Interpolation')
    plt.savefig(path_to_plot+'nearest.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=shepards_vel_mag, alpha=0.5, s=5,cmap='viridis')
    # sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=shepards_vel_mag, alpha=0.5, s=5,cmap='viridis', vmin=min(mag), vmax=max(mag))
    plt.gca().set_box_aspect([1,1,1])
    # plt.gca().set_aspect('equal')
    plt.colorbar(sctt, ax=ax)
    plt.title('Shepards Interpolation')
    plt.savefig(path_to_plot+'shepards.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=cs_vel_mag, alpha=0.5, s=5,cmap='viridis')
    # sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=cs_vel_mag, alpha=0.5, s=5,cmap='viridis', vmin=min(mag), vmax=max(mag))
    plt.gca().set_box_aspect([1,1,1])
    plt.colorbar(sctt, ax=ax)
    plt.title('Cubic Spline Interpolation')
    plt.savefig(path_to_plot+'cs.png')


    # plt.clf()
    # fig = plt.figure(figsize = (16, 9))
    # ax = plt.axes(projection ="3d")
    # sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=cs_vel_mag, alpha=0.5, s=5,cmap='viridis')
    # plt.colorbar(sctt, ax=ax)
    # plt.title('Cubic Spline Interpolation')
    # plt.savefig(path_to_plot+'cs.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=mag, alpha=0.5, s=5,cmap='viridis', vmin=min(mag), vmax=max(mag))
    plt.gca().set_box_aspect([1,1,1])
    plt.colorbar(sctt, ax=ax)
    plt.title('Avgerage Velocity')
    plt.savefig(path_to_plot+'vel_avg.png')







    
    


if __name__ == '__main__':
    main()