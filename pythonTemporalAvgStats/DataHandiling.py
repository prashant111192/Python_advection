import random
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import scipy.spatial as spatial
import sklearn.neighbors as NN
import math
# import pyinterp

#initialiation parameters
path_to_data = './data/'
path_to_plot = './plots_SPH/'
methods = ['Linear', 'Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

def shepards_method_all_parts(x, y, z, s, u, v, w, NN_idx, h):
    shepards = np.zeros(len(u))
    shepards_gms = np.zeros(len(u))
    cs = np.zeros(len(u))
    gaussian = np.zeros(len(u))
    qs = np.zeros(len(u))
    p = 3
    # for i in range(len(u)):
    #     shepards[i],shepards_gms[i], cs[i], gaussian[i], qs[i] = kernel_methods(x, y, z, s, u[i], v[i], w[i], p, NN_idx[i], h)
    return shepards, shepards_gms, cs, gaussian, qs

def kernel_methods(x, y, z, s, u, v, w, p, N_idx, h):
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
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
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
    print(np.shape(data))
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

    data_last = readDSPHdata(files[-1])
    mag = np.sqrt(data_last[:,4]**2+data_last[:,5]**2+data_last[:,6]**2)/len(files)
    vel_avg =   np.repeat(mag,len(methods), axis = 0)

    # for i in range(1):
    for i in range(len(files)-1):
        file = files[i]
        data = readDSPHdata(file)
        tree = spatial.KDTree(data[:,0:3])
        h = 0.005196
        nbrs = NN.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data[:,0:3])
        NN_idx = nbrs.radius_neighbors(data_last[:,0:3], radius=2*h, return_distance=False)

        show_neighbours(NN_idx, data_last[:,0:3], data[:,0:3], 100)
        
        vel_mag = np.sqrt(data[:,4]**2+data[:,5]**2+data[:,6]**2)
        # print('Linear Interpolation')
        # interpolated_fx_linear  = interp.LinearNDInterpolator((data[:,0], data[:,1], data[:,2]), vel_mag)
        # liner_vel_mag = interpolated_fx_linear((data_last[:,0], data_last[:,1], data_last[:,2]))
        # del(interpolated_fx_linear)

        print('Nearest Interpolation')
        interpolated_fx_nearest = interp.NearestNDInterpolator((data[:,0], data[:,1], data[:,2]), vel_mag)
        nearest_vel_mag = interpolated_fx_nearest((data_last[:,0], data_last[:,1], data_last[:,2]))
        del(interpolated_fx_nearest)

        print('Shepards Interpolation')
        shepards_vel_mag, shepards_vel_mag_gms, cs_vel_mag, gaussian_vel_mag, qs_vel_mag = shepards_method_all_parts(data[:,0], data[:,1], data[:,2], vel_mag, data_last[:,0], data_last[:,1], data_last[:,2], NN_idx, h)

        vel_avg = vel_avg + np.vstack((nearest_vel_mag, shepards_vel_mag, shepards_vel_mag_gms, cs_vel_mag, gaussian_vel_mag, qs_vel_mag))/len(files)
        

        

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
    plt.colorbar(sctt, ax=ax)
    plt.title('Nearest Interpolation')
    plt.savefig(path_to_plot+'nearest.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=shepards_vel_mag, alpha=0.5, s=5,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Shepards Interpolation')
    plt.savefig(path_to_plot+'shepards.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=cs_vel_mag, alpha=0.5, s=5,cmap='viridis')
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
    sctt = ax.scatter3D(data_last[:,0], data_last[:,1], data_last[:,2], c=vel_avg, alpha=0.5, s=5,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Avgerage Velocity')
    plt.savefig(path_to_plot+'vel_avg.png')







    
    

    # print_graphs = False
    # # number of points on 1 axis of the original data
    # min_size = 10
    # max_size = 100
    # domain_size = np.arange(min_size, max_size, 10)
    # seeds = [1294, 1248, 6076]
    # complete_error_stats = []
    # for seed in seeds:
    #     error_stats = []
    #     i = 0
    #     for num in domain_size:
    #         box_max = 1
    #         X,Y,Z = create_domain(num, box_max, seed)
    #         points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    #         s,s_rel = func(X, Y, Z)
    #         s = s.flatten()
    #         s_rel = s_rel.flatten()

    #         # The interpolated points
    #         interpolated_size = 1000
    #         u = np.random.rand(interpolated_size)  
    #         v = np.random.rand(interpolated_size)  
    #         w = np.random.rand(interpolated_size)
    #         random_points = np.array([u, v, w]).T
    
    #         kd_tree = spatial.KDTree(points)
    #         h = (box_max-0)/num
    #         print("h = ", h)
    #         nbrs = NN.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    #         NN_idx = nbrs.radius_neighbors(random_points, radius=2*h, return_distance=False)
    #         print(np.shape(NN_idx))
    #         print(NN_idx[100])
    #         show_neighbours(NN_idx, random_points, points, random.randint(0,interpolated_size))

    #         s_truth, s_rel_truth = func(u, v, w)

    #         fig = plt.figure(figsize = (16, 9))
    #         ax = plt.axes(projection ="3d")
    #         sctt = ax.scatter3D(u,v,w,alpha = 0.5, c = s_truth, s=50)
    #         fig.colorbar(sctt, ax=ax)
    #         plt.title('Random Points')
    #         plt.savefig(path_to_plot+'random_points.png')
    #         plt.clf()
    #         fig = plt.figure(figsize = (16, 9))
    #         ax = plt.axes(projection ="3d")
    #         sctt = ax.scatter3D(X, Y,Z,alpha = 0.5, c=s,  cmap='viridis')
    #         fig.colorbar(sctt, ax=ax)
    #         plt.title('Actual Data')
    #         plt.savefig(path_to_plot+'actual_data.png')
    #         plt.clf()
    #         fig = plt.figure(figsize = (16, 9))
    #         ax = plt.axes(projection ="3d")
    #         sctt = ax.scatter3D(X, Y,Z,alpha = 0.5, c=s_rel,  cmap='viridis')
    #         fig.colorbar(sctt, ax=ax)
    #         plt.title('Linear/Nonlinear Element')
    #         plt.savefig(path_to_plot+'actual_data_rel.png')

    #         print('Linear Interpolation')
    #         interpolated_fx_linear  = interp.LinearNDInterpolator((points[:,0], points[:,1], points[:,2]), s)
    #         interpolated_scalars_linear = interpolated_fx_linear((u, v, w))
    #         del(interpolated_fx_linear)

    #         print('Nearest Interpolation')
    #         interpolated_fx_nearest = interp.NearestNDInterpolator((points[:,0], points[:,1], points[:,2]), s)
    #         interpolated_scalars_nearest = interpolated_fx_nearest((u, v, w))
    #         del(interpolated_fx_nearest)
    
    #         print('Shepards Interpolation')
    #         interpolated_scalars_shepards,interpolated_scalars_shepards_gms, interpolated_scalars_cs, interpolated_scalars_gaussian, interpolated_scalars_qs = shepards_method_all_parts(points[:,0], points[:,1], points[:,2], s, u, v, w, NN_idx, h)

    #         temp_linear = (interpolated_scalars_linear-s_truth)/s_truth
    #         temp_nearest = (interpolated_scalars_nearest-s_truth)/s_truth
    #         temp_shepards = (interpolated_scalars_shepards-s_truth)/s_truth
    #         temp_shepardsgms = (interpolated_scalars_shepards_gms-s_truth)/s_truth
    #         temp_cs = (interpolated_scalars_cs-s_truth)/s_truth
    #         temp_gaussian = (interpolated_scalars_gaussian-s_truth)/s_truth
    #         temp_qs = (interpolated_scalars_qs-s_truth)/s_truth
    #         if(print_graphs):
    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_linear, alpha=0.5,s=50, cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Linear Interpolation')
    #             plt.savefig(path_to_plot+'linear.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_nearest, alpha=0.5, s = 50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Nearest Interpolation')
    #             plt.savefig(path_to_plot+'nearest.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_shepards, alpha=0.5, s=50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Shepards Interpolation')
    #             plt.savefig(path_to_plot+'shepards.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_shepards_gms, alpha=0.5, s=50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Shepards GMS Interpolation')
    #             plt.savefig(path_to_plot+'shepards_gms.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_cs, alpha=0.5, s=50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Cubic Spline Interpolation')
    #             plt.savefig(path_to_plot+'cubic_spline.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_gaussian, alpha=0.5, s=50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Gaussian Interpolation')
    #             plt.savefig(path_to_plot+'gaussian.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_qs, alpha=0.5, s=50,cmap='viridis')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.title('Quintic Spline Interpolation')
    #             plt.savefig(path_to_plot+'quintic.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_linear, s=50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Linear Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_linear.png')


    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_nearest, s = 50, alpha=0.5, cmap='viridis')
    #             plt.title('Difference Nearest Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_nearest.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_shepards,  s = 50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Shepards Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_shepards.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_shepardsgms, s = 50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Shepards GMS Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_shepards_gsm.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_cs, s = 50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Cubic Spline Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_cubic_spline.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_gaussian, s = 50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Gaussian Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_gaussian.png')

    #             plt.clf()
    #             fig = plt.figure(figsize = (16, 9))
    #             ax = plt.axes(projection ="3d")
    #             sctt = ax.scatter3D(u, v,w, c=temp_qs, s = 50,alpha=0.5, cmap='viridis')
    #             plt.title('Difference Quintic Spline Interpolation')
    #             plt.colorbar(sctt, ax=ax)
    #             plt.savefig(path_to_plot+'diff_quintic_spline.png')


    #         # Combine all datasets into a list
    #         all_temps = [temp_linear, temp_nearest, temp_shepards, temp_shepardsgms, temp_cs, temp_gaussian, temp_qs]
    #         methods = ['Linear', 'Nearest', 'Shepards', 'Shepards GMS', 'Cubic Spline', 'Gaussian', 'Quintic Spline']

    #         # Create subplots
    #         fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    #         fig.suptitle('Histograms: '+str(domain_size[i]))
    #         x_min = min(min(data) for data in all_temps)
    #         x_max = max(max(data) for data in all_temps)
    #         # Plot each histogram in a separate subplot
    #         for j, ax in enumerate(axes.flatten()):
    #             if j < len(all_temps):
    #                 ax.hist(all_temps[j], bins=100)
    #                 ax.set_title(f'{methods[j]}')
    #                 ax.set_ylim(0, 250)
    #                 ax.set_xlim(x_min, x_max)

    #         # Adjust layout and save the plot
    #         plt.tight_layout()
    #         plt.subplots_adjust(top=0.85)  # Adjust the title position
    #         plt.savefig(path_to_plot+'hist_subplots'+str(domain_size[i])+'.png')

    #         print('Name \t\t Max \t\t Mean \t\t Median \t\t StdDev')
    #         temp =  ((interpolated_scalars_linear-s_truth)/s_truth)
    #         # temp =  np.abs((interpolated_scalars_linear-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = temp
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = temp2
    #         print('Linear \t\t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))

    #         temp = ((interpolated_scalars_nearest-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('Nearest \t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))
    
    #         temp = ((interpolated_scalars_shepards-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('Shepards \t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))

    #         temp = ((interpolated_scalars_shepards_gms-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('ShepardsGMS \t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))

    #         temp = ((interpolated_scalars_cs-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('Cubic \t\t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))

    #         temp = ((interpolated_scalars_gaussian-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('Gaussian \t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))  

    #         temp = ((interpolated_scalars_qs-s_truth)/s_truth)
    #         temp = temp[~np.isnan(temp)]
    #         box = np.vstack((box, temp))
    #         temp2 = [np.max(temp), np.mean(temp),np.median(temp),np.std(temp)]
    #         error = np.vstack((error, temp2))
    #         print('Quintic \t {:.5f} \t {:.5f} \t {:.5f} \t\t {:.5f}'.format(np.max(temp), np.mean(temp),np.median(temp),np.std(temp)))

    #         i = i+1
    #         error_stats.append(error)
    #     error_stats = np.array(error_stats)
    #     complete_error_stats.append(error_stats) 
        
    # complete_error_stats = np.array(complete_error_stats)
    # error_stats = np.mean(complete_error_stats, axis=0)

    # plt.clf()
    # rounded_domain_size = (int)(round(len(domain_size)/2, ndigits=0))
    # if(rounded_domain_size <len(domain_size)/2):
    #     rounded_domain_size = rounded_domain_size+1
    # fig, axes = plt.subplots(nrows=2, ncols=rounded_domain_size, figsize=((8*rounded_domain_size), 8), dpi=200)
    # fig.suptitle('Error Stats')
    # error_states= np.array(error_stats)
    # y_max = np.max(error_stats)
    # # Plot each histogram in a separate subplot
    # for j, ax in enumerate(axes.flatten()):
    #     x_axis = np.arange(len(methods))
    #     width_bar = 0.1
    #     if j < len(error_stats):
    #         ax.bar(x_axis+(0*width_bar), abs(error_stats[j][:,0]), width = width_bar, label='Max', color='indigo')
    #         ax.bar(x_axis+(1*width_bar), abs(error_stats[j][:,1]), width = width_bar, label='Mean', color='teal')
    #         ax.bar(x_axis+(2*width_bar), abs(error_stats[j][:,3]), width = width_bar, label='StdDev', color='gold')
    #         ax.set_yscale('log')
    #         ax.set_xticks(x_axis+width_bar, methods)
    #         ax.legend()
    #         ax.set_axisbelow(True)
    #         ax.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    #         ax.set_title(f'{domain_size[j]}')
    #         ax.set_ylim(0, y_max)
    # # Adjust layout and save the plot
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.85)  # Adjust the title position
    # plt.savefig(path_to_plot+'Error_stats.png')

if __name__ == '__main__':
    main()