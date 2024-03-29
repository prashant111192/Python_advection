import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
import scipy.spatial as spatial
import sklearn.neighbors as NN
import math
# import pyinterp

def shepards_method_all_parts(x, y, z, s, u, v, w, NN_idx, h):
    shepards = np.zeros(len(u))
    shepards_gms = np.zeros(len(u))
    cs = np.zeros(len(u))
    gaussian = np.zeros(len(u))
    qs = np.zeros(len(u))
    # shepards = np.zeros((len(u),2))
    # p for shepards method
    p = 3
    for i in range(len(u)):
        shepards[i],shepards_gms[i], cs[i], gaussian[i], qs[i] = kernel_methods(x, y, z, s, u[i], v[i], w[i], p, NN_idx[i], h)
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
    # print(w_basic_shepards)
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

        # Cubclic Spline Kernel
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

    # for i in range(len(N_idx)):
    #     w_kernel_cs[i] = kernel_based_weight(d[i], np.max(d))
    # w_kernel_based = w_kernel_based/np.sum(w_kernel_based)


    # return xx_return
    return shepards, shepards_gms, cs, gaussian, qs

def show_neighbours(NN_Idx, random_points, points, idx):
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
    
    plt.savefig('NN_area.png')
    # plt.show()
    plt.clf()
    # exit()



def func(x, y, z):
    temp = np.abs(np.sin(np.pi*x)*np.cos(np.pi*y)*z*1000)
    # temp = np.abs(np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*z)
    # temp = np.sin(x+y)+np.cos(z+10)
    # temp = (np.sin(2*np.pi*x*y*z))
    # temp = x**2+y**2+z**2
    # temp = temp/np.max(temp)
    return temp

def main():

    # number of points on 1 axis of the original data
    num = 40
    # number of points on 1 axis of the interpolated data
    interpolated_size = 1000
    np.random.seed(3274)
    box_max = 1
    x = np.linspace(0, box_max, num)
    x = np.sort(x)
    y = np.linspace(0, box_max, num)
    y = np.sort(y)
    z = np.linspace(0, box_max, num)
    z = np.sort(z)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    s = func(X, Y, Z)
    s = s.flatten()
    density = 1000
    mass = 0.001
    volume = mass/density

    u = np.random.rand(interpolated_size)  
    v = np.random.rand(interpolated_size)  
    w = np.random.rand(interpolated_size)
    random_points = np.array([u, v, w]).T
    
    kd_tree = spatial.KDTree(points)
    h = (box_max-0)/num
    print("h = ", h)
    nbrs = NN.NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    NN_idx = nbrs.radius_neighbors(random_points, radius=2*h, return_distance=False)
    print(np.shape(NN_idx))
    print(NN_idx[100])
    show_neighbours(NN_idx, random_points, points, random.randint(0,interpolated_size))

    s_truth = func(u, v, w)

    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u,v,w,alpha = 0.5, c = s_truth, s=50)
    plt.savefig('random_points.png')
    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(X, Y,Z,alpha = 0.5, c=s,  cmap='viridis')
    fig.colorbar(sctt, ax=ax)
    plt.title('Actual Data')
    plt.savefig('actual_data.png')

    print('Linear Interpolation')
    interpolated_fx_linear  = interp.LinearNDInterpolator((points[:,0], points[:,1], points[:,2]), s)
    interpolated_scalars_linear = interpolated_fx_linear((u, v, w))
    del(interpolated_fx_linear)

    print('Nearest Interpolation')
    interpolated_fx_nearest = interp.NearestNDInterpolator((points[:,0], points[:,1], points[:,2]), s)
    interpolated_scalars_nearest = interpolated_fx_nearest((u, v, w))
    del(interpolated_fx_nearest)
    
    print('Shepards Interpolation')
    interpolated_scalars_shepards,interpolated_scalars_shepards_gms, interpolated_scalars_cs, interpolated_scalars_gaussian, interpolated_scalars_qs = shepards_method_all_parts(points[:,0], points[:,1], points[:,2], s, u, v, w, NN_idx, h)

    print('Regular Interpolation')
    interpolated_fx_regular = interp.RegularGridInterpolator((points[:,0](), Y.flatten(), Z.flatten()), s)
    interpolated_scalars_regular = interpolated_fx_regular((u, v, w))
    del(interpolated_fx_regular)

    # interpolated_fx_linear  = interp.LinearNDInterpolator((X, Y, Z), s)
    # interpolated_scalars_linear = interpolated_fx_linear((u, v, w))
    # del(interpolated_fx_linear)
    # interpolated_fx_nearest = interp.RegularGridInterpolator((points[:,0](), Y.flatten(), Z.flatten()), s)
    # interpolated_scalars_nearest = interpolated_fx_nearest((u, v, w))
    # del(interpolated_fx_nearest)
    # interpolated_fx_regular = interp.NearestNDInterpolator((points[:,0](), Y.flatten(), Z.flatten()), s)
    # interpolated_scalars_regular = interpolated_fx_regular((u, v, w))
    # del(interpolated_fx_regular)



    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_linear, alpha=0.5,s=50, cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Linear Interpolation')
    plt.savefig('linear.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_nearest, alpha=0.5, s = 50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Nearest Interpolation')
    plt.savefig('nearest.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_regular, alpha=0.5, cmap='viridis')
    plt.colorbar()
    plt.title('Regular Interpolation')
    plt.savefig('regular.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_shepards, alpha=0.5, s=50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Shepards Interpolation')
    plt.savefig('shepards.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_shepards_gms, alpha=0.5, s=50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Shepards GMS Interpolation')
    plt.savefig('shepards_gms.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_cs, alpha=0.5, s=50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Cubic Spline Interpolation')
    plt.savefig('cubic_spline.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_gaussian, alpha=0.5, s=50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Gaussian Interpolation')
    plt.savefig('gaussian.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_qs, alpha=0.5, s=50,cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Quintic Spline Interpolation')
    plt.savefig('quintic.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_linear-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s=50,alpha=0.5, cmap='viridis')
    plt.title('Difference Linear Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_linear.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_linear.png')


    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_nearest-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s = 50, alpha=0.5, cmap='viridis')
    plt.title('Difference Nearest Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_nearest.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_nearest.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_regular-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, alpha=0.5, cmap='viridis')
    plt.title('Difference Regular Interpolation')
    plt.colorbar()
    plt.savefig('diff_regualr.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_regular.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_shepards-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp,  s = 50,alpha=0.5, cmap='viridis')
    plt.title('Difference Shepards Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_shepards.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_shepards.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_shepards_gms-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s = 50,alpha=0.5, cmap='viridis')
    plt.title('Difference Shepards GMS Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_shepards_gsm.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_shepards_gms.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_cs-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s = 50,alpha=0.5, cmap='viridis')
    plt.title('Difference Cubic Spline Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_cubic_spline.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_cubic_spline.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_gaussian-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s = 50,alpha=0.5, cmap='viridis')
    plt.title('Difference Gaussian Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_gaussian.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_gaussian.png')

    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    temp = (interpolated_scalars_qs-s_truth)/s_truth
    sctt = ax.scatter3D(u, v,w, c=temp, s = 50,alpha=0.5, cmap='viridis')
    plt.title('Difference Quintic Spline Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_quintic_spline.png')
    plt.clf()
    plt.hist(temp, bins=100)
    plt.savefig('hist_quintic_spline.png')

    temp =  np.abs((interpolated_scalars_linear-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = temp
    print('Maximum error for Linear: ',np.max(temp))
    print('Mean error for Linear: ',np.mean(temp))

    temp = np.abs((interpolated_scalars_nearest-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Nearest: ',np.max(temp))
    print('Mean error for Nearest: ',np.mean(temp))
    
    temp = np.abs((interpolated_scalars_regular-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for regular: ',np.max(temp))
    print('Mean error for Regular: ',np.mean(temp))

    temp = np.abs((interpolated_scalars_shepards-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Shepards: ',np.max(temp))
    print('Mean error for Shepards: ',np.mean(temp))
    print('Median error for Shepards: ',np.median(temp))

    temp = np.abs((interpolated_scalars_shepards_gms-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Shepards GMS: ',np.max(temp))
    print('Mean error for Shepards GMS: ',np.mean(temp))
    print('Median error for Shepards GMS: ',np.median(temp))


    temp = np.abs((interpolated_scalars_cs-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Cubic Spline: ',np.max(temp))
    print('Mean error for Cubic Spline: ',np.mean(temp))
    print('Median error for Cubic Spline: ',np.median(temp))

    temp = np.abs((interpolated_scalars_gaussian-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Gaussian: ',np.max(temp))
    print('Mean error for Gaussian: ',np.mean(temp))
    print('Median error for Gaussian: ',np.median(temp))


    temp = np.abs((interpolated_scalars_qs-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    box = np.vstack((box, temp))
    print('Maximum error for Quintic Spline: ',np.max(temp))
    print('Mean error for Quintic Spline: ',np.mean(temp))
    print('Median error for Quintic Spline: ',np.median(temp))

    plt.clf()
    plt.yscale('log')
    plt.boxplot(box.T, labels=['Linear', 'Nearest', 'regular','Shepards', 'Shepards_gms', 'Cubic Spline', 'Gaussian', 'Quintic Spline'])
    # plt.boxplot(box.T, labels=['Linear', 'Nearest', 'Shepards', 'Cubic Spline', 'Gaussian', 'Quintic Spline'])
    plt.grid()
    plt.title('Error Boxplot')
    plt.savefig('boxplot.png')

    # temp = (np.abs(interpolated_scalars_regular-s_truth)/s_truth)
    # temp = temp[~np.isnan(temp)]
    # print(np.max(temp))




if __name__ == '__main__':
    main()