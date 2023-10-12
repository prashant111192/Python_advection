import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.interpolate as interp

def main():

    # number of points on 1 axis of the original data
    num = 50
    # number of points on 1 axis of the interpolated data
    interpolated_size = 10000
    np.random.seed(3274)
    x = np.random.rand(num)
    x = np.sort(x)
    y = np.random.rand(num)
    y = np.sort(y)
    z = np.random.rand(num)
    z = np.sort(z)
    X, Y, Z = np.meshgrid(x, y, z)
    s = np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)*Z
    s = s.flatten()
    s = s/np.max(s)
    density = 1000
    mass = 0.001
    volume = mass/density

    u = np.random.rand(interpolated_size)  
    v = np.random.rand(interpolated_size)  
    w = np.random.rand(interpolated_size)

    s_truth = np.sin(2*np.pi*u)*np.cos(2*np.pi*v)*w
    s_truth = s_truth/np.max(s_truth)

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
    interpolated_fx_linear  = interp.LinearNDInterpolator((X.flatten(), Y.flatten(), Z.flatten()), s)
    interpolated_scalars_linear = interpolated_fx_linear((u, v, w))
    del(interpolated_fx_linear)
    print('Nearest Interpolation')
    interpolated_fx_nearest = interp.NearestNDInterpolator((X.flatten(), Y.flatten(), Z.flatten()), s)
    interpolated_scalars_nearest = interpolated_fx_nearest((u, v, w))
    del(interpolated_fx_nearest)
    print('Regular Interpolation')
    # interpolated_fx_regular = interp.RegularGridInterpolator((X.flatten(), Y.flatten(), Z.flatten()), s)
    # interpolated_scalars_regular = interpolated_fx_regular((u, v, w))
    # del(interpolated_fx_regular)

    # interpolated_fx_linear  = interp.LinearNDInterpolator((X, Y, Z), s)
    # interpolated_scalars_linear = interpolated_fx_linear((u, v, w))
    # del(interpolated_fx_linear)
    # interpolated_fx_nearest = interp.RegularGridInterpolator((X.flatten(), Y.flatten(), Z.flatten()), s)
    # interpolated_scalars_nearest = interpolated_fx_nearest((u, v, w))
    # del(interpolated_fx_nearest)
    # interpolated_fx_regular = interp.NearestNDInterpolator((X.flatten(), Y.flatten(), Z.flatten()), s)
    # interpolated_scalars_regular = interpolated_fx_regular((u, v, w))
    # del(interpolated_fx_regular)



    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_linear, alpha=0.5, cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Linear Interpolation')
    plt.savefig('linear.png')
    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_nearest, alpha=0.5, cmap='viridis')
    plt.colorbar(sctt, ax=ax)
    plt.title('Nearest Interpolation')
    plt.savefig('nearest.png')
    # plt.clf()
    # fig = plt.figure(figsize = (16, 9))
    # ax = plt.axes(projection ="3d")
    # sctt = ax.scatter3D(u, v,w, c=interpolated_scalars_regular, alpha=0.5, cmap='viridis')
    # plt.colorbar()
    # plt.title('Regular Interpolation')
    # plt.savefig('regular.png')
    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=(interpolated_scalars_linear-s_truth)*100/s_truth, alpha=0.5, cmap='viridis')
    plt.title('Difference Linear Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_linear.png')
    plt.clf()
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    sctt = ax.scatter3D(u, v,w, c=(interpolated_scalars_nearest-s_truth)*100/s_truth, alpha=0.5, cmap='viridis')
    plt.title('Difference Nearest Interpolation')
    plt.colorbar(sctt, ax=ax)
    plt.savefig('diff_nearest.png')
    # plt.clf()
    # fig = plt.figure(figsize = (16, 9))
    # ax = plt.axes(projection ="3d")
    # sctt = ax.scatter3D(u, v,w, c=(interpolated_scalars_regular-s_truth)*100/s_truth, alpha=0.5, cmap='viridis')
    # plt.title('Difference Regular Interpolation')
    # plt.colorbar()
    # plt.savefig('diff_regualr.png')
    temp =  np.abs((interpolated_scalars_linear-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    print('Maximum error for Linear: ',np.max(temp))
    print('Mean error for Linear: ',np.mean(temp))
    temp = np.abs((interpolated_scalars_nearest-s_truth)/s_truth)
    temp = temp[~np.isnan(temp)]
    print('Maximum error for Nearest: ',np.max(temp))
    print('Mean error for Nearest: ',np.mean(temp))
    # temp = (np.abs(interpolated_scalars_regular-s_truth)/s_truth)
    # temp = temp[~np.isnan(temp)]
    # print(np.max(temp))




if __name__ == '__main__':
    main()