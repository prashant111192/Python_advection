
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors as NN
import copy
import multiprocessing as mp

def kerDWendland(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
        return qr
    else:
        return 0

def compute(args):
    i, particle_coords, vel, h, dt, Conc, vol, NN_idx = args
    C_nn = 0
    for j in NN_idx[i]:
        dr = particle_coords[i]-particle_coords[j]
        r = np.linalg.norm(dr)
        if r==0:
            continue
        dw = kerDWendland(r,h)
        concx = Conc[j] - Conc[i]
        if concx<0:
            print("negative concentration")
        dot = np.dot(dr,vel)
        adv_component = concx * vol * dw  * dot
        C_nn += adv_component
    return C_nn


def main():
    dp = 0.01
    h = dp*3
    dt = 0.001
    t_end = 1
    boxsize = (1,0.1)

    single_thread = True
    # plot every x states
    plot_state = True
    if plot_state:
        plot_scatter = False
        plot_line = True


    #make a grid of particles with spacing dp and boxsize
    x_coords = np.arange(0,boxsize[0],dp)
    y_coords = np.arange(0,boxsize[1],dp)
    particle_coords = np.mgrid[0:boxsize[0]:dp, 0:boxsize[1]:dp].reshape(2,-1).T
    num_particles = len(particle_coords)

    #middle parallel to the x-axis
    mid_y = int(len(y_coords)/2)
    middle_particles = np.where(particle_coords[:,1]==y_coords[mid_y])

    #initialize the velocity of the particles
    vel = np.array([0.01,0])

    #initialize the concentration of the particles
    Conc = np.zeros(len(particle_coords))
    ConcOrig = np.zeros(len(particle_coords))
    for i in range(len(particle_coords)):
        Conc[i] = math.exp(-((particle_coords[i,0]-0.5)**2)/0.01)
        ConcOrig[i] = math.exp(-((particle_coords[i,0]-0.5)**2)/0.01)

    # volume of the particles
    vol = boxsize[0]*boxsize[1]/num_particles

    #initialize the nearest neighbor algorithm using h as the radius, return the index and distance
    nbrs = NN(radius=h, algorithm='ball_tree').fit(particle_coords)
    NN_dist, NN_idx = nbrs.radius_neighbors(particle_coords, return_distance=True)
    print("Average number of neighbors: ", np.mean([len(x) for x in NN_idx]))

    # the main loop for advection
    t = 0
    while(t<t_end):
        print("time = ", t)
        C_temp = np.zeros(len(Conc))
        args = (i, particle_coords, vel, h, dt, Conc, vol, NN_idx)
        if single_thread:
            for i in range(num_particles):
                C_temp[i] = compute(args)
        else:
            num_processes = mp.cpu_count()-4
            with mp.Pool(num_processes) as pool:
                ranges = [args for i in range(num_particles)]
                C_temp = pool.map(compute, ranges)

        C_temp = np.array(C_temp)
        C_temp = C_temp*dt
        Conc -= C_temp
        t += dt

        if(t%0.01<dt and plot_state):
            if plot_scatter:
                plt.clf()
                plt.scatter(particle_coords[:,0], particle_coords[:,1], c=C_temp)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title("t = " + str(t))
                plt.colorbar()
                plt.grid()
                plt.draw()
                plt.pause(0.01)
            if plot_line:
                plt.clf()
                plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles], 'r^',markevery = 5, label="Initial")
                plt.plot(particle_coords[middle_particles,0].T, Conc[middle_particles], label="Current")
                plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles]-Conc[middle_particles], label="Difference")
                plt.title("t = " + str(t))
                plt.xlabel("x position")
                plt.ylabel("Concentration")
                plt.legend()
                plt.grid()
                plt.draw()
                plt.pause(0.01)

    # Difference between initial and final concentration
    print("Difference between initial and final concentration: ", np.sum(ConcOrig-Conc))

    # Subplots to compare the initial, final, and concentration difference with colorbar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ini_plot = ax1.scatter(particle_coords[:,0], particle_coords[:,1], c=ConcOrig, s=ConcOrig*10)
    ax1.set_title("Initial concentration")
    ax1.set_aspect('equal', adjustable='box')
    plt.colorbar(ini_plot, ax=ax1)
    ax1.grid()
    Final_plot = ax2.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc, s=Conc*10)
    ax2.set_title("Final concentration")
    ax2.set_aspect('equal', adjustable='box')
    plt.colorbar(Final_plot, ax=ax2)
    ax2.grid()
    diff_plot = ax3.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc-ConcOrig, s=Conc*10)
    ax3.set_title("dc")
    ax3.set_aspect('equal', adjustable='box')
    plt.colorbar(diff_plot, ax=ax3)
    ax3.grid()
    plt.savefig("adv2d.png", dpi=300)


if __name__ == '__main__':
    main()