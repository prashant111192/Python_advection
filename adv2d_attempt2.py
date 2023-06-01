
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors as NN
import copy
import multiprocessing as mp

def kerDWendland(r,h, DIM):
    q = r/h
    if q >2:
        return 0
    elif DIM==1:
        alphaD = 5/(8*h*h)
        return -3*alphaD*q*((1-q/2)**2)
    elif DIM ==2:
        qq = 1-q/2
        qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
        return qr

def Lagrangian(args):
    i, particle_coords, vel, h, dt, Conc, vol, NN_idx, DIM = args
    C_nn = 0
    if DIM == 2 and (particle_coords[i,0]<0.2 or particle_coords[i,0]>0.8 or particle_coords[i,1]<0.001 or particle_coords[i,1]>0.009):
        return C_nn

    for j in NN_idx[i]:
        dr = particle_coords[i]-particle_coords[j]
        r = np.linalg.norm(dr)
        if r==0:        # if i=j, skip
            continue
        dw = kerDWendland(r,h, DIM)
        dot = np.dot(dr,vel[i])
        concx = Conc[j] - Conc[i]
        # directional change in concentration
        # temp_j = dw * vol * Conc[j]* dot/r
        # temp_i = dw * vol * Conc[i]* dot/r
        # if temp_i >0:
        #     C_nn += temp_i
        # if temp_j <0:
        #     C_nn += temp_j
        # non directional change in concentration
        adv_component = concx * vol * dw  * dot/r
        C_nn += adv_component
    return C_nn


def main():
    plt.style.use('dark_background')
    DIM = 1
    dp = 0.001

    dt = 0.01
    t_end = 2
    if DIM ==1:
        h = dp*8
        boxsize = (2,dp)
    if DIM ==2:
        h = dp*2
        boxsize = (2,0.01)
    if DIM ==3:
        h = dp*2
        boxsize = (1,0.01,0.01)

    single_thread = True
    # plot every x states
    plot_state = False
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
    #Single velocity
    # vel = np.array([0.01,0])
    #Multiple velocities
    vel = np.zeros((num_particles,2))


    mean = boxsize[0]/4
    #initialize the concentration of the particles
    Conc = np.zeros(len(particle_coords))
    ConcOrig = np.zeros(len(particle_coords))
    for i in range(len(particle_coords)):
        Conc[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        ConcOrig[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        vel[i,0] = particle_coords[i,0] * (-0.1)
        # vel[i,0] = math.exp(-((particle_coords[i,0]-(boxsize[0]/2))**2)/0.5)*0.01
    
    # volume of the particles
    if DIM==2:
        vol = boxsize[0]*boxsize[1]/num_particles
    if DIM==1:
        vol = boxsize[0]/num_particles

    print("volume of the particles: ", vol)
    #initialize the nearest neighbor algorithm using h as the radius, return the index and distance
    nbrs = NN(radius=2*h, algorithm='kd_tree').fit(particle_coords)
    NN_idx = nbrs.radius_neighbors(particle_coords)[1]
    print("Average number of neighbors: ", np.mean([len(x) for x in NN_idx]))

    # the main loop for advection
    t = 0
    while(t<t_end):
        print("time = ", t)
        total_C_before = np.sum(Conc)*vol*num_particles
        C_temp = np.zeros(len(Conc))
        if single_thread:
            for i in range(num_particles):
                args = (i, particle_coords, vel, h, dt, Conc, vol, NN_idx, DIM)
                C_temp[i] = Lagrangian(args)
        elif not single_thread:
            num_processes = mp.cpu_count()-4
            with mp.Pool(num_processes) as pool:
                ranges = [(i, particle_coords, vel, h, dt, Conc, vol, NN_idx, DIM) for i in range(num_particles)]
                C_temp = pool.map(Lagrangian, ranges)

        C_temp = np.array(C_temp)
        C_temp = C_temp*dt
        Conc -= C_temp
        total_C_after = np.sum(Conc)*vol*num_particles
        Conc = Conc/total_C_after*total_C_before
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
                plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles], 'r*',markevery = 10, label="Initial", fillstyle='none' )
                plt.plot(particle_coords[middle_particles,0].T, Conc[middle_particles],'g', label="Current")
                # plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles]-Conc[middle_particles], label="Difference")
                plt.ylim(-0.1,1.1)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    ini_plot = ax1.scatter(particle_coords[:,0], particle_coords[:,1], c=ConcOrig)
    # ini_plot = ax1.scatter(particle_coords[:,0], particle_coords[:,1], c=ConcOrig, s=ConcOrig*20)
    ax1.set_title("Initial concentration")
    # ax1.set_aspect('equal', adjustable='box')
    plt.colorbar(ini_plot, ax=ax1)
    ax1.grid()
    Final_plot = ax2.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc)
    ax2.set_title("Final concentration")
    # ax2.set_aspect('equal', adjustable='box')
    plt.colorbar(Final_plot, ax=ax2)
    ax2.grid()
    # diff_plot = ax3.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc-ConcOrig, s=Conc*10)
    # ax3.set_title("dc")
    # ax3.set_aspect('equal', adjustable='box')
    # plt.colorbar(diff_plot, ax=ax3)
    # ax3.grid()
    plt.savefig("adv2d_"+str(DIM)+".png", dpi=300)
    
    plt.clf()
    plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles], 'r*',markevery = 10, label="Initial", fillstyle='none' )
    plt.plot(particle_coords[middle_particles,0].T, Conc[middle_particles],'g', label="Final")
    plt.ylim(-0.1,1.1)
    plt.title("Concentration along the middle line")
    plt.xlabel("x position")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid()
    plt.savefig("adv2d_line_"+str(DIM)+".png", dpi=300)



    # difference between the peak position of Conc and ConcOrig
    print("Difference between the peak position of Conc and ConcOrig: ", (np.argmax(Conc)-np.argmax(ConcOrig))*dp)
    print("Final position of the peak of Conc: ", particle_coords[np.argmax(Conc[middle_particles]),])



if __name__ == '__main__':
    main()