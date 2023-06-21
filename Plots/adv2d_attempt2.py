
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors as NN
import copy
import multiprocessing as mp
from matplotlib.ticker import ScalarFormatter

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
    i, particle_coords, vel, h, dt, Conc, Conc_N_CD, Conc_N_SPH, Conc_UP, Conc_N_UP,vol, NN_idx, DIM, Conc_CD = args
    C_nn = [0, 0, 0, 0,0,0]
    if DIM == 2 and (particle_coords[i,0]<0.2 or particle_coords[i,0]>0.8 or particle_coords[i,1]<0.001 or particle_coords[i,1]>0.009):
        return C_nn
    if DIM == 1 and (particle_coords[i,0]<0.1 or particle_coords[i,0]>1.9):
        return C_nn

    for j in NN_idx[i]:
        dr = particle_coords[i]-particle_coords[j]
        r = np.linalg.norm(dr)
        if r==0:        # if i=j, skip
            continue
        dw = kerDWendland(r,h, DIM)
        dot = np.dot(dr,vel[i])
        concx = Conc[j] - Conc[i]
        concx_N_SPH = Conc_N_SPH[j] - Conc_N_SPH[i]
        # directional change in concentration
        # temp_j = dw * vol * Conc[j]* dot/r
        # temp_i = dw * vol * Conc[i]* dot/r
        # if temp_i >0:
        #     C_nn += temp_i
        # if temp_j <0:
        #     C_nn += temp_j
        # non directional change in concentration
        adv_component = concx * vol * dw  * dot/r
        adv_component_N = concx_N_SPH * vol * dw  * dot/r
        C_nn[0] += adv_component
        C_nn[1] += adv_component_N
    #CD methods
    dr = particle_coords[i,0]-particle_coords[i-1,0]
    C_nn[2] = vel[i,0]*(Conc_CD[i+1]-Conc_CD[i-1])/(2*dr)
    C_nn[3] = vel[i,0]*(Conc_N_CD[i+1]-Conc_N_CD[i-1])/(2*dr)
    C_nn[4] = vel[i,0]*(Conc_UP[i]-Conc_UP[i-1])/(dr)
    C_nn[5] = vel[i,0]*(Conc_N_UP[i]-Conc_N_UP[i-1])/(dr)
    return C_nn


def main():
    # plt.style.use('dark_background')
    DIM = 1
    dp = 0.001

    dt = 0.001
    t_end = 5
    if DIM ==1:
        h = dp*8
        boxsize = (2,dp)
    if DIM ==2:
        h = dp*2
        boxsize = (2,0.01)
    if DIM ==3:
        h = dp*2
        boxsize = (1,0.01,0.01)

    single_thread = False
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
    vel = np.zeros((num_particles,2))


    mean = boxsize[0]/4
    #initialize the concentration of the particles
    Conc = np.zeros(len(particle_coords))
    Conc_CD = np.zeros(len(particle_coords))
    Conc_UP = np.zeros(len(particle_coords))
    Conc_N_SPH = np.zeros(len(particle_coords))
    Conc_N_CD = np.zeros(len(particle_coords))
    Conc_N_UP = np.zeros(len(particle_coords))
    ConcOrig = np.zeros(len(particle_coords))
    for i in range(len(particle_coords)):
        Conc[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        ConcOrig[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        Conc_CD[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        Conc_UP[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        Conc_N_SPH[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        Conc_N_CD[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        Conc_N_UP[i] = math.exp(-((particle_coords[i,0]-mean)**2)/0.005)
        # if particle_coords[i,0] < 0.5 and particle_coords[i,0] > 0.4:
        #     Conc[i] = 1
        #     ConcOrig[i] = 1
        #     Conc_CD[i] = 1
        #     Conc_N_SPH[i] = 1
        #     Conc_N_CD[i] = 1
        # vel[i,0] = 0.7
        # vel[i,0] = (particle_coords[i,0]+0.01) * (0.1)
        # vel[i,0] = ((particle_coords[i,0]+5) * 0.01) 
        vel[i,0] = math.exp(-((particle_coords[i,0]-(boxsize[0]/2))**2)/0.08)*0.05
        # vel
    
    plt.clf()
    plt.plot(particle_coords[:,0], vel[:,0], "gold", marker="x", label="Velocity")
    plt.show()
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
    count = 0
    count_normal = 0
    while(t<t_end):

        total_C_before_CD = np.sum(Conc_N_CD)*vol*num_particles
        total_C_before_SPH = np.sum(Conc_N_SPH)*vol*num_particles
        total_C_before_UP = np.sum(Conc_N_SPH)*vol*num_particles

        if single_thread:
            for i in range(num_particles):
                args = (i, particle_coords, vel, h, dt, Conc, vol, NN_idx, DIM)
                C_temp[i] = Lagrangian(args)
        elif not single_thread:
            num_processes = mp.cpu_count()-6
            with mp.Pool(num_processes) as pool:
                ranges = [(i, particle_coords, vel, h, dt, Conc, Conc_N_CD, Conc_N_SPH,  Conc_UP, Conc_N_UP,vol, NN_idx, DIM, Conc_CD) for i in range(num_particles)]
                C_temp = pool.map(Lagrangian, ranges)

        C_temp = np.array(C_temp)
        C_temp = C_temp*dt

        Conc -= C_temp[:, 0]
        Conc_N_SPH -= C_temp[:, 1]
        Conc_CD -= C_temp[:, 2]
        Conc_N_CD -= C_temp[:, 3]

        Conc_UP -= C_temp[:, 2]
        Conc_N_UP -= C_temp[:, 3]

        total_C_after_CD = np.sum(Conc_N_CD)*vol*num_particles
        Conc_N_CD = Conc_N_CD/total_C_after_CD*total_C_before_CD

        total_C_after_SPH = np.sum(Conc_N_SPH)*vol*num_particles
        Conc_N_SPH = Conc_N_SPH/total_C_after_SPH*total_C_before_SPH

        total_C_after_UP = np.sum(Conc_N_UP)*vol*num_particles
        Conc_N_UP = Conc_N_UP/total_C_after_UP*total_C_before_UP

        count_normal += 1
        t += dt
        print("time = ", f'{t:.3f}', "SPH: ", sum(ConcOrig)-sum(Conc), "CD: ", sum(ConcOrig)-sum(Conc_CD), "UP: ", sum(ConcOrig)-sum(Conc_UP))

        if(t%0.01<dt and plot_state):
            t_precision = round(t,2)
            count += 1
            if plot_scatter:
                plt.clf()
                plt.scatter(particle_coords[:,0], particle_coords[:,1], c=C_temp, s=0.5)
                # plt.gca().set_aspect('equal', adjustable='box')
                plt.title("t = " + str(t_precision))
                plt.colorbar()
                plt.grid()
                # name_scatter = "scatter_"+str(count)+".png"
                name_scatter = "scatter_"+str(t_precision)+"_"+str(DIM)+".png"
                plt.savefig(name_scatter, dpi=300)
                # plt.draw()
                # plt.pause(0.01)
            if plot_line:
                plt.clf()
                plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles], 'r*',markevery = 10, label="Initial", fillstyle='none' )
                plt.plot(particle_coords[middle_particles,0].T, Conc[middle_particles],'g', label="Current")
                # plt.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles]-Conc[middle_particles], label="Difference")
                plt.ylim(-0.1,1.1)
                plt.title("t = " + str(t_precision))
                plt.xlabel("x position")
                plt.ylabel("Concentration")
                plt.legend()
                plt.grid()
                # name_line = "line_"+str(count)+".png"
                name_line = "line_"+str(t_precision)+"_"+str(DIM)+".png"
                plt.savefig(name_line, dpi=300)
                # plt.draw()
                # plt.pause(0.01)

    # Difference between initial and final concentration
    print("Difference between initial and final concentration_SPH: ", np.sum(ConcOrig-Conc))
    print("Difference between initial and final concentration_CD: ", np.sum(ConcOrig-Conc_CD))
    print("Difference between initial and final concentration_UP: ", np.sum(ConcOrig-Conc_UP))
    print("Difference between initial and final concentration_SPH_N_CD: ", np.sum(ConcOrig-Conc_N_CD))
    print("Difference between initial and final concentration_SPH_N_SPH: ", np.sum(ConcOrig-Conc_N_SPH))
    print("Difference between initial and final concentration_SPH_N_UP: ", np.sum(ConcOrig-Conc_N_UP))

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]})
    ini_plot = ax1.scatter(particle_coords[:,0], particle_coords[:,1], c=ConcOrig)
    ax1.set_title("Initial concentration")
    ax1.set_ylim(-0.01,0.01)
    plt.colorbar(ini_plot, ax=ax1)
    ax1.grid()
    Final_plot = ax2.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc)
    ax2.set_title("Final concentration")
    ax2.set_ylim(-0.01,0.01)
    plt.colorbar(Final_plot, ax=ax2)
    ax2.grid()
    plt.tight_layout()
    plt.savefig("adv2d_"+str(DIM)+".png", dpi=300)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    # Concentration plots
    ax1.plot(particle_coords[middle_particles, 0].T, ConcOrig[middle_particles], 'r', markevery=10, label="Initial", fillstyle='none')
    ax1.plot(particle_coords[middle_particles, 0].T, Conc_CD[middle_particles], 'k:', label="Central_diff", fillstyle='none')
    ax1.plot(particle_coords[middle_particles, 0].T, Conc_UP[middle_particles], 'g:', label="Upwind", fillstyle='none')
    ax1.plot(particle_coords[middle_particles, 0].T, Conc[middle_particles], 'b:', label="SPH")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("x position (m)")
    ax1.set_ylabel("Concentration")
    ax1.legend(loc="lower right")
    ax1.grid()
    # Velocity plot
    ax2 = ax1.twinx()
    ax2.plot(particle_coords[:, 0], vel[:,0], 'k.', markevery=80, label="Vel", fillstyle='none')
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend(loc="upper right")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.offsetText.set_fontsize(10)
    plt.title("Concentration")
    plt.tight_layout()
    plt.savefig("adv2d_lineNN_"+str(DIM)+".png", dpi=300)

    plt.clf()
    fig, ax1 = plt.subplots()
    # Concentration plots
    ax1.plot(particle_coords[middle_particles,0].T, ConcOrig[middle_particles], 'r',markevery = 10, label="Initial", fillstyle='none' )
    ax1.plot(particle_coords[middle_particles,0].T, Conc_N_CD[middle_particles],'k:', label="Central_diff", fillstyle='none')
    ax1.plot(particle_coords[middle_particles,0].T, Conc_N_UP[middle_particles],'g:', label="Upwind", fillstyle='none')
    ax1.plot(particle_coords[middle_particles,0].T, Conc_N_SPH[middle_particles],'b:', label="SPH")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("x position (m)")
    ax1.set_ylabel("Concentration")
    ax1.legend(loc="lower right")
    ax1.grid()
    # Velocity plot
    ax2 = ax1.twinx()
    ax2.plot(particle_coords[:, 0], vel[:,0], 'k.', markevery=80, label="Vel", fillstyle='none')
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend(loc="upper right")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.offsetText.set_fontsize(10)
    plt.title("Concentration")
    plt.tight_layout()
    plt.savefig("adv2d_lineN_"+str(DIM)+".png", dpi=300)



    # difference between the peak position of Conc and ConcOrig
    print("Difference between the peak position of Conc_NN)SPH and ConcOrig: ", (np.argmax(Conc)-np.argmax(ConcOrig))*dp)
    print("Difference between the peak position of Conc_CD and ConcOrig: ", (np.argmax(Conc_CD)-np.argmax(ConcOrig))*dp)
    print("Difference between the peak position of Conc_N_CD and ConcOrig: ", (np.argmax(Conc_N_CD)-np.argmax(ConcOrig))*dp)
    print("Difference between the peak position of Conc_N_SPH and ConcOrig: ", (np.argmax(Conc_N_SPH)-np.argmax(ConcOrig))*dp)
    print("Difference between the peak position of Conc_UP and ConcOrig: ", (np.argmax(Conc_UP)-np.argmax(ConcOrig))*dp)
    print("Difference between the peak position of Conc_N_UP and ConcOrig: ", (np.argmax(Conc_N_UP)-np.argmax(ConcOrig))*dp)
    print("Final position of the peak of Conc: ", particle_coords[np.argmax(Conc[middle_particles])])


    pos = particle_coords[middle_particles,0]
    pos = pos.reshape((pos.shape[1]))
    header_list = ['x','vel', 'ConcOrig', 'Conc', 'Conc_CD', 'Conc_N_CD', 'Conc_N_SPH', 'Conc_UP', 'Conc_N_UP']
    save = np.column_stack((pos, vel,ConcOrig[middle_particles],Conc[middle_particles],Conc_CD[middle_particles],Conc_N_CD[middle_particles],Conc_N_SPH[middle_particles],Conc_UP[middle_particles],Conc_N_UP[middle_particles]))

    np.savetxt("save_2.csv", save, delimiter=",", header=','.join(header_list))
    # print("pos shape: ", pos.shape)
    # pos = pos.flatten()
    # print("pos shape: ", pos.shape)
    # save = np.concatenate((pos,ConcOrig[middle_particles],Conc[middle_particles],Conc_CD[middle_particles],Conc_N_CD[middle_particles],Conc_N_SPH[middle_particles],Conc_UP[middle_particles],Conc_N_UP[middle_particles]))
    # save the 7 Conc arrays to separate csv files
    # np.savetxt("ConcOrig.csv", pos,ConcOrig[middle_particles], delimiter=",")
    # # np.savetxt("ConcOrig.csv", (particle_coords[middle_particles,0].T,ConcOrig[middle_particles]), delimiter=",")
    # np.savetxt("Conc.csv", np.array(particle_coords[middle_particles,0].T,Conc[middle_particles]), delimiter=",")
    # np.savetxt("Conc_CD.csv", np.array(particle_coords[middle_particles,0].T,Conc_CD[middle_particles]), delimiter=",")
    # np.savetxt("Conc_N_CD.csv", np.array(particle_coords[middle_particles,0].T,Conc_N_CD[middle_particles]), delimiter=",")
    # np.savetxt("Conc_N_SPH.csv", np.array(particle_coords[middle_particles,0].T,Conc_N_SPH[middle_particles]), delimiter=",")
    # np.savetxt("Conc_UP.csv", np.array(particle_coords[middle_particles,0].T,Conc_UP[middle_particles]), delimiter=",")
    # np.savetxt("Conc_N_UP.csv", np.array(particle_coords[middle_particles,0].T,Conc_N_UP[middle_particles]), delimiter=",")




if __name__ == '__main__':
    main()



# using ffmpeg to make a video from the images
# https://askubuntu.com/questions/610903/how-can-i-create-a-video-file-from-a-set-of-jpg-images
# ffmpeg -framerate 10 -i line_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
# 