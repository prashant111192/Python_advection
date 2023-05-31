import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors as NN

def sumConc(Conc, particle_coords, vol):
    sumTemp = 0
    for i in range(len(Conc)):
        if(particle_coords[i,0]<0.1 or particle_coords[i,0]>0.9):
            continue
        sumTemp = sumTemp + Conc[i]*vol
    return sumTemp

def kernel(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq*qq *(q+q+1)*(7/(4*np.pi*h*h))
        return qr
    else:
        return 0

def dkernelG(r,h, dim):
    q = r/h
    if(q<2):
        qr = -2*q*math.exp(-q*q)*(1/(np.pi*h*h))
        if (dim==2):
            return qr/h
        if dim==1:
            return qr


def dkernel(r,h, dim):
    q = r/h
    if (dim==2):
        if(q<2):
            qq = 1-q/2
            qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
            return qr
        else:
            return 0
    else: 
        if (dim==1):
            if(q<2):
                qq = 1-q/2
                qr = qq*qq *q *(-15)/(8*h*h)
                return qr
            else:
                return 0

def compute(args):
    i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx, dim, ConcOrig = args
    particle_spacing = 0.01
    ConcTemp = Conc[i]
    amount = 0
    for k in (NN_idx[i]):
        if(particle_coords[i,0]<0.1 or particle_coords[i,0]>0.9):
            continue
        dr = particle_coords[i] - particle_coords[k]
        r = np.linalg.norm(dr)
        dker = dkernel(r,kernelSize, dim)
        if(dker==0) or (r==0):
            continue
        absdr = np.abs(dr)/(np.linalg.norm(dr)**2 + 1e-20)
        adv_v_ij = np.dot(dr, np.array([vel_x, vel_y])*absdr)
        adv_v_ji = np.dot(dr, np.array([vel_x, vel_y])*absdr)
        adv = dker * dt * adv_v_ij *  mass/density
        adv = adv/particle_spacing
        if(adv>0):
            # Particle i gives 
            amount = amount+ (adv * Conc[i])
            # print("Gives, ker: ", dker, "adv: ", adv, "Conc: ", ConcTemp, "Amount: ", amount, "NN_idx: ", k)
        adv = dker * dt * adv_v_ji *  mass/density
        adv = adv/particle_spacing
        if(adv<0):
            # particle i gets
            amount = amount + (adv * Conc[k])
            # print("Gives, ker: ", dker, "adv: ", adv, "Conc: ", ConcTemp, "Amount: ", amount, "NN_idx: ", k)
    ConcTemp = ConcTemp - particle_spacing *amount
    return ConcTemp


def main():
    # set dimensions
    dim = 1
    # Set the particle spacing and box size
    particle_spacing = 0.001
    # length, width
    if(dim==2):
        box_size = (1, 0.2)
    else:
        if(dim==1):
            box_size = (1, particle_spacing)
    start_index = int(particle_spacing*0.2)
    end_index = int(particle_spacing*0.3)

    # Constants for setting up the particles and their concnetrations

    # creating the vector for original concentrations before any calcs
    # corig[start_index:end_index + 1] = gaussian_values

    # calculate the number of particles in each dimension
    num_particles_x = int(box_size[0] / particle_spacing)
    num_particles_y = int(box_size[1] / particle_spacing)

    # create a meshgrid of particle coordinates
    x_coords = np.linspace(0, box_size[0], num_particles_x)
    y_coords = np.linspace(0, box_size[1], num_particles_y)
    particle_coords = np.meshgrid(x_coords, y_coords)
    # find the middle point in y_coords
    tempy = len(y_coords)
    tempy = int(tempy/2)

    # reshape the particle coordinates into a 2d array
    particle_coords = np.array(particle_coords).reshape(2, -1).T

    num_particles = len(particle_coords)
    # create the particle velocities (assume all particles have the same velocity)
    vel_x = 10
    vel_y = 0
    # volume of each particle
    if dim==2:
        vol = (box_size[0] * box_size[1])/num_particles
    elif dim==1:
        vol = (box_size[0])/num_particles
    density = 1000
    mass = density * vol 


    print("number of particles: ", num_particles)
    # create the particle masses
    Conc = np.zeros(num_particles)
    gaussian_values = np.zeros(num_particles)
    #make gaussian distribution from start to end 

    counter = 0
    for i in range(num_particles):
        Conc[counter] = math.exp(-((i-(particle_coords[i,0])/2)**2)/(2*0.01))
        counter = counter + 1
        
        # if 0.2 < particle_coords[i, 0] < 0.3:
        # # if 0.2 < particle_coords[i, 0] < 0.3:
        #     Conc[i] = 1

    ConcOrig = copy.deepcopy(Conc)
    # ConcOrig = Conc
    dt = 0.01
    t = np.arange(0, 2, dt)
    kernelSize = particle_spacing * 3

    # NEAREST NEIGHBORS
    kdt = NN(radius=2*kernelSize, algorithm='kd_tree').fit(particle_coords)
    NN_idx = kdt.radius_neighbors(particle_coords)[1]

    # Get the number of available cores
    num_cores = mp.cpu_count()
    # Calculate the number of cores to use
    num_processes = max(num_cores - 2, 1)
    ## for line plot
    line_plot = False
    Flow = False
    if(Flow):
        plt.clf()
        plt.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc)
        plt.title("Particle grid")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.colorbar()
        plt.clim(0, 1)
        plt.grid()
        plt.draw()
        plt.pause(0.01)

    ConcTemp = Conc
    middle_conc= []
    kernelType = ["Gaussian", "Flat", "Wendland"]
    for s in t:
        # print("Time: ", s,"***************************************************")
        sumTempOld = sumConc(Conc, particle_coords, vol)
        for i in range(num_particles):
            args = (i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx, dim, ConcOrig)
            result = compute(args)
            ConcTemp[i]=(result)

        ## Parallelize the loop
        # with mp.Pool(num_processes) as pool:
        #     ranges = [(i, particle_coords, vel_x, vel_y, kernelSize, dt, Conc, density, mass, NN_idx, dim, ConcOrig) for i in range(num_particles)]
        #     ConcTemp = pool.map(compute, ranges)

        Conc= np.array(ConcTemp)
        sumTempNew = sumConc(Conc, particle_coords, vol)
        Conc = (Conc*sumTempOld)/sumTempNew

        temp_time = s*100 / dt 
        middle_particles = []
        middle_particles = np.where(particle_coords[:, 1] == y_coords[tempy])[0]

        # Extract the concentration values for these particles
        middle_conc = Conc[middle_particles]
        middle_conc_orig = ConcOrig[middle_particles]

        # Extract the x coordinates for these particles
        middle_x = particle_coords[middle_particles, 0]
        # print("Current time: {:.2f} s, Total Conc = {:.4f}".format(s, np.sum(Conc*vol)))
        if(temp_time%100==0):
        # if(temp_time%10000==0):
            # Make the FLOW plot
            if(Flow):
                plt.clf()
                plt.scatter(particle_coords[:,0], particle_coords[:,1], c=Conc)
                plt.title("Particle grid, Time: {:.4f} s, Total Conc = {:.4f}".format(s+dt, np.sum(Conc*vol)))
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.colorbar()
                plt.clim(0, 1)
                plt.grid()
                plt.draw()
                plt.pause(0.01)

            # Make the LINE plot
            if(line_plot):
                plt.clf()
                plt.plot(middle_x, middle_conc, "x")
                # plt.plot(middle_x, middle_conc_orig, "Orig")
                plt.xlabel('Position along x-axis')
                plt.ylabel('Concentration')
                plt.title(label='Time: {:.4f} s, Total Conc = {:.4f}'.format(s+dt, np.sum(Conc*vol)))
                plt.ylim(0, 1.5)
                plt.grid(which='both')
                plt.draw()
                plt.pause(0.01)

        pos_max_idx = np.argmax(middle_conc)
        # print("Position of maximum concentration: {:.15f}".format(middle_x[pos_max_idx]))

    plt.clf()
    plt.plot(middle_x, middle_conc,label="time 0" )
    plt.plot(middle_x, middle_conc_orig, label="time end")
    plt.xlabel('Position along x-axis')
    plt.ylabel('Concentration')
    plt.title(label='Time: {:.2f} s, Total Conc = {:.4f}'.format(s, np.sum(Conc*vol)))
    plt.legend()
    plt.grid(which='both')
    plt.ylim(0, 1.5)
    # plt.savefig("lineplot.png")
    plt.show()


if __name__ == '__main__':
    main()