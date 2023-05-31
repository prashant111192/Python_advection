
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors as NN
import copy
import multiprocessing as mp

def kerWendland(r,h):
    q = r/h
    awen = 5/(8*h)
    return awen*((1-q/2)**3)*(1.5*q+1)

def kerDWendland(r,h):
    q = r/h
    if(q<2):
        qq = 1-q/2
        qr = qq*qq*qq *(-5*q)*(7/(4*np.pi*h*h*h))
        return qr
    else:
        return 0
    # alphaD = 5/(8*h*h)
    # q = r/h
    # return -3*alphaD*q*((1-q/2)**2)


def compute(args):
    i , particle_coords, vel, h, dt, Conc, vol, NN_idx = args
    ConcNew = np.zeros(len(Conc))
    conc_temp = 0
    for j in NN_idx[i]:
        dr = particle_coords[i]-particle_coords[j]
        r = np.linalg.norm(dr)
        if r==0:
            continue
        dw = kerDWendland(r,h)
        concx = Conc[j] - Conc[i]
        dot = np.dot(dr,vel)
        # sign  = math.cos(dr/r)
        test = concx * vol * dw  * dot
        conc_temp += test
    ConcNew[i] = conc_temp
        
    dc = -conc_temp
    return dc

def lagrangian(particle_coords, vel, dt, Conc, vol, dp, t_end, NN_idx, h, num_particles, ConcOrig, y_coords, tempy):
    t = 0
    nx = 2*int(h/dp)
    while(t<t_end):
        ConcNew = np.zeros(len(Conc))
        dc = np.zeros(len(Conc))
        num_processes = mp.cpu_count()-4
        # with mp.Pool(num_processes) as pool:
        #     ranges = [(i, particle_coords, vel, h, dt, Conc, vol, NN_idx) for i in range(num_particles)]
        #     dc = pool.map(compute, ranges)
        dc = np.array(dc)
        # print(dc.shape)
        # print(dt.shape)
        tempx = dt*dc
        Conc = Conc + (dt*dc)
        # Plot the concentration each 100 time steps
        temp_time = t 
        middle_particles = []
        middle_particles = np.where(particle_coords[:, 1] == y_coords[tempy])[0]

        # Extract the concentration values for these particles
        middle_x = particle_coords[middle_particles, 0]
        middle_conc = Conc[middle_particles]
        middle_conc_orig = ConcOrig[middle_particles]
        plt.clf()
        plt.plot(middle_x, middle_conc, "x")
        # plt.plot(middle_x, middle_conc_orig, "Orig")
        plt.xlabel('Position along x-axis')
        plt.ylabel('Concentration')
        plt.title(label='Time: {:.4f} s, Total Conc = {:.4f}'.format(t+dt, np.sum(Conc*vol)))
        # plt.ylim(0, 1.5)
        plt.grid(which='both')
        plt.draw()
        plt.pause(0.01)
        # Extract the x coordinates for these particles

        t = t + dt
    # print("averenge number of neighbours: ", num/len(Conc))
    return Conc, ConcNew

def main():

    # set dimensions
    dim = 2
    # Set the particle spacing and box size
    dp = 0.001
    if dim ==2:
        box_size = (1,0.01)
    else:
        exit()
    t_end = 10
    dt =0.01
    h = dp*8
    # calculate the number of particles in each dimension
    num_particles_x = int(box_size[0] / dp)
    num_particles_y = int(box_size[1] / dp)

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
    vel = (0.1,0)
    # vel_y = 0
    if dim==2:
        vol = (box_size[0] * box_size[1])/num_particles

    print("number of particles: ", num_particles)

    # create the particle concentrations
    Conc = np.zeros(num_particles)

    for i in range(num_particles):
        Conc[i] = math.exp(-((i-(particle_coords[i,0])/2)**2)/(2*0.01))


    # lagrangian advection
    concLag = np.zeros(num_particles)
    conc_d = np.zeros(num_particles)
    concLag = copy.deepcopy(Conc)
    conc_d = copy.deepcopy(Conc)

    # nearest neighbours
    kdt = NN(radius=2*h, algorithm='kd_tree').fit(particle_coords)
    NN_idx = kdt.radius_neighbors(particle_coords)[1]

    tempy = len(y_coords)
    tempy = int(tempy/2)
    concLag, conc_d = lagrangian(particle_coords, vel, dt, concLag, vol, dp, t_end, NN_idx, h, num_particles, Conc, y_coords, tempy)
    
if __name__=='__main__':
    main()