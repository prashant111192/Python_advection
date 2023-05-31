import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors as NN
import copy

def kerWendland(r,h):
    q = r/h
    awen = 5/(8*h)
    return awen*((1-q/2)**3)*(1.5*q+1)

def kerDWendland(r,h):
    alphaD = 5/(8*h*h)
    q = r/h
    return -3*alphaD*q*((1-q/2)**2)
    # q = r/h
    # awen = 5/(8*h*h)
    # if(q<2):
    #     return awen*(-3)*q*(1-q/2)**2
    # else:
    #     return 0

def eularian(particle_coords, vel, dt, Conc, vol, dp, t_end):
    t = 0
    while(t<t_end):
        ConcNew = np.zeros(len(Conc))
        for i in range(1, len(Conc)-1):
            dr = (particle_coords[i]-particle_coords[i-1])
            ConcNew[i] = (Conc[i]-Conc[i-1])*vel/dr
        Conc = Conc - dt*ConcNew
        t = t + dt
    return Conc

def lagrangian(particle_coords, vel, dt, Conc, vol, dp, t_end, NN_idx, h):
    t = 0
    nx = 2*int(h/dp)
    while(t<t_end):
        num = 0
        ConcNew = np.zeros(len(Conc))
        for i in range(1, len(Conc)-1):
            # idjMin = int((i - nx)) if (i - nx) > 0 else 0
            # idjMax = int((i + nx)) if (i + nx) < len(Conc) else len(Conc)
            # idJ = np.arange(idjMin,idjMax)
            conc_temp = 0
            # for j in idJ:
            for j in NN_idx[i]:
                # if i==j: continue
                dr = particle_coords[i]-particle_coords[j]
                r = abs(dr)
                if r==0:
                    continue
                num = num + 1
                dw = kerDWendland(r,h)
                concx = Conc[j] - Conc[i]

                sign  = dr/r
                test = concx * dp * dw  * sign
                conc_temp += test
            ConcNew[i] = conc_temp
        
        dc = (-ConcNew *vel )
        Conc = Conc + (dc*dt)

        t = t + dt
    print("averenge number of neighbours: ", num/len(Conc))
    return Conc, ConcNew

def main():
    length = 1
    pos_start = 0.2
    pos_end = 0.3
    dp = 0.001
    t_end = 1
    dt =0.01
    h = dp*8
    plot_state = True

    pos = np.arange(0,length,dp)
    vol = length*dp/len(pos)
    vel = 0.01

    # give concs to particles
    conc = np.zeros(len(pos), dtype=float)
    counter = 0
    for i in pos:
        conc[counter] = math.exp(-((i-(length)/2)**2)/(2*0.01))
        counter = counter + 1

    # eularian advection
    concEul = np.zeros(len(pos))
    concEul = copy.deepcopy(conc)
    concEul = eularian(pos, vel, dt, concEul, vol, dp, t_end)

    # lagrangian advection
    concLag = np.zeros(len(pos))
    conc_d = np.zeros(len(pos))
    concLag = copy.deepcopy(conc)
    conc_d = copy.deepcopy(conc)
    # nearest neighbours
    kdt = NN(radius=2*h, algorithm='kd_tree').fit(pos.reshape(-1,1))
    NN_idx = kdt.radius_neighbors(pos.reshape(-1,1))[1]
    concLag, conc_d = lagrangian(pos, vel, dt, concLag, vol, dp, t_end, NN_idx, h)
    
    plt.clf()
    plt.plot(pos,concEul, label='Eul')
    plt.plot(pos,conc,'r^', markevery=20, label='t=0')
    plt.plot(pos,concLag, label='Lag')
    # plt.plot(pos,conc_d, 'g*', markevery=2,label='dc')
    plt.legend()
    # plt.ylim([-0.05,0.05])
    plt.grid()
    plt.savefig("adv1d.png")

if __name__=='__main__':
    main()