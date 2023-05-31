import math
import copy
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
import matplotlib.pyplot as plt


#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density



class particle1D:
    x = 0.0       # position in x-axis [m]
    vel = 0.0     # velocity in x direction [m/s]
    C = 0.0       # concentration of tracer [kg/m3]
    C_SPH = 0.0
    def __init__(self, posx, velx, Conc):
        self.x = posx
        self.vel = velx
        self.C = Conc
        self.C_SPH = Conc
    def setPos(self, posx):
        self.x = posx
    def setVel(self, velx):
        self.vel = velx
    def setConc(self, Conc):
        self.C = Conc
    def setConc_SPH(self, Conc):
        self.C_SPH = Conc
    
def dW(r,h):
    alphaD = 5/(8*h*h)
    q = r/h
    return -3*alphaD*q*((1-q/2)**2)

def calcGradC_SPH(particles, gradC_FD, h, NN_idx):
    gradC_FD[:] = 0
    N = len(particles)
    dx = (particles[-1].x - particles[0].x)/(N-1)
    nx = 2*int(h/dx)
    for i in range(N):
        # count_neighbors = 0
        sumWeight = 0.0
        idjMin = int((i - nx)) if (i - nx) > 0 else 0
        idjMax = int((i + nx)) if (i + nx) < N else N
        idJ = np.arange(idjMin,idjMax)
        # print(len(idJ))
        for j in idJ:
        # for j in NN_idx[i]:
            if i == j: continue
            rij = abs(particles[i].x - particles[j].x)
            xij = (particles[i].x - particles[j].x)
            # if rij == 0: continue
            sign = xij/rij
            dw = dW(rij,h)
            sumWeight += dx * sign * dw
            gradC_FD[i] += dx * (particles[j].C - particles[i].C) * sign * dw

def calcDC_SPH(particles, gradC_SPH, dC_SPH, NP):
    for i in range(NP):
        dC_SPH[i] = -particles[i].vel * gradC_SPH[i]

## Constant parameters ===========================
NP = 500
vx = 0.01
xMin = 0.0
xMax = 1.0
CMin = 1.0
CMax = 1.01
CLeft = 0.0
CRight = 1.0
h = 0.016

tMin = 0.0
tMax = 20.0
tNow = tMin
dt = 1e-2
dtLog = 0.25

## Pre-processing ================================
particles = np.empty(NP, dtype=object)
posx = np.linspace(xMin, xMax, NP, dtype=float)
Conc = np.full((NP), CMin, dtype=float)
Conc = normal_dist(posx , 0.5 , 0.08)
Conc_ini = copy.deepcopy(Conc)
for i in range(NP):
    particles[i] = particle1D(posx[i],vx,Conc[i])

kdt = NN(radius=h, algorithm='kd_tree').fit(posx.reshape(-1,1))
NN_idx = kdt.radius_neighbors(posx.reshape(-1,1))[1]
# kdt = NN(radius=h, algorithm='kd_tree').fit(particles.x.reshape(-1,1))
# NN_idx = kdt.radius_neighbors(particles.x.reshape(-1,1))[1]

Ctemp_FD = np.zeros(NP)
Ctemp_SPH = np.zeros(NP)
dC_FD = np.zeros(NP)
dC_SPH = np.zeros(NP)
gradC_FD = np.zeros(NP)
gradC_SPH = np.zeros(NP)
while(tNow<tMax):
    calcGradC_SPH(particles, gradC_SPH, h, NN_idx)
    calcDC_SPH(particles, gradC_SPH, dC_SPH, NP)
    for i in range(NP):
        Ctemp_SPH[i] = particles[i].C_SPH + dt*dC_SPH[i]
    for i in range(NP):
        particles[i].setConc_SPH(Ctemp_SPH[i])
        # particles[i].C_SPH = Ctemp_SPH[i] + dt*dC_SPH[i]
#     for x in range(len(particles)):
#         dC_SPH[x] = -particles[x].vel * gradC_SPH[x]
#         Ctemp_SPH[x] = particles[x].C_SPH + dt*dC_SPH[x]
#         particles[x].C_SPH = Ctemp_SPH[x]
    tNow += dt
    
for x in range(len(particles)):
    Conc[x] = particles[x].C_SPH

    
plt.plot(posx, Conc_ini, 'r', label='initial')
plt.plot(posx, Conc, 'b', label='final')
plt.legend()
plt.show()




