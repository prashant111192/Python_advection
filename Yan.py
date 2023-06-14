import math
import numpy as np
import matplotlib.pyplot as plt


#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = 0.1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean)/sd)**2)
    prob_density = prob_density/np.max(prob_density)
    # prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)

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


def calcDC_FD(particles, gradC_FD, dC_FD):
    for i in range(NP):
        dC_FD[i] = -particles[i].vel * gradC_FD[i]


# >>> 1st-order upwind scheme
def calcGradC_1st(particles, gradC_FD, CLeft, CRight):
    gradC_FD[0] = (particles[0].C - CLeft)/(particles[1].x-particles[0].x)
    for i in range(1,NP):
        gradC_FD[i] = (particles[i].C - particles[i-1].C)/(particles[i].x-particles[i-1].x)


# >>> 2nd-order upwind scheme
def calcGradC_2nd(particles, gradC_FD, CLeft, CRight):
    gradC_FD[0] = (3*particles[0].C - 3*CLeft)/(2*(particles[1].x-particles[0].x))
    gradC_FD[1] = (3*particles[1].C - 4*particles[0].C + CLeft)/(2*(particles[1].x-particles[0].x))
    for i in range(2,NP):
        gradC_FD[i] = (3*particles[i].C - 4*particles[i-1].C + particles[i-2].C)/(2*(particles[1].x-particles[0].x))


# >>> SPH Kernel Function
def W(r,h):
    alphaD = 5/(8*h)
    q = r/h
    return alphaD*((1-q/2)**3)*(1.5*q+1)


# >>> SPH Kernel Function Derivative
def dW(r,h):
    alphaD = 5/(8*h*h)
    q = r/h
    return -3*alphaD*q*((1-q/2)**2)


# >>>
def calcGradC_SPH(particles, gradC_FD, h):
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
            if i == j: continue
            rij = abs(particles[i].x - particles[j].x)
            xij = (particles[i].x - particles[j].x)
            sign = xij/rij
            dw = dW(rij,h)
            sumWeight += dx * sign * dw
            gradC_FD[i] += dx * (particles[j].C_SPH - particles[i].C_SPH) * sign * dw


# >>>
def calcDC_SPH(particles, gradC_SPH, dC_SPH):
    for i in range(NP):
        dC_SPH[i] = -particles[i].vel * gradC_SPH[i]


## Constant parameters ===========================
NP = 500
vx = 0.01
xMin = 0.0
xMax = 2.0
CMin = 1.0
CMax = 1.01
CLeft = 0.0
CRight = 1.0
h = 0.016


tMin = 0.0
tMax = 1
# tMax = 20.0
tNow = tMin
dt = 1e-2
dtLog = 0.25




## Pre-processing ================================
particles = np.empty(NP, dtype=object)
posx = np.linspace(xMin, xMax, NP, dtype=float)
Conc = np.full((NP), CMin, dtype=float)
Conc = np.full((NP), CMin, dtype=float)
# Conc[:int(NP*0.6)] = CMax
# Conc[:int(NP*0.1)] = CMin
# Conc = 0.5*(np.sin(6*posx)+1)
for i in range(NP):
    Conc[i] = math.exp(-((posx[i]-0.5)**2)/0.005)
# Conc = normal_dist(posx , 0.5 , 0.1)


for i in range(NP):

    v = posx[i]*0.01
    particles[i] = particle1D(posx[i],v,Conc[i])
    # particles[i] = particle1D(posx[i],vx,Conc[i])

# Plot the velocity of the particles
vel_plot = np.array([part.vel for part in particles])
plt.plot(posx,vel_plot)
plt.show()



Ctemp_FD = np.zeros(NP)
Ctemp_SPH = np.zeros(NP)
dC_FD = np.zeros(NP)
dC_SPH = np.zeros(NP)
gradC_FD = np.zeros(NP)
gradC_SPH = np.zeros(NP)


## Calculation ===================================
while tNow <= tMax:
    calcGradC_1st(particles, gradC_FD, CLeft, CRight)
    calcDC_FD(particles, gradC_FD, dC_FD)
    calcGradC_SPH(particles, gradC_SPH, h)
    calcDC_SPH(particles, gradC_SPH, dC_SPH)
    for i in range(NP):
        Ctemp_FD[i] = particles[i].C + dt*dC_FD[i]
        Ctemp_SPH[i] = particles[i].C_SPH + dt*dC_SPH[i]
    for i in range(NP):
        x = 0
        # particles[i].setConc(x)
        particles[i].setConc(Ctemp_FD[i])
        particles[i].setConc_SPH(Ctemp_SPH[i])
    tNow += dt
    print("t="+"{:.6f}".format(tNow),"FD: "+"{:.6f}".format(np.array([part.C for part in particles]).sum()),"SPH:"+"{:.6f}".format(np.array([part.C_SPH for part in particles]).sum()))
    if abs(dt-(tNow % dtLog)) < 1e-8: #((tNow % dtLog) <= (dtLog/1e2)):
        partPos = np.array([part.x for part in particles])
        partC_FD =  np.array([part.C for part in particles])
        partC_SPH =  np.array([part.C_SPH for part in particles])
        # print(partC_FD.sum()/NP)
        plt.clf()
        plt.plot(partPos, partC_FD, "k:")
        plt.plot(partPos, partC_SPH, "r^", markevery=5, fillstyle='none')
        # plt.plot(partPos, dC_FD, "k:")
        plt.plot(partPos, dC_SPH, "r^", markevery=20, fillstyle='none')
        # plt.ylim([-0.05,0.30])
        plt.draw()
        plt.pause(0.01)
        # plt.savefig("test_"+ "{:.6f}".format(tNow) +".png",dpi=300)
        # current_fig = plt.gcf()
        # plt.close(current_fig)

# make a new array with final fd concentration and position and save it to a csv
partPos = np.array([part.x for part in particles])
partC_FD =  np.array([part.C for part in particles])
partC_SPH =  np.array([part.C_SPH for part in particles])
np.savetxt("final_concentration.csv", np.column_stack((partPos, partC_FD, partC_SPH)), delimiter=",", header="Position, FD Concentration, SPH Concentration")


# plot the final concentration
plt.clf()
plt.plot(partPos, partC_FD, "k:")
plt.plot(partPos, partC_SPH, "r^", markevery=5, fillstyle='none')
plt.plot()
plt.grid()
plt.show()

