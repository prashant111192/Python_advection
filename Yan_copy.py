import copy
import math
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
    def __init__(self, posx, velx, Conc_s, Conc_f):
        self.x = posx
        self.vel = velx
        self.C = Conc_f
        self.C_SPH = Conc_s
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
NP = 1000
vx = 0.01
xMin = 0.0
xMax = 1.0
CMin = 1.0
CMax = 1.01
CLeft = 0.0
CRight = 1.0
h = 0.016


tMin = 0.0
tMax = 2.0
tNow = tMin
dt = 1e-2
dtLog = 0.25




## Pre-processing ================================
particles = np.empty(NP, dtype=object)
posx = np.linspace(xMin, xMax, NP, dtype=float)
Conc_f = np.full((NP), CMin, dtype=float)
Conc_s = np.full((NP), CMin, dtype=float)
Conc_f_out = np.full((NP), CMin, dtype=float)
Conc_s_out = np.full((NP), CMin, dtype=float)
Conc = np.full((NP), CMin, dtype=float)
Conc_ini = np.full((NP), CMin, dtype=float)
# Conc[:int(NP*0.6)] = CMax
# Conc[:int(NP*0.1)] = CMin
# Conc = 0.5*(np.sin(6*posx)+1)
for i in range(NP):
    Conc[i] = math.exp(-((posx[i]-0.5)**2)/0.005)
    Conc_s[i] = math.exp(-((posx[i]-0.5)**2)/0.005)
    Conc_f[i] = math.exp(-((posx[i]-0.5)**2)/0.005)
    Conc_ini[i] = math.exp(-((posx[i]-0.5)**2)/0.005)
    
# Plot Conc
plt.plot(posx, Conc, 'k', label = 'Initial')
plt.plot(posx, Conc_ini, 'b', label = 'S')
plt.legend()
plt.grid()
plt.savefig("test.png",dpi=300)

# Conc = normal_dist(posx , 0.5 , 0.08)
# Conc_s = normal_dist(posx , 0.5 , 0.08)
# Conc_f = normal_dist(posx , 0.5 , 0.08)
# Conc_ini = copy.deepcopy(Conc_s)
# Conc_ini = normal_dist(posx, 0.5, 0.08)


for i in range(NP):
    v = posx[i]*0.1
    particles[i] = particle1D(posx[i],v,Conc_s[i], Conc_f[i])


Ctemp_FD = np.zeros(NP)
Ctemp_SPH = np.zeros(NP)
dC_FD = np.zeros(NP)
dC_SPH = np.zeros(NP)
gradC_FD = np.zeros(NP)
gradC_SPH = np.zeros(NP)


## Calculation ===================================
while tNow <= tMax:
    tot_mass_before_FD = np.array([part.C for part in particles]).sum()
    tot_mass_before_SPH = np.array([part.C_SPH for part in particles]).sum()
    calcGradC_1st(particles, gradC_FD, CLeft, CRight)
    calcDC_FD(particles, gradC_FD, dC_FD)
    calcGradC_SPH(particles, gradC_SPH, h)
    calcDC_SPH(particles, gradC_SPH, dC_SPH)
    for i in range(NP):
        Ctemp_FD[i] = particles[i].C + dt*dC_FD[i]
        Ctemp_SPH[i] = particles[i].C_SPH + dt*dC_SPH[i]
    tot_mass_after_FD = Ctemp_FD.sum()
    tot_mass_after_SPH = Ctemp_SPH.sum()
    Ctemp_FD = Ctemp_FD * tot_mass_before_FD / tot_mass_after_FD
    Ctemp_SPH = Ctemp_SPH * tot_mass_before_SPH / tot_mass_after_SPH

    for i in range(NP):
        particles[i].setConc(Ctemp_FD[i])
        particles[i].setConc_SPH(Ctemp_SPH[i])

    tNow += dt
    print("t="+"{:.6f}".format(tNow),"FD: "+"{:.6f}".format(np.array([part.C for part in particles]).sum()),"SPH:"+"{:.6f}".format(np.array([part.C_SPH for part in particles]).sum()))
    if abs(dt-(tNow % dtLog)) < 1e-8: #((tNow % dtLog) <= (dtLog/1e2)):
        partPos = np.array([part.x for part in particles])
        partC_FD =  np.array([part.C for part in particles])
        partC_SPH =  np.array([part.C_SPH for part in particles])
    #     # print(partC_FD.sum()/NP)
    #     fig = plt.figure()
        plt.clf()
        plt.plot(partPos, partC_FD, "k:")
        plt.plot(partPos, partC_SPH, "r^", markevery=5, fillstyle='none')
        plt.draw()
        plt.pause(0.01)
    #     # plt.plot(partPos, dC_FD, "k:")
    #     # plt.plot(partPos, dC_SPH, "r^", markevery=20, fillstyle='none')
    #     plt.ylim([-0.05,0.30])
    #     plt.savefig("test_"+ "{:.6f}".format(tNow) +".png",dpi=300)
    #     current_fig = plt.gcf()
    #     plt.close(current_fig)
for x in range(len(particles)):
    Conc_s_out[x] = particles[x].C_SPH
    Conc_f_out[x] = particles[x].C

    
plt.clf()
plt.plot(posx, Conc_ini, 'k', label='Initial')
plt.plot(posx, Conc_f_out, 'b', label='FD')
plt.plot(posx, Conc_s_out, 'r', label='SPH')
plt.legend()
plt.grid()
plt.savefig("test_FD.png",dpi=300)

print("Initial mass: ", np.sum(Conc_ini)/xMax)
print("Difference between initial and final mass (FD): ", np.sum(Conc_ini-Conc_f_out)/xMax)
print("Difference between initial and final mass (SPH): ", np.sum(Conc_ini-Conc_s_out)/xMax)
print("Difference between the peak position of Conc and ConcOrig (FD): ", (np.argmax(Conc_f_out)-np.argmax(Conc_ini))*((xMax-xMin)/NP))
print("Difference between the peak position of Conc and ConcOrig (SPH): ", (np.argmax(Conc_s_out)-np.argmax(Conc_ini))*((xMax-xMin)/NP))
print("Position of the peak of Conc (FD): ", np.argmax(Conc_f_out)*((xMax-xMin)/NP))
print("Position of the peak of Conc (SPH): ", np.argmax(Conc_s_out)*((xMax-xMin)/NP))

