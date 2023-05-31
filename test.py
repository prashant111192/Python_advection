
import copy
import math
import numpy as np
import matplotlib.pyplot as plt


#Creating a Function.
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density



class particle1D_test:
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
    

def main():
    Conc =1
    Conc_s =1
    Conc_f =1
    posx = 1
    velx = 1

    a = particle1D(posx, velx, Conc_s, Conc_f)
    b = particle1D_test(posx, velx, Conc)

    a.C = a.C+1
    a.C_SPH = a.C_SPH+1
    b.C = b.C+1
    b.C_SPH = b.C_SPH+1

    print(a.C)
    print(a.C_SPH)
    print(b.C)
    print(b.C_SPH)


if __name__ == "__main__":
    main()