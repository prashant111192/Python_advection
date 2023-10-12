# To create a class for handling particle data, you can define a ParticleData class that encapsulates the functionality for reading and processing the data. Here's a basic structure for such a class:

# ```python
import numpy as np

class ParticleData:
    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.data = None
        self.pos = None
        self.idp = None
        self.vel = None
        self.density = None
        self.typeMk = None

    def read_data(self, filename):
        # Read and filter the data
        data = np.loadtxt(self.path_to_data + filename, delimiter=';', skiprows=4, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        mask = data[:, 8] == 3
        self.data = data[mask]
        self.pos = self.data[:, 0:3]
        self.idp = self.data[:, 3]
        self.vel = self.data[:, 4:7]
        self.density = self.data[:, 7]
        self.typeMk = self.data[:, 8:10]

    def get_particle_count(self):
        if self.data is not None:
            return len(self.data)
        else:
            return 0

    def get_particle_position(self, particle_index):
        if self.pos is not None and 0 <= particle_index < len(self.pos):
            return self.pos[particle_index]
        else:
            return None

    def get_particle_velocity(self, particle_index):
        if self.vel is not None and 0 <= particle_index < len(self.vel):
            return self.vel[particle_index]
        else:
            return None

    def get_particle_density(self, particle_index):
        if self.density is not None and 0 <= particle_index < len(self.density):
            return self.density[particle_index]
        else:
            return None

    def get_particle_typeMk(self, particle_index):
        if self.typeMk is not None and 0 <= particle_index < len(self.typeMk):
            return self.typeMk[particle_index]
        else:
            return None

# Usage
path_to_data = "path/to/your/data/"
particle_data_handler = ParticleData('./data/')
particle_data_handler.read_data("that_0297.csv")

# Example: Get the position of the first particle
if particle_data_handler.get_particle_count() > 0:
    first_particle_position = particle_data_handler.get_particle_position()
    print("Position of the first particle:", first_particle_position)

# This class allows you to read and store particle data, as well as retrieve specific information about particles by their index. Adjust the methods as needed to suit your specific data analysis needs.