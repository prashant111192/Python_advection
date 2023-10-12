import numpy as np

def main():
    print("Hello World!")
    particle_numbers = 100

    # Initialize the position of the particles
    pos = np.ones((3, particle_numbers))
    pos[0] = np.random.rand(particle_numbers)
    pos[1] = np.random.rand(particle_numbers)
    pos[2] = np.random.rand(particle_numbers)

    # Initialize the velocity of the particles
    vel = np.zeros((3, particle_numbers))
    vel[0] = np.random.rand(particle_numbers)
    vel[1] = np.random.rand(particle_numbers)
    vel[2] = np.random.rand(particle_numbers)

    # Initialize the pressure of the particles
    pressure = np.rand(particle_numbers)


if __name__ == "__main__":
    main()
