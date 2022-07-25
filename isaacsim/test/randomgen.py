import numpy as np

# White Box Size
x_min = -1
x_max = 1
z_min = -1
z_max = 1

# Robot init position
x_list = [0]
z_list = [0]

norm = 0.1 # Car or Obstacle size

def get_norm1(x1,z1,x2,z2):
    norm = np.abs(x1 - x2) + np.abs(z1 - z2)
    return norm

def get_random(x_list,z_list):
    x_random = np.random.uniform(x_min, x_max)
    z_random = np.random.uniform(z_min, z_max)

    for x,z in zip(x_list,z_list):
        if get_norm1(x,z,x_random,z_random) < norm: 
            x_random, z_random = get_random()
    return x_random, z_random

n = 5
for i in range(n):
    x_random, z_random = get_random(x_list,z_list)
    x_list.append(x_random)
    z_list.append(x_random)

# x_list, z_list <- final obstacle size
print("done")

