#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import typed
from numba import int32, int64, float64

from playground_fast import ray
# %%
def rot_vector(v, k, theta):
    return np.cos(theta)*v + np.cross(k, v)*np.sin(theta) + k*np.dot(k, v)*(1-np.cos(theta))

def initialize_rays_parallel_plane_fast(m, ray_density = 10, center=[0,0,0], a=15, b=15, phi=0, theta=0):
    
    ray_ls = typed.List()
    
    center = np.array(center)
    
    #Azimutal 
    n3 = np.array([np.cos(phi), -np.sin(phi), 0])
    n1 = np.array([np.sin(phi), np.cos(phi), 0])
    n2 = np.array([0,0,1])

    #Elevation
    n2 = rot_vector(n2, -n1, theta)
    n3 = rot_vector(n3, -n1, theta)

    p = center - (a/2)*n1 - (b/2)*n2

    x = np.linspace(0,1,ray_density)
    y = np.linspace(0,1,ray_density)
    x, y = np.meshgrid(x, y)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = p + a*x[i][j]*n1 + b*y[i][j]*n2

            start = r + 100*n3

            start = start.astype(np.float64)
            n3 = n3.astype(np.float64)

            ray_ls.append(ray(start, -n3, m))

    return ray_ls