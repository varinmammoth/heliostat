#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba.experimental import jitclass
from numba import int32, int64, float64, boolean, char, typed
from numba.types import Array, List
# %%
@jit
def add_to_dict(keys, values, key, value):
    keys.append(key)
    values.append(value)
    return keys, values

@jit
def rot_vector(v, k, theta):
    return np.cos(theta)*v + np.cross(k, v)*np.sin(theta) + k*np.dot(k, v)*(1-np.cos(theta))

ray_spec = [
    ('p', Array(float64, 1, 'C')),
    ('a', Array(float64, 1, 'C')),
    ('s0_keys', Array(float64, 1, 'C')),
    ('s0_mirrors_index', Array(float64, 1, 'C')),
    ('history_px',  Array(float64, 1, 'C')),
    ('history_py',  Array(float64, 1, 'C')),
    ('history_pz',  Array(float64, 1, 'C')),
    ('history_ax',  Array(float64, 1, 'C')),
    ('history_ay',  Array(float64, 1, 'C')),
    ('history_az',  Array(float64, 1, 'C')),
    ('absorbed', boolean),
    ('finished', boolean)
]

@jit
def generate_ray(p: Array, a: Array, m: int32):
    p = p
    a = a/np.linalg.norm(a)
    s0_keys = np.zeros(m, dtype=np.float64) #numba doesn't support dictionaries, so we specify a list for keys
    s0_mirrors_index = np.zeros(m, dtype=np.int32) #and another list for the values, in this case it is the index of the mirrors in playground.mirrors
    history_px = np.array([p[0]])
    history_py = np.array([p[1]])
    history_pz = np.array([p[2]])
    history_ax = np.array([a[0]])
    history_ay = np.array([a[1]])
    history_az = np.array([a[2]])
    absorbed = False #this is set to True when the ray no longer intersects any mirros
    finished = False
    return [p, a, s0_keys, s0_mirrors_index, history_px, history_py, history_pz, history_ax, history_ay, history_az, absorbed, finished]

mirror_spec = [
    ('n1', Array(float64, 1, 'C')),
    ('n2', Array(float64, 1, 'C')),
    ('n3', Array(float64, 1, 'C')),
    ('C', Array(float64, 1, 'C')),
    ('a', float64),
    ('b', float64),
    ('ray_count', int32),
    ('isReceiver', boolean),
    ('isGround', boolean),
    ('mirror_type', char)
]

@jit
def generate_mirror(x: float64, y: float64, z: float64, phi: float64, theta: float64, a: float64, b: float64, mirror_type='mirror'):
    #Azimutal 
    n3 = np.array([np.cos(phi), -np.sin(phi), 0])
    n1 = np.array([np.sin(phi), np.cos(phi), 0])
    n2 = np.array([0.,0.,1.])

    #Elevation
    n2 = rot_vector(n2, -n1, theta)
    n3 = rot_vector(n3, -n1, theta)

    n1 = n1
    n2 = n2
    n3 = n3

    x = np.float64(x)
    y = np.float64(y)
    z = np.float64(z)

    C = np.array([x, y, z])
    a = np.float64(a)
    b = np.float64(b)
    ray_count = np.float64(0)
    isReceiver = False
    isGround = False

    if mirror_type == 'mirror':
        mirror_type = 'mirror'
    elif mirror_type == 'absorber':
        mirror_type = 'absorber'
    elif mirror_type =='receiver':
        mirror_type = 'absorber'
        isReceiver = True
    elif mirror_type == 'ground':
        mirror_type = 'absorber'
        isGround = True

    return [n1, n2, n3, C, a, b, ray_count, isReceiver, isGround, mirror_type]

playground_spec = [
    ('mirrors', typed.List(mirror_spec)),
    ('rays', typed.List(ray_spec)),
]

@jitclass(playground_spec)
class playground:
    def __init__(self, mirrors, rays):
        self.mirrors = mirrors
        self.rays = rays
        return
# %%
