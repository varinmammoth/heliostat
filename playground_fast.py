#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba.experimental import jitclass
from numba import int32, int64, float64, boolean, char
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

@jitclass(ray_spec)
class ray():
    def __init__(self, p: Array, a: Array, m: int32):
        self.p = p
        self.a = a/np.linalg.norm(a)
        self.s0_keys = np.zeros(m, dtype=np.float64) #numba doesn't support dictionaries, so we specify a list for keys
        self.s0_mirrors_index = np.zeros(m, dtype=np.int32) #and another list for the values, in this case it is the index of the mirrors in playground.mirrors
        self.history_px = np.array([self.p[0]])
        self.history_py = np.array([self.p[1]])
        self.history_pz = np.array([self.p[2]])
        self.history_ax = np.array([self.a[0]])
        self.history_ay = np.array([self.a[1]])
        self.history_az = np.array([self.a[2]])
        self.absorbed = False #this is set to True when the ray no longer intersects any mirros
        self.finished = False
        return

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
# %%
