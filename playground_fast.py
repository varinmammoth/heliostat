#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba.experimental import jitclass
from numba import int32, int64, float64, boolean, char, typed, typeof
from numba.types import Array, List, string

import datetime as dt

from skyfield import api
ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac

import warnings
warnings.filterwarnings("ignore")

import time

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))
# %%
@jit
def add_to_dict(keys, values, key, value):
    keys.append(key)
    values.append(value)
    return keys, values

@jit
def sort_dict_by_keys(keys, values):
    n = len(keys)
    for i in range(n):
        already_sorted = True

        for j in range(n - i - 1):
            if keys[j] > keys[j + 1]:
                keys[j], keys[j + 1] = keys[j + 1], keys[j]
                values[j], values[j + 1] = values[j + 1], values[j]
                already_sorted = False

        if already_sorted:
            break

    return 

@jit
def rot_vector(v, k, theta):
    return np.cos(theta)*v + np.cross(k, v)*np.sin(theta) + k*np.dot(k, v)*(1-np.cos(theta))

ray_spec = [
    ('p', Array(float64, 1, 'C')),
    ('a', Array(float64, 1, 'C')),
    ('s0_keys', Array(float64, 1, 'C')),
    ('s0_mirrors_index', Array(int64, 1, 'C')),
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
    def __init__(self, p: Array, a: Array, m: int64):
        self.p = p
        self.a = a
        self.s0_keys = np.zeros(m, dtype=np.float64) #numba doesn't support dictionaries, so we specify a list for keys
        self.s0_mirrors_index = np.zeros(m, dtype=np.int64) #and another list for the values, in this case it is the index of the mirrors in playground.mirrors
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
    ('ray_count', float64),
    ('isReceiver', boolean),
    ('isGround', boolean),
    ('mirror_type', string)
]

@jitclass(mirror_spec)
class mirror():
    def __init__(self, x: float64, y: float64, z: float64, phi: float64, theta: float64, a: float64, b: float64, mirror_type='mirror'):
        #Azimutal 
        n3 = np.array([np.cos(phi), -np.sin(phi), 0])
        n1 = np.array([np.sin(phi), np.cos(phi), 0])
        n2 = np.array([0.,0.,1.])

        #Elevation
        n2 = rot_vector(n2, -n1, theta)
        n3 = rot_vector(n3, -n1, theta)

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        x = np.float64(x)
        y = np.float64(y)
        z = np.float64(z)

        self.C = np.array([x, y, z])
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.ray_count = np.float64(0)
        self.isReceiver = False
        self.isGround = False

        if mirror_type == 'mirror':
            self.mirror_type = 'mirror'
        elif mirror_type == 'absorber':
            self.mirror_type = 'absorber'
        elif mirror_type =='receiver':
            self.mirror_type = 'absorber'
            self.isReceiver = True
        elif mirror_type == 'ground':
            self.mirror_type = 'absorber'
            self.isGround = True

        return 

#making template to define playground_spec
template_ray = ray(np.array([1.,1.,1.]), np.array([2.,2.,2.]), 1)
template_ray_ls = typed.List()
template_ray_ls.append(template_ray)
template_mirror = mirror(2,2,2,2,2,2,2, 'mirror')
template_mirror_ls = typed.List()
template_mirror_ls.append(template_mirror)

playground_spec = [
    ('mirrors', typeof(template_mirror_ls)),
    ('rays', typeof(template_ray_ls)),
    ('num_rays', int64),
    ('finished_and_absorbed_rays', int64)
]

@jitclass(playground_spec)
class playground:
    def __init__(self, mirrors, rays):
        self.mirrors = mirrors
        self.rays = rays
        self.num_rays = len(rays)
        self.finished_and_absorbed_rays = 0 
        return

    def get_intersections(self):
        for ray_i in range(0, len(self.rays)):
            ray = self.rays[ray_i]
            for mirror_i in range(0, len(self.mirrors)):
                mirror = self.mirrors[mirror_i]
                s0 = np.dot((mirror.C - ray.p), mirror.n3)/np.dot(ray.a, mirror.n3)
                ray.s0_keys[mirror_i] = s0
                ray.s0_mirrors_index[mirror_i] = mirror_i
        return

    def propagate_rays(self):
        for ray_i in range(0, len(self.rays)):
            ray = self.rays[ray_i]
            if ray.absorbed == False and ray.finished == False:
                got_intersection = False
                sort_dict_by_keys(ray.s0_keys, ray.s0_mirrors_index)
                for s0_i in range(len(ray.s0_keys)):
                    if ray.s0_keys[s0_i] > 1e-8:
                        mirror_i = ray.s0_mirrors_index[s0_i]
                        mirror_i = np.int64(mirror_i)
                        mirror = self.mirrors[mirror_i]
                        r0 = ray.p + ray.s0_keys[s0_i]*ray.a
                        if np.abs(np.dot(r0-mirror.C, mirror.n1)) <= mirror.a/2 and np.abs(np.dot(r0-mirror.C, mirror.n2)) <= mirror.b/2:
                            ray.p = r0
                            ray.a = ray.a - 2*np.dot(ray.a, mirror.n3)*mirror.n3
                            got_intersection = True

                            mirror.ray_count += 1
                            
                            ray.history_px = np.append(ray.history_px, ray.p[0])
                            ray.history_py = np.append(ray.history_py, ray.p[1])
                            ray.history_pz = np.append(ray.history_pz, ray.p[2])
                            ray.history_ax = np.append(ray.history_ax, ray.a[0])
                            ray.history_ay = np.append(ray.history_ay, ray.a[1])
                            ray.history_az = np.append(ray.history_az, ray.a[2])
                            
                            if mirror.mirror_type == 'absorber':
                                ray.absorbed = True
                                self.finished_and_absorbed_rays += 1
                    
                    if got_intersection == True:
                        break

                if got_intersection == False:
                    ray.finished = True
                    self.finished_and_absorbed_rays += 1
        return

    def final_propagation(self):
        for ray_i in range(0, len(self.rays)):
            ray = self.rays[ray_i]
            if ray.absorbed == False:
                rfinal = ray.p + 20*ray.a
                ray.history_px = np.append(ray.history_px, rfinal[0])
                ray.history_py = np.append(ray.history_py, rfinal[1])
                ray.history_pz = np.append(ray.history_pz, rfinal[2])

    def simulate(self):
        i = 0
        while self.finished_and_absorbed_rays != self.num_rays:
            self.get_intersections()
            self.propagate_rays()
            i += 1
        self.final_propagation()
        return
    
    def get_history(self):
        return self.mirrors, self.rays

def visualize(mirror_ls, ray_ls, xlim=[-15,15], ylim=[-15,15], zlim=[-15,15], show_rays=True, show_mirrors=True, show_mirror_normals=True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    if show_mirrors == True:
        for mirror in mirror_ls:
            if mirror.isGround == False:
                P = mirror.C - (mirror.a/2)*mirror.n1 - (mirror.b/2)*mirror.n2
                dy = np.linspace(0, mirror.b, 100)
                x = []
                y = []
                z = []
                Pi = P
                for i in dy:
                    Pi = P + i*mirror.n2
                    x.append(Pi[0])
                    y.append(Pi[1])
                    z.append(Pi[2])
                    vector_end = Pi + mirror.a*mirror.n1
                    x.append(vector_end[0])
                    y.append(vector_end[1])
                    z.append(vector_end[2])
                    ax.plot(x, y, z, color='blue')

    if show_mirror_normals == True:
        for mirror in mirror_ls:
            x = []
            y = []
            z = []
            x.append(mirror.C[0])
            y.append(mirror.C[1])
            z.append(mirror.C[2])
            x.append(x[-1] + mirror.n3[0])
            y.append(y[-1] + mirror.n3[1])
            z.append(z[-1] + mirror.n3[2])
            ax.plot(x, y, z, color='black')

    if show_rays == True:
        for ray in ray_ls:
            x = []
            y = []
            z = []

            history_length = len(ray.history_px)
            for point_i in range(0, history_length):
                x.append(ray.history_px[point_i])
                y.append(ray.history_py[point_i])
                z.append(ray.history_pz[point_i])
            ax.plot(x, y, z, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return

