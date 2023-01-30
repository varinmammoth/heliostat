#%%
import numpy as np
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
# %%
def anti_clockwise_rotxy(a):
    return np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0,0,1]])

def rot(beta, gamma):
    """
    beta = rotation around y axis
    gamma = rotation around x axis
    """
    cb = np.cos(beta)
    sb = -np.sin(beta)
    cg = np.cos(gamma)
    sg = -np.sin(gamma)

    return np.array([[cb, sb*sg, sb*cg], [0, cg, -sg], [-sb, cb*sg, cb*cg]])

class ray():
    def __init__(self, p, a):
        """Creates a ray

        Args:
            s (float): length of ray
            p (np.array): starting position vector of ray
            a (np.array): direction vector of ray

        Returns:
            _type_: _description_
        """
        self.p = np.array(p)
        self.a = np.array(a)/np.linalg.norm(a)
        self.s0_ls = {}
        self.history = [self.p]
        self.a_history = [self.a]
        self.absorbed = False #this is set to True when the ray no longer intersects any mirros
        self.finished = False
        return

class mirror():
    def __init__ (self, x, y, z, alpha, beta, a, b, mirror_type='mirror'):
        self.n1 = rot(alpha, beta)@np.array([1,0,0])
        self.n2 = rot(alpha, beta)@np.array([0,1,0])
        self.n3 = rot(alpha, beta)@np.array([0,0,1])
        self.C = np.array([x, y, z])
        self.a = a
        self.b = b
        self.ray_count = 0

        if mirror_type == 'mirror':
            self.mirror_type = 'mirror'
        elif mirror_type == 'absorber':
            self.mirror_type = 'absorber'

        return

class playground():
    def __init__(self):
        self.mirrors = []
        self.rays = []

        self.num_rays = 0
        self.finished_and_absorbed_rays = 0

    def add_rect_mirror(self, x, y, z, beta, gamma, a, b, mirror_type='mirror'):
        self.mirrors.append(mirror(x, y, z, beta, gamma, a, b, mirror_type))
        return

    def add_ray(self, p, a):
        self.rays.append(ray(p, a))
        self.num_rays += 1
        return

    def get_intersections(self):
        for ray in self.rays:
            ray.s0_ls = {}
            for i, mirror in enumerate(self.mirrors):
                s0 = np.dot((mirror.C - ray.p), mirror.n3)/np.dot(ray.a, mirror.n3)
                if s0 in ray.s0_ls:
                    ray.s0_ls[s0].append(i)
                else:
                    ray.s0_ls[s0] = [i] 
        return

    def propagate_rays(self):
        for ray in self.rays:
            if ray.absorbed == False and ray.finished == False:
                got_intersection = False
                ray.s0_ls = collections.OrderedDict(sorted(ray.s0_ls.items()))
                for s0 in ray.s0_ls:
                    if s0 > 1e-8:
                        for mirror_number in ray.s0_ls[s0]:
                            r0 = ray.p + s0*ray.a
                            mirror = self.mirrors[mirror_number]
                            if np.abs(np.dot(r0-mirror.C, mirror.n1)) <= mirror.a/2 and np.abs(np.dot(r0-mirror.C, mirror.n2)) <= mirror.b/2:
                                ray.p = r0
                                ray.a = ray.a - 2*np.dot(ray.a, mirror.n3)*mirror.n3
                                got_intersection = True

                                mirror.ray_count += 1

                                ray.history.append(ray.p)
                                ray.a_history.append(ray.a)

                                if mirror.mirror_type == 'absorber':
                                    ray.absorbed = True
                                    self.finished_and_absorbed_rays += 1

                                break
                        if got_intersection == True:
                            break
                if got_intersection == False:
                    ray.finished = True
                    self.finished_and_absorbed_rays += 1

        return

    def final_propogation(self):
        for ray in self.rays:
            if ray.absorbed == False:
                rfinal = ray.p + 20*ray.a
                ray.history.append(rfinal)

    def simulate(self, N='not-specified'):
        if N != 'not-specified':
            print(f'Simulating for {N} iterations. N_rays = {len(self.rays)}. N_mirrors = {len(self.mirrors)}.')
            print(f'Total number of calculations is O({N*len(self.rays)*len(self.mirrors)}.)')
            for i in tqdm(range(0, N)):
                self.get_intersections()
                self.propagate_rays()
            self.final_propogation()
        else:
            print('N is not specified. Simulating until all rays are either absorbed or no longer intersect.')
            while self.finished_and_absorbed_rays != self.num_rays:
                self.get_intersections()
                self.propagate_rays()
            self.final_propogation()

    def display(self, xlim=[-15,15], ylim=[-15,15], zlim=[-15,15], show_rays=True, show_mirrors=True, show_mirror_normals=True):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # #display mirrors
        if show_mirrors == True:
            for mirror in self.mirrors:
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

        #display mirror normals
        if show_mirror_normals == True:
            for mirror in self.mirrors:
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

        #display rays
        if show_rays == True:
            for ray in self.rays:
                x = []
                y = []
                z = []
                for point in ray.history:
                    x.append(point[0])
                    y.append(point[1])
                    z.append(point[2])
                ax.plot(x, y, z, color='r')

        plt.show()
        return
#%%
