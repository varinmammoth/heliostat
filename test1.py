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
        self.history = []
        self.a_history = []
        self.finished = False #this is set to True when the ray no longer intersects any mirros
        return

class mirror():
    def __init__ (self, x, y, z, alpha, beta, a, b):
        self.n1 = rot(alpha, beta)@np.array([1,0,0])
        self.n2 = rot(alpha, beta)@np.array([0,1,0])
        self.n3 = rot(alpha, beta)@np.array([0,0,1])
        self.C = np.array([x, y, z])
        self.a = a
        self.b = b
        return

class playground():
    def __init__(self):
        self.mirrors = []
        self.rays = []

        self.finished_rays = 0

    def add_rect_mirror(self, x, y, z, beta, gamma, a, b):
        self.mirrors.append(mirror(x, y, z, beta, gamma, a, b))
        return

    def add_ray(self, p, a):
        self.rays.append(ray(p, a))
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
            got_intersection = False
            ray.history.append(ray.p)
            ray.a_history.append(ray.a)
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
                            break
                    if got_intersection == True:
                        break
            
            # if got_intersection == False:
            #     ray.finished == True
            #     self.finished_rays += 1
        return

    def final_propogation(self):
        for ray in self.rays:
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
            print('N is not specified. Please specify N.')
            # while self.finished_rays != len(self.rays):
            #     self.get_intersections()
            #     self.propagate_rays()

    def display(self, xlim=[-15,15], ylim=[-15,15], zlim=[-15,15]):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        #display rays
        for ray in self.rays:
            x = []
            y = []
            z = []
            for point in ray.history:
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
            ax.plot(x, y, z, color='r')

        # #display mirrors
        # for mirror in self.mirrors:
        #     x_ls = []
        #     y_ls = []
        #     z_ls = []

        #     x = np.linspace(mirror.C[0] - mirror.a, mirror.C[0] + mirror.a, 100)
        #     y = np.linspace(mirror.C[1] - mirror.b, mirror.C[1] + mirror.b, 100)
            
        #     d = np.dot(mirror.n3, mirror.C)

        #     for i in x:
        #         for j in y: 
        #             z = (d - mirror.C[0]*mirror.n2[0] - mirror.C[1]*mirror.n3[1])/mirror.n3[2]
        #             r0 = np.array([x,y,z])
        #             if np.abs(np.dot(r0-mirror.C, mirror.n1)) <= mirror.a/2 and np.abs(np.dot(r0-mirror.C, mirror.n2)) <= mirror.b/2:
        #                 x_ls.append(x)
        #                 y_ls.append(y)
        #                 z_ls.append(z)

        #     ax.plot(x, y, z, color='b')

        #display mirror normals;
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
        plt.show()
        return
#%%
start1 = np.array([1,2,0])
start2 = np.array([1,4,0])
start3 = np.array([1,-2,0])
start4 = np.array([3,-4,0])

test_playground = playground()
test_playground.add_rect_mirror(*start1, np.pi/4, 0,1,1)
# test_playground.add_rect_mirror(*start2, np.pi/4,0,1,1)
# test_playground.add_rect_mirror(*start3, np.pi/4,0,1,1)
# test_playground.add_rect_mirror(*start4, np.pi/4,0,1,1)

start1 = np.array([1,2,5])
# start2 = np.array([1,4,5])
# start3 = np.array([1,-2,5])
# start4 = np.array([3,-4,5])

test_playground.add_rect_mirror(*start1, np.pi/4, 0,1,1)
# test_playground.add_rect_mirror(*start2, np.pi/4,0,1,1)
# test_playground.add_rect_mirror(*start3, np.pi/4,0,1,1)
# test_playground.add_rect_mirror(*start4, np.pi/4,0,1,1)

start1 = [5,2,5]
test_playground.add_rect_mirror(*start1, np.pi/4, 0,1,1)

start1 = [5,2,10]
test_playground.add_rect_mirror(*start1, -np.pi/4, 0,1,1)

test_playground.add_ray([-10, 2, 0], [1,0,0])
# test_playground.add_ray([-10, 4, 0], [1,0,0])
# test_playground.add_ray([-10, -2, 0], [1,0,0])
# test_playground.add_ray([-10, -4, 0], [1,0,0])


test_playground.simulate(5)

test_playground.display()
# %%
test_playground2 = playground()
test_playground2.add_rect_mirror(0, 0, 0, 0, 0, 3, 3)
# test_playground2.add_rect_mirror(0, 0, 21, 0, 0, 1000, 1000)

x = np.arange(-3, 3, 1)
y = np.arange(-3, 3, 1)
for i in y:
    for j in x:
        test_playground2.add_ray([j,i,20], [1,0,-5])

test_playground2.simulate(1000)
test_playground2.display(xlim=[-5,8], ylim=[-5,5], zlim=[0,20])
# %%
