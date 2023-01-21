#%%
import numpy as np
import collections
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
        self.p = p
        self.a = a
        self.s0_ls = {}
        self.history = []
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
                ray.s0_ls[s0] = i
        return

    def propagate_rays(self):
        for ray in self.rays:
            ray.history.append(ray.p)
            ray.s0_ls = collections.OrderedDict(sorted(ray.s0_ls.items()))
            for s0 in ray.s0_ls:
                if s0 >= 0:
                    # r0 = ray.p + s0*ray.a
                    r0 = ray.p + ray.a
                    mirror = self.mirrors[ray.s0_ls[s0]]
                    if np.abs(np.dot(r0-mirror.C, mirror.n1)) <= mirror.a/2 and np.abs(np.dot(r0-mirror.C, mirror.n2)) <= mirror.b/2:
                        ray.p = r0
                        ray.a = ray.a - 2*np.dot(ray.a, mirror.n3)*mirror.n3
                        print('yes')
        return
#%%
start1 = np.array([5,2,0])
start2 = np.array([5,4,0])
start3 = np.array([5,-2,0])
start4 = np.array([5,-4,0])

test_playground = playground()
test_playground.add_rect_mirror(*start1, np.pi/4,0,5,5)
test_playground.add_rect_mirror(*start2, np.pi/4,0,5,5)
test_playground.add_rect_mirror(*start3, np.pi/4,0,5,5)
test_playground.add_rect_mirror(*start4, np.pi/4,0,5,5)

end1 = test_playground.mirrors[0].n3 + start1
end2 = test_playground.mirrors[1].n3 + start2
end3 = test_playground.mirrors[2].n3 + start3
end4 = test_playground.mirrors[3].n3 + start4

vector1 = np.append(start1, end1)
vector2 = np.append(start2, end2)
vector3 = np.append(start3, end3)
vector4 = np.append(start4, end4)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

soa = np.array([vector1, vector2, vector3, vector4])

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
plt.show()
# %%
