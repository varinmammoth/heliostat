#%%
import numpy as np
import collections

# %%
def anti_clockwise_rot(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])

def get_phi_hat(phi):
    """Gets the phi hat unit vector.

    Args:
        phi (float): The angle phi between x axis and R_hat.

    Returns:
        float: Phi hat unit vector.
    """
    return np.array([-np.sin(phi), np.cos(phi), 0])
# %%
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
    def __init__ (self, R, phi, h, alpha, beta, a, b):
        """The user specifies the position of the mirror through R and phi.
        The user specifies the orientation of the mirror through alpha and beta.
        The user specifices the boundaries of the mirror through as and b.

        Args:
            R (float): R vector.
            phi (float): phi angle
            alpha (float): alpha angle
            beta (float): beta angle
            a (float): length in xy direction
            b (float): length in the other direction

        Returns:
            array: [R, phi, n1, n2, n3, a, b]: All information about the mirror.
        """
        self.n1 = anti_clockwise_rot(alpha)@get_phi_hat(phi)
        self.n3 = anti_clockwise_rot(beta)@np.array([0,0,1])
        self.n2 = np.cross(self.n1, self.n3)
        self.C = np.array([R, phi, h])
        self.a = a
        self.b = b
        return

class playground():
    def __init__(self):
        self.mirrors = []
        self.rays = []

    def add_rect_mirror(self, R, phi, h, alpha, beta, a, b):
        self.mirrors.append(mirror(R, phi, h, alpha, beta, a, b))
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
                    r0 = ray.p + s0*ray.a
                    mirror = self.mirrors[self.rays_ls[s0]]
                    if np.abs(np.dot(r0-mirror.C, mirror.n1)) <= mirror.a/2 and np.abs(np.dot(r0-mirror.C, mirror.n2)) <= mirror.b/2:
                        ray.p = r0
                        ray.a = ray.a - 2*np.dot(ray.a, mirror.n3)*mirror.n3
        return
#%%
test_playground = playground()
test_playground.add_rect_mirror(5,0,0,5,5)
test_playground.add_ray([-100,0,0])
test_playground.propagate_rays()



    

    

    
# %%
