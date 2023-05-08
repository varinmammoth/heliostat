import numpy as np
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import datetime as dt
from skyfield import api
ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac


from mpl_toolkits.mplot3d import Axes3D

"""The following part is the core of the code. It describes the playground. It is from the former Playgroundcode file."""

def rot_vector(v, k, theta):
    return np.cos(theta)*v + np.cross(k, v)*np.sin(theta) + k*np.dot(k, v)*(1-np.cos(theta))

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
    def __init__ (self, x, y, z, phi, theta, a, b, mirror_type='mirror'):
        """A mirror object.
        Args:
            x (float): x coordinate of mirror center
            y (float): y coordinate of mirror center
            z (float): z coordinate of mirror center
            alpha (float): actually beta angle, gotta fix naming convention
            beta (float): actually gamma angle, gotta fix naming convention
            a (float): length of mirror along n1
            b (float): length of mirror along n2
            mirror_type (str, optional): Mirror type. Can be "mirror", "absorber", "receiver", "ground".
                                        "mirror" reflects.
                                        "absorber" absorbs.
                                        "receiver" absorbs, and adds to the total power extracted by the system.
                                        "ground" absorbs, but is not plotted during display.
                                        Defaults to 'mirror'.
        """
        #Azimutal
        n3 = np.array([np.cos(phi), -np.sin(phi), 0])
        n1 = np.array([np.sin(phi), np.cos(phi), 0])
        n2 = np.array([0,0,1])

        #Elevation
        n2 = rot_vector(n2, -n1, theta)
        n3 = rot_vector(n3, -n1, theta)

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.C = np.array([x, y, z])
        self.a = a
        self.b = b
        self.ray_count = 0
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

class playground():
    def __init__(self):
        self.mirrors = []
        self.rays = []

        self.num_rays = 0
        self.finished_and_absorbed_rays = 0

    def add_rect_mirror(self, x, y, z, phi, theta, a, b, mirror_type='mirror'):
        self.mirrors.append(mirror(x, y, z, phi, theta, a, b, mirror_type))
        return

    def add_ray(self, p, a):
        self.rays.append(ray(p, a))
        self.num_rays += 1
        return

    def add_cubic_receiver(self, p, l=5, w=5, h=5):
        #See convention in notes.
        x, y, z = p
        self.add_rect_mirror(x+w/2, y, z, 0, 0, l, h, 'receiver')
        self.add_rect_mirror(x-w/2, y, z, 0, np.pi, l, h, 'receiver')
        self.add_rect_mirror(x, y, z+h/2, 0, np.pi/2, l, w, 'receiver')
        self.add_rect_mirror(x, y, z-h/2, 0, -np.pi/2, l, w, 'receiver')
        self.add_rect_mirror(x, y+l/2, z, -np.pi/2, 0, w, h, 'receiver')
        self.add_rect_mirror(x, y-l/2, z, np.pi/2, 0, w, h, 'receiver')

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
            # print(f'Simulating for {N} iterations. N_rays = {len(self.rays)}. N_mirrors = {len(self.mirrors)}.')
            # print(f'Total number of calculations is O({N*len(self.rays)*len(self.mirrors)}.)')
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

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return

    def get_receiver_power(self):
        #Returns the total number of rays absorbed by the receiver.
        rays_received = 0
        for mirror in self.mirrors:
            if mirror.isReceiver == True:
                rays_received += mirror.ray_count
        return rays_received


"""The following part describes the mathematics of 
 a 3D rotation around any arbitrary axis. This is from the former Matrix.py file."""


def rotation(t, u):
    return np.matrix([[np.cos(t) + u[0]**2 * (1 - np.cos(t)), u[0]*u[1]*(1 - np.cos(t)) - u[2]*np.sin(t), u[0]*u[2]*(1 - np.cos(t)) + u[1]*np.sin(t)],
                    [u[1]*u[0]*(1 - np.cos(t)) + u[2]*np.sin(t), np.cos(t) + u[1]**2 * (1 - np.cos(t)), u[1]*u[2]*(1 - np.cos(t))-u[0]*np.sin(t)],
                    [u[2]*u[0]*(1 - np.cos(t)) - u[1]*np.sin(t), u[2]*u[1]*(1 - np.cos(t)) + u[0]*np.sin(t), np.cos(t) + u[2]**2 * (1 - np.cos(t))]])

#SOURCE: https://www.wikiwand.com/en/Rotation_matrix#Rotation_matrix_from_axis_and_angle

def mirror_normal(input, reflected):

    sun = input/np.linalg.norm(input)
    # direction of Sun, normalized

    receiver = reflected/np.linalg.norm(reflected)
    # direction of receiver, normalized

    rotation_axis = np.cross(receiver, sun)/np.linalg.norm(np.cross(receiver, sun))
    # takes the cross product of the incoming and reflected ray, this defines the rotation axis

    magnitude = np.dot(receiver, sun)
    # as the directionvectprs are normalized, this is the cosine of the inbetween angle

    angle = np.arccos(magnitude)/2
    # we need to divide the angle by 2 (law of reflections)

    mirror_normal = np.array(rotation(angle, rotation_axis)@receiver)[0]

    print('MIRROR NORMAL=', mirror_normal)

    return mirror_normal

"""The following part defines new functions so we can initialize the system. It is the former initialize file."""


# %%
def initialize_rays(playground, xlim=[-10, 10], ylim=[-10, 10], ray_density=10, start=[100, 100, 100]):
    """Initializes rays that originate from a single point. The rays will end up on the ground
    at the coordinates (x,y), where (x,y) are equally spaced points in the range xlim and ylim.
    Args:
        playground (playground obj): Playground object
        xlim (list, optional): x range of ray endpoints on the ground. Defaults to [-10,10].
        ylim (list, optional): y range of ray endpoints on the ground. Defaults to [-10,10].
        ray_density (int, optional): Ray density. e.g. 10 means there will be 10*10 rays in total. Defaults to 10.
        start (list, optional): Starting point of all rays. Defaults to [100,100,100].
    """
    start = np.array(start)

    x = np.linspace(*xlim, ray_density)
    y = np.linspace(*ylim, ray_density)
    x, y = np.meshgrid(x, y)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = np.array([x[i][j], y[i][j], 0])
            a = r - start

            playground.add_ray(start, a)

    return


def initialize_rays_parallel(playground, xlim=[-10, 10], ylim=[-10, 10], ray_density=10, phi=np.pi / 4,
                             theta=np.pi / 4):
    """Initializes paralle rays that end up on coordinates (x,y) on the ground, where (x,y) is
    same as above function. The rays follow along the z_hat vector rotated by beta and gamma,
    where beta and gamma are defined using the convention of this project.
    Args:
        playground (playgound obj): Playground object
        xlim (list, optional): x range. Defaults to [-10,10].
        ylim (list, optional): y ragne. Defaults to [-10,10].
        ray_density (int, optional): Ray density. Defaults to 10.
        beta (float, optional): Beta angle of ray direction. Defaults to np.pi/4.
        gamma (float, optional): Gamma angle of ray direction. Defaults to np.pi/4.
    """
    x = np.linspace(*xlim, ray_density)
    y = np.linspace(*ylim, ray_density)
    x, y = np.meshgrid(x, y)

    n1 = np.array([np.sin(phi), np.cosh(phi), 0])
    a = np.array([np.cos(phi), -np.sin(phi), 0])
    a = rot_vector(a, -n1, theta)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = np.array([x[i][j], y[i][j], 0])

            start = r + 100 * a

            playground.add_ray(start, -a)

    return


def initialize_rays_parallel_plane(playground, ray_density=10, center=[0, 0, 0], a=15, b=15, phi=0, theta=0):
    """Initialise parrallel rays that are equally spaced on a plane. The plane has the normal vector
    defined by the azimutal and elevation angle (same as solar convention). It has dimensions a and b
    along the n1 and n2 directions respectively, initially with the n1 direction along the y-axis and n2 along
    the z-axis.
    Preferably, should choose a = b so that rays are uniformly distributed across the plane.
    Args:
        playground (obj): Playground object
        ray_density (int, optional): Total number of rays = ray_density^2. Defaults to 10.
        center (list, optional): Center of the plane. Defaults to [0,0,0].
        a (int, optional): Dimension of plane along n1. Defaults to 15.
        b (int, optional): Dimension of plane along n2. Defaults to 15.
        phi (float, optional): Azimutal angle. Defaults to 0.
        theta (float, optional): Elevation angle. Defaults to 0.
    """
    center = np.array(center)

    # Azimutal
    n3 = np.array([np.cos(phi), -np.sin(phi), 0])
    n1 = np.array([np.sin(phi), np.cos(phi), 0])
    n2 = np.array([0, 0, 1])

    # Elevation
    n2 = rot_vector(n2, -n1, theta)
    n3 = rot_vector(n3, -n1, theta)

    p = center - (a / 2) * n1 - (b / 2) * n2

    x = np.linspace(0, 1, ray_density)
    y = np.linspace(0, 1, ray_density)
    x, y = np.meshgrid(x, y)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = p + a * x[i][j] * n1 + b * y[i][j] * n2

            start = r + 100 * n3

            playground.add_ray(start, -n3)

    # playground.add_rect_mirror(*center, phi, theta, a, b, 'absorber')
    return


def get_solar_positions(date, lat='48.324777 N', long='11.405610 E', elevation=0, N_datapoints=12):
    """Gets the azimutal and elevation angle of the sun at a particuliar lat, long, and elevation for
    a given date. N_datapoints are given, with the points equally spaced in time between the sunrise and
    sunset times on that date.
    Args:
        date (datetime): Date.
        lat (str, optional): Latitude. Defaults to '48.324777 N'.
        long (str, optional): Longtitudww. Defaults to '11.405610 E'.
        elevation (float, optional): Elevation. Defaults to 0.
        N_datapoints (int, optional): Number of equally spaced (in time) values. Defaults to 12.
    Returns:
        tuple: Tuple of np.arrays. Time in datetime format, time after sunrise, azimutal angle at that time, elevation angle at that time.
    """
    start = date - 1
    end = date + 1
    location = api.Topos(lat, long, elevation_m=elevation)
    t, y = almanac.find_discrete(start, end, almanac.sunrise_sunset(eph, location))  # sunrise and sunset times

    for i in range(0, len(t)):
        if y[i] == 1:
            sunrise = t[i]
            sunset = t[i + 1]
            break

    sun = eph["Sun"]
    earth = eph["Earth"]

    delta_t = (sunset - sunrise) / (N_datapoints - 1)

    t_ls = [sunrise]
    for n in range(1, N_datapoints):
        t_ls.append(t_ls[-1] + delta_t)

    phi = []  # azimuth
    theta = []  # elevation

    # Gets the azimuth and elevation at different times
    for t in t_ls:
        sun_pos = (earth + location).at(t).observe(sun).apparent()
        elevation, azimuth, distance = sun_pos.altaz()
        phi.append(azimuth)
        theta.append(elevation)

    # Converts units
    for i in range(0, len(t_ls)):
        t_ls[i] = t_ls[i].utc_strftime()
        phi[i] = phi[i].radians
        theta[i] = theta[i].radians

    for i in range(0, len(t_ls)):
        t_ls[i] = dt.datetime.strptime(t_ls[i][0:-4], '%Y-%m-%d %H:%M:%S')

    t_since_sunrise = [0]
    for i in range(1, len(t_ls)):
        t_since_sunrise.append((t_ls[i] - t_ls[0]).total_seconds())

    return np.array(t_ls), np.array(t_since_sunrise), np.array(phi), np.array(theta)

"""The function below was Varin's original solution for mirror initialization. 
It has a feature that Zeteny's function still not doing, that is, adding a mirror in the end. 
For this reason it is left here as inspiration how to modify his version until it is done."""


def initialise_mirrors(playground, position_ls, a, b, receiver_position, phi, theta, mirror_type='mirror'):
    position_ls = np.array(position_ls)
    receiver_position = np.array(receiver_position)

    # So what u have to do is find n3 of the mirror,
    # and convert that into azimutal and elevation,
    # because this is what our add mirror function uses.

    for i in range(0, position_ls.size):
        # Original vectors for mirror. n3 points to the sun.
        # Azimutal

        input = np.array([-np.sin(phi), np.cos(phi), np.tan(theta)]) #sun position
        #convention: North points to positive y direction, East towards poisitive x direction

        reflected = receiver_position - position_ls[i]

        n3 = mirror_normal(input, reflected)

        # Check for Nan due to n3 having 0 x-component
        # if n3[0], add a small number to it.
        if n3[0] == 0:
            n3[0] += 1e-5

        # Get azimutal and elevation for n3 vector
        # This is done using usual spherical coordinates.
        # But we must care about the convention

        z_hat = np.array([0, 0, 1])
        y_hat = np.array([0, 1, 0])
        n3_xy = np.array([n3[0], n3[1], 0])/np.linalg.norm(np.array([n3[0], n3[1], 0])) #n3 projection to xy plane, normalized

        n3_theta = np.pi/2 - np.arccos(np.dot(z_hat, n3))
        n3_phi = np.pi - np.arccos(y_hat, n3_xy)

        # Now, add the mirror with these mirrors to playground
        playground.add_rect_mirror(*position_ls[i], n3_phi, n3_theta, a, b, mirror_type=mirror_type)

    return

"""The function below is Zeteny's solution for mirror tracking. It lacks one feature, mentioned above. 
See: the last line of the previous function, and compare it with this below."""

def mirror_normal_calculator(position_ls, receiver_position, phi, theta):
    position_ls = np.array(position_ls)
    receiver_position = np.array(receiver_position)

    input = np.array([np.cos(phi),-np.sin(phi), np.tan(theta)]) #sun position

    #convention: North points to positive y direction, East towards poisitive x direction

    reflected = receiver_position - position_ls #receiver - mirror

    n3 = mirror_normal(input, reflected) #this is the normal of the mirror

    # Check for Nan due to n3 having 0 x-component
    # if n3[0], add a small number to it.
    if n3[0] == 0:
        n3[0] += 1e-5

    x_hat = np.array([1, 0, 0]) #this is where we measure the azimuth angle from. Add a link to an explaining graph.
    n3_xy = np.array([n3[0], n3[1], 0])/np.linalg.norm(np.array([n3[0], n3[1], 0])) #n3 projection to xy plane, normalized


    sign_detector2 = np.cross(n3_xy, x_hat) #takes care of the azimuth angle, as we need to be careful as if it would be bigger than Pi, we need to change it (see below)

    get_n3_theta = np.dot(n3_xy, n3) #elevation of the mirror normal to the ground, using the angle between its xy projection and itself

    if np.abs(get_n3_theta) <= 1e-3:
        get_n3_theta += 1e-3 #to avoid nan values

    n3_theta = np.arccos(get_n3_theta) #the actual elevation

    get_n3_phi = np.dot(x_hat, n3_xy)#get the azimuth angle

    if np.abs(get_n3_phi) <= 1e-5:
        get_n3_phi += 1e-5

    n3_phi = np.arccos(get_n3_phi)

    if sign_detector2[2] > 0:
        n3_phi = n3_phi
    else:
        n3_phi = 2 * np.pi - n3_phi


    if np.abs(n3_theta) <=  1e-5:
        n3_theta += 1e-5


    # Now, add the mirror with these mirrors to playground
    #playground.add_rect_mirror(*position_ls[i], n3_phi, n3_theta, a, b, mirror_type=mirror_type)

    return n3_phi, n3_theta

""""This code below is just a test example so u can play with the code and see what it is doing."""

day = ts.utc(2023, 2, 9) #GIVE ANY DATE HERE. THIS WILL BE THE DAY YOU SIMULATE.

lat = '47.3167 N' #GIVE ANY LATITUDE AND LONGITUDE ON THE GLOBE. THE SUNRAYS WILL HAVE THE ANGLES DEFINED BY THIS LOCATON, AND THE DATE ABOVE.
long = '18.9114 E'

elevation = 1.5 #Elevation above sea level. Unit is not known yet (Varin?)

n = 5 # Divide the time spent in daylight on the given day to n equal timesteps. It will then generate these events.

t_ls, t_since_sunrise, phi_ls, sun_theta_ls = get_solar_positions(day, lat, long, elevation, n) #gives you the data you need


for i in range(0, n):


    print('CYCLE =', i)

    receiver_phi = 3*np.pi/2 #receiver position
    receiver_r = 40

    receiver = receiver_r * np.array([np.cos(receiver_phi), -np.sin(receiver_phi), 0.5]) #0 -15 15 surely works, actually negative x works, positive x fixed


    test_playground = playground()

    position_ls = np.array([0, 0, 0]) #mirror center #1 4 2 surely works

    input = np.array([np.cos(phi_ls[i]), -np.sin(phi_ls[i]), np.tan(sun_theta_ls[i])])

    reflected = (receiver-position_ls) #reflected

    ray_count_ls = []

    test_playground = playground()
    initialize_rays_parallel_plane(test_playground, 5, [0,0,0], 15, 15, phi_ls[i], sun_theta_ls[i])

    normal_phi = mirror_normal_calculator(position_ls, receiver, phi_ls[i], sun_theta_ls[i])[0]
    normal_theta =  mirror_normal_calculator(position_ls, receiver, phi_ls[i], sun_theta_ls[i])[1]


    test_playground.add_cubic_receiver(receiver, 5, 5, 5)
    test_playground.add_rect_mirror(position_ls[0], position_ls[1], position_ls[2], normal_phi,normal_theta,15,15,mirror_type='mirror')
    test_playground.simulate()
    test_playground.display(xlim=[-50,50], ylim=[-50,50], zlim=[0,45], show_mirrors=True, show_mirror_normals=True)
    #ray_count_ls.append(test_playground.mirrors[0].ray_count)