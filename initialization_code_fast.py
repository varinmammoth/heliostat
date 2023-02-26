#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import typed
from numba import int32, int64, float64

from playground_fast import *

import time

from skyfield import api
ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac

from tqdm import tqdm
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

    n3 = n3.astype(np.float64)
    # n3 = -1*n3

    p = center - (a/2)*n1 - (b/2)*n2

    x = np.linspace(0,1,ray_density)
    y = np.linspace(0,1,ray_density)
    x, y = np.meshgrid(x, y)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = p + a*x[i][j]*n1 + b*y[i][j]*n2

            start = r + 100*n3

            # start = start.astype(np.float64)

            ray_ls.append(ray(start, -n3, m))

    return ray_ls

def initialize_rays_parallel(m, xlim=[-10,10], ylim=[-10,10], ray_density = 10, phi=np.pi/4, theta=np.pi/4):
    mirror_ls = typed.List()

    x = np.linspace(*xlim, ray_density)
    y = np.linspace(*ylim, ray_density)
    x, y = np.meshgrid(x, y)

    a = np.array([np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)])

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = np.array([x[i][j], y[i][j], 0])

            start = r + 100*a

            mirror_ls.append(ray(start, -a, m))

    return mirror_ls

def initialise_mirrors(position_ls, a, b, receiver_position, phi, theta, mirror_type='mirror'):
    
    mirror_ls = typed.List()
    
    position_ls = np.array(position_ls)
    receiver_position = np.array(receiver_position)

    for i in range (0, len(position_ls)):
        #Original vectors for mirror. n3 points to the sun.
        #Azimutal
        n3 = np.array([np.cos(phi), -np.sin(phi), 0])
        n1 = np.array([np.sin(phi), np.cos(phi), 0])

        #Elevation
        n3 = rot_vector(n3, -n1, theta)
    
        # The vector x that points from mirror center to receiver
        x = receiver_position - position_ls[i]
        x = x/np.linalg.norm(x)
        # #The cross product of n3 and x
        N = np.cross(n3, x)
        # #The angle alpha between n3 and x
        alpha = np.arccos(np.dot(n3, x))
        
        # #Now, apply Rodriguez's formula to rotate n3 by amount alpha/2 towards the reciever,
        # # ie. rotate about N in clockwise fashion    
        n3 = rot_vector(n3, N, alpha/2)

        #Check for Nan due to n3 having 0 x-component
        #if n3[0], add a small number to it.
        if n3[0] == 0:
            n3[0] += 1e-5
        
        # Get azimutal and elevation for n3 vector
        # This is done using usual spherical coordinates.
        # But we must care about the convention
        n3_theta = np.pi/2 - np.arctan(np.sqrt(n3[0]**2 + n3[1]**2)/n3[2])
        n3_phi = -np.arctan(n3[1]/n3[0])

        #Now, add the mirror with these mirrors to playground
        mirror_ls.append(mirror(*position_ls[i], n3_phi, n3_theta, a, b, mirror_type=mirror_type))

    return mirror_ls

def add_receiver(mirror_ls, p, l=5, w=5, h=5):
    x, y, z = p
    mirror_ls.append(mirror(x+w/2, y, z, 0, 0, l, h, 'receiver'))
    mirror_ls.append(mirror(x-w/2, y, z, 0, np.pi, l, h, 'receiver'))
    mirror_ls.append(mirror(x, y, z+h/2, 0, np.pi/2, l, w, 'receiver'))
    mirror_ls.append(mirror(x, y, z-h/2, 0, -np.pi/2, l, w, 'receiver'))
    mirror_ls.append(mirror(x, y+l/2, z, -np.pi/2, 0, w, h, 'receiver'))
    mirror_ls.append(mirror(x, y-l/2, z, np.pi/2, 0, w, h, 'receiver'))
    return mirror_ls

def create_circular_positions(R, num_mirrors_ls):
    position_ls = []
    dr = R/len(num_mirrors_ls)
    for layer in range(0, len(num_mirrors_ls)):
        r = (layer+1)*dr
        d_angle = 2*np.pi/num_mirrors_ls[layer]
        for i in range(0, num_mirrors_ls[layer]):
            x = r*np.cos(i*d_angle)
            y = r*np.sin(i*d_angle)
            position_ls.append([x, y, 5])
    return position_ls

def initialise_mirrors_optimal(position_ls, receiver_position, phi, theta, a=4, b=4):
    def rotation(t, u):
        return np.matrix([[np.cos(t) + u[0]**2 * (1 - np.cos(t)), u[0]*u[1]*(1 - np.cos(t)) - u[2]*np.sin(t), u[0]*u[2]*(1 - np.cos(t)) + u[1]*np.sin(t)],
                        [u[1]*u[0]*(1 - np.cos(t)) + u[2]*np.sin(t), np.cos(t) + u[1]**2 * (1 - np.cos(t)), u[1]*u[2]*(1 - np.cos(t))-u[0]*np.sin(t)],
                        [u[2]*u[0]*(1 - np.cos(t)) - u[1]*np.sin(t), u[2]*u[1]*(1 - np.cos(t)) + u[0]*np.sin(t), np.cos(t) + u[2]**2 * (1 - np.cos(t))]])

    def mirror_normal(input, reflected):
        sun = input/np.linalg.norm(input) # direction of Sun, normalized
        receiver = reflected/np.linalg.norm(reflected) # direction of receiver, normalized
        rotation_axis = np.cross(receiver, sun)/np.linalg.norm(np.cross(receiver, sun)) # takes the cross product of the incoming and reflected ray, this defines the rotation axis
        magnitude = np.dot(receiver, sun) # as the directionvectprs are normalized, this is the cosine of the inbetween angle
        angle = np.arccos(magnitude)/2 # we need to divide the angle by 2 (law of reflections)
        mirror_normal = np.array(rotation(angle, rotation_axis)@receiver)[0]
        return mirror_normal
    
    mirror_ls = typed.List()

    receiver_position = np.array(receiver_position, dtype=np.float64)
    position_ls = np.array(position_ls, dtype=np.float64)

    input = np.array([np.cos(phi),-np.sin(phi), np.tan(theta)]) #sun position

    for i in range(0, len(position_ls)):
        reflected = receiver_position - position_ls[i]

        n3 = mirror_normal(input, reflected) #this is the normal of the mirror

        # Check for Nan due to n3 having 0 x-component, if n3[0], add a small number to it.
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

        mirror_ls.append(mirror(*position_ls[i], n3_phi, n3_theta, a, b))
    
    return mirror_ls

def initialise_rays_cone(rays_ls_old, N, omega_sun, m):
    rays_ls_new = typed.List()

    for old_ray in rays_ls_old:
        for i in range(0, N):
            p = old_ray.p
            
            a = old_ray.a
            a0 = np.tan(omega_sun/2)
            z = (a[0]+a[1])/a[2]
            x = np.array([-1.,-1., z], dtype=np.float64)
            
            alpha = np.random.rand()
            beta = 2*np.pi*np.random.rand()

            A = alpha*a0*x/np.linalg.norm(x)
            A = rot_vector(A, a, beta)

            a_cone = a + A
            a_cone = a_cone/np.linalg.norm(a_cone)

            rays_ls_new.append(ray(p, a_cone, m))

    return rays_ls_new

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
    t, y = almanac.find_discrete(start, end, almanac.sunrise_sunset(eph, location)) #sunrise and sunset times
    
    for i in range(0, len(t)):
        if y[i] == 1:
            sunrise = t[i]
            sunset = t[i+1]
            break

    sun = eph["Sun"]
    earth = eph["Earth"]

    delta_t = (sunset - sunrise)/(N_datapoints-1)

    t_ls = [sunrise]
    for n in range(1, N_datapoints):
        t_ls.append(t_ls[-1] + delta_t)

    phi = [] #azimuth
    theta = [] #elevation
    distance_ls = []

    #Gets the azimuth and elevation at different times
    for t in t_ls:
        sun_pos = (earth + location).at(t).observe(sun).apparent()
        elevation, azimuth, distance = sun_pos.altaz()
        phi.append(azimuth)
        theta.append(elevation)

        d = earth.at(t).observe(sun).apparent().distance()
        distance_ls.append(d.m)

    #Converts units
    for i in range(0, len(t_ls)):
        t_ls[i] = t_ls[i].utc_strftime()
        phi[i] = phi[i].radians
        theta[i] = theta[i].radians

    for i in range(0, len(t_ls)):
        t_ls[i] = dt.datetime.strptime(t_ls[i][0:-4], '%Y-%m-%d %H:%M:%S')

    t_since_sunrise = [0]
    for i in range(1, len(t_ls)):
        t_since_sunrise.append((t_ls[i]-t_ls[0]).total_seconds())

    return np.array(t_ls), np.array(t_since_sunrise), np.array(phi), np.array(theta), np.array(distance_ls)
# %%
