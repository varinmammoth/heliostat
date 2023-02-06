#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
import datetime as dt

from skyfield import api
ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac
# %%
def initialize_rays(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, start=[100,100,100]):
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

def initialize_rays_parallel(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, phi=np.pi/4, theta=np.pi/4):
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

            start = r + 100*a

            playground.add_ray(start, -a)

    return 

def initialize_rays_parallel_plane(playground, ray_density = 10, center=[0,0,0], a=15, b=15, phi=0, theta=0):
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

    #Gets the azimuth and elevation at different times
    for t in t_ls:
        sun_pos = (earth + location).at(t).observe(sun).apparent()
        elevation, azimuth, distance = sun_pos.altaz()
        phi.append(azimuth)
        theta.append(elevation)

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

    return np.array(t_ls), np.array(t_since_sunrise), np.array(phi), np.array(theta)

def initialise_mirrors(playground, position_ls, a, b, receiver_position, phi, theta, mirror_type='mirror'):
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
        playground.add_rect_mirror(*position_ls[i], n3_phi, n3_theta, a, b, mirror_type=mirror_type)

    return
# %%
test_playground = playground()
position_ls = [[10,0,0], [20,0,0], [-10,0,0], [-20,0,0]]
initialise_mirrors(test_playground, position_ls, 2, 2, [0,0,10], 0, np.pi/4)
test_playground.add_cubic_receiver([0,0,10], 1,1,1)
%matplotlib ipympl
test_playground.display()
# %%
