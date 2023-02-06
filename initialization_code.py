#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
from pysolar.solar import *
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
# %%

# %%
