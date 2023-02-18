#%%
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from numba import typed

from playground_fast import *
from initialization_code_fast import *
# %%

def performance(position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, earth_sun_distance_ls, light_cone=True, N=10):
    #phi_ls and theta_ls are the sun's position we would like to simulate for
    
    power_ls = []

    for i in range(0, len(phi_ls)):
        #First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls,[0,0,15], theta=theta_ls[i], phi=phi_ls[i],  a=mirror_dim[0], b=mirror_dim[1])
        print('1')
        ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        print('2')
        playground1 = playground(mirror_ls, ray_ls)
        print('3')
        playground1.simulate()
        print('4')
        #Return the list of propagated rays
        ray_ls = playground1.rays
        print('5')
        #if light_cone == True, split the rays into N rays within the lightcone
        if light_cone == True:
            omega_sun = np.pi*(696340000)**2/earth_sun_distance_ls[i]**2
            ray_ls = initialise_rays_cone(ray_ls, 100, omega_sun)

        mirror_ls = typed.List() #reset mirror_ls to now only include the receiver
        mirror_ls = add_receiver(mirror_ls, receiver_pos)
        print('6')
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()
        print('7')

        num_rays_receiver = playground1.get_receiver_power() #number of rays received by the receiver
        print('8')
        if light_cone == True:
            #if ray has been split into N rays in a cone, then the power of each ray will be reduced by a factor of N
            power = num_rays_receiver/N 
            power_ls.append(power)
        else:
            power_ls.append(num_rays_receiver)

        print(f'Finished {i}/{len(phi_ls)} iterations.')
    
    return power_ls


def ground_power(ground_dim, theta_ls, phi_ls):
    
    power_ls = []

    for i in range(0, len(phi_ls)):
        ground = mirror(0, 0, 0, 0, np.pi/2, *ground_dim, 'receiver')
        mirror_ls = typed.List()
        mirror_ls.append(ground)
        ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        power = playground1.get_receiver_power()
        power_ls.append(power)

    return power_ls
#%%
'''
Test performance and ground_power functions
'''
day = ts.utc(2014, 12, 19) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation
t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, 3)
#%%
position_ls = create_circular_positions(10, [4,5,6])
power_ls = performance(position_ls, [0,0,10],[4,4],[3,3,3],phi,theta,distance, light_cone=False)
#%%
ground_power_ls = ground_power([15,15], theta, phi)
# %%

# %%
