#%%
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from numba import njit
from numba import typed
from numba import int32, int64, float64
from scipy.interpolate import CubicSpline

from playground_fast import *
from initialization_code_fast import *
# %%

def performance(position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, earth_sun_distance_ls, light_cone=True, N=10):
    #phi_ls and theta_ls are the sun's position we would like to simulate for
    
    power_ls = []

    for i in range(0, len(phi_ls)):
        #First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls,[0,0,15], theta=theta_ls[i], phi=phi_ls[i],  a=mirror_dim[0], b=mirror_dim[1])

        # ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-7.5,7.5], ylim=[-7.5,7.5], ray_density=10, phi=phi_ls[i], theta=theta_ls[i])

        playground1 = playground(mirror_ls, ray_ls)

        playground1.simulate()

        #Return the list of propagated rays
        ray_ls = playground1.rays

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
        
        #if light_cone == True, split the rays into N rays within the lightcone
        if light_cone == True:
            omega_sun = np.pi*(696340000)**2/earth_sun_distance_ls[i]**2
            ray_ls = initialise_rays_cone(ray_ls, 100, omega_sun, len(mirror_ls))
        else:
            ray_ls_new = typed.List()
            for j in ray_ls:
                ray_ls_new.append(ray(j.p, j.a, len(mirror_ls)))
            ray_ls = ray_ls_new

        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        visualize(*playground1.get_history())

        num_rays_receiver = playground1.get_receiver_power() #number of rays received by the receiver

        if light_cone == True:
            #if ray has been split into N rays in a cone, then the power of each ray will be reduced by a factor of N
            power = num_rays_receiver/N 
            power_ls.append(power*np.cos(np.pi/2-theta_ls[i]))
        else:
            power_ls.append(num_rays_receiver*np.cos(np.pi/2-theta_ls[i]))

        print(f'Finished {i+1}/{len(phi_ls)} iterations.')
    
    return power_ls

@jit
def performance_no_cone(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density, ground_length):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length/2

    for i in range(0, len(phi_ls)):
        #First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],  a=mirror_dim[0], b=mirror_dim[1])
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)

        # ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=ray_density, phi=phi_ls[i], theta=theta_ls[i])

        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        power = playground1.get_receiver_power()*np.cos(np.pi/2-theta_ls[i])
        power_ls.append(power)
        output_t_ls.append(t_ls[i])

        mirror_ls = playground1.mirrors
        count_subls = []
        for j in mirror_ls:
            count_subls.append(j.ray_count)
        count_ls.append(count_subls)

        print(f'Iteration: {i+1}/{len(phi_ls)}')

    return output_t_ls, power_ls, count_ls

def ground_power(ray_density, theta_ls, phi_ls, ground_length):
    N_rays = ray_density**2
    ground_lim = ground_length/2

    power_ls = []

    for i in range(0, len(phi_ls)):

        mirror_ls = typed.List()
        ground = mirror(0.,0.,0.,0.,np.pi/2,ground_length,ground_length,'ground')
        mirror_ls.append(ground)
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=ray_density, phi=phi_ls[i], theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        power_ls.append(playground1.mirrors[0].ray_count*np.cos(np.pi/2-theta_ls[i]))

    return power_ls

def ground_power_pred(ray_density, theta_ls):
    """Prediction for total power on the ground.

    Args:
        ray_density (int): Linear ray density
        theta_ls (list): List of theta angles.

    Returns:
        list: Prediction for total power on the ground.
    """
    return (ray_density**2)*np.cos(np.pi/2 - theta_ls)

def mirror_power_pred(ray_density, ground_area, mirror_area, N_mirrors, theta_ls):
    """Prediction for the upperbound to the power received by all mirrors in the system.
    It is the upperbound because this situation corresponds to the mirror being completely
    flat on the floor, i.e. mirror is in same orientation as ground, so the total power is
    just the fraction of the area covered by mirrors to area of ground.

    If the "size" of a mirror is in the same order as the "size" of the receiver, then
    this should roughly give the power received by receiver since all rays would be reflected
    onto the receiver.

    Args:
        ray_density (int): Linear ray density
        ground_area (float): Area of ground
        mirror_area (float): Area of a single mirror
        N_mirrors (int): Total number of mirrors
        theta_ls (list): List of theta angles

    Returns:
        list: Prediction for the upperbound to the power received by all mirrors.
    """
    return (ray_density**2)/(ground_area)*N_mirrors*mirror_area*np.cos(np.pi/2-theta_ls)

def mirror_power_upperbound(ray_density, ground_length, mirror_dim, N_mirrors, theta_ls):
    d = ground_length/ray_density
    output_ls = []
    for theta in theta_ls:
        x = min(float(ray_density), mirror_dim[0]/(d*np.sin(theta)))
        y = min(float(ray_density), mirror_dim[1]/(d*np.sin(theta)))
        output_ls.append(N_mirrors*x*y*np.cos(np.pi/2-theta))
    return np.array(output_ls)
#%%
'''
Test performance and ground_power functions
'''
day = ts.utc(2014, 12, 25) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation
t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, 20)
#%%
ray_density = 100
ground_length = 30
mirror_num_ls = [4,8,16,32,64]
mirror_dim = [1.,1.]
receiver_dim = [1.,1.,1.]
receiver_pos = [0.,0.,15.]

#Performance of same mirror config, but placing the mirrors farther from receiver/from each other
R_list = [6, 7, 10, 12, 14]
power_scenario_ls = []
count_scenario_ls = []

for radius in R_list:
    position_ls = create_circular_positions(radius, mirror_num_ls)
    t_out, power_ls, count_ls = performance_no_cone(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length)
    power_scenario_ls.append(power_ls)
    count_scenario_ls.append(count_ls)
#%%
#Ground power
ground_power_ls = ground_power(ray_density, theta, phi, ground_length)

#Estimate predictions
# from estimate_predictions import *
ground_power_pred_ls = ground_power_pred(ray_density, theta)
mirror_power_estimate_pred = mirror_power_pred(ray_density, ground_length**2, mirror_dim[0]*mirror_dim[1], np.sum(mirror_num_ls), theta)
mirror_power_upperbound_pred = mirror_power_upperbound(ray_density, ground_length, mirror_dim, np.sum(mirror_num_ls), theta)
#%%
#Plot the result
plt.clf()
plt.figure(dpi=800)
t = np.linspace(0, t_sunrise[-1], 40000)
for radius_i, radius in enumerate(R_list):
    plt.plot(t_sunrise, power_scenario_ls[radius_i], '.', color="C{}".format(radius_i))
    func = CubicSpline(t_sunrise, power_scenario_ls[radius_i])
    plt.plot(t, func(t), color="C{}".format(radius_i), label=f'R={radius}')

plt.plot(t_sunrise, ground_power_ls, '.', label='Total power incident on area (Simulated)')
plt.plot(t_sunrise, ground_power_pred_ls, '--', alpha=0.5, label='Total power incident on area (Predicted)')
plt.plot(t_sunrise, mirror_power_estimate_pred, '--', alpha=0.5, label='Estimated power collected by receiver (Predicted)')
# plt.plot(t_sunrise, mirror_power_upperbound_pred, '-.', alpha=0.5, label='Upperbound power collected by receiver (Predicted)')

plt.xlabel('''Time since sunrise (s)

R = Total radius of mirror configuration
''')
plt.ylabel('Power (Arbitrary units)')
plt.legend(bbox_to_anchor =(0.5,-0.63), loc='lower center')

plt.ylim([0,2000])
plt.show()



# %%
def histogram_generator(data, N_points):
    hist, bin_edges = np.histogram(data, bins=N_points)

    bin_locations = []
    for i in range(1, len(bin_edges)):
        bin_locations.append((bin_edges[i-1]+bin_edges[i])/2)

    hist_new = []
    bins_locations_new = []
    for i in range(0, len(bin_locations)):
        if hist[i] != 0:
            hist_new.append(hist[i])
            bins_locations_new.append(bin_locations[i])

    return bins_locations_new, hist_new

mean_ls = []
std_ls = []
for i, count_subls in enumerate(count_ls):
    # x, y = histogram_generator(count_subls[0:-6], 10)
    # plt.bar(x, y, label=f'{t_out[i]}')
    # # plt.hist(count_subls[0:-6], label=f'{t_out[i]}')
    # plt.legend()
    # plt.show()
    mean_ls.append(np.mean(count_subls))
    std_ls.append(np.std(count_subls))
#%%
plt.clf()
plt.figure(dpi=800)
plt.plot(t_sunrise, power_ls, '.', label='Total power at receiver')
plt.plot(t_sunrise, ground_power_ls, '.', label='Total power incident on area')
plt.xlabel('''Time since sunrise (s)
For a given ground area, how well does this configuration perform?
The numbers represent average number of rays reflected by each mirror.
''')

for t in range(0, len(t_out)):
    plt.text(t_sunrise[t]-500, power_ls[t]+100, f'{mean_ls[t]:.1f}')


plt.show()
# %%
k = 9
phi1 = phi[k]
theta1 = theta[k]
position_ls = create_circular_positions(10, [4,8,16,32,50])
mirror_ls = initialise_mirrors_optimal(position_ls, [0,0,25], phi=phi1, theta=theta1, a=1,b=1)
mirror_ls = add_receiver(mirror_ls, [0,0,25])
# mirror_ls.append(mirror(0,0,-5,0,np.pi/2,30,30,'ground'))
ray_ls = initialize_rays_parallel(len(mirror_ls), ray_density=500, xlim=[-20,20], ylim=[-20,20],phi=phi1, theta=theta1)
p = playground(mirror_ls, ray_ls)
p.simulate()
#%%
%matplotlib ipympl
visualize(*p.get_history(), show_rays=True)
# %%
'''
rough model vs simulated data to validate correctness of simualiton
'''