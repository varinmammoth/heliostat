#%%
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from numba import njit
from numba import typed
from numba import int32, int64, float64
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint, quad
from scipy.stats import chisquare

from playground_fast import *
from initialization_code_fast import *

plt.rc('font', size=13)
plt.rc('axes', labelsize=13)
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

def performance_no_cone_2step(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density, ground_length):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length/2

    for i in range(0, len(phi_ls)):

        #First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],  a=mirror_dim[0], b=mirror_dim[1])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=ray_density, phi=phi_ls[i], theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
        ray_ls_new = typed.List()
        for old_ray in playground1.rays:
            ray_ls_new.append(ray(old_ray.p, old_ray.a, len(mirror_ls)))

        playground1 = playground(mirror_ls, ray_ls_new)
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

def performance_cone_2step(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density, ground_length, N_raycone):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length/2

    for i in range(0, len(phi_ls)):

        #First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],  a=mirror_dim[0], b=mirror_dim[1])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=ray_density, phi=phi_ls[i], theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
        ray_ls_new = typed.List()
        for old_ray in playground1.rays:
            ray_ls_new.append(ray(old_ray.p, old_ray.a, len(mirror_ls)))
        sun_angle = 0.533*np.pi/180
        ray_ls_new = initialise_rays_cone(ray_ls_new, N_raycone, sun_angle, len(mirror_ls))

        playground1 = playground(mirror_ls, ray_ls_new)
        playground1.simulate()

        power = playground1.get_receiver_power()*np.cos(np.pi/2-theta_ls[i])/N_raycone
        power_ls.append(power)
        output_t_ls.append(t_ls[i])

        mirror_ls = playground1.mirrors
        count_subls = []
        for j in mirror_ls:
            count_subls.append(j.ray_count)
        count_ls.append(count_subls)

        print(f'Iteration: {i+1}/{len(phi_ls)}')
    
    return output_t_ls, power_ls, count_ls

def get_T(t, P, Tc, alpha=1, c=1, k=1):
    #Interpolate the power
    P_func_spline = CubicSpline(t, P)
    #Due to cubic spline, we may get negative values, set these to 0
    def P_func(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(0, P_func_spline(i)))
            return np.array(output)
        else:
            return max(0, P_func_spline(t))
    
    #Define dT/dt
    dTdt = lambda T, t: (alpha/c)*P_func(t) - k*(T-Tc)

    #Solve the ODE
    t_ls = np.linspace(t[0], t[-1], 100) 
    T0 = Tc
    T = odeint(dTdt, T0, t_ls)
    
    #Interpolate the ODE solution
    T_func_spline = CubicSpline(t_ls, T)
    #Due to cubic spline, we may get T < Tc, set these to 0
    def T_func(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(Tc, T_func_spline(i)))
            return np.array(output)
        else:
            return max(Tc, T_func_spline(t))
        
    #Intergrate to get average temperature
    avg_T = quad(T_func, t_ls[0], t_ls[-1])[0]
    avg_T /= (t_ls[-1]-t_ls[0])

    # #Uncertainty in T
    frac = 0.05
    T_err_func = lambda t: (alpha*frac)/(c*k)
    
    return avg_T, T_func, T_err_func

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
#%%
'''
Test performance and ground_power functions
'''
day = ts.utc(2014, 12, 25) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation
t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, 20)

for i, val in enumerate(theta):
    if val < 0:
        theta[i] = 0
#%%
ray_density = 150
ground_length = 30
mirror_num_ls = [4,8,16,32,64]
mirror_dim = [1.,1.]
receiver_dim = [1.,1.,1.]
receiver_pos = [0.,0.,15.]

#Performance of same mirror config, but placing the mirrors farther from receiver/from each other
# R_list = [6, 7, 10, 12, 14]
R_list = [10]
power_scenario_ls = []
count_scenario_ls = []

start = time.time()
for radius in R_list:
    position_ls = create_circular_positions(radius, mirror_num_ls)
    t_out, power_ls, count_ls = performance_no_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length)
    power_scenario_ls.append(power_ls)
    count_scenario_ls.append(count_ls)
end = time.time()
print(f'Elapsed time: {end-start}')
#%%
#Ground power
ground_power_ls = ground_power(ray_density, theta, phi, ground_length)

#Estimate predictions
# from estimate_predictions import *
ground_power_pred_ls = ground_power_pred(ray_density, theta)
mirror_power_upperbound_ls = mirror_power_pred(ray_density, ground_length**2, mirror_dim[0]*mirror_dim[1], np.sum(mirror_num_ls), theta)
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
plt.plot(t_sunrise, mirror_power_upperbound_ls, '--', alpha=0.5, label='Estimated power collected by receiver (Predicted)')

plt.xlabel('''Time since sunrise (s)

R = Total radius of mirror configuration
''')
plt.ylabel('Power (Arbitrary units)')
plt.legend(bbox_to_anchor =(0.5,-0.63), loc='lower center')

plt.ylim([-100,3000])
plt.show()
#%%
'''
From power, get temperature
'''
Tc = 300
Tfunc_ls = []
Tavg_ls = []
Terrfunc_ls = []
for power in power_scenario_ls:
    power = np.array(power)

    t_test = np.linspace(t_sunrise[0], t_sunrise[-1], 1000)
    power_test_func = CubicSpline(t_sunrise, power)

    Tavg, Tfunc, Terrfunc = get_T(t_sunrise, power, Tc=Tc)
    Tavg_ls.append(Tavg)
    Tfunc_ls.append(Tfunc)
    Terrfunc_ls.append(Terrfunc)

t_plot = np.linspace(t_sunrise[0], t_sunrise[-1])
for i, func in enumerate(Tfunc_ls):
    y = func(t_plot)/Tc
    y = np.array(y, dtype=float)
    plt.plot(t_plot, y, label=f'R={R_list[i]}')
    yerr = Terrfunc_ls[i](t_plot)
    yerr = np.array(yerr, dtype=float)
    plt.fill_between(t_plot, y-yerr, y+yerr, alpha=0.5)
plt.xlabel('Time')
plt.ylabel(r'$T/T_{c}$')
plt.legend()
plt.show()

t_plot = np.linspace(t_sunrise[0], t_sunrise[-1])
for i, func in enumerate(Tfunc_ls):
    plt.plot(t_plot, 1-Tc/func(t_plot), label=f'R={R_list[i]}')
plt.xlabel('Time')
plt.ylabel(r'$\eta_{carnot}$')
plt.legend()
plt.show()

# %%
'''
Now do the same thing but with the ray separating into the light cone
'''
N_raycone = 10
power_cone_scenario_ls = []
count_cone_scenario_ls = []

for radius in R_list:
    position_ls = create_circular_positions(radius, mirror_num_ls)
    t_out, power_ls, count_ls = performance_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length, N_raycone)
    power_cone_scenario_ls.append(power_ls)
    count_cone_scenario_ls.append(count_ls)
# %%
#Plot the result
plt.clf()
plt.figure(dpi=800)
t = np.linspace(0, t_sunrise[-1], 40000)
for radius_i, radius in enumerate(R_list):
    plt.plot(t_sunrise, power_cone_scenario_ls[radius_i], '.', color="C{}".format(radius_i))
    func = CubicSpline(t_sunrise, power_cone_scenario_ls[radius_i])
    plt.plot(t, func(t), color="C{}".format(radius_i), label=f'R={radius}')

plt.plot(t_sunrise, ground_power_ls, '.', label='Total power incident on area (Simulated)')
plt.plot(t_sunrise, ground_power_pred_ls, '--', alpha=0.5, label='Total power incident on area (Predicted)')
plt.plot(t_sunrise, mirror_power_upperbound_ls, '--', alpha=0.5, label='Estimated power collected by receiver (Predicted)')

plt.xlabel('''Time since sunrise (s)

R = Total radius of mirror configuration
''')
plt.ylabel('Power (Arbitrary units)')
plt.legend(bbox_to_anchor =(0.5,-0.63), loc='lower center')

plt.ylim([0,2000])
plt.show()

# %%
'''
Categorizing uncertainty from using various number of points
'''
ray_density = 150
ground_length = 30
mirror_num_ls = [4,8,16,32,64]
mirror_dim = [1.,1.]
receiver_dim = [1.,1.,1.]
receiver_pos = [0.,0.,15.]

R_list = [10]

t_scenario_ls2 = []
power_scenario_ls2 = []
count_scenario_ls2 = []

day = ts.utc(2014, 12, 25) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation

N_points_ls = [3, 5, 10, 15, 20, 25]
#%%
start = time.time()
for i, N_points in enumerate(N_points_ls):
    t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, N_points)

    for j, val in enumerate(theta):
        if val < 0:
            theta[j] = 0

    for radius in R_list:
        position_ls = create_circular_positions(radius, mirror_num_ls)
        t_out, power_ls, count_ls = performance_no_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length)
        power_scenario_ls2.append(power_ls)
        count_scenario_ls2.append(count_ls)
        t_scenario_ls2.append(t_sunrise)

    print(i+1)
   
end = time.time()
print(end-start)
# %%
#Get the time list (delete later, forgot to put it in above)
t_scenario_ls2 = []
for i, N_points in enumerate(N_points_ls):
    t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, N_points)
    t_scenario_ls2.append(t_sunrise)

#Interpolate each situation and find the chi-square with the previous scenario
power_interp_ls = []
t_ls = np.linspace(0, t_sunrise[-1], 100)
plt.figure(figsize=(7,5), dpi=800)
for i, power_ls in enumerate(power_scenario_ls2):
    func = CubicSpline(t_scenario_ls2[i], power_ls)
    def func2(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(0, func(i)))
            return np.array(output)
        else:
            return max(0, func(t))
    power_interp_ls.append(func)
    plt.plot(t_ls, func2((t_ls)), label=f'{N_points_ls[i]}')
plt.legend(title=r'$N_{interp.}$')
plt.xlabel('Time since sunrise (s)')
plt.ylabel('Power (arbitrary units)')
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::2]))
for label in temp:
    label.set_visible(False)
plt.show()
#%%
# Now compute the chi square
t_ls = np.linspace(0, t_sunrise[-1], 100)
chi_ls = []
for i in range(1, len(power_interp_ls)):
    chi_score = np.sum((power_interp_ls[i](t_ls) - power_interp_ls[i-1](t_ls))**2)/100**2
    chi_ls.append(chi_score)

plt.figure(figsize=(7,5), dpi=800)
plt.plot(N_points_ls[0:-1], chi_ls, '.', c='black', markersize=8)
plt.xlabel('Number of points on P. vs. t graph')
plt.ylabel(r'$\frac{(P_{i}-P_{i-1})^{2}}{N_{sample}}$')
plt.show()
# %%
'''
Get final results
Longest day in London
'''
day = ts.utc(2022, 6, 21) #Date
lat = '51.5072 N' #Location
long = '0.1276 W'
elevation = 1.5 #Elevation
t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, 20)

for i, val in enumerate(theta):
    if val < 0:
        theta[i] = 0
#%%
ray_density = 150
ground_length = 30
mirror_num_ls = [4,8,16,32,64,128]
mirror_dim = [1.,1.]
receiver_dim = [1.,1.,1.]
receiver_pos = [0.,0.,15.]

#Performance of same mirror config, but placing the mirrors farther from receiver/from each other
R_list = [6, 7, 8, 9, 10, 11, 12, 13, 14]
power_scenario_ls = []
count_scenario_ls = []

start = time.time()
for radius in R_list:
    position_ls = create_circular_positions(radius, mirror_num_ls)
    t_out, power_ls, count_ls = performance_no_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length)
    power_scenario_ls.append(power_ls)
    count_scenario_ls.append(count_ls)
end = time.time()
print(f'Elapsed time: {end-start}')
#%%
#Ground power
ground_power_ls = ground_power(ray_density, theta, phi, ground_length)

#Estimate predictions
# from estimate_predictions import *
ground_power_pred_ls = ground_power_pred(ray_density, theta)
mirror_power_upperbound_ls = mirror_power_pred(ray_density, ground_length**2, mirror_dim[0]*mirror_dim[1], np.sum(mirror_num_ls), theta)
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
plt.plot(t_sunrise, mirror_power_upperbound_ls, '--', alpha=0.5, label='Estimated power collected by receiver (Predicted)')

plt.xlabel('''Time since sunrise (s)

R = Total radius of mirror configuration
''')
plt.ylabel('Power (Arbitrary units)')
plt.legend(bbox_to_anchor =(0.5,-0.63), loc='lower center')

plt.ylim([-100,3000])
plt.show()
#%%
'''
From power, get temperature
'''
Tc = 300
Tfunc_ls = []
Tavg_ls = []
Terrfunc_ls = []
for power in power_scenario_ls:
    power = np.array(power)

    t_test = np.linspace(t_sunrise[0], t_sunrise[-1], 1000)
    power_test_func = CubicSpline(t_sunrise, power)

    Tavg, Tfunc, Terrfunc = get_T(t_sunrise, power, Tc=Tc)
    Tavg_ls.append(Tavg)
    Tfunc_ls.append(Tfunc)
    Terrfunc_ls.append(Terrfunc)

t_plot = np.linspace(t_sunrise[0], t_sunrise[-1])
for i, func in enumerate(Tfunc_ls):
    y = func(t_plot)/Tc
    y = np.array(y, dtype=float)
    plt.plot(t_plot, y, label=f'R={R_list[i]}')
    yerr = Terrfunc_ls[i](t_plot)
    yerr = np.array(yerr, dtype=float)
    plt.fill_between(t_plot, y-yerr, y+yerr, alpha=0.5)
plt.xlabel('Time')
plt.ylabel(r'$T/T_{c}$')
plt.legend()
plt.show()

for i, Tavg in enumerate(Tavg_ls):
    plt.plot(R_list[i], Tavg, '.')
    plt.xlabel('R')
    plt.ylabel(r'$T_{avg}$')

t_plot = np.linspace(t_sunrise[0], t_sunrise[-1])
for i, func in enumerate(Tfunc_ls):
    plt.plot(t_plot, 1-Tc/func(t_plot), label=f'R={R_list[i]}')
plt.xlabel('Time')
plt.ylabel(r'$\eta_{carnot}$')
plt.legend()
plt.show()
