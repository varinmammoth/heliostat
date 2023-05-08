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

import pickle
#%%
def load_pickle(filename: str):
    with open(f'{filename}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
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
            if len(old_ray.history_px) > 2: 
                ray_ls_new.append(ray(old_ray.p, old_ray.a, len(mirror_ls)))
        print(len(ray_ls_new))
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
    frac = 0.02
    T_err_func = lambda t: (alpha*frac*P_func(t))/(c*k)
    
    return avg_T, T_func, T_err_func

def avg_power(t, power):
    func_spline = CubicSpline(t, power)
    def P_func(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(0, func_spline(i)))
            return np.array(output)
        else:
            return max(0, func_spline(t))
    
    avg_P = quad(P_func, t[0], t[-1])[0]
    avg_P /= (t[-1]-t[0])
    return avg_P

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
# %%
N_ls = np.array([8, 22, 28])
SA = 100 #surface area

day = ts.utc(2022, 6, 21) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation
t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat, long, elevation, 40)

for i, val in enumerate(theta):
    if val < 0:
        theta[i] = 0

ray_density = 150
ground_length = 30
receiver_dim = [2.,2.,2.]
receiver_pos = [0.,0.,15.]
#%%
power_scenario_ls = []
start = time.time()
for N in N_ls:
    position_ls = create_circular_positions(13, [N])
    mirror_dim = [np.sqrt(100/N), np.sqrt(100/N)]
    t_out, power_ls, count_ls = performance_no_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,receiver_dim,phi,theta,ray_density,ground_length)
    power_scenario_ls.append(power_ls)
end = time.time()
print(f'Elapsed time: {end-start}')
#%%
#Ground power
ground_power_ls = ground_power(ray_density, theta, phi, ground_length)

#Estimate predictions
# from estimate_predictions import *
ground_power_pred_ls = ground_power_pred(ray_density, theta)
# mirror_power_upperbound_ls = mirror_power_pred(ray_density, ground_length**2, mirror_dim[0]*mirror_dim[1], np.sum(N_), theta)
# %%
#Plot the result
max_power = np.max(ground_power_pred_ls)
P_avg_ls = []
plt.clf()
plt.figure(dpi=800, figsize=(7,5))
t = np.linspace(0, t_sunrise[-1], 40000)

for N_i, N in enumerate(N_ls):
    plt.plot(t_sunrise, power_scenario_ls[N_i]/max_power, '.', color="C{}".format(N_i))
    func_spline = CubicSpline(t_sunrise, power_scenario_ls[N_i])
    def func(t):
        output = func_spline(t)
        out = []
        for i in output:
            if i < 0:
                out.append(0)
            else:
                out.append(i)
        return np.array(out)
    y = func(t)/max_power
    yerr = np.ones(len(y))*0.003
    plt.plot(t, y, color="C{}".format(N_i), label=f'$N={{{N}}}\ $')
    plt.fill_between(t, y-yerr, y+yerr, alpha=0.3)
    P_avg_ls.append(avg_power(t_sunrise, power_scenario_ls[N_i]))

# plt.plot(t_sunrise, ground_power_pred_ls/max_power, '--', c='grey', label=r'$P_{ground}^{pred.}/P_{max}$')
# plt.plot(t_sunrise, ground_power_ls/max_power, '.', c='black', label=r'$P_{ground}^{sim.}/P_{max}$')

plt.xlabel(r'$t_{sunrise}\ (s)$')
plt.ylabel(r'$P/P_{max}$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylim([-0.001,0.06])
plt.show()
# %%
P_avg_ls = np.array(P_avg_ls)
plt.figure(dpi=800, figsize=(7,5))
fit, cov = np.polyfit(N_ls[0:7], P_avg_ls[0:7]/max_power, 1, cov=True)
func = np.poly1d(fit)
x = np.linspace(0, 31, 3)
plt.hlines(y=P_avg_ls[-1]/max_power, xmin=0, xmax=31, linestyle='--', color='red', label=r'$y = 0.061 \pm 0.008$')
plt.plot(x, func(x), '--', label=r'y =$(0.0040 \pm 0.0001) \cdot x + (0.005 \pm 0.001$)')
plt.errorbar(N_ls, P_avg_ls/max_power, yerr=0.003, fmt='.', c='black', capsize=2, label='Data')
plt.xlim([0,31])
plt.ylim([0, 0.07])
plt.xlabel(r'$N$')
plt.ylabel(r'$\langle P \rangle_{t}/P_{max}$')
plt.legend()
plt.show()

# %%
max_power = 22160.628010443357
N_ls = [ 2,  5, 10, 15, 20, 25, 30, 3,  7, 12, 17, 13, 27, 8, 22, 28]
P_avg_ls = [ 250.17785637,  597.59489366, 1026.88536683, 1274.16033753,
       1345.14229546, 1369.5378347 , 1371.83414451, 387.5590031540344,
 789.0953157407229,
 1147.8456576219887,
 1318.553255816324,
 1202.9024104575356,
 1371.0015259542845, 842.0908775659841, 1359.88481042715, 1370.1092481036121]

P_avg_ls = [x for _, x in sorted(zip(N_ls, P_avg_ls))]
N_ls.sort()
# %%
'''
Spot diagrams
'''
from scipy.spatial import ConvexHull
import matplotlib

hull_ls = []
points_ls = []

hull_area_ls = []

N_ls = [5, 7, 10, 12, 15, 17, 20, 22, 25, 30]

# N_ls = [5]
# plt.figure(figsize=(5,5), dpi=500)
# currentAxis = plt.gca()
for i, N in enumerate(N_ls):
    position_ls = create_circular_positions(13, [N])
    mirror_dim = [np.sqrt(100/N), np.sqrt(100/N)]
    
    mirror_ls = initialise_mirrors_optimal(position_ls, [-30,0,15], theta=np.pi/4, phi=0,  a=mirror_dim[0], b=mirror_dim[1])
    # mirror_ls = add_receiver(mirror_ls, [0,0,15], 5, 5, 5)
    mirror_ls.append(mirror(-30., 0., 0., 0., 0, 10000., 10000.))
    ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-15, 15], ylim=[-15,15], ray_density=150, phi=0, theta=np.pi/4)
    playground1 = playground(mirror_ls, ray_ls)
    playground1.simulate()

    points = []
    for r in playground1.rays:
        if r.p[2] > 0:
            points.append([r.p[1], r.p[2]])
    hull = ConvexHull(points)
    hull_ls.append(hull)
    points_ls.append(points)

    points = np.array(points)
    plt.figure(figsize=(5,5), dpi=500)
    currentAxis = plt.gca()
    points = np.array(points)
    hull = ConvexHull(points)
    currentAxis.plot(points[:, 0], points[:, 1], '.')
    for simplex in hull.simplices:
        currentAxis.plot(points[simplex, 0], points[simplex, 1], color='red')
    # currentAxis.plot(points[simplex, 0], points[simplex, 1], label=f'$N={{{N}}}$', color="C{}".format(i))
        
    # currentAxis.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5),
    #                                     1, 1,
    #                                     color='red', alpha=1, zorder=2, fill=None))

    currentAxis.set_xlim(xmin=-4, xmax=4)
    currentAxis.set_ylim(ymin=12, ymax=18)
    currentAxis.set_aspect('equal',adjustable='box')
    currentAxis.set_xlabel('y')
    currentAxis.set_ylabel('z')
    currentAxis.set_title(f'$N={{{N}}}$')
# currentAxis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    hull_area_ls.append(hull.volume)
    plt.show()
    
# %%
currentAxis = plt.gca()
points_ls = np.array(points_ls)
for i, h in enumerate(hull_ls):
    currentAxis.plot(points[:, 0], points[:, 1], '.')
    for simplex in h.simplices:
        currentAxis.plot(points_ls[i][simplex, 0], points_ls[i][simplex, 1], 'k-')

plt.show()
# %%
plt.figure(dpi=500)
fit, cov = np.polyfit(100/np.array(N_ls), hull_area_ls, 1, cov=True)
func = np.poly1d(fit)
x = np.linspace(0, 22, 3)
plt.plot(x, func(x), label=r'$$')
plt.errorbar(100/np.array(N_ls), hull_area_ls, yerr=[0.5]*len(hull_area_ls), fmt='.', capsize=2, c='black', label='Data')
plt.xlabel('Area of a single mirror')
plt.ylabel('Convex hull area')
plt.xlim([0, 21])
plt.show()
# %%
