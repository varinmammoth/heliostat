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

from plot_generation import load_pickle
#%%
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
#%%

'''
1 mirror, but keep moving it farther and farther away from
receiver
To test ray cone effect
'''
N_cone = 300
N_trials = 3
distance_ls = np.arange(-400, 100, 50)
angle_count = []
angle_count_std = []
angle_ls = [0, np.pi/8, np.pi/6, np.pi/4]
#%%
for angle in angle_ls:
    count = []
    count_std = []
    for distance in distance_ls:
        temp = []
        for i in range(0, N_trials):
            mirror_ls = typed.List()
            mirror_ls.append(mirror(0., 0., float(distance), 0., np.pi/2-angle, 1., 1.))
            ray_ls = initialize_rays_parallel(1, xlim=[-0.5, 0.5], ylim=[-0.5,0.5], ray_density=20, phi=0., theta=np.pi/2)
            ray_ls = initialise_rays_cone(ray_ls, N_cone, 0.5*np.pi/180, 1)
            playground1 = playground(mirror_ls, ray_ls)
            playground1.simulate()
            temp.append(playground1.mirrors[-1].ray_count/N_cone)
            print('yes')
        count.append(np.mean(temp))
        count_std.append(np.std(temp)/np.sqrt(N_trials))
    angle_count.append(count)
    angle_count_std.append(count_std)
    print(f'{angle} done!')

# %%
from scipy.optimize import curve_fit
#%%

frac_estimate = lambda x, a:  a**2/(a+2*x*np.tan(0.5*np.pi/180/2))**2
frac_estimate_circle = lambda x, a: a**2/np.pi/(a+x*np.tan(0.5*np.pi/180/2))**2
frac_estimate_rounded = lambda x, a: a**2/((a+2*x*np.tan(0.5*np.pi/180/2))**2 - (4-np.pi)*(x*np.tan(0.5*np.pi/180/2))**2)

x_plot = np.linspace(0, 550, 100)

plt.figure(figsize=(7,5), dpi=800)
plt.plot(x_plot, frac_estimate(x_plot, 1), c='grey', alpha=0.8, label=r'$Square\ Est.$')
plt.plot(x_plot, frac_estimate_circle(x_plot, 1), '--', c='grey', alpha=0.8, label=r'$Circle\ Est.$')
plt.plot(x_plot, frac_estimate_rounded(x_plot, 1), linestyle='dotted', c='grey', alpha=0.8, label=r'$Rounded\ Est.$')


count = np.array(angle_count[0])
count_std = np.array(angle_count_std[0])
plt.errorbar(100-distance_ls, count/400, yerr=count_std/15, fmt='.', capsize=2, label=r'$\alpha=0$')
count = np.array(angle_count[1])
count_std = np.array(angle_count_std[1])
plt.errorbar(100-distance_ls, count/400, yerr=count_std/15, fmt='.', capsize=2, label=r'$\alpha=\pi/8$')
count = np.array(angle_count[2])
count_std = np.array(angle_count_std[2])
plt.errorbar(100-distance_ls, count/400, yerr=count_std/15, fmt='.', capsize=2, label=r'$\alpha=\pi/6$')
count = np.array(angle_count[3])
count_std = np.array(angle_count_std[3])
plt.errorbar(100-distance_ls, count/400, yerr=count_std/15, fmt='.', capsize=2, label=r'$\alpha=\pi/4$')

plt.xlabel(r'$Distance\ from\ receiver\ (m)$')
plt.ylabel(r'$P/P_{no\ cone}$')
plt.xlim([0, 520])
plt.ylim([0, 1])
plt.legend()
plt.show()
# %%
'''
spot diagram
'''
from scipy.spatial import ConvexHull
import matplotlib

rect = matplotlib.patches.Rectangle((-0.5, -0.5),
                                     1, 1,
                                     fill=None, alpha=1)


#%%
from scipy.spatial import ConvexHull
import matplotlib

distance_ls = np.arange(-400, 100, 50)
angle_ls = [0, np.pi/8, np.pi/6, np.pi/4]

hull_area_ls = []
for angle in angle_ls:
    temp = []
    for distance in distance_ls:

        mirror_ls = typed.List()
        mirror_ls.append(mirror(0., 0., distance, 0., np.pi/2-angle, 10000., 10000.))
        ray_ls = initialize_rays_parallel(1, xlim=[-0.5, 0.5], ylim=[-0.5,0.5], ray_density=20, phi=0., theta=np.pi/2)
        ray_ls = initialise_rays_cone(ray_ls, N_cone, 0.5*np.pi/180, 1)
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        points = []

        for ray in playground1.rays:
            points.append([ray.p[0], ray.p[1]])

        plt.figure(figsize=(5,5), dpi=500)
        currentAxis = plt.gca()
        points = np.array(points)
        hull = ConvexHull(points)
        currentAxis.plot(points[:, 0], points[:, 1], '.')
        for simplex in hull.simplices:
            currentAxis.plot(points[simplex, 0], points[simplex, 1], 'k-')
        currentAxis.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5),
                                            1, 1,
                                            color='red', alpha=1, zorder=2, fill=None))

        currentAxis.set_xlim(xmin=-3, xmax=3)
        currentAxis.set_ylim(ymin=-3, ymax=3)
        currentAxis.set_aspect('equal',adjustable='box')
        currentAxis.set_xlabel('y')
        currentAxis.set_ylabel('z')
        currentAxis.set_title(f'd={100-distance}')
        plt.show()

        temp.append(hull.volume)

        print('yes')

    hull_area_ls.append(temp)

# %%
hull_area_ls = np.array(hull_area_ls)

frac_estimate = lambda x, a:  a**2/(a+2*x*np.tan(0.5*np.pi/180/2))**2
frac_estimate_circle = lambda x, a: a**2/np.pi/(a+x*np.tan(0.5*np.pi/180/2))**2
frac_estimate_rounded = lambda x, a: a**2/((a+2*x*np.tan(0.5*np.pi/180/2))**2 - (4-np.pi)*(x*np.tan(0.5*np.pi/180/2))**2)

x_plot = np.linspace(0, 550, 100)

plt.figure(figsize=(7,5), dpi=800)

plt.plot(x_plot, frac_estimate_rounded(x_plot, 1), linestyle='dotted', c='grey', alpha=0.8, label=r'$Rounded\ Est.$')

yerr_ls = [0.01]*10
plt.errorbar(100-distance_ls, 1/hull_area_ls[0], yerr=yerr_ls, fmt='.', capsize=2, label=r'$Convex\ hull\ \alpha=0$')
plt.errorbar(100-distance_ls, 1/hull_area_ls[1], yerr=yerr_ls, fmt='.', capsize=2, label=r'$Convex\ hull\ \alpha=\pi/8$')
plt.errorbar(100-distance_ls, 1/hull_area_ls[2], yerr=yerr_ls, fmt='.', capsize=2, label=r'$Convex\ hull\ \alpha=\pi/6$')
plt.errorbar(100-distance_ls, 1/hull_area_ls[3], yerr=yerr_ls, fmt='.', capsize=2, label=r'$Convex\ hull\ \alpha=\pi/4$')

plt.xlabel(r'$Distance\ from\ receiver\ (m)$')
plt.ylabel(r'$A_{mirror}/A_{hull}$')
plt.xlim([0, 520])
plt.ylim([0, 1])
plt.legend()
plt.show()
# %%
mirror_ls = typed.List()
mirror_ls.append(mirror(0., 0., 95, 0., np.pi/2-0, 10000., 10000.))
ray_ls = initialize_rays_parallel(1, xlim=[-0.5, 0.5], ylim=[-0.5,0.5], ray_density=20, phi=0., theta=np.pi/2)
ray_ls = initialise_rays_cone(ray_ls, N_cone, 0.5*np.pi/180, 1)
playground1 = playground(mirror_ls, ray_ls)
playground1.simulate()

points = []

for ray in playground1.rays:
    points.append([ray.p[0], ray.p[1]])

plt.figure(figsize=(5,5), dpi=500)
currentAxis = plt.gca()
points = np.array(points)
hull = ConvexHull(points)
currentAxis.plot(points[:, 0], points[:, 1], '.')
# for simplex in hull.simplices:
    # currentAxis.plot(points[simplex, 0], points[simplex, 1], 'k-')
currentAxis.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5),
                                    1, 1,
                                    color='red', alpha=1, zorder=2, fill=None))

currentAxis.set_xlim(xmin=-1, xmax=1)
currentAxis.set_ylim(ymin=-1, ymax=1)
currentAxis.set_aspect('equal',adjustable='box')
currentAxis.set_xlabel('y')
currentAxis.set_ylabel('z')
currentAxis.set_title(f'd={100-distance}')
plt.show()
# %%
