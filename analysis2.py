# %%
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from numba import njit
from numba import typed
from numba import int32, int64, float64
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint, quad
from scipy.stats import chisquare
import pickle

from playground_fast import *
from initialization_code_fast import *


plt.rc('font', size=13)
plt.rc('axes', labelsize=13)


# %%

def performance(position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, earth_sun_distance_ls,
                light_cone=True, N=10):
    # phi_ls and theta_ls are the sun's position we would like to simulate for

    power_ls = []

    for i in range(0, len(phi_ls)):
        # First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, [0, 0, 15], theta=theta_ls[i], phi=phi_ls[i],
                                               a=mirror_dim[0], b=mirror_dim[1])

        # ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-7.5, 7.5], ylim=[-7.5, 7.5], ray_density=10,
                                          phi=phi_ls[i], theta=theta_ls[i])

        playground1 = playground(mirror_ls, ray_ls)

        playground1.simulate()

        # Return the list of propagated rays
        ray_ls = playground1.rays

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)

        # if light_cone == True, split the rays into N rays within the lightcone
        if light_cone == True:
            omega_sun = np.pi * (696340000) ** 2 / earth_sun_distance_ls[i] ** 2
            ray_ls = initialise_rays_cone(ray_ls, 100, omega_sun, len(mirror_ls))
        else:
            ray_ls_new = typed.List()
            for j in ray_ls:
                ray_ls_new.append(ray(j.p, j.a, len(mirror_ls)))
            ray_ls = ray_ls_new

        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        visualize(*playground1.get_history())

        num_rays_receiver = playground1.get_receiver_power()  # number of rays received by the receiver

        if light_cone == True:
            # if ray has been split into N rays in a cone, then the power of each ray will be reduced by a factor of N
            power = num_rays_receiver / N
            power_ls.append(power * np.cos(np.pi / 2 - theta_ls[i]))
        else:
            power_ls.append(num_rays_receiver * np.cos(np.pi / 2 - theta_ls[i]))

        print(f'Finished {i + 1}/{len(phi_ls)} iterations.')

    return power_ls


@jit
def performance_no_cone(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density,
                        ground_length):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length / 2

    for i in range(0, len(phi_ls)):

        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],
                                               a=mirror_dim[0], b=mirror_dim[1])
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)

        # ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], phi=phi_ls[i], theta=theta_ls[i])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim],
                                          ylim=[-ground_lim, ground_lim], ray_density=ray_density, phi=phi_ls[i],
                                          theta=theta_ls[i])

        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        power = playground1.get_receiver_power() * np.cos(np.pi / 2 - theta_ls[i])
        power_ls.append(power)
        output_t_ls.append(t_ls[i])

        mirror_ls = playground1.mirrors
        count_subls = []
        for j in mirror_ls:
            count_subls.append(j.ray_count)
        count_ls.append(count_subls)

        print(f'Iteration: {i + 1}/{len(phi_ls)}')

    return output_t_ls, power_ls, count_ls


def performance_no_cone_2step(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density,
                              ground_length):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length / 2

    for i in range(0, len(phi_ls)):

        # First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],
                                               a=mirror_dim[0], b=mirror_dim[1])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim],
                                          ylim=[-ground_lim, ground_lim], ray_density=ray_density, phi=phi_ls[i],
                                          theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
        ray_ls_new = typed.List()
        for old_ray in playground1.rays:
            ray_ls_new.append(ray(old_ray.p, old_ray.a, len(mirror_ls)))

        playground1 = playground(mirror_ls, ray_ls_new)
        playground1.simulate()

        power = playground1.get_receiver_power() * np.cos(np.pi / 2 - theta_ls[i])
        power_ls.append(power)
        output_t_ls.append(t_ls[i])

        mirror_ls = playground1.mirrors
        count_subls = []
        for j in mirror_ls:
            count_subls.append(j.ray_count)
        count_ls.append(count_subls)

        print(f'Iteration: {i + 1}/{len(phi_ls)}')

    return output_t_ls, power_ls, count_ls


def performance_cone_2step(t_ls, position_ls, receiver_pos, mirror_dim, receiver_dim, phi_ls, theta_ls, ray_density,
                           ground_length, N_raycone):
    output_t_ls = []
    power_ls = []
    count_ls = []

    ground_lim = ground_length / 2

    for i in range(0, len(phi_ls)):

        # First do it without the receiver (assume amount of radiation receiver obtains is negligible)
        mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=theta_ls[i], phi=phi_ls[i],
                                               a=mirror_dim[0], b=mirror_dim[1])
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim],
                                          ylim=[-ground_lim, ground_lim], ray_density=ray_density, phi=phi_ls[i],
                                          theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        mirror_ls = typed.List()
        mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
        ray_ls_new = typed.List()
        for old_ray in playground1.rays:
            ray_ls_new.append(ray(old_ray.p, old_ray.a, len(mirror_ls)))
        sun_angle = 0.533 * np.pi / 180
        ray_ls_new = initialise_rays_cone(ray_ls_new, N_raycone, sun_angle, len(mirror_ls))

        playground1 = playground(mirror_ls, ray_ls_new)
        playground1.simulate()

        power = playground1.get_receiver_power() * np.cos(np.pi / 2 - theta_ls[i]) / N_raycone
        power_ls.append(power)
        output_t_ls.append(t_ls[i])

        mirror_ls = playground1.mirrors
        count_subls = []
        for j in mirror_ls:
            count_subls.append(j.ray_count)
        count_ls.append(count_subls)

        print(f'Iteration: {i + 1}/{len(phi_ls)}')

    return output_t_ls, power_ls, count_ls


def get_T(t, P, Tc, alpha=1, c=1, k=1):
    # Interpolate the power
    P_func_spline = CubicSpline(t, P)

    # Due to cubic spline, we may get negative values, set these to 0
    def P_func(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(0, P_func_spline(i)))
            return np.array(output)
        else:
            return max(0, P_func_spline(t))

    # Define dT/dt
    dTdt = lambda T, t: (alpha / c) * P_func(t) - k * (T - Tc)

    # Solve the ODE
    t_ls = np.linspace(t[0], t[-1], 100)
    T0 = Tc
    T = odeint(dTdt, T0, t_ls)

    # Interpolate the ODE solution
    T_func_spline = CubicSpline(t_ls, T)

    # Due to cubic spline, we may get T < Tc, set these to 0
    def T_func(t):
        if type(t) == np.ndarray or type(t) == list:
            output = []
            for i in t:
                output.append(max(Tc, T_func_spline(i)))
            return np.array(output)
        else:
            return max(Tc, T_func_spline(t))

    # Intergrate to get average temperature
    avg_T = quad(T_func, t_ls[0], t_ls[-1])[0]
    avg_T /= (t_ls[-1] - t_ls[0])

    # #Uncertainty in T
    frac = 0.02
    T_err_func = lambda t: (alpha * frac * P_func(t)) / (c * k)

    return avg_T, T_func, T_err_func


def ground_power(ray_density, theta_ls, phi_ls, ground_length):
    N_rays = ray_density ** 2
    ground_lim = ground_length / 2

    power_ls = []

    for i in range(0, len(phi_ls)):
        mirror_ls = typed.List()
        ground = mirror(0., 0., 0., 0., np.pi / 2, ground_length, ground_length, 'ground')
        mirror_ls.append(ground)
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim],
                                          ylim=[-ground_lim, ground_lim], ray_density=ray_density, phi=phi_ls[i],
                                          theta=theta_ls[i])
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()

        power_ls.append(playground1.mirrors[0].ray_count * np.cos(np.pi / 2 - theta_ls[i]))

    return power_ls


def ground_power_pred(ray_density, theta_ls):
    """Prediction for total power on the ground.
    Args:
        ray_density (int): Linear ray density
        theta_ls (list): List of theta angles.
    Returns:
        list: Prediction for total power on the ground.
    """
    return (ray_density ** 2) * np.cos(np.pi / 2 - theta_ls)


def mirror_power_pred(ray_density, ground_area, mirrormaterial, theta_ls):
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
    #return (ray_density ** 2) / (ground_area) * N_mirrors * mirror_area * np.cos(np.pi / 2 - theta_ls)
    return (ray_density ** 2) / (ground_area) * mirrormaterial * np.cos(np.pi / 2 - theta_ls)

'''
Get final results
Longest day in London
'''

# for mon in range(2):
#
#     if mon == 0:
#         m = 3
#     else:
#         m = 6

simulated_power_on_the_day_list = []

expected_power_on_the_day_list = []

#switcher = 3

lat_list = ['47.4979 N', '37.9838 N', '28.65195 N', '13.7563 N', '3.139 N'] # Bp #Ath #Delh , '28.65195 N'
long_list = ['19.0402 E', '23.7275 E', '77.23149 E', '100.5017 E', '101.6868 E']
elevation_list = [102, 20, 300, 1.5, 63]  # Elevation
placelist = np.array(['Budapest', 'Athens', 'Delhi', 'Bangkok', 'Kuala Lumpur'])

startzero = time.time()

mirrorsetuplist = ['Traditional Circular', 'Sunflower', 'Semicircular', 'Radial Circular']

for p in range(len(lat_list)):
#for p in range(0,1):

    for switcher in range(4):


        simulated_power_on_the_day_list_collec = []
        expected_power_on_the_day_list_collec = []

        for mo in range(1, 13):

            arrangement = []

            print('DATE: 2022', mo, '21, LOCATION:', placelist[p], 'ARRANGEMENT:', mirrorsetuplist[switcher])

            day = ts.utc(2022, mo, 21)  # Date

            t, t_sunrise, phi, theta, distance = get_solar_positions(day, lat_list[p], long_list[p], elevation_list[p], 5)

            for i, val in enumerate(theta):
                if val < 0:
                    theta[i] = 0
            # %%
            ray_density = 10
            ground_length = 30
            mirror_material = 44
            mirror_num_ls = [[4, 8, 12, 24]] #circular
            mirror_num_ls_semicirc = [6, 10, 14, 18] #semicircular
            mirror_num_ls_crosscirc = [0, 12, 12, 12, 12]

            f = 1

            receiver_dim = [f, f, f]
            receiver_pos = [0., 0., 15.]

            critical_receiverlength = np.sqrt(receiver_dim[0]**2 + receiver_dim[1]**2)/2

            colorlist = ['orange', 'red', 'green', 'purple', 'blue', 'black', 'cyan', 'yellow']

            if len(colorlist) >= len(mirror_num_ls):
                pass
            else:
                print('NOT ENOUGH COLORS!!!')
                print('ABORT!!!!!!!!!')

                quit()

            # Performance of same mirror config, but placing the mirrors farther from receiver/from each other
            R_list = [8.8] #7, 8, 9, 10, 11, 12, 13, 14
            power_scenario_ls = []
            count_scenario_ls = []

            #start = time.time()

            for j in range(len(mirror_num_ls)):

                mirrorj = sum(mirror_num_ls[j]) #number of mirrors in the jth setup

                mirror_dim = np.array([np.sqrt(mirror_material/mirrorj), np.sqrt(mirror_material/mirrorj)])

                mirror_diagonal = np.sqrt(mirror_dim[0] ** 2 + mirror_dim[1] ** 2)

                if 1.1 * mirror_material/mirrorj > (receiver_dim[0] * receiver_dim[1]) > 1.05 * mirror_material/mirrorj:
                    #print('Cubic receiver-mirror side area ratio is fine. Ratio is:', receiver_dim[0]/mirror_dim[0])
                    pass
                else:
                    if 1.05 * mirror_material/mirrorj > (receiver_dim[0] * receiver_dim[1]):
                        print('Cubic receiver is too small. The minimum allowed side length is', np.sqrt((1.05 * mirror_material/mirrorj)),
                              'and the maximum allowed side length is', np.sqrt(1.1 * mirror_material/mirrorj))
                        quit()
                    else:
                        print('Cubic receiver is too large. The maximum allowed side length is', np.sqrt(1.1 * mirror_material/mirrorj),
                              'and the minimum allowed side length is', np.sqrt(1.05 * mirror_material/mirrorj))
                        quit()

                #print('The minimum allowed distance between 2 mirrors', 'for setup', j,  'is:', mirror_diagonal)
                #print('Square-shaped mirror dimensions:',mirror_dim)

                critical_mirrorlength = np.linalg.norm(mirror_dim) / 2 #ensures the mirrors do not overlap with receiver


                for radius in R_list:
                    if switcher == 0:
                        position_ls = create_circular_positions(radius, mirror_num_ls[j])
                        arrangement.append('Traditional Circular')
                    if switcher == 1:
                        position_ls = create_sunflower_positions(2, sum(mirror_num_ls[j]))
                        arrangement.append('Sunflower')
                    if switcher == 2:
                        position_ls = create_semicircular_positions(radius, 0, np.pi, mirror_num_ls_semicirc)
                        arrangement.append('Semicircular')
                    if switcher == 3:
                        position_ls = create_circular_positions(radius, mirror_num_ls_crosscirc)
                        arrangement.append('Radial Circular')

                    # print(position_ls)
                    #
                    # x_pos = []
                    # y_pos = []
                    #
                    # for b in range(len(position_ls)):
                    #     x_pos.append(position_ls[b][0])
                    #     y_pos.append(position_ls[b][1])
                    #
                    # plt.figure(figsize=(6, 6))  # default is (8,6)
                    # plt.plot(x_pos, y_pos, 'x')
                    # plt.ylim([-30, 30])
                    # plt.xlim([-30, 30])
                    # plt.grid()
                    # plt.show()

                    for position in position_ls:
                        r = np.linalg.norm(position)

                        if r > critical_mirrorlength + critical_receiverlength:
                            pass
                        else:
                            print('MIRROR AT', position, 'OVERLAPS WITH RECEIVER!!!')
                            quit()

                    for k in range(len(position_ls)):
                        position_ls[k] = np.array(position_ls)[k]
                        for l in range(len(position_ls)):
                            position_ls[l] = np.array(position_ls)[l]
                            if l > k:
                                r = ((position_ls[l] - position_ls[k]))
                                mirror_distance = np.sqrt((r[0] ** 2 + r[1] ** 2))
                                if mirror_distance > mirror_diagonal:
                                    pass
                                else:
                                    print('THE FOLLOWING MIRRORS IN MIRROR LIST ARE TOO CLOSE:', 'mirror', k, 'at', position_ls[k], 'and','mirror', l, 'at', position_ls[l])
                                    print('Distance of mirror', k, 'and', l, 'is', mirror_distance)
                                    quit()
                            else:
                                pass

                    t_out, power_ls, count_ls = performance_no_cone_2step(t_sunrise, position_ls, receiver_pos, mirror_dim,
                                                                          receiver_dim, phi, theta, ray_density, ground_length)
                    power_scenario_ls.append(power_ls)
                    count_scenario_ls.append(count_ls)



            t = np.linspace(0, t_sunrise[-1], 4000)
            for radius_i, radius in enumerate(R_list):

                plt.figure(figsize=(14,8), dpi=100)

                plt.plot(t_sunrise/3600, power_scenario_ls[j], '.', color='red', label='Simulated Values (Received Power)')

                func = CubicSpline(t_sunrise, power_scenario_ls[j])

                integral = np.trapz(func(t), t)

                #print('RECEIVED POWER:', integral, 'SETUP:', j)

                simulated_power_on_the_day_list_collec.append(integral)

                plt.plot(t/3600, func(t), label=f'Polynomial Fit ({int(integral)} au.)', linewidth = 0.5)


            # %%
            # Ground power
            ground_power_ls = ground_power(ray_density, theta, phi, ground_length)

            # Estimate predictions
            # from estimate_predictions import *
            ground_power_pred_ls = ground_power_pred(ray_density, theta)
            mirror_power_upperbound_ls = mirror_power_pred(ray_density, ground_length ** 2, mirror_material, theta)

            #Plot the result

            func2 = CubicSpline(t_sunrise, mirror_power_upperbound_ls)
            integral_expected = np.trapz(func2(t), t) #this is the expected power received

            plt.plot(t_sunrise/3600, ground_power_ls, '.', label='Simulated Total Power Incident on Area') # label='Total power incident on area (Simulated)
            plt.plot(t_sunrise/3600, ground_power_pred_ls, '--', alpha=0.5, label='Predicted Total Power Incident on Area') #, label='Total power incident on area (Predicted)'
            plt.plot(t_sunrise/3600, mirror_power_upperbound_ls, '--', alpha=0.5, label=f'Predicted Power ({int(integral_expected)} au.)') #label='Estimated power collected by receiver (Predicted)'


            expected_power_on_the_day_list_collec.append(integral_expected)


            string1 = ''
            plt.title(f'{placelist[p]}, 2022.{mo}.21., {string1.join(arrangement)} Setup')

            plt.xlabel('Time since sunrise (h)')
            plt.ylabel('Power (Arbitrary Units)')
            plt.legend(loc='upper right', fontsize='small')

            plt.ylim([-5, max(2 * mirror_power_upperbound_ls)])
            plt.grid()
            #plt.show()
            plt.savefig(f'test_plots\_{placelist[p]}_date_2022_{mo}_21_setup_{arrangement}_power_{integral}.png')
            plt.clf()
            end = time.time()
            print(f'Elapsed time: {end - startzero}')

            #print(simulated_power_on_the_day_list_collec)
        simulated_power_on_the_day_list.append(simulated_power_on_the_day_list_collec)
        expected_power_on_the_day_list.append(expected_power_on_the_day_list_collec)


    months = np.array(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']) #, 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    X_axis = np.arange(len(months))


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(14,8), dpi=100)

    plt.suptitle(f'Different Mirror Arrangements in {placelist[p]}', fontsize = '25')


    ax1.bar(X_axis - 0.4, simulated_power_on_the_day_list[0 + 4*p], 0.4, label = 'Simulated')
    ax1.bar(months, expected_power_on_the_day_list[0 + 4*p], 0.4, label = 'Expected')
    ax2.bar(X_axis - 0.4, simulated_power_on_the_day_list[1 + 4*p], 0.4)
    ax2.bar(months, expected_power_on_the_day_list[1 + 4*p], 0.4)
    ax3.bar(X_axis - 0.4, simulated_power_on_the_day_list[2 + 4*p], 0.4)
    ax3.bar(months, expected_power_on_the_day_list[2 + 4*p], 0.4)
    ax4.bar(X_axis - 0.4, simulated_power_on_the_day_list[3 + 4*p], 0.4)
    ax4.bar(months, expected_power_on_the_day_list[3 + 4*p], 0.4)
    ax1.title.set_text('Traditional Circular')
    ax2.title.set_text('Sunflower')
    ax3.title.set_text('Semicircular')
    ax4.title.set_text('Radial Circular')

    ax1.grid(axis = 'y')
    ax2.grid(axis = 'y')
    ax3.grid(axis = 'y')
    ax4.grid(axis = 'y')

    f.text(0.5, 0.04, 'Date', ha='center', fontsize = '17')
    f.text(0.04, 0.5, 'Received Power (Arbitrary Units)', va='center', rotation='vertical', fontsize = '17')

    f.legend()


    plt.savefig(f'test_plots\_test_powerbar_{placelist[p]}.png')
    plt.clf()


with open("simulated_power_list.pkl", "wb") as fp:   #Pickling
    pickle.dump(simulated_power_on_the_day_list, fp)

np.savetxt("simulated_power_list.csv", simulated_power_on_the_day_list, delimiter=",")

with open("expected_power_list.pkl", "wb") as fp:   #Pickling
    pickle.dump(expected_power_on_the_day_list, fp)

np.savetxt("expected_power_list.csv", expected_power_on_the_day_list, delimiter=",")