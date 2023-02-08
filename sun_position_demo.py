#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
from initialization_code import *

from scipy.interpolate import interp1d
from scipy.integrate import quad
#%%
#Getting real solar angles 
day = ts.utc(2014, 12, 19) #Date
lat = '13.7563 N' #Location
long = '100.5018 E'
elevation = 1.5 #Elevation
t_ls, t_since_sunrise, phi_ls, theta_ls = get_solar_positions(day, lat, long, elevation, 20)
plt.figure(dpi=800)
plt.plot(t_since_sunrise, phi_ls, '.', label=r'$Azimuth\ \phi$', marker='.', c='black')
plt.plot(t_since_sunrise, theta_ls, '.', label=r'$Elevation\ \theta$', marker='v', c='black')
plt.legend()
plt.grid()
plt.xlabel(r'$Time\ since\ sunrise\ (s)$')
plt.ylabel(r'$Solar\ angles$')
plt.show()
# %%
#Plotting the flux on the ground throughout the day
ray_count_ls = []
for i in range(0, len(phi_ls)):
    test_playground5 = playground()
    initialize_rays_parallel_plane(test_playground5, 100, [0,0,0], 15, 15, phi=phi_ls[i], theta=theta_ls[i])
    test_playground5.add_rect_mirror(0,0,0,0,np.pi/2,15,15,mirror_type='receiver')
    test_playground5.simulate()
    ray_count_ls.append(test_playground5.mirrors[0].ray_count)
# %%
#We can also interpolate the flux to get the total energy received in that day.
flux_func = interp1d(t_since_sunrise, ray_count_ls)
energy_ls = [0]
for i in range(1, len(t_since_sunrise)):
    energy_ls.append(energy_ls[-1] + quad(flux_func, t_since_sunrise[i-1], t_since_sunrise[i])[0])
#%%
fig,ax = plt.subplots(dpi=800)
ax.scatter(t_since_sunrise, ray_count_ls, marker=".", label='Flux', c='black')
ax.set_xlabel('Time since sunrise (s)')
ax.set_ylabel('Flux (arbitrary units)')
ax2=ax.twinx()
ax2.scatter(t_since_sunrise, energy_ls, marker="v", label='Energy', c='black')
ax2.set_ylabel("Energy (arbitrary units)")
fig.legend(loc=2, bbox_to_anchor=(0.15,0.85))
plt.grid()
plt.show()

# %%
