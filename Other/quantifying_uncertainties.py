#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from playground_fast import *
from initialization_code_fast import *
# %%
mirror_dim = [1, 1]
ground_lim = 16
N_iter = 500 #number of iterations to per linear_ray_density

np.random.seed(15)
linear_ray_density_ls = np.arange(10, 155, 5) #list of linear ray densities to try
linear_ray_density_ls = linear_ray_density_ls.tolist()

received_mean_ls = [] #mean number of rays received
received_std_ls = [] #std of the number of rays received
received_theory_ls = [] #expected number of rays received
time_mean_ls = [] #time required to run
time_std_ls = []

# %%
for i, linear_ray_density in enumerate(linear_ray_density_ls):
    mirror_ls = typed.List()
    ground = mirror(0,0,0,0,np.pi/2, ground_lim, ground_lim)
    mirror_ls.append(ground)
    ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=linear_ray_density, phi=0, theta=np.pi/2)
    playground1 = playground(mirror_ls, ray_ls)
    playground1.simulate()

    received_theory_ls.append(
        (mirror_dim[0]*mirror_dim[1])/(ground_lim**2)*(playground1.mirrors[0].ray_count)
    )

for i, linear_ray_density in enumerate(linear_ray_density_ls):
    

    x_coords = np.random.uniform(low=-ground_lim+1, high=ground_lim-1, size=N_iter) #list of random x coordinates
    y_coords = np.random.uniform(low=-ground_lim+1, high=ground_lim-1, size=N_iter) #list of random y coordinates

    temp = []
    temp2 = []
    for j in range(0, N_iter):
        start = time.time()
        mirror_ls = typed.List()
        test_mirror = mirror(x_coords[j], y_coords[j], 0, 0, np.pi/2, *mirror_dim)
        mirror_ls.append(test_mirror)
        ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=linear_ray_density, phi=0, theta=np.pi/2)
        playground1 = playground(mirror_ls, ray_ls)
        playground1.simulate()
        end = time.time()

        temp.append(playground1.mirrors[0].ray_count)
        temp2.append(end-start)

    received_mean_ls.append(np.mean(temp))
    received_std_ls.append(np.std(temp))
    time_mean_ls.append(np.mean(temp2))
    time_std_ls.append(np.std(temp2))

    print(f'Finished {i+1} iterations')

#%%
received_mean_ls = np.array(received_mean_ls)
received_std_ls = np.array(received_std_ls)
received_theory_ls = np.array(received_theory_ls)
time_mean_ls = np.array(time_mean_ls)
time_std_ls = np.array(time_std_ls)
ratio_ls = received_mean_ls/received_theory_ls
errbar = received_std_ls/np.sqrt(N_iter)/received_theory_ls
time_errbar = time_std_ls/np.sqrt(N_iter)

plt.rc('font', size=13)
plt.rc('axes', labelsize=13)

plt.clf()
plt.figure(figsize=(7,5), dpi=800)
plt.errorbar(linear_ray_density_ls, ratio_ls, yerr=errbar, fmt='.', capsize=2, markersize=8, c='black')
plt.hlines(y=1, xmin=0, xmax=500, linestyles='--')
plt.xlim([0, 160])
plt.xlabel('Number of rays along an axis')
plt.ylabel(r'$N_{sim.}/N_{pred.}$')      
plt.show()
#%%
plt.clf()
plt.figure(figsize=(7,5), dpi=800)
plt.errorbar(linear_ray_density_ls, time_mean_ls, fmt='.', capsize=2, markersize=8, c='black')
plt.xlabel('Number of rays along an axis')
plt.ylabel('Test runtime (s)')
plt.show()

plt.clf()
plt.figure(figsize=(7,5), dpi=800)
systematic_err = np.abs(1- ratio_ls)
total_err = systematic_err + errbar
plt.plot(linear_ray_density_ls, total_err, '.', markersize=8, c='black')
plt.xlabel('Number of rays along an axis')
plt.ylabel('''Random + Systematic err.
(Fractional)''')
plt.show()
# %%
