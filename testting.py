#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
from initialization_code import *
#%%
#Test the initialize ray function
test_playground = playground()
initialize_rays(test_playground, [-15,15], [-15,15], 10, start=[0,0,50])
# test_playground.add_rect_mirror(0,0,-10,np.pi/3,np.pi/4,25,15,'mirror')
test_playground.add_rect_mirror(0,0,0,np.pi/4,np.pi/4,15,15,'mirror')
test_playground.simulate()
%matplotlib ipympl
test_playground.display(show_mirrors=True, show_mirror_normals=True)
# %%
#Test the initialize ray parallel function
test_playground2 = playground()
initialize_rays_parallel(test_playground2, [-15,15], [-15,15], 10, 0, np.pi/2)
test_playground2.add_rect_mirror(0,0,0,0,np.pi/4,15,15,'mirror')
test_playground2.simulate()
%matplotlib ipympl
test_playground2.display(show_mirrors=True, show_mirror_normals=True)
# %%
#Test add receiver
test_playground3 = playground()
test_playground3.add_cubic_receiver([0,0,5], 5, 5, 10)
# test_playground3.add_rect_mirror(0,0,0,np.pi/7,0,45,45,'mirror')
initialize_rays_parallel(test_playground3, [-20,20], [-20,20], 20, 0, np.pi/4)
test_playground3.simulate()
%matplotlib ipympl
test_playground3.display(show_mirrors=True, show_mirror_normals=True)
print(test_playground3.get_receiver_power())
#%%
test_playground4 = playground()
test_playground4.add_cubic_receiver([0,0,0], 1, 1, 1)
test_playground4.add_ray([-10,0,0],[1,0,0])
test_playground4.add_ray([-10,0,0.25],[1,0,0])
test_playground4.add_ray([-10,0,3],[1,0,0])
test_playground4.simulate()
%matplotlib ipympl
test_playground4.display()
print(test_playground4.get_receiver_power())
# %%
#Test plane initialization
sun_theta_ls = np.linspace(0,np.pi,25)
ray_count_ls = []
for sun_theta in sun_theta_ls:
    test_playground5 = playground()
    initialize_rays_parallel_plane(test_playground5, 100, [0,0,0], 15, 15, 0, sun_theta)
    test_playground5.add_rect_mirror(0,0,0,0,np.pi/2,15,15,mirror_type='receiver')
    test_playground5.simulate()
    ray_count_ls.append(test_playground5.mirrors[0].ray_count)
#%%
fig = plt.figure()
plt.plot(sun_theta_ls, ray_count_ls, '.')
plt.xlabel(r'$Elevation\ \theta$')
plt.ylabel(r'$Flux\ (arbitrary\ units)$')
plt.grid()
plt.show()

# %%

# %%
