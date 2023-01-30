#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot
# %%
def initialize_rays(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, start=[100,100,100]):
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

def initialize_rays_parallel(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, beta=np.pi/4, gamma=np.pi/4):
    x = np.linspace(*xlim, ray_density)
    y = np.linspace(*ylim, ray_density)
    x, y = np.meshgrid(x, y)

    a = rot(beta, gamma)@np.array([0,0,1])

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = np.array([x[i][j], y[i][j], 0])

            start = r + 100*a

            playground.add_ray(start, -a)

    return 
# %%
#Test the initialize ray function
test_playground = playground()
initialize_rays(test_playground, [-15,15], [-15,15], 10, start=[0,0,50])
# test_playground.add_rect_mirror(0,0,-10,np.pi/3,np.pi/4,25,15,'mirror')
test_playground.add_rect_mirror(3,2,5,np.pi/7,np.pi/4,45,45,'mirror')
test_playground.simulate()
%matplotlib ipympl
test_playground.display(show_mirrors=True, show_mirror_normals=True)
# %%
#Test the initialize ray parallel function
test_playground2 = playground()
initialize_rays_parallel(test_playground2, [-15,15], [-15,15], 10, 0, 0)
test_playground2.add_rect_mirror(3,2,5,np.pi/7,np.pi/4,45,45,'mirror')
test_playground2.simulate(500)
%matplotlib ipympl
test_playground2.display(show_mirrors=True, show_mirror_normals=True)
# %%
