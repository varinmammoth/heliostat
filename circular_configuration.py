#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
from initialization_code import *

from scipy.interpolate import interp1d
from scipy.integrate import quad
#%%
def create_circular_positions(R, num_mirrors_ls):
    position_ls = []
    dr = R/len(num_mirrors_ls)
    for layer in range(0, len(num_mirrors_ls)):
        r = (layer+1)*dr
        d_angle = 2*np.pi/num_mirrors_ls[layer]
        for i in range(0, num_mirrors_ls[layer]):
            x = r*np.cos(i*d_angle)
            y = r*np.sin(i*d_angle)
            position_ls.append([x, y, 5])
    return position_ls

test_playground = playground()

position_ls = create_circular_positions(7, [5,8])
receiver_pos = [0, 0, 20]
sun_phi = 0
sun_theta = np.pi/5
# initialize_rays_parallel_plane(test_playground, 20, [0,0,0], 15, 15, sun_phi, sun_theta)
# initialise_mirrors(test_playground, position_ls, 1, 1, receiver_pos, sun_phi, sun_theta)
test_playground.add_cubic_receiver(receiver_pos, 1, 1, 1)
# test_playground.add_rect_mirror(0,0,0,0,np.pi/2,15,15,'ground')
test_playground.simulate()
%matplotlib ipympl
test_playground.display()
# %%
