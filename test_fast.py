#%%
import numpy as np
from playground_fast import *
from initialization_code_fast import *
# %%
ray_ls = initialize_rays_parallel_plane_fast(1, 100, center=[0,0,15])
ground = mirror(0,0,0,0,0,15,15, 'ground')
mirror_ls = typed.List()
mirror_ls.append(ground)
test_playground = playground(mirror_ls, ray_ls)
# %%
test_playground.simulate()
# %%
