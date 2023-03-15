#%%
import numpy as np
from playground_fast import *
from initialization_code_fast import *
# %%
start = time.time()

position_ls = create_circular_positions(10, [4,5,6])
mirror_ls = initialise_mirrors_optimal(position_ls,[0,0,30], 0, np.pi/6)
# mirror_ls = typed.List()
# ground = mirror(0,0,0,0,np.pi/2,15,15, 'ground')
# mirror_ls.append(ground)
mirror_ls = add_receiver(mirror_ls, [0,0,30], 3, 3, 3)
ray_ls = initialize_rays_parallel_plane_fast(len(mirror_ls), 10, center=[0,0,0], a=15, b=15, phi=0, theta=np.pi/6)

end = time.time()
print("Elapsed = %s" % (end - start))
#%%
test_playground = playground(mirror_ls, ray_ls)
# %%
start = time.time()
test_playground.simulate()
end = time.time()
print("Elapsed = %s" % (end - start))
#%%
%matplotlib ipympl
visualize(*test_playground.get_history(), show_rays=True)
# %%
'''
Test the ray cone generator
'''
#Add ground and a single ray to the system
start = time.time()

mirror_ls = typed.List()
ground = mirror(0,0,0,0,np.pi/2,15,15, 'ground')
mirror_ls.append(ground)
ray_ls = typed.List()
ray_ls.append(ray(np.array([0.,0.,0.]), np.array([-1.,0.,1.]), 1))
test_playground = playground(mirror_ls, ray_ls)
test_playground.simulate()

end = time.time()
print("Elapsed = %s" % (end - start))

start = time.time()
ray_ls_old = test_playground.rays
omega_sun = 5*np.pi/180
ray_cone = initialise_rays_cone(ray_ls_old, 100, omega_sun, 1)
test_playground = playground(mirror_ls, ray_cone)
test_playground.simulate()
end = time.time()
print("Elapsed = %s" % (end - start))

%matplotlib ipympl
visualize(*test_playground.get_history(), show_rays=True)
# %%

