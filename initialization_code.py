#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
# %%
def initialize_rays(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, start=[100,100,100]):
    """Initializes rays that originate from a single point. The rays will end up on the ground
    at the coordinates (x,y), where (x,y) are equally spaced points in the range xlim and ylim.

    Args:
        playground (playground obj): Playground object
        xlim (list, optional): x range of ray endpoints on the ground. Defaults to [-10,10].
        ylim (list, optional): y range of ray endpoints on the ground. Defaults to [-10,10].
        ray_density (int, optional): Ray density. e.g. 10 means there will be 10*10 rays in total. Defaults to 10.
        start (list, optional): Starting point of all rays. Defaults to [100,100,100].
    """
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

def initialize_rays_parallel(playground, xlim=[-10,10], ylim=[-10,10], ray_density = 10, phi=np.pi/4, theta=np.pi/4):
    """Initializes paralle rays that end up on coordinates (x,y) on the ground, where (x,y) is
    same as above function. The rays follow along the z_hat vector rotated by beta and gamma,
    where beta and gamma are defined using the convention of this project.

    Args:
        playground (playgound obj): Playground object
        xlim (list, optional): x range. Defaults to [-10,10].
        ylim (list, optional): y ragne. Defaults to [-10,10].
        ray_density (int, optional): Ray density. Defaults to 10.
        beta (float, optional): Beta angle of ray direction. Defaults to np.pi/4.
        gamma (float, optional): Gamma angle of ray direction. Defaults to np.pi/4.
    """
    x = np.linspace(*xlim, ray_density)
    y = np.linspace(*ylim, ray_density)
    x, y = np.meshgrid(x, y)

    n1 = np.array([np.sin(phi), np.cosh(phi), 0])
    a = np.array([np.cos(phi), -np.sin(phi), 0])
    a = rot_vector(a, -n1, theta)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = np.array([x[i][j], y[i][j], 0])

            start = r + 100*a

            playground.add_ray(start, -a)

    return 

def initialize_rays_parallel_plane(playground, ray_density = 10, center=[0,0,0], a=15, b=15, phi=0, theta=0):
    center = np.array(center)
    
    #Azimutal 
    n3 = np.array([np.cos(phi), -np.sin(phi), 0])
    n1 = np.array([np.sin(phi), np.cos(phi), 0])
    n2 = np.array([0,0,1])

    #Elevation
    n2 = rot_vector(n2, -n1, theta)
    n3 = rot_vector(n3, -n1, theta)

    p = center - (a/2)*n1 - (b/2)*n2

    x = np.linspace(0,1,ray_density)
    y = np.linspace(0,1,ray_density)
    x, y = np.meshgrid(x, y)

    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            r = p + a*x[i][j]*n1 + b*y[i][j]*n2

            start = r + 100*n3

            playground.add_ray(start, -n3)

    # playground.add_rect_mirror(*center, phi, theta, a, b, 'absorber')

    return
# %%
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
sun_phi_ls = np.linspace(0,np.pi,25)
ray_count_ls = []
for sun_phi in sun_phi_ls:
    test_playground5 = playground()
    initialize_rays_parallel_plane(test_playground5, 100, [0,0,0], 15, 15, 0, sun_phi)
    test_playground5.add_rect_mirror(0,0,0,0,np.pi/2,7.5,7.5,mirror_type='receiver')
    test_playground5.simulate()
    ray_count_ls.append(test_playground5.mirrors[0].ray_count)

fig = plt.figure()
plt.plot(sun_phi_ls, ray_count_ls, '.')
plt.show()

# %%
