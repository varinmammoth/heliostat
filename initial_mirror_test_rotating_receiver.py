import numpy as np
import matplotlib.pyplot as plt
from playgroundcode import playground
from playgroundcode import rot_vector
from initializationcode_final import *
from Matrix import mirror_normal

from scipy.interpolate import interp1d
from scipy.integrate import quad
from initializationcode_final import get_solar_positions


day = ts.utc(2023, 3, 21) #Date
lat = '0.3167 N' #Location
long = '18.9114 E'
elevation = 1.5 #Elevation

n = 6 #we divide the time spent in daylight to this many equal timesteps

t_ls, t_since_sunrise, phi_ls, sun_theta_ls = get_solar_positions(day, lat, long, elevation, n)

print(phi_ls, sun_theta_ls)


for i in range(0, 9):

    a = i + 0.0001
    b = 1
    print('CIKLUS =', i)
    number = 4 #for manual calcultions, this gives you which timestep you are interested in

    receiver_phi = a * np.pi/4 #receiver position


    receiver_r = 40

    test_phi = 0 #7 to 6.67 does not work if theta is pi/90
    test_theta = 0 #sulight angle

    receiver = receiver_r * np.array([np.cos(receiver_phi), -np.sin(receiver_phi), 0]) #0 -15 15 surely works, actually negative x works, positive x fixed


    test_playground = playground()

    position_ls = np.array([0, 0, 0]) #mirror center #1 4 2 surely works

    #input = np.array([np.cos(phi_ls[number]), -np.sin(phi_ls[number]), np.tan(sun_theta_ls[number])]) #sun position manual
    input = np.array([np.cos(test_phi), -np.sin(test_phi), np.tan(test_theta)])
    reflected = (receiver-position_ls) #reflected

    print('INPUT, REFLECTED =',input, reflected)

    n3_phi_actually = np.arccos(np.dot(input/np.linalg.norm(input), reflected/np.linalg.norm(reflected)))

    print('N3_PHI ATUALLY', n3_phi_actually/2)

    ray_count_ls = []

    #for i in range(n):

    test_playground = playground()
    initialize_rays_parallel_plane(test_playground, 10, [0,0,0], 15, 15, test_phi, test_theta)

    normal_phi = mirror_normal_calculator(position_ls, receiver, test_phi, test_theta)[0]
    normal_theta =  mirror_normal_calculator(position_ls, receiver, test_phi, test_theta)[1]


    test_playground.add_cubic_receiver(receiver, 5, 5, 5)
    test_playground.add_rect_mirror(position_ls[0], position_ls[1], position_ls[2], normal_phi,normal_theta,15,15,mirror_type='mirror')
    test_playground.simulate()
    test_playground.display(xlim=[-50,50], ylim=[-50,50], zlim=[0,45], show_mirrors=True, show_mirror_normals=True)
    #ray_count_ls.append(test_playground.mirrors[0].ray_count)

