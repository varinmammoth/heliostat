from initializationcode import *
from initializationcode import get_solar_positions


day = ts.utc(2023, 3, 21) #Date
lat = '47.3167 N' #Location
long = '18.9114 E'
elevation = 1.5 #Elevation

n = 2 #we divide the time spent in daylight to this many equal timesteps

t_ls, t_since_sunrise, phi_ls, sun_theta_ls = get_solar_positions(day, lat, long, elevation, n)


for i in range(0, n):


    print('CIKLUS =', i)

    receiver_phi = 3*np.pi/2 #receiver position
    receiver_r = 40

    receiver = receiver_r * np.array([np.cos(receiver_phi), -np.sin(receiver_phi), 0.5]) #0 -15 15 surely works, actually negative x works, positive x fixed


    test_playground = playground()

    position_ls = np.array([0, 0, 0]) #mirror center #1 4 2 surely works

    input = np.array([np.cos(phi_ls[i]), -np.sin(phi_ls[i]), np.tan(sun_theta_ls[i])])

    reflected = (receiver-position_ls) #reflected

    ray_count_ls = []

    test_playground = playground()
    initialize_rays_parallel_plane(test_playground, 10, [0,0,0], 15, 15, phi_ls[i], sun_theta_ls[i])

    normal_phi = mirror_normal_calculator(position_ls, receiver, phi_ls[i], sun_theta_ls[i])[0]
    normal_theta =  mirror_normal_calculator(position_ls, receiver, phi_ls[i], sun_theta_ls[i])[1]


    test_playground.add_cubic_receiver(receiver, 5, 5, 5)
    test_playground.add_rect_mirror(position_ls[0], position_ls[1], position_ls[2], normal_phi,normal_theta,15,15,mirror_type='mirror')
    test_playground.simulate()
    test_playground.display(xlim=[-50,50], ylim=[-50,50], zlim=[0,45], show_mirrors=True, show_mirror_normals=True)
    #ray_count_ls.append(test_playground.mirrors[0].ray_count)

