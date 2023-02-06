#%%
import numpy as np
import matplotlib.pyplot as plt
from playground_code import playground 
from playground_code import rot_vector
from initialization_code import *
#%%
day = ts.utc(2014, 6, 19)
lat = '13.7563 N'
long = '100.5018 E'
elevation = 1.5
t_ls, t_since_sunrise, phi, theta = get_solar_positions(day, lat, long, elevation, 20)
plt.plot(t_since_sunrise, phi, '.', label=r'$Azimuth\ \phi$')
plt.plot(t_since_sunrise, theta, '.', label=r'$Elevation\ \theta$')
plt.legend()
plt.grid()
plt.xlabel(r'$Time\ since\ sunrise\ (s)$')
plt.ylabel(r'$Solar\ angles$')
plt.show()
# %%
