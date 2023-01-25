#%%
import matplotlib.pyplot as plt
import numpy as np

def plane(cx, cy, cz, nx, ny, nz, length, width):
    x = np.linspace(cx - length/2, cx + length/2)
    y = np.linspace(cy - width/2, cy + width/2)
    X, Y = np.meshgrid(x, y)
    Z = (nx*cx + ny*cy + nz*cz)/nz - (nx/nz)*X - (ny/nz)*Y
    return X, Y, Z

# coordinates of the center point
cx, cy, cz = 0, 0, 0
# components of the normal vector
nx, ny, nz = 1, 1e-8, 1e-8
# dimensions of the rectangle
length, width = 2, 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = plane(cx, cy, cz, nx, ny, nz, length, width)
ax.plot_surface(X, Y, Z, alpha=1)
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
plt.show()
# %%
