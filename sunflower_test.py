import numpy as np
import matplotlib.pyplot as plt

def create_sunflower_positions(safety_distance, mirror_number):
    golden = (3 - np.sqrt(5)) * np.pi
    x_pozik = []
    y_pozik = []
    position_ls = []
    for k in range(safety_distance, safety_distance + mirror_number):
        x = np.sqrt(k+1)*np.sin((k+1)*golden)
        y = np.sqrt(k+1)*np.cos((k+1)*golden)
        position_ls.append([x, y, 5])
        x_pozik.append(x)
        y_pozik.append(y)
    return position_ls

position_ls = create_sunflower_positions(4, sum([8, 16, 24]))

x_pos = []
y_pos = []

for b in range(len(position_ls)):
    x_pos.append(position_ls[b][0])
    y_pos.append(position_ls[b][1])

plt.figure(figsize=(6,6)) # default is (8,6)
plt.plot(x_pos, y_pos, 'x')
plt.ylim([-30, 30])
plt.xlim([-30, 30])
plt.show()

normals = []

for position in position_ls:
    normals.append(np.linalg.norm(position))

print('MAXIMUM', max(normals))

print(sum([8, 16, 24]))