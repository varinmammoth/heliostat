#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from playground_fast import *
from initialization_code_fast import *
# %%
def load_pickle(filename: str):
    with open(f'{filename}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# plt.rc('font', size=13)
# plt.rc('axes', labelsize=13)
# # %%
# ground_length = 30
# ground_lim = ground_length/2
# radus = 10
# mirror_num_ls = [4]
# mirror_dim = [1.,1.]
# receiver_dim = [1.,1.,1.]
# receiver_pos = [0.,0.,15.]
# position_ls = create_circular_positions(15, mirror_num_ls)
# # %%
# mirror_ls = initialise_mirrors_optimal(position_ls, receiver_pos, theta=np.pi/4, phi=0,  a=mirror_dim[0], b=mirror_dim[1])
# mirror_ls = add_receiver(mirror_ls, receiver_pos, *receiver_dim)
# ray_ls = initialize_rays_parallel(len(mirror_ls), xlim=[-ground_lim, ground_lim], ylim=[-ground_lim,ground_lim], ray_density=30, phi=0, theta=np.pi/4)
# playground1 = playground(mirror_ls, ray_ls)
# playground1.simulate()
# # %%
# visualize(*playground1.get_history(), show_rays=False)
# # %%
# R_list = [6, 7, 8, 9, 10, 11, 12, 13, 14]
# bangkok_pavg = load_pickle('bangkok_pavg')
# london_pavg = load_pickle('london_pavg')

# max_power_bangkok = 22099.928241735983
# max_power_london = 19379.433

# plt.figure(figsize=(7,5), dpi=800)
# plt.errorbar(R_list, bangkok_pavg/max_power_bangkok, yerr=(1.02*bangkok_pavg-bangkok_pavg)/max_power_bangkok, fmt='.', capsize=2, label='Bangkok')
# plt.errorbar(R_list, london_pavg/max_power_london, yerr=(1.02*london_pavg-london_pavg)/max_power_london, fmt='.', capsize=2, label='London')
# plt.legend()
# plt.xlabel(r'$R\ (m)$')
# plt.ylabel(r'$\langle P \rangle _{t}/P_{max}$')
# plt.show()
# %%
