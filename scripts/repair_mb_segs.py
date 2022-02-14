import numpy as np
from matplotlib import pyplot as plt
from utils import data_utils


def add_points_to_seg(in_tomo, out_tomo, point_coords):
    tomo = data_utils.load_tomogram(in_tomo)
    for coord in point_coords:
        tomo[coord] = 1
    tomo = np.transpose(tomo, (2,1,0))
    data_utils.store_tomogram(out_tomo, tomo)

def remove_z_slice_from_seg(in_tomo, out_tomo, z_slices):
    tomo = data_utils.load_tomogram(in_tomo)
    for z_slice in z_slices:
        tomo[:, :, z_slices] = 0
    tomo = np.transpose(tomo, (2,1,0))
    data_utils.store_tomogram(out_tomo, tomo)


def visualize_z_slices(in_tomo, z_slices):
    tomo = data_utils.load_tomogram(in_tomo)
    plt.figure()
    for k,z in enumerate(z_slices):
        if len(z_slices) <= 5:
            plt.subplot(1,len(z_slices), k + 1)
        else:
            plt.subplot(2, int(len(z_slices) / 2), k+1)
        plt.title(z)
        plt.imshow(tomo[:,:, z])
    plt.show()



for i in range(24, 28):
    print(i)
    in_tomo = '/fs/pool/pool-engel/Spinach_project/folder_for_lorenz/190601/Tomo_0017/all_no_positions/membranes/T17S1C2M' + str(i) + '.mrc'
    out_tomo = '/fs/pool/pool-engel/Spinach_project/folder_for_lorenz/190601/Tomo_0017/all_no_positions/membranes/T17S1C2M' + str(i) + '_V2.mrc'
    point_coords = [(472, 207, 259), (472, 208, 259), (472, 209, 259),  (472, 210, 259),  (472, 211, 259),  (472, 212, 259),
                    (472, 207, 260), (472, 208, 260), (472, 209, 260), (472, 210, 260), (472, 211, 260)]
    # remove_z_slice_from_seg(in_tomo, out_tomo, [196, 261])
    # add_points_to_seg(in_tomo, out_tomo, point_coords)
    visualize_z_slices(out_tomo, range(258, 263))