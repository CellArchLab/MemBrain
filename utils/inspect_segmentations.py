import utils.star_utils as star_utils
import os
import numpy as np
from utils import data_utils
import matplotlib.pyplot as plt
from config import *
from sklearn.decomposition import PCA


point_list = []
def line_picker(line, mouseevent):
    """
    find the points within a certain distance from the mouseclick in
    data coords and attach some extra attributes, pickx and picky
    which are the data points that were picked
    """
    if mouseevent.xdata is None:
        return False, dict()
    point_list[-1] = np.array((mouseevent.xdata, mouseevent.ydata))
    print("Adding point", point_list[-1])
    return True, dict()

def find_mean_idx_z(seg_mask):
    z_idcs = np.argwhere(np.sum(seg_mask, axis=(0, 1)) != 0)
    return int(round(np.mean(z_idcs)))


def find_xy_lim(image):
    x_y_idcs = np.argwhere(image == np.max(image))
    ylim = (np.min(x_y_idcs[:, 0]) - 20, np.max(x_y_idcs[:, 0]) + 20)
    xlim = (np.min(x_y_idcs[:, 1]) - 20, np.max(x_y_idcs[:, 1]) + 20)
    return xlim, ylim


def fuse_segmentations_together(star_file, temp_folder):
    """
    Fuses all segmentations of a tomogram into one single membrane segmentation file. Different membranes will have
    different values, sorted from 1 to N.
    The all_mbs file can then be opened by the "inspect_segmentations" scripts in order to inspect them more quickly.
    """
    data_dict = star_utils.read_star_file_as_dict(star_file)
    tomo_tokens = data_dict['tomoToken']
    seg_paths = data_dict['segPath']
    mb_tokens = data_dict['mbToken']
    stack_tokens = data_dict['stackToken']
    prev_token = ''
    print("All Tomo -- Stack -- Mb Combinations")
    for i, tomo_token in enumerate(tomo_tokens):
        print(tomo_token, stack_tokens[i],  mb_tokens[i])
    all_segs = None
    mem_count = 0
    for i, tomo_token in enumerate(tomo_tokens):
        if tomo_token != prev_token:
            seg = data_utils.load_tomogram(seg_paths[i])
            if all_segs is None:
                all_segs = np.array(seg != 0) * 1.0
            else:
                all_segs += np.array(seg != 0) * 1.0 * (mem_count+1)
            mem_count += 1
        else:
            data_utils.store_tomogram(os.path.join(temp_folder, tomo_token + '_all_mbs.mrc'),
                                      np.array(all_segs, dtype=np.float32))
            all_segs = np.array(seg != 0) * 1.0
        data_utils.store_tomogram(os.path.join(temp_folder, tomo_token + '_all_mbs.mrc'),
                                  np.array(all_segs, dtype=np.float32))


def inspect_segmentation_before(star_file, out_star_file, temp_folder):
    """
    Visualizes membrane segmentations with tomogram densities.
    The user can click on one side of the membrane to choose the side of interest.
    """
    print(star_file)
    data_dict = star_utils.read_star_file_as_dict(star_file)
    tomo_tokens = data_dict['tomoToken']
    tomo_paths = data_dict['tomoPath']
    mb_tokens = data_dict['mbToken']
    stack_tokens = data_dict['stackToken']
    origin_pos_z = []
    prev_token = ''
    tomo = None
    for i, tomo_token in enumerate(tomo_tokens):
        print(tomo_token, mb_tokens[i])
    mem_count = 0
    for i, tomo_token in enumerate(tomo_tokens):
        if tomo_token != prev_token:
            tomo = data_utils.load_tomogram(tomo_paths[i])
            tomo_shape = tomo.shape
            all_seg = data_utils.load_tomogram(os.path.join(temp_folder, tomo_token + '_all_mbs.mrc'))
            prev_token = tomo_token

        seg = all_seg == (mem_count + 1)

        mem_count += 1

        z_idx = find_mean_idx_z(seg)
        origin_pos_z.append(z_idx)
        image = np.transpose(tomo[:, :, z_idx], (1, 0)).copy()
        seg_img = np.transpose(seg[:, :, z_idx], (1, 0))

        if PICK_ON_BOTH_SIDES:
            seg_idcs = np.argwhere(seg_img)
            pca = PCA(n_components=1)
            pca.fit(seg_idcs)
            pca_vec = pca.components_[0]
            add_vec = np.array((pca_vec[1], -pca_vec[0]))
            center = np.mean(seg_idcs, axis=0)
            pointA_y = center[0] - add_vec[0] * 300
            pointB_y = center[0] + add_vec[0] * 300
            pointA_x = center[1] - add_vec[1] * 300
            pointB_x = center[1] + add_vec[1] * 300
            point_list.append(np.array((pointA_x, pointA_y)))
            point_list.append(np.array((pointB_x, pointB_y)))
            origin_pos_z.append(z_idx)
        else:
            image[seg_img] = np.max(image)
            xlim, ylim = find_xy_lim(seg_img)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_title('Tomogram: ' + tomo_token + ', Membrane: ' + mb_tokens[i] + 'Stack: ' + stack_tokens[i] +
                          "\nPlease specify the picking side by clicking somewhere on the correct side\n and closing this "
                          "window. The next membrane will appear. \n(hint: click with a large distance to the membrane for "
                          "more stability)")
            point_list.append(None)
            ax1.imshow(image, cmap='gray', origin='lower', picker=line_picker)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()
    origin_pos_x = [point[0] for point in point_list]
    origin_pos_y = [point[1] for point in point_list]
    star_utils.copy_star_file(star_file, out_star_file)
    if PICK_ON_BOTH_SIDES:
        star_utils.spread_lines(out_star_file, line_spread=2)
        star_utils.split_membranes_for_both_side_picks(out_star_file)
    star_utils.add_or_change_column_to_star_file(out_star_file, 'origin_pos_x', origin_pos_x)
    star_utils.add_or_change_column_to_star_file(out_star_file, 'origin_pos_y', origin_pos_y)
    star_utils.add_or_change_column_to_star_file(out_star_file, 'origin_pos_z', origin_pos_z)



def inspect_segmentation_after(mics_dir, segs_dir, temp_folder):
    """
    Visualizes membrane segmentations, as well as membrane sides, together with tomogram densities.
    This step allows the user to verify that the picked membrane sides are correct.
    """
    mics_paths, mics_tokens, segs_paths, segs_tokens = [], [], [], []
    for file in os.listdir(mics_dir):
        if file.endswith('mrc'):
            mics_tokens.append(file)
    for file in os.listdir(segs_dir):
        if file.endswith('mrc'):
            segs_tokens.append(file)
    for file in segs_tokens:
        if file not in mics_tokens:
            segs_tokens.remove(file)
    for file in mics_tokens:
        if file not in segs_tokens:
            mics_tokens.remove(file)
    for i, file in enumerate(mics_tokens):
        mics_paths.append(os.path.join(mics_dir, file))
        segs_paths.append(os.path.join(segs_dir, file))

    for i, tomo_file in enumerate(mics_paths):
        tomo = data_utils.load_tomogram(tomo_file)
        seg = data_utils.load_tomogram(segs_paths[i])
        z_idx = find_mean_idx_z(seg)
        temp_tomo = tomo.copy()
        tomo[seg == 1] = 200
        tomo[seg == 2] = 300
        temp_tomo[seg == 1] = 60
        temp_tomo[0, 0, z_idx] = 60
        image = np.transpose(tomo[:, :, z_idx], (1, 0))
        image_orig = np.transpose(temp_tomo[:, :, z_idx], (1, 0))
        plt.subplot(1,2,1)
        plt.imshow(image_orig, cmap='gray', origin='lower')
        # plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(image, cmap='gray', origin='lower')
        # plt.colorbar()
        plt.show()



# star_file = '/fs/pool/pool-engel/Lorenz/Matthias_project/pipeline/initial_stars/Tomo02_matthias.star'
# inspect_segmentation_after('/fs/pool/pool-engel/Lorenz/Matthias_project/pipeline/mics', '/fs/pool/pool-engel/Lorenz/Matthias_project/pipeline/segs')
# inspect_segmentation_after('/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/mics', '/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/segs')
# inspect_segmentation_after('/fs/pool/pool-engel/Lorenz/real_data/mics/all_tomos_all_mbs_bin2/inspection_temp/', '/fs/pool/pool-engel/Lorenz/real_data/segs/all_tomos_all_mbs_bin2/inspection_temp/')