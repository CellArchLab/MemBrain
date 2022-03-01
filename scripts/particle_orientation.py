from utils import data_utils, process_meshes
import numpy as np
from scripts import add_labels_and_distances
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from utils.parameters import ParameterSettings
import os
from mpl_toolkits.mplot3d import Axes3D
import utils.star_utils as star_utils
from scipy.spatial.transform import Rotation as R
from scripts.add_labels_and_distances import read_GT_data_membranorama_xml
import re
from sklearn.metrics.pairwise import euclidean_distances
from config import *


def get_single_cluster_orientation(points, angles, return_euler_angles=True):
    points = points.copy()
    pca = PCA(n_components=1)
    pca.fit(points)
    pca_vec = pca.components_[0]
    angles = angles[[2, 1, 0]]
    # angle = 0
    # angles = np.deg2rad(angles)

    if return_euler_angles:
        pca_vec_BU = pca_vec.copy()
        rot = R.from_euler('zyz', angles, degrees=True)
        Z1_rot = add_labels_and_distances.z_rot_matrix(np.deg2rad(angles[2]))
        Y_rot = add_labels_and_distances.y_rot_matrix(np.deg2rad(angles[1]))
        Z2_rot = add_labels_and_distances.z_rot_matrix(np.deg2rad(angles[0]))
        rot2 = np.dot(np.dot(Z1_rot, Y_rot), Z2_rot)
        pca_vec = rot.apply(pca_vec)
        pca_vec = np.dot(rot2, pca_vec_BU)
        pca_vec[2] = 0
        pca_vec = pca_vec / np.linalg.norm(pca_vec)

        # polar coordinates give us the angle
        angle = np.rad2deg(np.arctan2(pca_vec[0], pca_vec[1])) * -1 + 90
        # angle = np.rad2deg(np.arccos(np.dot(pca_vec[0], pca_vec[1])))
        # angle = 0
        return pca_vec_BU, angle
    return pca_vec


def compute_particle_orientations(cluster_centers, points_with_labels, filename, angles, visualize=False):
    if cluster_centers.shape[1] == 4:
        cluster_centers = cluster_centers[:, :3]
    orientations = np.zeros((cluster_centers.shape[0], 3))
    euler_angles = np.zeros((cluster_centers.shape[0], 3))
    euler_angles[:, 1:] = angles[:, 1:]
    unique_center_idcs = np.unique(points_with_labels[:, 3])
    for i in range(cluster_centers.shape[0]):
        cur_center_idx = unique_center_idcs[i]
        cur_points = points_with_labels[points_with_labels[:, 3] == cur_center_idx][:, :3]
        # orientations[i], euler_angles[i, 0] = np.array(get_single_cluster_orientation(cur_points, angles[i], return_euler_angles=True))
        ors, angs = get_single_cluster_orientation(cur_points, angles[i], return_euler_angles=True)
        orientations[i] = np.array(ors)
        euler_angles[i, 0] = np.array(angs)
        # orientations[i], euler_angles[i, 0] = np.array(get_single_cluster_orientation(cur_points, angles[i], return_euler_angles=True))
        if visualize:
            visualize_outputs(center=cluster_centers[i], points=cur_points, pca=orientations[i])
    mask = np.ones(cluster_centers.shape[0])
    # idcs = [3,4,8,10]
    # mask[idcs] = 0
    mask = mask == 1
    data_utils.store_np_in_vtp(np.concatenate((cluster_centers, orientations), 1)[mask], filename)
    out_csv_file = os.path.splitext(filename)[0] + '.csv'
    data_utils.store_array_in_csv(out_csv_file, np.concatenate((cluster_centers, orientations, euler_angles), 1))
    return out_csv_file


def load_centers_and_points_and_angles(cur_cluster_path, cur_associated_points_path, tomo_token, mb_token, cluster_dir, settings, mode='MS'):
    assert isinstance(settings, ParameterSettings)
    centers = np.array(data_utils.get_csv_data(cur_cluster_path), dtype=np.float)[:, :3]
    out_file = os.path.splitext(os.path.basename(cur_cluster_path))[0] + '.vtp'
    points = np.array(data_utils.get_csv_data(cur_associated_points_path), dtype=np.float)
    angles = np.array(data_utils.get_csv_data(cur_cluster_path), dtype=np.float)[:, 6:9]
    return centers, points, out_file, angles


def compute_all_orientations(cluster_dir, settings, out_dir, mode='MS', visualize=False):
    assert isinstance(settings, ParameterSettings)
    star_dict = star_utils.read_star_file_as_dict(settings.star_file)
    tomo_tokens = settings.tomo_tokens
    mb_tokens = settings.mb_tokens
    out_csv_files = []
    for tomo_token in tomo_tokens:
        for mb_token in mb_tokens[tomo_token]:
            tomo_mb_idx = star_utils.find_tomo_mb_idx(tomo_token, mb_token, star_dict)
            cur_cluster_path = star_dict['clusterPath'][tomo_mb_idx]
            cur_associated_points_path = star_dict['pointsWithClusterLabels'][tomo_mb_idx]

            cur_centers, cur_points, cur_file, cur_angles = load_centers_and_points_and_angles(cur_cluster_path, cur_associated_points_path, tomo_token, mb_token, cluster_dir, settings,
                                                                        mode=mode)
            cur_file = os.path.join(out_dir, cur_file)
            out_csv_file = compute_particle_orientations(cur_centers, cur_points, cur_file, cur_angles, visualize=visualize)
            out_csv_files.append(out_csv_file)
    star_token = os.path.basename(settings.star_file)
    star_dict['clusterPath'] = out_csv_files
    out_star_file = os.path.join(out_dir, star_token)
    star_utils.write_star_file_from_dict(out_star_file, star_dict)
    return out_star_file



def visualize_outputs(center, points, pca):
    fig = plt.figure()
    axes = plt.gca()
    ax = Axes3D(fig)
    ax.set_xlim([center[0] - 20, center[0] + 20])
    ax.set_ylim([center[1] - 20, center[1] + 20])
    ax.set_zlim([center[2] - 20, center[2] + 20])
    ax.scatter(center[0], center[1], center[2], s=100)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=20)
    ax.quiver(center[0], center[1], center[2], 10 * pca[0], 10 * pca[1], 10 * pca[2])
    plt.show()


def get_tokens_from_filename(filename):
    print(filename)
    mb_token = filename.split('M')[1]
    mb_token = 'M' + re.split(r'\D+', mb_token)[0]
    if re.split(r'\D+', mb_token)[1][0] in ['a', 'b', 'c']:
        mb_token +=re.split(r'\D+', mb_token)[1][0]
    tomo_token = filename.split('T')[1]
    tomo_token = 'T' + re.split(r'\D+', tomo_token)[0]
    stack_token = filename.split('S')[1]
    stack_token = 'S' + re.split(r'\D+', stack_token)[0]
    return tomo_token, stack_token, mb_token


def get_correct_file_from_gt_dir(gt_dir, tomo_token, mb_token, stack_token):
    for file in os.listdir(os.path.join(gt_dir, tomo_token, 'as_xml')):
        filename = os.path.join(gt_dir, tomo_token, 'as_xml', file)
        if os.path.isdir(filename):
            continue
        tokens = get_tokens_from_filename(file)
        if all((mb_token == tokens[2], stack_token == tokens[1])):
            return filename
    else: return  None
def get_correct_file_from_obj_dir(gt_dir, tomo_token, mb_token, stack_token):
    for file in os.listdir(gt_dir):
        filename = os.path.join(gt_dir, file)
        if os.path.isdir(filename):
            continue
        tokens = get_tokens_from_filename(file)
        if all((mb_token == tokens[2], stack_token == tokens[1])):
            return filename
    else: return  None


