from utils import data_utils, mahalanobis_covariance, process_meshes
import numpy as np
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
        Z1_rot = mahalanobis_covariance.z_rot_matrix(np.deg2rad(angles[2]))
        Y_rot = mahalanobis_covariance.y_rot_matrix(np.deg2rad(angles[1]))
        Z2_rot = mahalanobis_covariance.z_rot_matrix(np.deg2rad(angles[0]))
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
        orientations[i], euler_angles[i, 0] = np.array(get_single_cluster_orientation(cur_points, angles[i], return_euler_angles=True))
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
    # all_files = os.listdir(cluster_dir)
    # for file in all_files:
    #     tomo_token_temp, mb_token_temp = data_utils.get_tomo_and_mb_from_file_name(file, settings)
    #     if tomo_token_temp is not None and mb_token_temp is not None:
    #         if tomo_token == tomo_token_temp and mb_token == mb_token_temp:
    #             if file.startswith(mode) and not '_all_coords' in file and '_Tomo' in file:
    #                 centers = np.array(data_utils.get_csv_data(os.path.join(cluster_dir, file)), dtype=np.float)[:, :3]
    #                 out_file = os.path.splitext(file)[0] + '.vtp'
    #             if file.startswith(mode) and  '_all_coords' in file:
    #                 points = np.array(data_utils.get_csv_data(os.path.join(cluster_dir, file)), dtype=np.float)
    return centers, points, out_file, angles


def compute_all_orientations(cluster_dir, settings, out_dir, mode='MS', visualize=False):
    assert isinstance(settings, ParameterSettings)
    star_dict = star_utils.read_star_file_as_dict(settings.star_file)
    tomo_tokens = settings.tomo_tokens
    mb_tokens = settings.mb_tokens
    out_csv_files = []
    for tomo_token in tomo_tokens:
        for mb_token in mb_tokens[tomo_token]:
            # if not (tomo_token == 'Tomo1L1' and '2' == mb_token):
            #     continue
            # if not mb_token == 'M34':
            #     continue
            tomo_mb_idx = star_utils.find_tomo_mb_idx(tomo_token, mb_token, star_dict)
            cur_cluster_path = star_dict['clusterPath'][tomo_mb_idx]
            cur_associated_points_path = star_dict['pointsWithClusterLabels'][tomo_mb_idx]

            cur_centers, cur_points, cur_file, cur_angles = load_centers_and_points_and_angles(cur_cluster_path, cur_associated_points_path, tomo_token, mb_token, cluster_dir, settings,
                                                                        mode=mode)
            cur_file = os.path.join(out_dir, cur_file)
            print cur_file
            out_csv_file = compute_particle_orientations(cur_centers, cur_points, cur_file, cur_angles, visualize=visualize)
            out_csv_files.append(out_csv_file)
    star_token = os.path.basename(settings.star_file)
    star_dict['clusterPath'] = out_csv_files
    star_utils.write_star_file_from_dict(os.path.join(out_dir, star_token), star_dict)



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

def quantify_orientations(in_star):
    star_dict = star_utils.read_star_file_as_dict(in_star)
    settings = ParameterSettings(in_star)
    all_diffs_abs = []
    all_diffs_mod = []
    all_diffs_random = []

    for tomo_token, mb_token, stack_token, gt_dir, mesh_path, cluster_path in zip(star_dict['tomoToken'], star_dict['mbToken'],
                                                                          star_dict['stackToken'], star_dict['gtPath'],
                                                                          star_dict['objDir'], star_dict['clusterPath']):
        gt_file = get_correct_file_from_gt_dir(gt_dir, tomo_token, mb_token, stack_token)
        objfile = get_correct_file_from_obj_dir(mesh_path, tomo_token, mb_token, stack_token)
        gt_dict, ori_dict = read_GT_data_membranorama_xml(gt_file, settings=settings, return_ids=True, return_orientation=True)
        gt_pos = gt_dict['PSII']
        gt_ori = ori_dict['PSII']
        cur_mesh = process_meshes.read_obj_file_to_triangles(objfile)

        tri_ids = gt_pos[:, 3]
        gt_pos = gt_pos[:, :3] * 4
        cluster_centers = np.array(data_utils.get_csv_data(cluster_path), dtype=np.float)


        dist_mat = euclidean_distances(cluster_centers[:, :3], gt_pos)
        nn_gts = np.argmin(dist_mat, axis=1)
        nearenough_mask = np.min(dist_mat, axis=1) < 18.0
        nn_gts = nn_gts[nearenough_mask]
        cluster_centers = cluster_centers[nearenough_mask]


        for cur_cen, nn_gt in zip(cluster_centers, nn_gts):
            cur_ori = gt_ori[nn_gt]
            cur_tri_id = tri_ids[nn_gt]
            plane_mat = cur_mesh.get_triangle(cur_tri_id).get_plane_matrix()
            # plane_mat = inv(plane_mat)
            vec = cur_cen[3:6]
            print(vec)
            print(np.dot(plane_mat, vec))
            vec =np.dot(plane_mat, vec)
            # alpha = np.arccos(vec[2])
            beta = np.arctan2(vec[0], vec[1])
            # beta = np.arctan2(vec[1], vec[0])

            int1 = np.random.randint(0, 360)
            int2 = np.random.randint(0, 360)
            # all_diffs.append( np.rad2deg(beta)%180 -  cur_ori[0]%180)
            # all_diffs.append((np.rad2deg(beta)%180 - cur_ori[0]%180))
            # all_diffs_mod.append(np.abs((np.rad2deg(beta)%180 - cur_ori[0]%180)%180 - 108.8))
            difference = ((np.rad2deg(beta)%180 - cur_ori[0]%180)%180) -120.1
            if difference < -90:
                difference = 90 - (-90 - difference)

            all_diffs_mod.append(difference)
            all_diffs_abs.append(np.abs(difference))
            # ori = cur_ori[0] % 180
            # beta = (np.rad2deg(beta) +42) % 180
            # beta = (np.rad2deg(beta) - 148) % 180
            # diff = np.abs(beta - ori) % 180



            # if diff > 90:
            #     res = 180 - diff
            # else:
            #     res = diff
            # a = ori
            # b = beta
            # sign = (((a - b >= 0 and a - b <= 90) or (a - b <= -90 and a - b >= -180)) - 0.5) * 2
            # res *= sign

            # all_diffs.append((np.rad2deg(beta)%180 - cur_ori[0]%180))
            # all_diffs.append((np.rad2deg(beta) - cur_ori[0])%180)
            # all_diffs_random.append((int1 - int2)%180)
            # all_diffs_mod.append(np.abs(res))

            print(np.rad2deg(beta)%180, cur_ori[0]%180, (np.rad2deg(beta)%180 - cur_ori[0]%180))
            print("")
    # print(np.median(all_diffs))
    print(np.mean(all_diffs_mod))
    print(np.mean(all_diffs_abs))
    print(np.std(all_diffs_mod))
        # for cur_gt_pos
    # plt.subplot(1,3,1)
    # plt.hist(all_diffs, bins=15)
    plt.subplot(1,2,1)
    # Axes.hist(all_diffs_abs, bins=20)
    plt.hist(all_diffs_abs, bins=20, rwidth=1.0, label="Absolute error")
    # plt.title(("Absolute error"))
    plt.subplot(1,2,2)
    # Axes.hist(all_diffs_abs, bins=20)
    plt.hist(all_diffs_mod, bins=20, density=False, label="Error", )
    # plt.title("Error")
    plt.show()


cases = ['Chlamy', 'spinach', 'synthetic', 'synthetic_bin4', 'spinach_val', 'chlamy_bin4', 'new_spinach']
case = 'new_spinach'
assert case in cases

if case == 'spinach':
    in_star = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/cluster_centers/cluster_centers_classified/chlamy_bw23.star'
    cluster_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/cluster_centers/cluster_centers_classified/'
    out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/cluster_centers/cluster_centers_with_orientation/'

if case == 'spinach_val':
    # in_star = '/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/cluster_centers_classified/chlamy_bw23.star'
    in_star = '/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/cluster_centers/cluster_centers_classified/chlamy_bw23.star'
    cluster_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/cluster_centers/'
    out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach_validation/cluster_centers/cluster_centers_with_orientation/'

elif case == 'Chlamy':
    cluster_dir = '/fs/pool/pool-engel/Lorenz/pipeline_spinach/cluster_centers/'
    out_dir = '/fs/pool/pool-engel/Lorenz/orientations'

elif case == 'chlamy_bin4':
    in_star = '/fs/pool/pool-engel/Lorenz/pipeline_Chlamy_bin4/cluster_centers/cluster_centers_classified/chlamy_bw23.star'
    # in_star = '/fs/pool/pool-engel/Lorenz/pipeline_Chlamy_bin4/cluster_centers/chlamy_bw23.star'
    cluster_dir = None
    out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_Chlamy_bin4/cluster_centers/cluster_centers_with_orientation'

elif case == 'new_spinach':
    in_star = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/new_spinach.star'
    # in_star = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/cluster_centers_classified/new_spinach_test.star'
    cluster_dir = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/'
    out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/cluster_centers_with_orientation/'


settings = ParameterSettings(in_star, cluster_centers=True)
compute_all_orientations(cluster_dir, settings, out_dir, visualize=True)
# quantify_orientations('/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/cluster_centers_with_orientation/new_spinach_test.star')
quantify_orientations('/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/cluster_centers/cluster_centers_with_orientation/new_spinach.star')


# test_center = np.array([2,3,4])
# test_points = np.array([[2,3,1], [2,3,7], [2,3,3], [4,3,2],[1,1,5], [2,4,1], [2,3,2]])
# # visualize_outputs(test_center, test_points, [1,1,1])
# ax.scatter(test_points[:,0], test_points[:,1], test_points[:,2])
# # plt.show()
# # get_single_cluster_orientation(test_center, test_points)
#
# settings = ParameterSettings()
# tomo_tokens = settings.tomo_tokens
# mb_tokens = settings.mb_tokens
# tomo_token = tomo_tokens[0]
# mb_token = mb_tokens[tomo_token][1]
# print tomo_token, mb_token
# cluster_dir = '/fs/pool/pool-engel/Lorenz/cluster_center_test/'
# cen, poin = load_centers_and_points(tomo_token, mb_token, cluster_dir, settings)
# print cen.shape, poin.shape
# get_particle_orientations(cen, poin)

