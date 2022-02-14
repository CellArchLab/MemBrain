from utils.parameters import ParameterSettings
from utils import star_utils, data_utils, process_meshes
import numpy as np
import os
from preprocessing import create_DL_data
from utils.mahalanobis_covariance import z_rot_matrix


in_star = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach_all/cluster_centers/new_spinach_tomo17.star'
out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach_all/gt_coords/Tomo_0017/post_processing'
settings = ParameterSettings(in_star)
settings.objPaths['Tomo_0017'] = '/fs/pool/pool-engel/Spinach_project/tomo17_analysis/s1_cut1/Tomo_0017_stats_format/'
settings.gt_paths['Tomo_0017'] = '/fs/pool/pool-engel/Spinach_project/tomo17_analysis/positions/Tomo_0017_stats_format/'

in_star = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach_all/cluster_centers/new_spinach_tomo17.star'
in_star = '/fs/pool/pool-engel/Lorenz/pipeline_Chlamy_bin4/cluster_centers/cluster_centers_classified/chlamy_bw23.star'
out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach_all/gt_coords/Tomo_0017/post_processing'
out_dir = '/fs/pool/pool-engel/Lorenz/pipeline_Chlamy_bin4/gt_coords/post_processing/'
settings = ParameterSettings(in_star)
settings.objPaths['Tomo_0017'] = '/fs/pool/pool-engel/Spinach_project/tomo17_analysis/s1_cut1/Tomo_0017_stats_format/'
settings.gt_paths['Tomo_0017'] = '/fs/pool/pool-engel/Spinach_project/tomo17_analysis/positions/Tomo_0017_stats_format/'
settings.objPaths['Tomo_1'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_1/meshes'
settings.objPaths['Tomo_21'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_21/meshes'
settings.objPaths['Tomo_8'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_8/meshes'
settings.objPaths['Tomo_9'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_9/meshes'
settings.gt_paths['Tomo_1'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_1/Tomo1_positions'
settings.gt_paths['Tomo_21'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_21/Tomo21_positions'
settings.gt_paths['Tomo_8'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_8/Tomo8_positions'
settings.gt_paths['Tomo_9'] = '/fs/pool/pool-engel/Lorenz/Chlamy_stats_S1/Tomo_9/Tomo9_positions'
star_dict = star_utils.read_star_file_as_dict(in_star)
for case in ['PSII', 'b6f', 'UK']:
    new_column = []
    gt_tomos = settings.tomo_tokens
    mesh_dict = {}
    gt_mbs = settings.mb_tokens
    gt_stacks = settings.stack_tokens
    for tomo_token in np.unique(gt_tomos):
        for stack_token in np.unique(gt_stacks[tomo_token]):
            mesh_dict[(tomo_token, stack_token)] = {}
            for mb_token in np.unique(gt_mbs[tomo_token]):
                # print tomo_token, mb_token
                for file in os.listdir(settings.objPaths[tomo_token]):
                    cur_tomo, cur_mb = data_utils.get_tomo_and_mb_from_file_name(
                        os.path.join(settings.objPaths[tomo_token], file), settings)
                    cur_stack = data_utils.get_stack_from_file_name(os.path.join(settings.objPaths[tomo_token], file), settings)
                    if cur_tomo == tomo_token and cur_mb == mb_token and cur_stack == stack_token:
                        cur_mesh_path = os.path.join(settings.objPaths[tomo_token], file)
                        cur_mesh = process_meshes.read_obj_file_to_triangles(cur_mesh_path)
                        mesh_dict[(tomo_token, stack_token)][mb_token] = cur_mesh
                        print tomo_token, mb_token, cur_mesh_path
                        break

    for tomo_token, mb_token, stack_token in zip(star_dict['tomoToken'], star_dict['mbToken'], star_dict['stackToken']):
        gt_dir = settings.gt_paths[tomo_token]
        for file in os.listdir(gt_dir):
            if file.endswith('.xml') and mb_token == data_utils.get_tomo_and_mb_from_file_name(os.path.join(gt_dir, file), settings)[1]:
                break
        out_file_name = os.path.join(out_dir, tomo_token + '_' + stack_token + '_' + mb_token + '_' + case + '_gt.csv')
        print(os.path.join(gt_dir, file))
        pos_dict, ori_dict = create_DL_data.read_GT_data_xml(os.path.join(gt_dir, file), settings, return_ids=True, return_orientation=True)
        psii_pos = np.zeros((0, 4))
        psii_ori = np.zeros((0, 3))
        for key in pos_dict.keys():
            if key != case:
                continue
            # pos_dict[key][:, :3] *= 4
            psii_pos = np.concatenate((psii_pos, pos_dict[key]), 0)
            psii_ori = np.concatenate((psii_ori, ori_dict[key]), 0)

        psii_ori_new = np.zeros((0, 9))
        psii_pos_new = np.zeros((0, 4))
        print(tomo_token, mb_token, stack_token)
        for i in range(psii_ori.shape[0]):
            angles = psii_ori[i]
            angles = np.deg2rad(angles)
            # print(tomo_token, stack_token, mb_token)
            cur_mesh = mesh_dict[(tomo_token, stack_token)][mb_token]
            cur_tri_ID = psii_pos[i, 3]
            assert isinstance(cur_mesh, process_meshes.Mesh)
            if cur_tri_ID >= len(cur_mesh):
                print(case)
                print("particle has no corresponding ground truth")
                continue
            cur_triangle = cur_mesh.get_triangle(cur_tri_ID)
            assert isinstance(cur_triangle, process_meshes.Triangle)
            rot = cur_triangle.get_plane_matrix()
            z_rot = z_rot_matrix(angles[0])
            rot = np.dot(z_rot, rot)
            rot = np.transpose(rot, (1, 0))
            rot = np.concatenate((rot[:, 0], rot[:, 1], rot[:, 2]), 0)
            psii_pos_new = np.concatenate((psii_pos_new, np.expand_dims(psii_pos[i], 0)), 0)
            psii_ori_new = np.concatenate((psii_ori_new, np.expand_dims(rot, 0)), 0)
        psii_pos = psii_pos_new
        # psii_pos = pos_dict['PSII'] * 4
        add_stack = np.ones((psii_pos.shape[0], 1)) * (1 if case == 'PSII' else 2 if case == 'b6f' else 3)
        temp_stack = np.expand_dims(psii_pos[:, 3], 1)
        psii_pos[:, 3] = add_stack[:,0]
        # psii_pos = np.concatenate((psii_pos, temp_stack), 1)
        psii_pos = np.concatenate((psii_pos, psii_ori_new), 1)
        data_utils.store_array_in_csv(out_file_name, psii_pos)
        new_column.append(out_file_name)
    star_utils.add_or_change_column_to_star_file(in_star, 'gt' + case, new_column)


def fuse_csv_files(csv_files, out_csv):
    out_data = np.zeros((0, 13))
    for csvfile in csv_files:
        cur_data = np.array(data_utils.get_csv_data(csvfile))
        out_data = np.concatenate((out_data, cur_data))
    data_utils.store_array_in_csv(out_csv, out_data)


star_dict = star_utils.read_star_file_as_dict(in_star)
all_columns = []
for i in range(len(star_dict['gtPSII'])):
    cur_PSII_col = star_dict['gtPSII'][i]
    cur_PSII_col_out = cur_PSII_col[:-11] + 'all_gt.csv'
    print(cur_PSII_col_out)
    fuse_csv_files([star_dict['gtPSII'][i], star_dict['gtb6f'][i], star_dict['gtUK'][i]], cur_PSII_col_out)
    all_columns.append(cur_PSII_col_out)
star_utils.add_or_change_column_to_star_file(in_star, 'gtall', all_columns)

