# from preprocessing import create_DL_data
from utils.parameters import ParameterSettings
import numpy as np
import h5py
import lxml.etree as ET
import os
from utils import star_utils, data_utils, process_meshes
from numpy.linalg import inv


def load_hdf5_files(settings, split=None):
    """
    Load a .h5 file containing positions and subvolumes into a Python dictionary
    @param settings: Parameter settings given y star files
    @return: data dictionary
    """
    assert isinstance(settings, ParameterSettings)
    subvolume_paths = settings.sub_volume_paths
    tomo_tokens = subvolume_paths.keys()
    consider_bin = 'bin' + str(settings.consider_bin)
    data_dict = {}
    for tomo_token in tomo_tokens:
        data_dict[tomo_token] = {consider_bin: {}}
        tomo_path = subvolume_paths[tomo_token]
        # print("Loading " + tomo_path)
        with h5py.File(tomo_path, 'r') as sub_file:
            combo_keys = sub_file.get(consider_bin).keys()
            for combo_token in combo_keys:
                mb_token = 'M' + combo_token.split('M')[1]
                stack_token = combo_token.split('M')[0]
                if split is not None and settings.data_splits[tomo_token][(stack_token, mb_token)] != split:
                    continue
                tomo_token_temp = sub_file.get(consider_bin).get(combo_token).get('tomo_token')[0].decode('UTF-8')
                mb_token_temp = sub_file.get(consider_bin).get(combo_token).get('mb_token')[0].decode('UTF-8')
                assert tomo_token == tomo_token_temp and mb_token_temp == mb_token

                positions = np.array(sub_file.get(consider_bin).get(combo_token).get('positions'))
                subvolumes = np.array(sub_file.get(consider_bin).get(combo_token).get('subvolumes'))
                normals = np.array(sub_file.get(consider_bin).get(combo_token).get('normals'))
                angles = np.array(sub_file.get(consider_bin).get(combo_token).get('angles'))
                data_dict[tomo_token][consider_bin][(stack_token, mb_token)] = {
                    'positions': positions,
                    'subvolumes': subvolumes,
                    'normals': normals,
                    'angles': angles
                }
                for dist_key in sub_file.get(consider_bin).get(combo_token).keys():
                    if dist_key.startswith('dist_'):
                        data_dict[tomo_token][consider_bin][(stack_token, mb_token)][dist_key] = \
                            np.array(sub_file.get(consider_bin).get(combo_token).get(dist_key))
    return data_dict

def store_dict_in_hdf5(data_dict, settings):
    subvolume_paths = settings.sub_volume_paths
    for tomo_token in data_dict.keys():
        tomo_path = subvolume_paths[tomo_token]
        with h5py.File(tomo_path, 'w') as f:
            for bin_token in data_dict[tomo_token].keys():
                bin_group = f.create_group(bin_token)
                for stack_token, mb_token in data_dict[tomo_token][bin_token].keys():
                    cur_keys = data_dict[tomo_token][bin_token][(stack_token, mb_token)].keys()
                    mb_group = bin_group.create_group(stack_token + mb_token)
                    mb_group.create_dataset('tomo_token', shape=(1,), data=tomo_token)
                    mb_group.create_dataset('mb_token', shape=(1,), data=mb_token)
                    mb_group.create_dataset('stack_token', shape=(1,), data=stack_token)
                    mb_group.create_dataset('bin', shape=(1,), data=bin_token[-1])
                    mb_group.create_dataset('subvolumes', data=data_dict[tomo_token][bin_token][(stack_token, mb_token)]['subvolumes'])
                    mb_group.create_dataset('positions', data=data_dict[tomo_token][bin_token][(stack_token, mb_token)]['positions'])
                    if 'normals' in cur_keys and data_dict[tomo_token][bin_token][(stack_token, mb_token)]['normals'] \
                            is not None:
                        mb_group.create_dataset('normals', data=data_dict[tomo_token][bin_token][(stack_token, mb_token)]['normals'])
                    if 'angles' in cur_keys and data_dict[tomo_token][bin_token][(stack_token, mb_token)]['angles'] \
                            is not None:
                        mb_group.create_dataset('angles', data=data_dict[tomo_token][bin_token][(stack_token, mb_token)]['angles'])
                    min_dists = data_dict[tomo_token][bin_token][(stack_token, mb_token)]['distances']
                    for unique_gt in min_dists.keys():
                        mb_group.create_dataset('dist_' + unique_gt, data=min_dists[unique_gt])

def adjust_GT_data_membranorama_xml(gt_file_name, out_file_name, mode='multiply_pos', mult_fac=1.0):
    print(gt_file_name)
    tree = ET.parse(gt_file_name)
    root = tree.getroot()
    for i, elem in enumerate(root):
        if elem.tag == 'PointGroups':
            coords_id = i
            break
    point_groups = root[coords_id]
    for particle_group in point_groups:
        for point in particle_group:
            cur_pos = np.array(point.attrib['Position'].split(','), dtype=np.float)
            if mode == 'multiply_pos':
                cur_pos *= mult_fac
            point.attrib['Position'] = str(cur_pos[0]) + ',' + str(cur_pos[1]) + ',' + str(cur_pos[2])
    with open(out_file_name, 'wb') as out_f:
        out_f.write(ET.tostring(root, pretty_print=True))



def read_GT_data_membranorama_xml(gt_file_name, settings=None, prot_tokens=None, return_orientation=False, return_ids=False):
    """
    Returns a dictionary containing the positions of the ground truth particles in the specified file
    :param gt_file_name: file name of ground truth positions (should be xml formatted)
    :param settings: settings container (instance of ParameterSettings)
    :param prot_tokens: dicitonary with keys incidating names of proteins classes. Each key value is a list of possible
            spellings of the respective protein
    :param return_orientation: flag whether to return the orientation vectors (in ori_dict)
    :param return_ids: return ids of associated triangles in corresponding mesh
    :return: numpy array with x,y,z positions
    """
    if settings is not None:
        assert isinstance(settings, ParameterSettings)
        pixel_spacing_bin1 = settings.pixel_spacing
        unbinned_offset_Z = settings.unbinned_offset_Z
    else:
        pixel_spacing_bin1 = 1.
        unbinned_offset_Z = 0.
    shape2 = (4 if return_ids else 3)
    if prot_tokens is not None:
        pos_dict = {key: np.zeros((0, shape2)) for key in prot_tokens.keys()}
        orientation_dict = {key: np.zeros((0, 3)) for key in prot_tokens.keys()}
    else:
        pos_dict = {}
        orientation_dict = {}
    tree = ET.parse(gt_file_name)
    root = tree.getroot()
    for i, elem in enumerate(root):
        if elem.tag == 'PointGroups':
            coords_id = i
            break
    point_groups = root[coords_id]
    for particle_group in point_groups:
        if return_ids:
            positions = np.zeros((0, 4))
        else:
            positions = np.zeros((0, 3))
        orientations = np.zeros((0, 3))
        for point in particle_group:
            cur_pos = np.expand_dims(np.array(point.attrib['Position'].split(','), dtype=np.float), 0)
            if return_ids:
                cur_id = np.expand_dims(np.expand_dims(np.array(point.attrib['ID'], dtype=np.float), 0), 0)
                cur_pos = np.concatenate((cur_pos, cur_id), 1)
            positions = np.concatenate((positions, cur_pos), 0)
            if return_orientation:
                cur_orientation = np.expand_dims(np.array(point.attrib['Orientation'].split(','), dtype=np.float), 0)
                orientations = np.concatenate((orientations, cur_orientation), 0)


        # if settings.adjust_positions:
            #TODO: When changing from synthetic to real data, change the following:
        positions[:, 2] -= unbinned_offset_Z  # Uncomment
        positions[:, :3] = positions[:, :3] / (pixel_spacing_bin1)  # Uncomment

        if prot_tokens is not None:
            for prot_class in prot_tokens.keys():
                if any(token in particle_group.attrib['Name'] for token in prot_tokens[prot_class]):
                    store_token = prot_class
        else:
            store_token = particle_group.attrib['Name']
            if store_token not in pos_dict.keys():
                pos_dict[store_token] = np.zeros((0, shape2))
        pos_dict[store_token] = np.concatenate((pos_dict[store_token], positions), axis=0)

        if return_orientation:
            orientations = np.rad2deg(orientations)
            if store_token not in orientation_dict.keys():
                orientation_dict[store_token] = np.zeros((0, 3))
            orientation_dict[store_token] = np.concatenate((orientation_dict[store_token], orientations), axis=0)

    if return_orientation:
        return pos_dict, orientation_dict
    return pos_dict


def z_rot_matrix(phi):
    return np.array(np.array([[np.cos(phi), -1* np.sin(phi), 0.],[np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]]))

def y_rot_matrix(theta):
    return np.array(np.array([[np.cos(theta), 0., np.sin(theta)],[0., 1., 0.], [-1 * np.sin(theta), 0., np.cos(theta)]]))

def zyz_rot_matrix(phi, theta, psi):
    a = np.cos(phi)
    b = np.sin(phi)
    c = np.cos(theta)
    d = np.sin(theta)
    e = np.cos(psi)
    f = np.sin(psi)
    return np.array([[e*a*c - b*f, -a*c*f - e*b, a*d], [a*f + e*b*c, e*a - b*c*f, b*d], [-e*d, d*f, c]])



def get_rotated_inversed_covariance_matrix(angles, particle_type, cur_tri_ID, cur_tomo, cur_mb, cur_stack, mesh_dict, avoid_inversion=False):
    if particle_type == 'PSII':
        cov = psii_cov
    elif particle_type == 'b6f':
        cov = b6f_cov
    elif particle_type in ['UK', 'cube']:
        cov = uk_cov
    else:
        raise IOError('Specified particle not found! Use either \'PSII\' or \'b6f\' or \'UK\'!')
    angles = np.deg2rad(angles)
    # print mesh_dict[cur_tomo].keys()
    cur_mesh = mesh_dict[cur_tomo][cur_stack, cur_mb]
    assert isinstance(cur_mesh, process_meshes.Mesh)
    cur_triangle = cur_mesh.get_triangle(cur_tri_ID)
    assert isinstance(cur_triangle, process_meshes.Triangle)
    rot = cur_triangle.get_plane_matrix()
    z_rot = z_rot_matrix(angles[0])
    rot = np.dot(z_rot, rot)
    rot = np.transpose(rot, (1,0))
    # rot = eulerAnglesToRotationMatrix(angles)
    rot_cov = np.dot(np.dot(rot, cov), np.transpose(rot, (1, 0)))
    if avoid_inversion:
        return rot_cov
    else:
        cov_inv = inv(rot_cov)
        return cov_inv


def zyz_from_rotation_matrix(rotation_matrix):
    """
    Returns Z-Y-Z Euler angels from a given rotation matrix.
    """
    phi = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 0])
    psi = np.arctan2(rotation_matrix[1, 2], -rotation_matrix[0, 2])
    theta = np.arctan2(rotation_matrix[2, 0] * np.cos(phi) + rotation_matrix[2,1] * np.sin(phi), rotation_matrix[2,2])

    ## Above version is more robust
    # theta = np.arccos(rotation_matrix[2,2])
    # phi = np.arccos(rotation_matrix[2,0] / np.sin(theta))
    # psi = np.arccos(-1 * rotation_matrix[0,2] / np.sin(theta))
    return phi, theta, psi


def compute_Euler_angles_from_membranorama(orientation_dict, gt_dict, mesh_file, return_as_dict=False):
    """
    Computes actual Euler angles from the angles and meshes given in a membranorama file.
    @return: array of Euler angles
    """
    mesh = process_meshes.read_obj_file_to_triangles(mesh_file)
    euler_arrays = []
    euler_dict = {}
    for key, prot_orientations in orientation_dict.items():
        cur_ids = gt_dict[key][:, 3]
        prot_eulers = []
        for cur_id, cur_orientation in zip(cur_ids, prot_orientations):
            cur_triangle = mesh.get_triangle(cur_id)
            rot = cur_triangle.get_plane_matrix()
            z_rot = z_rot_matrix(np.deg2rad(cur_orientation[0]))
            rot = np.dot(z_rot, rot)
            rot = np.transpose(rot, (1, 0))
            phi, theta, psi = zyz_from_rotation_matrix(rot.T)
            euler_arrays.append(np.array((phi, theta, psi)))
            prot_eulers.append(np.array((phi, theta, psi)))
        if len(prot_eulers) > 0:
            euler_dict[key] = np.stack(prot_eulers)
    euler_arrays = np.stack(euler_arrays)
    if return_as_dict:
        return euler_dict
    return euler_arrays



def convert_membranorama_gt_to_csv(project_dir, prot_tokens, settings: ParameterSettings, convert_orientation=True):
    """
    Converts the normal membranorama output to a more readable csv file.
    @param convert_orientation: if True, membranorama orientations are also stored after conversion to proper Euler
        angles (corresponding mesh file required)
    """
    star_dict = star_utils.read_star_file_as_dict(settings.star_file)
    mb_names = [os.path.basename(mem_seg)[:-3] for mem_seg in star_dict['segPath']]
    tomo_tokens = star_dict['tomoToken']
    xml_paths = [os.path.join(project_dir, 'gt_coords', tomo_token, 'as_xml', mb_name + 'xml') for
                 (tomo_token, mb_name) in zip(tomo_tokens, mb_names)]
    csv_paths_out = [os.path.join(project_dir, 'gt_coords', tomo_token, 'as_csv', mb_name + 'csv') for
                 (tomo_token, mb_name) in zip(tomo_tokens, mb_names)]
    mesh_files = star_dict['objDir']

    for xml_path, csv_path_out, mesh_file in zip(xml_paths, csv_paths_out, mesh_files):

        header = ['Protein', 'X', 'Y', 'Z']
        if convert_orientation:
            header += ['Phi', 'Tilt', 'Psi']
        gt_dict = read_GT_data_membranorama_xml(xml_path, settings, prot_tokens, return_orientation=convert_orientation,
                                                          return_ids=convert_orientation)
        if convert_orientation:
            orientation_dict = gt_dict[1]
            gt_dict = gt_dict[0]
        data_array = np.zeros((0, 4))
        for key, temp_array in gt_dict.items():
            prot_col = np.expand_dims(np.array([key] * temp_array.shape[0]), 1)
            conc_temp = np.concatenate((prot_col, temp_array), 1)
            data_array = np.concatenate((data_array, conc_temp[:, :4]))

        if convert_orientation:
            euler_angles = compute_Euler_angles_from_membranorama(orientation_dict, gt_dict, mesh_file)
            data_array = np.concatenate((data_array, euler_angles), 1)
        data_utils.store_array_in_csv(csv_path_out, data_array, header=header)


def compute_covariance_for_particle(particle):
    occ = np.where(particle > 0.1)
    occ = np.array(np.transpose(occ, (1, 0)), dtype=np.float)
    occ -= np.mean(occ, axis=0)
    occ /= 4
    cov = np.cov(np.transpose(occ, (1, 0)))
    return cov

def initialize_global_shapes(shapes):
    shape_dict = {}
    for key, shape in shapes.items():
        if not shape.startswith('sphere'):
            cur_shape = data_utils.load_tomogram(shape)
        else:
            radius = float(shape[6:])
            cur_shape = np.zeros((int(radius * 2) + 1, int(radius * 2) + 1, int(radius * 2) + 1))
            for x in range(cur_shape.shape[0]):
                for y in range(cur_shape.shape[0]):
                    for z in range(cur_shape.shape[0]):
                        if np.linalg.norm(np.array((x,y,z)) - np.array((radius, radius, radius))) < radius:
                            cur_shape[x,y,z] = 1
        cov = compute_covariance_for_particle(cur_shape)
        shape_dict[key] = cov
    return shape_dict

def mahalanobis_for_matrix_to_point(X, point, inv_cov):
    diff = X - point
    maha = np.sqrt(np.einsum('nj,jk,nk->n', diff, inv_cov, diff))
    return maha


def compute_distances_to_gt(star_file, particle_orientations=False, prot_shapes=None):
    if prot_shapes is not None:
        shape_dict = initialize_global_shapes(prot_shapes)
    settings = ParameterSettings(star_file)
    obj_paths = settings.objPaths
    star_dict = star_utils.read_star_file_as_dict(star_file)

    subvol_paths = settings.sub_volume_paths
    tomo_tokens = settings.tomo_tokens
    mb_tokens = settings.mb_tokens
    stack_tokens = settings.stack_tokens
    data_dict = load_hdf5_files(settings)
    for tomo_token in tomo_tokens:
        cur_data_dict = data_dict[tomo_token]['bin' + str(settings.consider_bin)]
        for subvol_path, stack_token, mb_token in zip(subvol_paths[tomo_token], stack_tokens[tomo_token],
                                                      mb_tokens[tomo_token]):
            cur_data_points = cur_data_dict[(stack_token, mb_token)]['positions']
            cur_data_dict[(stack_token, mb_token)]['distances'] = {}
            mb_name = os.path.basename(obj_paths[tomo_token][(stack_token, mb_token)])[:-3]
            csv_name = os.path.join(settings.gt_paths[tomo_token], mb_name + 'csv')
            cur_gt_points = data_utils.get_csv_data(csv_name, with_header=True)
            gt_types = cur_gt_points[:, 0]
            unique_gt_types = np.unique(gt_types)
            gt_positions = np.array(cur_gt_points[:, 1:4], dtype=np.float)
            if particle_orientations:
                gt_orientations = np.array(cur_gt_points[:, 4:], dtype=np.float)

            all_dists = np.zeros((cur_data_points.shape[0], gt_positions.shape[0]))
            for k, (gt_position, gt_type, gt_orientation) in enumerate(zip(gt_positions, gt_types, gt_orientations)):
                phi, theta, psi = tuple(gt_orientation[0:3])
                rot = zyz_rot_matrix(phi, theta, psi)
                cov = shape_dict[gt_type]
                rot_cov = np.dot(np.dot(rot, cov), np.transpose(rot, (1, 0)))
                rot_cov = inv(rot_cov)
                dists = mahalanobis_for_matrix_to_point(cur_data_points, gt_position, rot_cov)
                all_dists[:, k] = dists

            for unique_gt_type in unique_gt_types:
                mask = gt_types == unique_gt_type
                cur_gt_dists = all_dists.T[mask].T
                cur_min_dists = np.min(cur_gt_dists, axis=1)
                cur_data_dict[(stack_token, mb_token)]['distances'][unique_gt_type] = cur_min_dists
    store_dict_in_hdf5(data_dict, settings)
            # dists = np.min(all_dists, axis=1)
            # test_array = np.concatenate((cur_data_points, np.expand_dims(dists, 1)),1)
            # data_utils.store_array_in_csv('/Users/lorenz.lamm/PhD_projects/MemBrain_stuff/test_pipeline/trythis/'
            #                               'temp_files/testcsv.csv', test_array)


def add_labels_and_distances(star_file, project_directory, membranorama_xmls=False, prot_tokens=None, prot_shapes=None,
                            particle_orientations=False):
    settings = ParameterSettings(star_file)
    star_dict = star_utils.read_star_file_as_dict(star_file)
    seg_paths = star_dict['segPath']
    obj_dirs = [os.path.join(os.path.dirname(os.path.dirname(seg_path)), 'meshes', os.path.basename(seg_path)[:-3] +
                             'obj') for seg_path in seg_paths]
    star_utils.add_or_change_column_to_star_file(star_file, 'objDir', obj_dirs)
    if membranorama_xmls:
        convert_membranorama_gt_to_csv(project_directory, prot_tokens, settings,
                                       convert_orientation=particle_orientations)
    compute_distances_to_gt(star_file, particle_orientations, prot_shapes)
