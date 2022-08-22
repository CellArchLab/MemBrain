import numpy as np

from config import *
from scripts import particle_orientation
from utils.parameters import ParameterSettings
from utils import data_utils, process_meshes, star_utils
import os

def adjust_paths_to_windows_machine(path_list):
    paths = [pathfile.replace('/scicore/home/engel0006/GROUP/pool-engel/', 'Z:\\') for pathfile in path_list]
    paths = [pathfile.replace("/","\\") for pathfile in paths]
    return paths


def transform_to_membranorama_angles(positions, angles, orientations, mesh_path):
    """ Compute the Membranorama angles from MemBrain outputs.
    How it works:
    For each detected cluster center & orientation:
        1. Find the closest triangle on the mesh. This triangle will be used to compute the plane matrix (the plane should ideally have the same normal as our cluster center)
        2. For the detected triangle, get the plane matrix
        3. Apply the plane matrix to the MemBrain orientation vector. This will transform the orientation vector (which lies in the triangle plane) 
            back to the x-y plane (z component will be minimal)
        4. Arctan2 of the computed vector will give the angle between the transformed vector and the positive x-axis
        5. Subtract 121.1 degrees from the computed angle. This will align 0° with the long axis of PSII
        6. Now compute the difference between the MemBrain in-plane orientation angle and the above computed angle. #TODO: Do I actually need this? The orientation vector should theoretically be enough!
    The idea is now:
        1. When the orientation is opened in Membranorama, it will first perform the in-plane orientation to get the offset from 0°
        2. Then, this rotated structure is transformed to lie in the triangle plane


    Args:
        positions (_type_): positions array (N, 3)
        angles (_type_): computed Euler angles -- only 1st one is important (N, 3) #TODO: Do I actually need this? The orientation vector should theoretically be enough!
        orientations (_type_): orientation vectors from MemBrain (N, 3)
        mesh_path (_type_): path to the corresponding mesh file

    Returns:
        _type_: _description_
    """
    mesh = process_meshes.read_obj_file_to_triangles(mesh_path)
    new_angles = []
    tri_ids = []
    for pos, angle, orientation in zip(positions, angles, orientations):
        tri_id = mesh.find_closest_triangle(pos)
        triangle = mesh.get_triangle(tri_id)
        plane_mat = triangle.get_plane_matrix()
        vec = orientation
        vec = np.dot(plane_mat, vec)
        beta = np.arctan2(vec[0], vec[1])
        beta = np.rad2deg(beta)%180 - 121.1
        beta = np.deg2rad(beta)
        new_angles.append(np.array((beta, 0, 0)))
        tri_ids.append(tri_id)
    return np.stack(new_angles), np.stack(tri_ids)


def convert_orientations_to_membranorama(star_file, out_dir_membranorama):
    star_dict = star_utils.read_star_file_as_dict(star_file)
    tomo_paths = star_dict['tomoPath']
    obj_paths = star_dict['objDir']
    position_paths = star_dict['clusterPath']
    out_dir = out_dir_membranorama

    new_obj_paths = []
    for obj_path in obj_paths:
        out_file = os.path.join(out_dir, 'meshes', os.path.basename(obj_path))
        mesh = process_meshes.read_obj_file_to_triangles(obj_path)
        if np.max(mesh.vertices) < 1000.:
            mesh.vertices = (np.array(mesh.vertices) * 14.08).tolist()
        mesh.store_in_file(out_file)
        new_obj_paths.append(out_file)
    obj_paths_bkp = new_obj_paths.copy()
    obj_paths = adjust_paths_to_windows_machine(new_obj_paths)

    # obj_paths = adjust_paths_to_windows_machine(obj_paths)
    # position_paths = adjust_paths_to_windows_machine(position_paths)
    new_tomo_paths = []
    for tomo_path in tomo_paths:
        print('Restoring tomo', tomo_path)
        out_file = os.path.join(out_dir, 'tomos', os.path.basename(tomo_path))
        new_tomo_paths.append(out_file)
        if os.path.isfile(out_file):
            continue
        tomo = data_utils.load_tomogram(tomo_path)
        data_utils.store_tomogram_valid_header(out_file, tomo)
    tomo_paths = adjust_paths_to_windows_machine(new_tomo_paths)

    for tomo_path, obj_path, obj_path_orig, pos_path in zip(tomo_paths, obj_paths, obj_paths_bkp, position_paths):
        positions = data_utils.get_csv_data(pos_path)
        angles, tri_ids = transform_to_membranorama_angles(positions[:, :3] * 3.52, positions[:, -3:], positions[:, 3:6], obj_path_orig)
        tri_ids = np.expand_dims(tri_ids, 1)
        pos_dict = {'PSII': np.concatenate((positions[:, :3] * 3.52,tri_ids), axis=1)}
        ori_dict = {'PSII': angles}
        particle_model_dict = {key: 'Z:\\Spinach_project\\structures\\C2_spinach_14A_centered.mrc' for key in pos_dict.keys()}
        out_xml = os.path.join(out_dir, os.path.basename(obj_path.replace('\\', '/'))[:-3] + 'xml')
        mesh = process_meshes.read_obj_file_to_triangles(obj_path_orig)
        if np.max(mesh.vertices) < 1000.:
            mesh.vertices = (np.array(mesh.vertices) * 14.08)
        mean_pos = np.mean(mesh.vertices, axis=0)
        data_utils.save_as_membranorama_xml_with_header(out_xml, pos_dict, obj_path, tomo_path, particle_model_dict, ori_dict=ori_dict, CameraTarget=mean_pos)


def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    pos_star_file = os.path.join(os.path.join(project_directory, 'cluster_centers', 'plain'),
                                 PROJECT_NAME + '_with_inner_outer_bw' + str(int(CLUSTER_BANDWIDTHS[0])) + '.star')
    cluster_dir = os.path.join(os.path.join(project_directory, 'cluster_centers'))
    out_dir = os.path.join(cluster_dir, 'with_orientation')
    settings = ParameterSettings(pos_star_file, is_cluster_center_file=True)
    out_star_file = particle_orientation.compute_all_orientations(cluster_dir, settings, out_dir, visualize=False)
    out_dir_membranorama = os.path.join(cluster_dir, 'with_orientation_membranorama')
    convert_orientations_to_membranorama(out_star_file, out_dir_membranorama)



if __name__ == '__main__':
    main()