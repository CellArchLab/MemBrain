from turtle import pos
import gdist
import os
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from scripts.add_labels_and_distances import read_GT_data_membranorama_xml, compute_Euler_angles_from_membranorama, zyz_rot_matrix
from utils import star_utils
from config import *
from utils import process_meshes, data_utils
from scipy.spatial import distance_matrix

project_dir = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)

star_file = '/Users/lorenz.lamm/PhD_projects/pipeline/sample_pipeline/cluster_centers/with_orientation/sample_pipeline_with_inner_outer_bw18.star'
star_dict = star_utils.read_star_file_as_dict(star_file)

mesh_files = star_dict['objDir']
tomo_tokens = star_dict['tomoToken']
mb_tokens = star_dict['mbToken']
stack_tokens = star_dict['stackToken']
pred_position_paths = star_dict['clusterPath']
mb_names = [os.path.basename(mem_seg)[:-3] for mem_seg in star_dict['segPath']]
xml_paths = [os.path.join(project_dir, 'gt_coords', tomo_token, 'as_xml', mb_name + 'xml') for
                 (tomo_token, mb_name) in zip(tomo_tokens, mb_names)]

def find_closest_triangle_ids(occs, mesh):
    ids = []
    for occ in occs:
        id = mesh.find_closest_triangle(occ)
        ids.append(id)
    ids = np.unique(ids)
    return ids

def find_closest_vertex_ids(occs, mesh):
    dists = distance_matrix(occs, mesh.vertices)
    min_ids = np.argmin(dists, axis=1)
    min_ids = np.unique(min_ids)
    return min_ids

def get_occs_for_current_particle(particle, rotation_matrix, positions, idx, mesh, normal_offset=0., make_occs_sparse=False):
    occ = np.where(particle > 0.1)
    occ = np.array(np.transpose(occ, (1, 0)), dtype=np.float64)
    occ -= np.mean(occ, axis=0)
    occ = occ[occ[:, 2] > 0]
    occ *= 3.42
    occ = np.dot(rotation_matrix, occ.T).T

    pos = positions[idx, :3]
    occ += pos
    
    normal = mesh.get_triangle(int(positions[idx, 3])).compute_normal()
    occ += normal_offset * normal  # Move points further away from the mesh s.t. structure does not touch both sides of the membrane
    if make_occs_sparse:
        occ = occ[np.arange(occ.shape[0]) % 10 == 0]  # make sparse
    return occ

def get_rotation_matrix_for_entry(eulers, token, id):
    gt_orientation = eulers[token][id, :]
    phi, theta, psi = tuple(gt_orientation)
    rot = zyz_rot_matrix(phi, theta, psi)
    return rot

def get_e2e_idcs_for_cur_prot_dict_entry(vert_dict_start, vert_dict_end, entry, is_same_prot):
    target_idcs = vert_dict_start[entry]
    start_idcs = []
    for key in vert_dict_end.keys():
        if key != entry or not is_same_prot:
            start_idcs += vert_dict_end[key]
    return np.array(start_idcs, dtype=np.int32), np.array(target_idcs, dtype=np.int32)


def get_shapes(shapes):
    shape_dict = {}
    for key, shape in shapes.items():
        if not shape.startswith('sphere'):
            cur_shape = data_utils.load_tomogram(shape, verbose=False)
        else:
            radius = float(shape[6:])
            cur_shape = np.zeros((int(radius * 2) + 1, int(radius * 2) + 1, int(radius * 2) + 1))
            for x in range(cur_shape.shape[0]):
                for y in range(cur_shape.shape[0]):
                    for z in range(cur_shape.shape[0]):
                        if np.linalg.norm(np.array((x,y,z)) - np.array((radius, radius, radius))) < radius:
                            cur_shape[x,y,z] = 1
        shape_dict[key] = cur_shape
    return shape_dict

def e2e_distances_for_mb(start_token, end_token, mesh, vert_dict):
    print("\nEdge-to-Edge from", start_token, "to", end_token)
    assert not (any((start_token == 'all', end_token == 'all')) and any((start_token != 'all', end_token != 'all')))
    min_dists = []
    if isinstance(start_token, list):
        vert_dict, start_token = merge_dict_classes(vert_dict, start_token)
    if isinstance(end_token, list):
        vert_dict, end_token = merge_dict_classes(vert_dict, end_token)
    for idx in vert_dict[start_token].keys():
        e2e_ids_start, e2e_ids_target = get_e2e_idcs_for_cur_prot_dict_entry(vert_dict[start_token], vert_dict[end_token], idx, is_same_prot=start_token==end_token)
        dists = gdist.compute_gdist(np.array(mesh.vertices), np.array(mesh.triangle_combos, dtype=np.int32), 
                                    source_indices=e2e_ids_start, target_indices=e2e_ids_target, max_distance=50000, is_one_indexed=True)
        min_dist = np.min(dists)
        print(min_dist)
    return min_dists

def merge_dict_classes(vert_dict, classes):
    count_dict = {}
    new_dict = vert_dict.copy()
    count = 0
    out_token = classes[0]
    for entry in classes[1:]:
        out_token += '_' + entry
    if classes == 'all':
        classes = list(vert_dict.keys())
        out_token = 'all'
    for key in classes:
        for key2 in vert_dict[key].keys():
            count_dict[count] = vert_dict[key][key2]
            count += 1
    new_dict[out_token] = count_dict
    return new_dict, out_token

def c2c_distances_for_mb(start_token, end_token, mesh, pos_dict):
    print("\nCenter-to-Center from", start_token, "to", end_token)
    assert not (any((start_token == 'all', end_token == 'all')) and any((start_token != 'all', end_token != 'all')))
    min_dists = []
    if isinstance(start_token, list):
        start_pos = np.zeros((0, 4))
        for token in start_token:
            start_pos = np.concatenate((start_pos, pos_dict[token]), axis=0)
    else:
        start_pos = pos_dict[start_token]
    if isinstance(end_token, list):
        end_pos = np.zeros((0, 4))
        for token in end_token:
            end_pos = np.concatenate((end_pos, pos_dict[token]), axis=0)
    else:
        end_pos = pos_dict[end_token]

    start_verts = []
    end_verts = []
    for pos in start_pos:
        idx = find_closest_vertex_ids(np.expand_dims(pos[:3], 0), mesh)
        start_verts.append(idx[0])
    for pos in end_pos:
        idx = find_closest_vertex_ids(np.expand_dims(pos[:3], 0), mesh)
        end_verts.append(idx[0])
    min_dists = []
    for idx in range(len(start_verts)):
        start_idcs = np.array(np.expand_dims(start_verts[idx], 0), dtype=np.int32)
        if start_token == end_token:
            end_idcs = np.array(end_verts, dtype=np.int32)[np.arange(len(end_verts)) != idx]
        else:
            end_idcs = np.array(end_verts, dtype=np.int32)
        dists = gdist.compute_gdist(np.array(mesh.vertices), np.array(mesh.triangle_combos, dtype=np.int32), 
                                    source_indices=end_idcs, target_indices=start_idcs, max_distance=50000, is_one_indexed=True)
        min_dists.append(dists)
    print(min_dists)
    return min_dists
        



def main():
    for tomo_token, stack_token, mb_token, mesh_file, gt_file, pred_pos_file in zip(tomo_tokens, stack_tokens, mb_tokens, mesh_files, xml_paths, pred_position_paths):
        print('\n', '\n', tomo_token, stack_token, mb_token)
        pos_dict, ori_dict = read_GT_data_membranorama_xml(gt_file, return_orientation=True, return_ids=True, prot_tokens=PROT_TOKENS)
        mesh = process_meshes.read_obj_file_to_triangles(mesh_file)
        eulers = compute_Euler_angles_from_membranorama(ori_dict, pos_dict, mesh_file, return_as_dict=True)

        all_prot_dict = {}
        shape_dict = get_shapes(PROT_SHAPES)
        all_occ_tris = {}
        for prot_token in PROT_SHAPES.keys():
            all_occs = np.zeros((0,3))
            particle = shape_dict[prot_token]
            cur_positions, cur_orientations = pos_dict[prot_token], ori_dict[prot_token]
            cur_prot_dict = {}
            cur_occ_tris = []
            for idx in range(cur_positions.shape[0]):
                rot = get_rotation_matrix_for_entry(eulers, token=prot_token, id=idx) #TODO: Check idx: eulers are stored as a stack for the entire dictionary
                occ = get_occs_for_current_particle(particle, rot, pos_dict[prot_token], idx, mesh, normal_offset=-5., make_occs_sparse=True)
                vt_ids = find_closest_vertex_ids(occ, mesh).tolist()
                tri_ids = find_closest_triangle_ids(occ, mesh).tolist()
                cur_occ_tris += tri_ids
                cur_prot_dict[idx] = vt_ids
                all_occs = np.concatenate((all_occs, occ), 0)
            all_prot_dict[prot_token] = cur_prot_dict
            all_occ_tris[prot_token] = cur_occ_tris



        all_prot_dict, _ = merge_dict_classes(all_prot_dict, 'all')

        c2c_distances_for_mb('PSII', 'PSII', mesh, pos_dict)
        c2c_distances_for_mb('PSII', 'b6f', mesh, pos_dict)
        c2c_distances_for_mb('b6f', 'b6f', mesh, pos_dict)
        c2c_distances_for_mb('PSII', 'UK', mesh, pos_dict)
        c2c_distances_for_mb('UK', 'UK', mesh, pos_dict)
        c2c_distances_for_mb(['PSII', 'UK'], ['PSII', 'UK'], mesh, pos_dict)
        exit()
        e2e_distances_for_mb('PSII', 'PSII', mesh, all_prot_dict)
        e2e_distances_for_mb('PSII', 'b6f', mesh, all_prot_dict)
        e2e_distances_for_mb('b6f', 'PSII', mesh, all_prot_dict)
        e2e_distances_for_mb('b6f', 'b6f', mesh, all_prot_dict)
        e2e_distances_for_mb('all', 'all', mesh, all_prot_dict)
        e2e_distances_for_mb(['PSII'], ['PSII'], mesh, all_prot_dict)
        e2e_distances_for_mb(['PSII', 'UK'], ['PSII', 'UK'], mesh, all_prot_dict)

        continue

        exit()
        min_dists = []
        for idx in cur_prot_dict.keys():
            e2e_ids_start, e2e_ids_target = get_e2e_distances_for_cur_prot_dict_entry(cur_prot_dict, idx)
            dists = gdist.compute_gdist(np.array(mesh.vertices), np.array(mesh.triangle_combos, dtype=np.int32), 
                                        source_indices=e2e_ids_start, target_indices=e2e_ids_target, max_distance=50000, is_one_indexed=True)
            min_dist = np.min(dists)
            min_dists.append(min_dist)
    

if __name__ == '__main__':
    main()