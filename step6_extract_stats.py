from turtle import pos
import gdist
import os
import numpy as np
from scripts.add_labels_and_distances import read_GT_data_membranorama_xml, compute_Euler_angles_from_membranorama, zyz_rot_matrix
from utils import star_utils
from config import *
from utils import process_meshes, data_utils

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
    print(ids)
    print(ids.shape)

for tomo_token, stack_token, mb_token, mesh_file, gt_file, pred_pos_file in zip(tomo_tokens, stack_tokens, mb_tokens, mesh_files, xml_paths, pred_position_paths):
    pos_dict, ori_dict = read_GT_data_membranorama_xml(gt_file, return_orientation=True, return_ids=True, prot_tokens=PROT_TOKENS)
    euler_angles = compute_Euler_angles_from_membranorama(ori_dict, pos_dict, mesh_file)
    mesh = process_meshes.read_obj_file_to_triangles(mesh_file)
    testID1 = pos_dict['PSII'][0, 3]
    testID2 = pos_dict['PSII'][1, 3]
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangle_combos, dtype=np.int32)

    source_indices = np.array([testID1], dtype=np.int32)
    target_indices = np.array([testID2], dtype=np.int32)
    #source_indices = np.array([pos_dict['PSII'][0, 3], pos_dict['PSII'][1, 3], pos_dict['PSII'][2, 3]], dtype=np.int32)
    source_indices = np.array([triangles[int(pos_dict['PSII'][0, 3])][0], triangles[int(pos_dict['PSII'][1, 3])][0], triangles[int(pos_dict['PSII'][2, 3])][0]], dtype=np.int32)
    target_indices = np.array([triangles[int(pos_dict['PSII'][3, 3])][0], triangles[int(pos_dict['PSII'][4, 3])][0], triangles[int(pos_dict['PSII'][5, 3])][0]], dtype=np.int32)

    dist = gdist.compute_gdist(vertices, triangles, source_indices=source_indices, target_indices=target_indices, max_distance=50000, is_one_indexed=True)


    prot_shapes = PROT_SHAPES
    psii_shape = prot_shapes['PSII']

    eulers = compute_Euler_angles_from_membranorama(ori_dict, pos_dict, mesh_file, return_as_dict=True)
    print(eulers)
    gt_orientation = eulers['PSII'][0, :]
    phi, theta, psi = tuple(gt_orientation)
    rot = zyz_rot_matrix(phi, theta, psi)

    particle = data_utils.load_tomogram(psii_shape)
    occ = np.where(particle > 0.1)
    occ = np.array(np.transpose(occ, (1, 0)), dtype=np.float64)
    occ -= np.mean(occ, axis=0)
    occ = np.dot(occ, rot)

    pos = pos_dict['PSII'][0, :3]
    print(occ[:10], "<-.--")
    occ += pos
    
    normal = mesh.get_triangle(int(pos_dict['PSII'][0, 3])).compute_normal()
    occ += 5. * normal  # Move points further away from the mesh s.t. structure does not touch both sides of the membrane
    occ = occ[np.arange(occ.shape[0]) % 10 == 0]  # make sparse
    
    find_closest_triangle_ids(occ, mesh)




    # data_utils.store_array_in_csv('/Users/lorenz.lamm/PhD_projects/MemBrain_post_stats/test_stuff/test_occs.csv', occ)
    break







