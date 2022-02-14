from utils import data_utils, process_meshes
import numpy as np
from numpy.linalg import inv


def compute_covariance_for_particle(particle):
    occ = np.where(particle > 0.1)
    occ = np.array(np.transpose(occ, (1, 0)), dtype=np.float)
    occ -= np.mean(occ, axis=0)
    occ /= 4
    cov = np.cov(np.transpose(occ, (1, 0)))
    return cov


def z_rot_matrix(phi):
    return np.array(np.array([[np.cos(phi), -1* np.sin(phi), 0],[np.sin(phi), np.cos(phi), 0], [0, 0, 1]]))

def y_rot_matrix(psi):
    return np.array([[np.cos(psi), 0, -1 * np.sin(psi)], [0, 1, 0], [np.sin(psi), 0, np.cos(psi)]])

def x_rot_matrix(alpha):
    return np.array([[1, 0, 0],[0, np.cos(alpha), -1 * np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

def zyz_rot_matrix(phi, the, psi):
    return np.dot(np.dot(z_rot_matrix(psi), y_rot_matrix(the)), z_rot_matrix(phi))

def zyz_to_zxz(phi, the, psi):
    mat = zyz_rot_matrix(phi, the, psi)
    the = np.arccos(mat[2,2])
    phi = np.arctan2(mat[2,0], mat[2,1])
    psi = np.arctan2(mat[0,2], mat[1,2])
    the = np.arctan2(np.cos(phi)*mat[2,1] + np.sin(phi) * mat[2,0],mat[2,2])
    # phi = np.arccos(mat[2,1] / np.sin(the))
    # psi = np.arccos(mat[1,2] / np.sin(the))
    return phi, the, psi


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


psii_particle = data_utils.load_tomogram('/fs/pool/pool-engel/Lorenz/4Lorenz/structures/Chlamy_C2_14A.mrc')
b6f_particle = data_utils.load_tomogram('/fs/pool/pool-engel/Lorenz/4Lorenz/structures/Cyt b6f_14A_center.mrc')
b6f_occ = np.where(b6f_particle > 0.1)
b6f_spans = np.max([np.max(b6f_occ[0]) - np.min(b6f_occ[0]), np.max(b6f_occ[1]) - np.min(b6f_occ[1]), np.max(b6f_occ[2]) - np.min(b6f_occ[2])])
uk_particle = np.zeros([b6f_spans] * 3)
uk_cen = np.array([[0.5 * b6f_spans] * 3])
for i in range(b6f_spans):
    for j in range(b6f_spans):
        for k in range(b6f_spans):
            if np.linalg.norm(np.array([i, j, k]) - uk_cen) < 0.5 * b6f_spans:
                uk_particle[i, j, k] = 1

psii_cov = compute_covariance_for_particle(psii_particle)
b6f_cov = compute_covariance_for_particle(b6f_particle)
uk_cov = compute_covariance_for_particle(uk_particle)
