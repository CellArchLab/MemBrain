import numpy as np
import csv
from utils import data_utils
import utils.star_utils
import os
import time
from sklearn.metrics.pairwise import euclidean_distances
import multiprocessing as mp


def __load_csv_for_normal_voting(in_path, delimiter=','):
    pos_array = np.zeros((0,3))
    normal_array = np.zeros((0,3))
    with open(in_path) as in_file:
        csv_reader = csv.reader(in_file, delimiter=delimiter)
        pos_list = []
        normal_list = []
        for i, row in enumerate(csv_reader):
            pos_list.append(np.array(row[:3], dtype=np.float))
            normal_list.append(np.array(row[3:6], dtype=np.float))
            # pos_array = np.concatenate((pos_array, np.expand_dims(np.array(row[:3], dtype=np.float), 0)))
            # normal_array = np.concatenate((normal_array, np.expand_dims(np.array(row[3:6], dtype=np.float), 0)))
        pos_array = np.asarray(pos_list)
        normal_array = np.asarray(normal_list)
    return pos_array, normal_array


def __store_normal_voting_in_csv(out_path, pos_array, normal_array, delimiter=','):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=delimiter)
        for i in range(pos_array.shape[0]):
            pos = pos_array[i]
            norm = normal_array[i]
            row = [pos[0], pos[1], pos[2], norm[0], norm[1], norm[2]]
            csv_writer.writerow(row)


def __distance_between_2_3Dpoints(point1, point2):
    distance = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2
    distance = np.sqrt(distance)
    return distance


def __get_distances(pos_array, single_pos):
    distances = np.zeros(0)
    for i in range(pos_array.shape[0]):
        temp_pos = pos_array[i]
        distance = __distance_between_2_3Dpoints(temp_pos, single_pos)
        distances = np.concatenate((distances, np.expand_dims(distance, 0)), 0)
    return distances


def __compute_normal_contribution_single_pos(point, normal, distance, single_pos):
    conn_vec = point - single_pos
    if np.linalg.norm(conn_vec) != 0:
        cosTheta = (-1) * np.dot(normal, conn_vec) / (distance + 1e-5)
        contribution = normal + 2 * cosTheta * conn_vec / np.linalg.norm(conn_vec)
    else:
        contribution = normal
    return contribution


def __compute_normal_contributions(neighbor_points, neighbor_normals, neighbor_distances, single_pos):
    contributions = np.zeros((0, 3))
    for i in range(neighbor_points.shape[0]):
        temp_cont = __compute_normal_contribution_single_pos(neighbor_points[i], neighbor_normals[i],
                                                           neighbor_distances[i], single_pos)
        contributions = np.concatenate((contributions, np.expand_dims(temp_cont, 0)), 0)
    return contributions


def __compute_covariance_matrix(contributions, weights):
    matrix = np.zeros((3, 3))
    for i in range(contributions.shape[0]):
        cont = contributions[i]
        wgt = weights[i]
        matrix += wgt * np.outer(cont, cont)
    return matrix


def __compute_weights(normals, distances, decay_param, len_thres):
    normal_lengths = np.linalg.norm(normals, axis=1)
    weights = np.exp((-1) *distances / decay_param)
    thres_mask = normal_lengths < len_thres
    if np.sum(thres_mask) != normal_lengths.shape[0]:
        normal_lengths[normal_lengths < len_thres] = 0
    weights = weights * normal_lengths
    return weights


def __compute_normal_vote(covariance_matrix, single_normal):
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    max_idx = np.argmax(eigen_values)
    normal_vote = eigen_vectors[:, max_idx]
    if np.dot(normal_vote, single_normal) < 0: #np.cos(-1 * np.dot(normal_vote, single_normal)):
        normal_vote = normal_vote * (-1)
    return normal_vote


def __normal_voting_for_single_position(pos_array, single_pos, normal_array, single_normal, distances, neighbor_threshold,
                                      decay_param, len_thres):
    neighbor_mask = distances < neighbor_threshold
    neighbor_distances = distances[neighbor_mask]
    neighbor_points = pos_array[neighbor_mask]
    neighbor_normals = normal_array[neighbor_mask]
    norm = np.transpose(np.tile(np.linalg.norm(neighbor_normals, axis=1), (3, 1)))
    neighbor_normals_normed = neighbor_normals / norm
    contributions = __compute_normal_contributions(neighbor_points, neighbor_normals_normed, neighbor_distances, single_pos)
    weights = __compute_weights(neighbor_normals, neighbor_distances, decay_param, len_thres)
    cov_matrix = __compute_covariance_matrix(contributions, weights)
    normal_vote = __compute_normal_vote(cov_matrix, single_normal)
    return normal_vote


def _get_mp_split(pos_array_shape, npr):
    if npr == 1:
        return np.zeros(pos_array_shape[0])
    else:
        out_mask = np.zeros(pos_array_shape[0])
        step = int(pos_array_shape[0] / npr)
        for i in range(npr - 1):
            out_mask[i*step:(i+1)*step] = i
        out_mask[(i+1) * step:] = npr - 1
        return out_mask


def normal_voting_for_file(in_path, out_path, in_del=',', out_del=',', neighbor_threshold=20, decay_param=100,
                           len_thres=4, tomo_token=None, mb_token=None, npr=1):
    abs_zero = time.time()
    # if tomo_token is None or mb_token is None:
    #     settings = ParameterSettings()
    #     tomo_token, mb_token = data_utils.get_tomo_and_mb_from_file_name(in_path, settings)
    print('Performing normal voting for tomo: ', tomo_token, '  Membrane: ', mb_token)
    pos_array, normal_array = __load_csv_for_normal_voting(in_path, in_del)
    new_normals = normal_voting_for_pos_and_normal_array(pos_array, normal_array, neighbor_threshold, decay_param, len_thres, npr=npr)
    __store_normal_voting_in_csv(out_path, pos_array, new_normals, out_del)
    print('Total amount of time for this single membrane: ', time.time() - abs_zero)


def normal_voting_process(pos_array, normal_array, process_mask, p_nr, return_dict, dist_matrix, compare_array, neighbor_threshold, decay_param, len_thres):
    new_normals = np.zeros((np.sum(process_mask == p_nr), 4))
    cur_entry = 0
    print('Process', p_nr, ':  Processing', np.sum(process_mask == p_nr), 'positions')
    for i in range(pos_array.shape[0]):
        if process_mask[i] != p_nr:
            continue

        temp_pos = pos_array[i]
        temp_normal = normal_array[i]
        distances = dist_matrix[i]
        normal_vote = __normal_voting_for_single_position(pos_array, temp_pos, normal_array, temp_normal, distances,
                                                          neighbor_threshold,
                                                          decay_param, len_thres)
        new_normals[cur_entry, :3] = normal_vote
        new_normals[cur_entry, 3] = i
        cur_entry += 1
    return_dict[p_nr] = new_normals


def normal_voting_for_pos_and_normal_array(pos_array, normal_array, neighbor_threshold, decay_param, len_thres, compare_array=None, npr=1):
    print('Processing', pos_array.shape[0], 'points')
    new_normals = np.zeros_like(pos_array)
    print('Computing distance matrix')
    time_zero = time.time()
    if compare_array is None:
        dist_matrix = euclidean_distances(pos_array)
    else:
        dist_matrix = euclidean_distances(pos_array, compare_array)

    if compare_array is None:
        process_points_masks = _get_mp_split(pos_array.shape, npr)
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for pr_id in range(npr):
            pr = mp.Process(target=normal_voting_process,
                            args=(pos_array, normal_array, process_points_masks, pr_id, return_dict, dist_matrix, compare_array, neighbor_threshold, decay_param, len_thres))
            pr.start()
            processes.append(pr)
        for pr_id in range(npr):
            pr = processes[pr_id]
            pr.join()
        unordered_normals = np.concatenate([return_dict[pr_id] for pr_id in range(npr)], 0)
        order = np.array(unordered_normals[:, 3], dtype=np.int32)
        new_normals = unordered_normals[:, :3][order]

    else:
        for i in range(pos_array.shape[0]):
            if i % 2500 == 0:
                print(i, '/', pos_array.shape[0])
                # print 'these 500 took', time.time() - time_zero, 'seconds.'
                time_zero = time.time()

            temp_pos = pos_array[i]
            temp_normal = normal_array[i]
            distances = dist_matrix[i]

            normal_vote = __normal_voting_for_single_position(compare_array, temp_pos, normal_array, temp_normal, distances,
                                                                  neighbor_threshold,
                                                                  decay_param, len_thres)
            new_normals[i] = normal_vote
    return new_normals


def normal_voting_for_star(in_star, out_dir, npr=1):
    """
    Performs normal correction for all csv files specified in a star file (key "particleCSV"), ideally output from
    MemBrain sampling step.
    in_star: path to star file
    out_dir: output directory for corrected normals + star file
    n_pr: number of processes for multi-processing
    """
    star_dict = utils.star_utils.read_star_file_as_dict(in_star)
    csv_list = star_dict['particleCSV']
    tomo_tokens = star_dict['tomoToken']
    mb_tokens = star_dict['mbToken']
    new_csvs_list = []
    for i, csv_file in enumerate(csv_list):
        tomo_token = tomo_tokens[i]
        mb_token = mb_tokens[i]
        filename = os.path.basename(csv_file)
        out_csv = os.path.join(out_dir, filename)
        normal_voting_for_file(csv_file, out_csv, neighbor_threshold=40, tomo_token=tomo_token, mb_token=mb_token, npr=npr)
        data_utils.convert_csv_to_vtp(out_csv, out_csv[:-4] + '.vtp')
        data_utils.convert_csv_to_vtp(csv_file, csv_file[:-4] + '.vtp')
        new_csvs_list.append(out_csv)
    star_dict['oldCSV'] = csv_list
    star_dict['particleCSV'] = new_csvs_list
    out_star_name = os.path.join(out_dir, os.path.basename(in_star))
    utils.star_utils.write_star_file_from_dict(out_star_name, star_dict)
    return out_star_name


def main():
    pass


if __name__ == '__main__':
    main()
