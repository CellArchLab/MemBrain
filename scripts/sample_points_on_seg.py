import numpy as np
from utils import star_utils, data_utils
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from scipy.ndimage.filters import convolve
from scipy.spatial import distance_matrix
from sklearn.linear_model import LinearRegression
import os
from time import time

def _get_out_star_dict():
    """
    initialize an empty star file
    """
    return {
        'tomoPath': [],
        'segPath': [],
        'particleCSV': [],
        'tomoToken': [],
        'mbToken': [],
        'stackToken': [],
        'rangeStartX': [],
        'rangeEndX': [],
        'rangeStartY': [],
        'rangeEndY': [],
        'rangeStartZ': [],
        'rangeEndZ': [],
        'gtPath': [],
        'pixelSpacing': [],
        'unbinnedOffsetZ': [],
        'tomoPathDenoised': [],
        'tomoPathDimi': [],
    }


def _extrapolate_linearly_1D(x_vals, y_vals, pred_xs):
    """
    1-dimensional extrapolation:
    creates a regression model based on x- and y_vals.
    directly predicts y's for pred_xs and returns them
    """
    pred_xs = np.array(pred_xs).reshape(-1, 1)
    x_vals = np.array(x_vals).reshape(-1, 1)
    y_vals = np.array(y_vals)
    x_vals = x_vals[: y_vals.shape[0]]
    reg = LinearRegression().fit(x_vals, y_vals)
    return reg.predict(pred_xs)


def normalize_vecs(vectors):
    """
    Normalize a matrix row-wise.
    """
    normed_vectors = np.zeros_like(vectors)
    for i, vector in enumerate(vectors):
        normed_vectors[i] = vector / np.linalg.norm(vector)
    return normed_vectors


def _extrapolate(temp_seg, conn_comp_idcs, max_mb_dst):
    """
    Extrapolates a given segmentation of a membrane on both sides in order "cut" the surrounding area in two halves.

    temp_seg: slice of a membrane segmentation
    conn_comp_idcs: indices corresponding to the membrane points of the current connected component
    max_mb_dst: radius of considered field around membranes (determines the radius of picked points)
    """
    dist_x = np.max(conn_comp_idcs[:, 0]) - np.min(conn_comp_idcs[:, 0])
    dist_y = np.max(conn_comp_idcs[:, 1]) - np.min(conn_comp_idcs[:, 1])
    if dist_x > dist_y:
        elongated_direction = 'x_dir'
    else:
        elongated_direction = 'y_dir'
    if elongated_direction == 'x_dir':
        x_vals_min = range(np.min(conn_comp_idcs[:, 0]), np.min(conn_comp_idcs[:, 0]) + 10)
        y_vals_min = []
        x_vals_max = range(np.max(conn_comp_idcs[:, 0] - 10), np.max(conn_comp_idcs))
        y_vals_max = []

        for i in range(10):
            y_idcs_min = np.argwhere(conn_comp_idcs[:, 0] == (np.min(conn_comp_idcs[:, 0] + i)))
            if len(y_idcs_min) == 0:
                break
            y_vals_min_temp = conn_comp_idcs[:, 1][y_idcs_min]
            y_vals_min.append(np.mean(y_vals_min_temp))

            y_idcs_max = np.argwhere(conn_comp_idcs[:, 0] == (np.max(conn_comp_idcs[:, 0] - i)))
            if len(y_idcs_max) == 0:
                break
            y_vals_max_temp = conn_comp_idcs[:, 1][y_idcs_max]
            y_vals_max.append(np.mean(y_vals_max_temp))

        y_vals_max.reverse()
        min_start_x = np.maximum(np.min(conn_comp_idcs[:, 0]) - (10 + max_mb_dst), 0)
        max_end_x = np.minimum(np.max(conn_comp_idcs[:, 0]) + 10 + max_mb_dst, temp_seg.shape[0])
        new_ys_min = _extrapolate_linearly_1D(x_vals_min, y_vals_min, [range(min_start_x, np.minimum(np.min(conn_comp_idcs[:, 0]) + 10, temp_seg.shape[0] - 1))])
        new_ys_max = _extrapolate_linearly_1D(x_vals_max, y_vals_max, [range(np.maximum(np.max(conn_comp_idcs[:, 0] - 10), 0), max_end_x)])

        new_ys_min = np.minimum(np.maximum(np.around(new_ys_min), 1), temp_seg.shape[1]-1)
        new_ys_max = np.minimum(np.maximum(np.around(new_ys_max), 1), temp_seg.shape[1]-1)
        inds_min = np.array(range(min_start_x, np.min(conn_comp_idcs[:, 0] + 10)))
        inds_max = np.array(range(np.max(conn_comp_idcs[:, 0] - 10), max_end_x))


        new_ys_min = new_ys_min[:inds_min.shape[0]]
        new_ys_max = new_ys_max[:inds_max.shape[0]]

        new_ys_min = np.concatenate((new_ys_min - 1, new_ys_min, new_ys_min +1), axis=0)
        new_ys_max = np.concatenate((new_ys_max - 1, new_ys_max, new_ys_max +1), axis=0)
        inds_min = np.concatenate((inds_min, inds_min, inds_min))
        inds_max = np.concatenate((inds_max, inds_max, inds_max))

        coords_min = np.stack((inds_min, new_ys_min), axis=1).T
        coords_max = np.stack((inds_max, new_ys_max), axis=1).T
    else:
        y_vals_min = range(np.min(conn_comp_idcs[:, 1]), np.min(conn_comp_idcs[:, 1]) + 10)
        x_vals_min = []
        y_vals_max = range(np.max(conn_comp_idcs[:, 1] - 10), np.max(conn_comp_idcs))
        x_vals_max = []

        for i in range(10):
            x_idcs_min = np.argwhere(conn_comp_idcs[:, 1] == (np.min(conn_comp_idcs[:, 1] + i)))
            if len(x_idcs_min) == 0:
                break
            x_vals_min_temp = conn_comp_idcs[:, 0][x_idcs_min]
            x_vals_min.append(np.mean(x_vals_min_temp))

            x_idcs_max = np.argwhere(conn_comp_idcs[:, 1] == (np.max(conn_comp_idcs[:, 1] - i)))
            if len(x_idcs_max) == 0:
                break
            x_vals_max_temp = conn_comp_idcs[:, 0][x_idcs_max]
            x_vals_max.append(np.mean(x_vals_max_temp))
        x_vals_max.reverse()

        min_start_y = np.maximum(np.min(conn_comp_idcs[:, 1]) - (10 + max_mb_dst), 0)
        max_end_y = np.minimum(np.max(conn_comp_idcs[:, 1]) + 10 + max_mb_dst, temp_seg.shape[1])
        new_xs_min = _extrapolate_linearly_1D(y_vals_min, x_vals_min, [
            range(min_start_y, np.minimum(np.min(conn_comp_idcs[:, 1]) + 10, temp_seg.shape[1] - 1))])
        new_xs_max = _extrapolate_linearly_1D(y_vals_max, x_vals_max,
                                              [range(np.maximum(np.max(conn_comp_idcs[:, 1] - 10), 0), max_end_y)])

        new_xs_min = np.minimum(np.maximum(np.around(new_xs_min), 1), temp_seg.shape[0] - 1)
        new_xs_max = np.minimum(np.maximum(np.around(new_xs_max), 1), temp_seg.shape[0] - 1)
        inds_min = np.array(range(min_start_y, np.min(conn_comp_idcs[:, 1] + 10)))
        inds_max = np.array(range(np.max(conn_comp_idcs[:, 1] - 10), max_end_y))

        new_xs_min = new_xs_min[:inds_min.shape[0]]
        new_xs_max = new_xs_max[:inds_max.shape[0]]

        new_xs_min = np.concatenate((new_xs_min - 1, new_xs_min, new_xs_min + 1), axis=0)
        new_xs_max = np.concatenate((new_xs_max - 1, new_xs_max, new_xs_max + 1), axis=0)
        inds_min = np.concatenate((inds_min, inds_min, inds_min))
        inds_max = np.concatenate((inds_max, inds_max, inds_max))

        coords_min = np.stack((new_xs_min, inds_min), axis=1).transpose()
        coords_max = np.stack((new_xs_max, inds_max), axis=1).transpose()

    temp_segregation_mask = np.zeros_like(temp_seg)
    temp_segregation_mask[tuple(np.array(coords_min, dtype=np.int))] = 1

    temp_segregation_mask[tuple(np.array(coords_max, dtype=np.int))] = 1
    temp_segregation_mask = binary_dilation(temp_segregation_mask)

    segregation_mask = temp_seg
    segregation_mask[temp_segregation_mask > 0] = 1

    return segregation_mask

def _ismember(coords1, coords2):
    """
    Computes a mask indicating of size coords1.shape[0], indicating whether the coords of coords1 are contained in
    coords2 (row-wise)
    """
    mask = (coords1[:, None] == coords2).all(-1).any(-1)
    return mask


def _find_batch_points(new_mb_points, point_batch):
    point_batch = np.array(point_batch)
    idcs = np.argwhere(_ismember(new_mb_points, point_batch))
    cur_mb_points = new_mb_points[tuple(idcs.T)]
    points_idcs_list = []
    for i in range(point_batch.shape[0]):
        cur_point = point_batch[i, :]
        cur_idcs = idcs[np.all(cur_mb_points == cur_point, axis=1)]
        points_idcs_list.append(cur_idcs)
    return points_idcs_list


def _get_segmentation_side(in_seg_path, tomo, max_mb_dist, mb_ht, orig_pos):
    """
    Segments the surrounding of the membrane on the correct side.
    in_seg_path: path to membrane segmentation (binary mrc file)
    tomo: raw tomogram
    max_mb_dist: radius of surrounding around membrane to be considered --> also inflences the robustness of picked normals
    mb_ht: thickness of membrane
    orig_pos: reference point specifying the correct side of the membrane.
    """
    pre_seg = data_utils.load_tomogram(in_seg_path) > 0
    seg_coords = np.argwhere(pre_seg)
    add_range = 60
    rangeX = range(np.maximum(np.min(seg_coords[:, 0]) - add_range, 0), np.minimum(np.max(seg_coords[:, 0]) + add_range,
                                                                                   pre_seg.shape[0]-1))
    rangeY = range(np.maximum(np.min(seg_coords[:, 1]) - add_range, 0), np.minimum(np.max(seg_coords[:, 1]) + add_range,
                                                                                   pre_seg.shape[1]-1))
    rangeZ = range(np.maximum(np.min(seg_coords[:, 2]) - add_range, 0), np.minimum(np.max(seg_coords[:, 2]) + add_range,
                                                                                   pre_seg.shape[2]-1))
    orig_pos -= np.array((rangeX[0], rangeY[0], rangeZ[0]))

    ranges = [rangeX, rangeY, rangeZ]
    pre_seg = pre_seg[rangeX[0]:rangeX[-1], rangeY[0]: rangeY[-1], rangeZ[0]:rangeZ[-1]]
    tomo = tomo[rangeX[0]:rangeX[-1], rangeY[0]: rangeY[-1], rangeZ[0]:rangeZ[-1]]

    Mbu = np.zeros(pre_seg.shape)

    print("Finding correct segmentation side.")
    for z in range(pre_seg.shape[2]): # go through the tomogram slice-wise
        if z % 10 == 0:
            print("Current slice:", z,'/',  pre_seg.shape[2])
        if np.sum(pre_seg[:, :, z]) < 5:
            continue
        conn_comp_array, tot_conn_comps = label(pre_seg[:, :, z], return_num=True, connectivity=2)

        for conn_comp_nr in range(1, tot_conn_comps + 1):
            ## Get all points in the surrounding of the membrane segmentation
            H = np.zeros(pre_seg[:,:,z].shape)
            dist_trafo, mb_idcs = distance_transform_edt(conn_comp_array != conn_comp_nr, return_indices=True)
            H_temp = H.copy()
            H_temp[dist_trafo <= (max_mb_dist + mb_ht)] = 2
            H_temp[dist_trafo <= mb_ht] = 1

            ## Separate the surrounding using extrapolation of the membrane ends
            segregation_mask = _extrapolate(H_temp, np.argwhere(conn_comp_array == conn_comp_nr), max_mb_dist)
            mask = (H_temp == 2) * 1.0
            mask[segregation_mask == 1] = 2
            conn_comps_seg, conn_comps_seg_num = label(mask == 1, return_num=True)
            if conn_comps_seg_num != 2:
                print("WARNING! The membrane sides are not separated properly! Check whether sampled points are correct.")

            """ Extract indices, their connection vectors to the membranes and the connection vectors to the origin point
            Compare these two vectors using the dot product
            Intention: If the dot product is high, the vectors point in the same direction"""

            half1_idcs = np.argwhere(conn_comps_seg == 1)
            half2_idcs = np.argwhere(conn_comps_seg == 2)
            surr_coords = np.concatenate((half1_idcs, half2_idcs), axis=0)
            mb_idcs = np.transpose(mb_idcs, (1, 2, 0))

            vec_conn_mb = surr_coords - mb_idcs[tuple(surr_coords.T)]
            vec_conn_mb = normalize_vecs(vec_conn_mb)

            vec_conn_cen = surr_coords - orig_pos[:2]
            vec_conn_cen = normalize_vecs(vec_conn_cen)

            dots = np.einsum('ij,ij->i', vec_conn_cen, vec_conn_mb)

            dots_comp1 = dots[:half1_idcs.shape[0]]
            dots_comp2 = dots[half1_idcs.shape[0]:]
            if np.mean(dots_comp1) < np.mean(dots_comp2):
                idcs_choice = half1_idcs
            else:
                idcs_choice = half2_idcs
            H[tuple(idcs_choice.T)] = 2
        dist_trafo = distance_transform_edt(pre_seg[:, :, z] == 0)
        H[dist_trafo <= mb_ht] = 1
        Mbu[:, :, z] = H
    return Mbu, tomo, ranges


def sample_uniformly(tomo_seg_path, out_path, tomo_token, stack_token, mb_token, shrink_thres, ranges):
    """
    sample points uniformly on previously computed membrane segmentation (with mb sides).

    tomo_seg_path: path to segmentation file (should also include picked side, i.e. all points on the correct side
    should be labeled as '2' in the segmentation
    out_path: directory to store coordinates
    tomo_token: token of tomogram
    mb_token: token of membrane
    threshold: threshold for cropping the prediction from the edges; should be around 170
    """
    print('Sampling points.')
    time_zero = time()
    tomo_seg = data_utils.load_tomogram(tomo_seg_path)
    dist_trafo, idcs = distance_transform_edt(tomo_seg != 1, return_indices=True)
    idcs = np.transpose(idcs, (1,2,3,0))
    mask = np.logical_and(dist_trafo > 0.5, dist_trafo < 8.)

    new_seg = np.zeros_like(tomo_seg)
    new_seg[mask] = 1

    new_idcs = np.argwhere(new_seg == 1)
    mask10th = np.array(range(new_idcs.shape[0])) % 10 == 0
    new_idcs = new_idcs[mask10th]
    mb_half_idcs = np.argwhere(tomo_seg == 2)

    mb_half_mask = (new_idcs[:, None] == mb_half_idcs).all(-1).any(-1)
    new_idcs = new_idcs[mb_half_mask]

    new_mb_points = idcs[tuple(new_idcs.T)]
    new_dt = dist_trafo[tuple(new_idcs.T)]

    conn_vecs = new_idcs - new_mb_points
    unique_mb_points = np.unique(new_mb_points, axis=0)

    ## Find edges of membrane

    mb_idcs = np.argwhere(tomo_seg == 1)

    conv_kernel = np.ones((7,7,7))
    temp_tomo = convolve((tomo_seg == 1)*1.0, conv_kernel)
    mask = temp_tomo > shrink_thres

    print("Shrinking borders of sampled points.")
    for i in range(2):
        new_tomo_seg = (tomo_seg == 1) * 1.0
        new_tomo_seg *= mask
        temp_tomo = convolve(new_tomo_seg, conv_kernel)
        mask = temp_tomo > shrink_thres

    new_tomo_seg = (tomo_seg == 1) * 1.0
    new_tomo_seg *= mask
    removed_parts = (tomo_seg == 1) - new_tomo_seg
    removed_idcs = np.argwhere(removed_parts)

    remove_mask = np.array(1 - (unique_mb_points[:, None] == removed_idcs).all(-1).any(-1))
    remove_mask = remove_mask == 1

    unique_mb_points2 = unique_mb_points[remove_mask]

    ## Sample points to decrease density

    points = unique_mb_points2.copy()
    points = np.concatenate((points, np.expand_dims(np.array(range(points.shape[0])), 1)), 1)
    remove_mask = np.ones(points.shape[0])
    print("Thinning sampled points.")
    while points.shape[0] > 1:
        idx = np.random.randint(points.shape[0])
        point = points[idx]
        take_id = point[3]
        mask = distance_matrix(np.expand_dims(point[:3], 0), points[:, :3]) < 1.5
        mask = np.squeeze(mask)
        remove_idcs = points[mask][:, 3]
        remove_mask[remove_idcs] = 0
        remove_mask[take_id] = 1
        points = points[(1 - mask) > 0]

    unique_mb_points2 = unique_mb_points2[remove_mask > 0]

    ## Shrink new_mb_points, conn_vecs, new_dt

    mask =  (new_mb_points[:, None] == unique_mb_points2).all(-1).any(-1)
    new_mb_points = new_mb_points[mask]
    conn_vecs = conn_vecs[mask]
    new_dt = new_dt[mask]

    ## Compute normals for each point
    print("Computing normals for each point.")
    points_and_normals = []
    for i in range(unique_mb_points2.shape[0]):
        if i % 1000 == 0:
            point_batch = []
            coord_batch = []
        point = unique_mb_points2[i]
        point_batch.append(point)
        if i % 1000 == 999 or i == unique_mb_points2.shape[0] - 1:
            cur_points_and_normals = []
            point_idcs_list = _find_batch_points(new_mb_points, point_batch)

            for j in range(len(point_idcs_list)):
                coord = point_batch[j]
                coord[0] += ranges[0][0]
                coord[1] += ranges[1][0]
                coord[2] += ranges[2][0]
                point_idcs = point_idcs_list[j]
                vecs = conn_vecs[tuple(point_idcs.T)]
                dists = new_dt[tuple(point_idcs.T)]
                max_idcs = np.argwhere(dists == np.amax(dists))
                longest_vecs = vecs[tuple(max_idcs.T)]
                avg_longest_vec = np.mean(longest_vecs, axis=-2)
                cur_points_and_normals.append([coord, avg_longest_vec])
            points_and_normals += (cur_points_and_normals)

    points = np.array(points_and_normals)[:, 0, :]
    normals = np.array(points_and_normals)[:, 1, :]
    points_and_normals = np.concatenate((points, normals), axis=1)
    out_name = os.path.join(out_path, tomo_token + '_' + stack_token + '_' + mb_token + '_pred_positions.csv')
    data_utils.store_array_in_csv(out_name, points_and_normals)
    data_utils.convert_csv_to_vtp(out_name, out_name[:-3] + 'vtp')

    return points_and_normals




def sample_points_on_seg(out_dir, in_star, out_path, max_mb_dist=60, mh_ht=0, out_bin=4, shrink_thres=118):
    star_dict = star_utils.read_star_file_as_dict(in_star)
    tomo_tokens = star_dict['tomoToken']
    tomo_paths = star_dict['tomoPath']
    mb_tokens = star_dict['mbToken']
    stack_tokens = star_dict['stackToken']
    seg_paths = star_dict['segPath']
    orig_posX = star_dict['origin_pos_x']
    orig_posY = star_dict['origin_pos_y']
    orig_posZ = star_dict['origin_pos_z']

    out_star = os.path.join(out_path, os.path.basename(in_star))
    prev_tomo_token = ''
    particle_csvs = []
    for i, tomo_token in enumerate(tomo_tokens):
        tomo_path = tomo_paths[i]
        mb_token = mb_tokens[i]
        # if mb_token == 'M2':
        #     continue
        stack_token = stack_tokens[i]
        # tomo_bin = tomo_bins[i]
        in_seg_path = seg_paths[i]
        orig_pos = np.array((orig_posX[i], orig_posY[i], orig_posZ[i]), dtype=np.float)
        # gt_path = gt_paths[i]

        if prev_tomo_token != tomo_token:
            tomo = data_utils.load_tomogram(tomo_path)
            Mbu, tomo, ranges = _get_segmentation_side(in_seg_path, tomo, max_mb_dist, mh_ht, orig_pos)

            mic_out_path = os.path.join(out_dir, 'mics', tomo_token + '_' + stack_token + '_' + mb_token + '.mrc')
            seg_out_path = os.path.join(out_dir, 'segs', tomo_token + '_' + stack_token + '_' + mb_token + '.mrc')
            data_utils.store_tomogram(mic_out_path, tomo)
            data_utils.store_tomogram(seg_out_path, Mbu)
            sample_uniformly(seg_out_path, out_path, tomo_token, stack_token, mb_token, shrink_thres, ranges)

            particle_csv = os.path.join(out_path, tomo_token + '_' + stack_token + '_' + mb_token + '_pred_positions.csv')
            particle_csvs.append(particle_csv)

    star_dict_out = star_dict.copy()
    star_dict_out['particleCSV'] = particle_csvs
    del star_dict_out['origin_pos_x']
    del star_dict_out['origin_pos_y']
    del star_dict_out['origin_pos_z']
    star_utils.write_star_file_from_dict(out_star, star_dict_out)
    return out_star

