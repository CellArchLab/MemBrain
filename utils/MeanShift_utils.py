import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eu_dist
import multiprocessing as mp
from numpy.linalg import norm


cluster_centers = []
max_it = 10000
convergence_thres = 1e-3
min_score = -1
max_score = 1


def reset_globals():
    global cluster_centers, min_score, max_score
    cluster_centers = []
    min_score = -1
    max_score = 1


def get_process_points_masks(points, n_pr):
    masks = []
    step = int(points.shape[0] / n_pr)
    for i in range(n_pr - 1):
        mask_i = np.zeros(points.shape[0])
        mask_i[i * step: (i + 1) * step] = 1
        mask_i = mask_i > 0.5
        masks.append(mask_i)
    mask_last = np.zeros(points.shape[0])
    mask_last[(n_pr - 1) * step:] = 1
    mask_last = mask_last > 0
    masks.append(mask_last)
    return masks


def compute_weights(neigh_scores, weighting):
    if weighting is None:
        return np.zeros_like(neigh_scores)
    min_max_range = max_score - min_score
    weight = (neigh_scores - min_score) / min_max_range
    if weighting == 'simple':
        return weight
    if weighting == 'quadratic':
        return weight**2
    return np.zeros_like(neigh_scores)


def compute_kernel(neigh_dists, bandwidth, kernel):
    if kernel == 'gauss':
        return np.exp(-1 * neigh_dists**2 / (2 * bandwidth) ** 2)
    else:
        raise IOError('No valid kernel specified: ', kernel)


def compute_weighted_average(neighbors, neigh_dists, neigh_scores, bandwidth, kernel, weighting):
    kernels = compute_kernel(neigh_dists, bandwidth, kernel)
    weights = compute_weights(neigh_scores, weighting)
    weighted_kernels = weights * kernels
    denominator = np.sum(weighted_kernels) + 1e-7
    numerator = np.sum(neighbors * np.tile(np.expand_dims(weighted_kernels, 1), (1,3)), axis=0)
    return numerator / denominator


def process_single_point(point, points, scores, bandwidth, weighting, kernel, center_list):
    converged = False
    it = 0
    cur_cen = point
    while not converged and it < max_it:
        it += 1
        cur_distances = eu_dist(np.expand_dims(cur_cen,0), points)
        cur_distances = cur_distances[0]
        mask = cur_distances < bandwidth
        neighbors = points[mask]
        neigh_dists = cur_distances[mask]
        neigh_scores = scores[mask]
        new_cen = compute_weighted_average(neighbors, neigh_dists, neigh_scores, bandwidth, kernel, weighting)
        if norm(new_cen - cur_cen) < convergence_thres:
            converged = True
        cur_cen = new_cen
    center_list.append(cur_cen)
    return center_list


def process_points_and_scores(pr_id, return_dict, points, scores, mask, bandwidth, weighting, kernel):
    pr_points = points[mask]
    center_list = []
    for i in range(pr_points.shape[0]):
        center_list = process_single_point(pr_points[i], points, scores, bandwidth, weighting, kernel, center_list)
    return_dict[pr_id] = center_list


def post_cleanup(all_centers, dist_thres=1):
    idcs = np.array(range(all_centers.shape[0]))
    cluster_idcs = -1 * np.ones_like(idcs)
    cluster_centers = np.array(all_centers)
    new_cluster_centers = np.zeros((0, 3))
    while cluster_centers.shape[0] != 0:
        center = cluster_centers[0]
        close_centers_mask = (eu_dist(np.expand_dims(center, 0), cluster_centers) < dist_thres)
        close_centers_mask = close_centers_mask[0]
        cluster_idcs[idcs[close_centers_mask]] = new_cluster_centers.shape[0]
        close_centers = cluster_centers[close_centers_mask]
        close_centers_mask = 1 - close_centers_mask
        cluster_centers = cluster_centers[close_centers_mask > 0]
        idcs = idcs[close_centers_mask > 0]
        new_cluster_centers = np.concatenate((new_cluster_centers, np.expand_dims(np.mean(close_centers, axis=0), 0)))
    return new_cluster_centers, cluster_idcs


def mean_shift(points, scores, bandwidth, weighting=None, kernel='gauss', n_pr=1, fuse_dist=40):
    assert points.shape[0] == scores.shape[0]
    reset_globals()
    global min_score, max_score
    min_score = np.min(scores)
    max_score = np.max(scores)
    process_points_masks = get_process_points_masks(points, n_pr)
    processes = []
    manager = mp.Manager()
    return_dict = manager.dict()
    queue = mp.Queue()
    queue.put([])
    for pr_id in range(n_pr):
        mask = process_points_masks[pr_id]
        pr = mp.Process(target=process_points_and_scores, args=(pr_id, return_dict, points, scores, mask, bandwidth, weighting, kernel))
        pr.start()
        processes.append(pr)
    for pr_id in range(n_pr):
        pr = processes[pr_id]
        pr.join()
    all_centers = np.concatenate([return_dict[pr_id] for pr_id in range(n_pr)], 0)
    cluster_centers, cluster_idcs = post_cleanup(all_centers, dist_thres=fuse_dist)
    return cluster_centers, cluster_idcs


