import numpy as np
from sklearn.metrics.pairwise import euclidean_distances



def chamfer_distance(array1, array2, prot_tokens, all_gt_types=None, cham_self=False):
    if not cham_self:
        mask = np.zeros(len(all_gt_types))
        for prot in prot_tokens:
            mask[np.array(all_gt_types) == prot] = 1
    else:
        mask = np.ones(array1.shape[0], dtype=np.bool)
    mask = mask > 0
    dist_matrix = euclidean_distances(array1, array2, squared=False)
    if cham_self:
        for i in range(dist_matrix.shape[0]):
            dist_matrix[i,i] = 100000
    min_dist_pred = np.min(dist_matrix[mask], axis=1) # distances to selected GTs
    min_dist_gt = np.min(dist_matrix[mask], axis=0)
    min_dist_pred[min_dist_pred > 40] = 40
    min_dist_gt[min_dist_gt > 40] = 40
    cham_pred = np.mean(min_dist_pred)
    cham_gt = np.mean(min_dist_gt)
    cham_total = cham_pred + cham_gt
    return cham_total, cham_pred, cham_gt


def confusion_matrix(all_gt_points, cluster_points, threshold, prot_tokens, all_gt_types=None, return_hit_idcs=False):
    mask = np.zeros(len(all_gt_types))
    for prot in prot_tokens:
        mask[np.array(all_gt_types) == prot] = 1
    mask = mask > 0

    dist_matrix = euclidean_distances(all_gt_points, cluster_points, squared=False)
    cham1 = np.min(dist_matrix[mask], axis=1) < threshold
    cham2 = np.min(dist_matrix[mask], axis=0) < threshold
    gt_hits_idcs = np.where(cham1)
    gt_misses_idcs = np.where(np.logical_not(cham1))
    pred_hits_idcs = np.where(cham2)
    pred_misses_idcs = np.where(np.logical_not(cham2))
    gt_hits = np.sum(cham1)
    gt_misses = np.sum(1 - cham1)
    pred_hits = np.sum(cham2)
    pred_misses = np.sum(1 - cham2)
    conf_mat = [gt_hits, gt_misses, pred_hits, pred_misses]
    hit_idcs = [gt_hits_idcs, gt_misses_idcs, pred_hits_idcs, pred_misses_idcs]
    if return_hit_idcs:
        return conf_mat, hit_idcs
    return conf_mat, None


def confusion_matrices_for_thresholds(all_gt_points, cluster_points, prot_tokens, all_gt_types=None):
    out_list = []
    for i in range(1, 100):
        out_list.append(confusion_matrix(all_gt_points, cluster_points, threshold=i, prot_tokens=prot_tokens,
                                         all_gt_types=all_gt_types)[0])
    return out_list