import numpy as np
from utils.data_utils import get_csv_data, check_mkdir
from utils.MeanShift_utils import mean_shift
from utils import star_utils, data_utils
from sklearn.metrics.pairwise import euclidean_distances
import os
import csv
from scripts import normal_voting, rotator
from utils.parameters import ParameterSettings
from scripts.add_labels_and_distances import read_GT_data_membranorama_xml
from utils.evaluation_metrics import chamfer_distance, confusion_matrix, confusion_matrices_for_thresholds
from time import time
from config import *

class MeanShift_clustering(object):
    def __init__(self, star_file, out_dir, detection_by_classification=False, pos_thres=-3.0):
        self.recluster_thres = None
        self.recluster_bw = None
        self.star_file = star_file
        self.settings = ParameterSettings(self.star_file)
        check_mkdir(out_dir)
        self.out_dir = out_dir
        self.detection_by_classification = detection_by_classification
        self.__initialize_metrics()
        self.pos_thres = pos_thres

    def __initialize_metrics(self):
        self.all_metrics = {
            'Chamfer': [],
            'Chamfer_GT': [],
            'Chamfer_pred': [],
            'bandwidth': [],
            'confusion_matrix': [],
            'confusion_matrix_list': [],
            'Cham_self': []
        }

    def start_clustering(self, bandwidth, recluster_thres=None, recluster_bw=None, convention='zyz'):
        self.recluster_thres = recluster_thres
        self.recluster_bw = recluster_bw
        star_dict = star_utils.read_star_file_as_dict(self.star_file)
        heatmap_paths = star_dict['heatmapDir']
        particle_csv_paths = star_dict['particleCSV']
        cluster_paths, points_with_labels_paths = [], []
        for heatmap_file, particle_csv  in zip(heatmap_paths, particle_csv_paths):
            out_path, points_with_labels_path = self.__cluster_NN_output(heatmap_file, particle_csv,
                                                                         bandwidth=bandwidth, convention=convention)
            cluster_paths.append(out_path)
            points_with_labels_paths.append(points_with_labels_path)

        star_dict['clusterPath'] = cluster_paths
        star_dict['pointsWithClusterLabels'] = points_with_labels_paths
        out_star_token = os.path.splitext(os.path.basename(self.star_file))[0] + '_bw' + str(bandwidth) + '.star'
        project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
        cluster_star = os.path.join(project_directory, 'cluster_centers', 'plain', out_star_token)
        star_utils.write_star_file_from_dict(cluster_star, star_dict)
        return cluster_star

    def evaluate_clustering(self, star_file, bandwidth, distance_thres=18, miccai_eval=False,
                            store_mb_wise=False):
        star_dict = star_utils.read_star_file_as_dict(star_file)
        settings = ParameterSettings(star_file)
        cluster_paths = star_dict['clusterPath']
        associated_points_paths = star_dict['pointsWithClusterLabels']
        gt_dirs = star_dict['gtPath']
        tomo_tokens = star_dict['tomoToken']
        mb_tokens = star_dict['mbToken']
        stack_tokens = star_dict['stackToken']
        xml_files = []

        for i in range(len(mb_tokens)):
            for file in os.listdir(os.path.join(gt_dirs[i], tomo_tokens[i], 'as_xml')):
                if (mb_tokens[i] + '_' in file or mb_tokens[i] + '.' in file) and stack_tokens[i] in file:
                    xml_files.append(os.path.join(gt_dirs[i], tomo_tokens[i], 'as_xml', file))
                    break
        assert len(mb_tokens) == len(xml_files)

        cur_chamfer, cur_chamfer_gt, cur_chamfer_pred, cur_cham_self, cur_conf_mats, \
        cur_conf_mats_lists,  mb_token_list, stack_token_list, tomo_token_list, hits_idcs_list = [], [], [], [], [], \
                                                                                                 [], [], [], [], [],
        for k, (cluster_path, xml_file, associated_points_path) in enumerate(
                zip(cluster_paths, xml_files, associated_points_paths)):
            mb_token = mb_tokens[k]
            stack_token = stack_tokens[k]
            tomo_token = tomo_tokens[k]
            print('Tomo:', tomo_token, '  Membrane:', mb_token, 'Stack:', stack_token)

            gt_dict = read_GT_data_membranorama_xml(xml_file, settings, PROT_TOKENS)
            all_gt_points = np.concatenate([gt_dict[key] for key in gt_dict.keys()], axis=0)
            all_gt_points *= 4

            # TODO: for Chlamy data: uncomment
            # all_gt_points[:, 2] -= 3173.74
            # all_gt_points /= 3.42
            all_gt_types = []
            for key in gt_dict.keys():
                all_gt_types += [key] * gt_dict[key].shape[0]
            if not os.path.isfile(cluster_path):
                cluster_points = np.zeros((1, 3))
            else:
                cluster_points = np.array(data_utils.get_csv_data(cluster_path), dtype=np.float)
            cluster_points = cluster_points[:, :3]
            # if miccai_eval:
            #     cluster_points *= 4

            cham, cham_gt, cham_pred = chamfer_distance(all_gt_points, cluster_points, PROT_TOKENS_PRED, PROT_TOKENS_GT, all_gt_types=all_gt_types)
            cham_self, _, _ = chamfer_distance(cluster_points, cluster_points, PROT_TOKENS_PRED, PROT_TOKENS_GT, all_gt_types=all_gt_types, cham_self=True)
            time_zero = time()
            conf_mat, hits_idcs = confusion_matrix(all_gt_points, cluster_points, threshold=distance_thres, prot_tokens_pred=PROT_TOKENS_PRED, prot_tokens_gt=PROT_TOKENS_GT, all_gt_types=all_gt_types,
                                                   return_hit_idcs=miccai_eval)
            conf_mats_list = confusion_matrices_for_thresholds(all_gt_points, cluster_points, PROT_TOKENS_PRED, PROT_TOKENS_GT, all_gt_types=all_gt_types)
            assiciated_points = np.array(data_utils.get_csv_data(associated_points_path), dtype=np.float)[:, :3]
            ##TODO: What are these lines?
            # if (cluster_points.shape[0] == 1 and cluster_points[0, 0] == 0):
            #     conf_mat[2], conf_mat[3] = 0, 0
            #     for entry in conf_mats_list:
            #         entry[2], entry[3] = 0, 0

            cur_conf_mats.append(conf_mat)
            cur_conf_mats_lists.append(conf_mats_list)
            hits_idcs_list.append(hits_idcs)
            cur_chamfer.append(cham)
            cur_chamfer_pred.append(cham_pred)
            cur_chamfer_gt.append(cham_gt)
            cur_cham_self.append(cham_self)
            mb_token_list.append(mb_token)
            tomo_token_list.append(tomo_token)
            stack_token_list.append(stack_token)

        mb_out_stats_path = os.path.dirname(cluster_path) + '/mb_stats.csv'
        mb_stats = np.array(cur_conf_mats)
        tomo_token_array = np.array(tomo_token_list)
        tomo_token_array = np.expand_dims(tomo_token_array, 1)
        mb_token_array = np.array(mb_token_list)
        mb_token_array = np.expand_dims(mb_token_array, 1)
        mb_stats = np.concatenate((tomo_token_array, mb_token_array, mb_stats), 1)
        data_utils.store_array_in_csv(mb_out_stats_path, mb_stats)
        conf_mat = np.sum([conf for conf in cur_conf_mats], axis=0)
        conf_mats_out_list = np.sum(np.array(cur_conf_mats_lists), axis=0)

        self.all_metrics['bandwidth'].append(bandwidth)
        self.all_metrics['Chamfer'].append(np.mean(cur_chamfer))
        self.all_metrics['Chamfer_GT'].append(np.mean(cur_chamfer_gt))
        self.all_metrics['Chamfer_pred'].append(np.mean(cur_chamfer_pred))
        self.all_metrics['confusion_matrix'].append(conf_mat)
        self.all_metrics['confusion_matrix_list'].append(conf_mats_out_list)
        self.all_metrics['Cham_self'].append(np.mean(cur_cham_self))
        if store_mb_wise:
            self.store_single_mb_metrics(tomo_token_list, stack_token_list, mb_token_list, bandwidth, cur_chamfer,
                                         cur_chamfer_gt, cur_chamfer_pred, cur_conf_mats, cur_conf_mats_lists, cur_cham_self)
        if miccai_eval:
            return self.all_metrics, hits_idcs_list, mb_token_list, stack_token_list
        return self.all_metrics

    def store_single_mb_metrics(self, tomo_tokens, stack_tokens, mb_tokens, bandwidth, chamfer, chamfer_gt,
                                chamfer_pred, conf_mat, conf_mats_list, cham_self):
        out_file = os.path.join(self.out_dir, 'stats_single_mbs_bw' + str(bandwidth) + '.csv')
        header = ['bandwidth', 'tomo_token', 'stack_token', 'mb_token', 'Chamfer', 'Chamfer_GT', 'Chamfer_pred',
                  'Cham_self', 'gt_hits', 'gt_misses', 'pred_hits', 'pred_misses'] + list(range(1, 100)) + \
                 list(range(1, 100)) + list(range(1, 100)) + list(range(1, 100))
        with open(out_file, 'w') as out_csv:
            csv_writer = csv.writer(out_csv, delimiter=',')
            csv_writer.writerow(header)
            for i in range(len(tomo_tokens)):
                row = np.array(list((bandwidth, tomo_tokens[i], stack_tokens[i], mb_tokens[i], chamfer[i], chamfer_gt[i],
                                chamfer_pred[i], cham_self[i], conf_mat[i][0], conf_mat[i][1], conf_mat[i][2],
                                conf_mat[i][3])) + list(np.array(conf_mats_list[i])[:, 0]) + list(np.array(conf_mats_list[i])[:, 1]) +
                               list(np.array(conf_mats_list[i])[:, 2]) + list(np.array(conf_mats_list[i])[:, 3]))
                csv_writer.writerow(row)

    def store_metrics(self):
        self.all_metrics['confusion_matrix'] = np.array(self.all_metrics['confusion_matrix'])
        out_metrics = np.zeros((len(self.all_metrics['bandwidth']), 9))
        out_metrics[:, 0] = self.all_metrics['bandwidth']
        out_metrics[:, 1] = self.all_metrics['Chamfer']
        out_metrics[:, 2] = self.all_metrics['Chamfer_GT']
        out_metrics[:, 3] = self.all_metrics['Chamfer_pred']
        out_metrics[:, 4] = self.all_metrics['Cham_self']
        out_metrics[:, 5] = self.all_metrics['confusion_matrix'][:, 0]
        out_metrics[:, 6] = self.all_metrics['confusion_matrix'][:, 1]
        out_metrics[:, 7] = self.all_metrics['confusion_matrix'][:, 2]
        out_metrics[:, 8] = self.all_metrics['confusion_matrix'][:, 3]

        out_thresholds_GT_hits = np.array(self.all_metrics['confusion_matrix_list'])[:, :, 0]
        out_thresholds_GT_misses = np.array(self.all_metrics['confusion_matrix_list'])[:, :, 1]
        out_thresholds_pred_hits = np.array(self.all_metrics['confusion_matrix_list'])[:, :, 2]
        out_thresholds_pred_misses = np.array(self.all_metrics['confusion_matrix_list'])[:, :, 3]
        out_metrics = np.concatenate((out_metrics, out_thresholds_GT_hits, out_thresholds_GT_misses,
                                      out_thresholds_pred_hits, out_thresholds_pred_misses), axis=1)

        header = ['bandwidth', 'Chamfer', 'Chamfer_GT', 'Chamfer_pred', 'Cham_self', 'gt_hits',
                  'gt_misses', 'pred_hits', 'pred_misses' ]
        header = header + list(range(1, 100)) + list(range(1, 100)) + list(range(1, 100)) + list(range(1, 100))
        data_utils.store_array_in_csv(os.path.join(self.out_dir, 'tuning_stats.csv'), out_metrics,
                                      header=header)

    def __cluster_NN_output(self, score_file, particle_csv, bandwidth=20, convention='zyz'):
        data, header = get_csv_data(score_file, with_header=True, return_header=True)
        # data = get_csv_data(score_file)#, with_header=True, return_header=True)
        data = np.array(data, dtype=np.float)
        # all_coords, scores = extract_coords_and_scores(data, header=None)
        all_coords, scores = extract_coords_and_scores(data, header)
        if not self.detection_by_classification:
            scores *= -1
        pos_thres = (0.0 if self.detection_by_classification else self.pos_thres)
        pos_mask = scores > pos_thres
        if np.sum(pos_mask) == 0:
            out_path = store_cluster_centers(score_file, self.out_dir, None, bandwidth, out_stem='meanshift')
            points_with_labels_path = store_points_with_labels(score_file, self.out_dir, None, None, bandwidth, out_stem='meanshift')
            print("Did not find any points with acceptable predicted distance. Returning 0-sized array.")
            return out_path, points_with_labels_path
        coords = all_coords[pos_mask]
        print(pos_mask.shape)
        print("Start clustering.")
        cluster_centers, cluster_labels = mean_shift(all_coords[pos_mask], scores[pos_mask], bandwidth=bandwidth,
                                                     weighting='quadratic', kernel='gauss', n_pr=8)


        if self.recluster_thres is not None:
            cluster_centers, cluster_labels = self.__refine_large_clusters(all_coords[pos_mask], cluster_centers,
                                                                    cluster_labels, scores[pos_mask])

        cluster_centers = add_center_quantity(cluster_centers, cluster_labels, coords)
        cluster_centers, cluster_labels, all_coords_masked = exclude_small_centers(cluster_centers, cluster_labels,
                                                                                   all_coords[pos_mask],
                                                                                   threshold=3)
        points_with_labels_path = store_points_with_labels(score_file, self.out_dir, all_coords_masked, cluster_labels,
                                                           bandwidth, out_stem='meanshift')
        cluster_centers_with_normals = compute_normals_and_angles_for_centers(cluster_centers, particle_csv,
                                                                              self.settings, convention=convention)
        if cluster_centers_with_normals.shape[0] == 0:
            cluster_centers_with_normals = np.ones((0, 10))
        out_path = store_cluster_centers(score_file, self.out_dir, cluster_centers_with_normals, bandwidth=bandwidth,
                                         out_stem='meanshift')
        return out_path, points_with_labels_path

    def __refine_large_clusters(self, all_coords, cluster_centers, cluster_labels, scores):
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            labelmask = cluster_labels == label
            cur_points = all_coords[labelmask]
            cur_scores = scores[labelmask]
            dist_mat = euclidean_distances(cur_points, cur_points, squared=False)
            max_val = np.amax(dist_mat)
            if max_val > self.recluster_thres:
                print("Refining one cluster because of its size.")
                new_cluster_centers, new_cluster_labels = mean_shift(cur_points, cur_scores, bandwidth=self.recluster_bw,
                                                                     weighting='quadratic', kernel='gauss', n_pr=1)
                if new_cluster_centers.shape[0] == 1:  ## This means that no more cluster centers were detected
                    print("-> No further splits.")
                    continue
                print("--> Split into", new_cluster_centers.shape[0], 'clusters.')
                new_cluster_labels_BU = new_cluster_labels.copy()
                for i, center in enumerate(new_cluster_centers):
                    if i == 0:
                        cluster_centers[label] = center
                        new_cluster_labels[new_cluster_labels_BU == 0] = label
                    else:
                        cluster_centers = np.concatenate((cluster_centers, np.expand_dims(center, axis=0)), axis=0)
                        new_cluster_labels[new_cluster_labels_BU == i] = cluster_centers.shape[0] - 1
                cluster_labels[labelmask] = new_cluster_labels
        return cluster_centers, cluster_labels


def extract_coords_and_scores(data, header):
    if header is not None:
        volume_x_id = np.squeeze(np.argwhere(np.array(header) == 'posX'))
        volume_y_id = np.squeeze(np.argwhere(np.array(header) == 'posY'))
        volume_z_id = np.squeeze(np.argwhere(np.array(header) == 'posZ'))
        for k, entry in enumerate(header):
            if entry.startswith('predDist'):
                score_id = k
                break
    else:
        volume_x_id = 0
        volume_y_id = 1
        volume_z_id = 2
        score_id = 3
    x_coords = np.expand_dims(data[:, volume_x_id], 1)
    y_coords = np.expand_dims(data[:, volume_y_id], 1)
    z_coords = np.expand_dims(data[:, volume_z_id], 1)
    coords = np.concatenate((x_coords, y_coords, z_coords), 1)
    scores = data[:, score_id]
    return coords, scores


def add_center_quantity(cluster_centers, cluster_labels, coords):
    out_centers = np.concatenate((cluster_centers, np.zeros((cluster_centers.shape[0], 1))), 1)
    for i in range(cluster_centers.shape[0]):
        labels_mask = cluster_labels == i
        temp_coords = coords[labels_mask]
        unique_coords = np.unique(temp_coords, axis=0)
        cen_quan = unique_coords.shape[0]
        out_centers[i, 3] = cen_quan
    return out_centers


def exclude_small_centers(cluster_centers, cluster_labels,all_coords, threshold):
    mask = cluster_centers[:, 3] > threshold
    keep_idcs = np.argwhere(mask)
    labels_mask = np.array([cluster_labels[i] in keep_idcs for i in range(cluster_labels.shape[0])])
    out_centers = cluster_centers[mask]
    out_labels = cluster_labels[labels_mask]
    out_coords = all_coords[labels_mask]
    print('Excluding', np.sum(1 - mask), 'centers because they are too small.')
    return out_centers, out_labels, out_coords


def store_points_with_labels(score_file, out_dir, coords, labels, bandwidth, out_stem=None, out_del=','):
    csv_stem = os.path.basename(score_file)
    csv_stem = 'all_coords_' + csv_stem
    if out_stem is not None:
        csv_stem = out_stem + '_bw' + str(bandwidth) + '_' + csv_stem
    out_path = os.path.join(out_dir, csv_stem)
    if labels is not None and len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)
    if coords is not None:
        out_coords = np.concatenate((coords, labels), 1)
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=out_del)
        if coords is not None:
            for i in range(out_coords.shape[0]):
                cur_coord = out_coords[i]
                csv_writer.writerow(cur_coord)
    return out_path


def compute_normals_and_angles_for_centers(cluster_centers, particle_csv, settings=None, convention='zyz'):
    '''Cluster centers should be given in bin1, i.e. also heatmaps should be computed in bin1!'''
    if settings is not None:
        scale = settings.consider_bin * 1.0 / 2
    else: scale = 2.
    all_array = data_utils.get_csv_data(particle_csv)
    pos_array = np.array(all_array[:, :3], dtype=np.float) * scale
    normal_array = np.array(all_array[:, 3:6], dtype=np.float)
    new_normals = normal_voting.normal_voting_for_pos_and_normal_array(cluster_centers[:, :3], normal_array, neighbor_threshold=50, decay_param=100, len_thres=4, compare_array=pos_array * 2)
    if convention == 'zyz':
        euler_angles = np.array(rotator.compute_Euler_angle_for_single_normals_array(new_normals))
    else:
        euler_angles = np.array(rotator.compute_zxz_Euler_angle_for_single_normals_array_testversion(new_normals))
    out_array = np.concatenate((cluster_centers[:, :3], new_normals, euler_angles, np.expand_dims(cluster_centers[:, 3], 1)), axis=1)
    return out_array


def store_cluster_centers(score_file, out_dir, cluster_centers, bandwidth, out_stem=None, out_del=','):
    csv_stem = os.path.basename(score_file)
    if out_stem is not None:
        csv_stem = out_stem + '_bw' + str(bandwidth) + '_' + csv_stem
    if RUN_TOKEN is not None:
        csv_stem = csv_stem[:-4] + "_" + RUN_TOKEN + ".csv"
    out_path = os.path.join(out_dir, csv_stem)
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=out_del)
        if not cluster_centers is None:
            for i in range(cluster_centers.shape[0]):
                cur_cen = cluster_centers[i]
                csv_writer.writerow(cur_cen)
    if not cluster_centers is None:
        out_path_vtk = out_path[:-3] + 'vtp'
        data_utils.convert_csv_to_vtp(out_path, out_path_vtk)
    return out_path

