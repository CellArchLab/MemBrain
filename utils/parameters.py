import os
from utils import star_utils
import numpy as np


class ParameterSettings():
    def initialize_meta_params(self, star_dict):
        self.pixel_spacing = float(star_dict['pixelSpacing'][0])
        if 'unbinnedOffsetZ' in star_dict.keys():
            self.unbinned_offset_Z = float(star_dict['unbinnedOffsetZ'][0])
        else:
            self.unbinned_offset_Z = 0.
        self.consider_bin = int(star_dict['tomoBin'][0])

    def initialize_tokens(self, star_dict):
        all_tomo_tokens = star_dict['tomoToken']
        all_mb_tokens = star_dict['mbToken']
        all_stack_tokens = star_dict['stackToken']
        tomo_tokens = np.unique(all_tomo_tokens).tolist()

        self.tomo_tokens = tomo_tokens
        self.mb_tokens = {}
        self.stack_tokens = {}

        for tomo_token in tomo_tokens:
            mb_tokens_for_tomo_idcs = np.argwhere(np.array(all_tomo_tokens) == tomo_token)
            mb_tokens_for_tomo = np.array(all_mb_tokens)[np.array(mb_tokens_for_tomo_idcs.T, dtype=np.int8)]
            stack_tokens_for_tomo = np.array(all_stack_tokens)[np.array(mb_tokens_for_tomo_idcs.T, dtype=np.int8)]
            mb_tokens_for_tomo = np.concatenate((mb_tokens_for_tomo.T, stack_tokens_for_tomo.T), axis=1)
            unique_mb_stack_tokens = np.unique(mb_tokens_for_tomo, axis=0)
            unique_mb_tokens = unique_mb_stack_tokens[:, 0].tolist()
            unique_stack_tokens = unique_mb_stack_tokens[:, 1].tolist()
            self.mb_tokens[tomo_token] = unique_mb_tokens
            self.stack_tokens[tomo_token] = unique_stack_tokens

    def initialize_pred_paths(self, star_dict, is_cluster_center_file):
        all_tomo_tokens = star_dict['tomoToken']
        tomo_tokens = np.unique(all_tomo_tokens).tolist()
        self.pred_paths = {}
        if is_cluster_center_file:
            self.cluster_paths = star_dict['clusterPath']
            self.associated_cluster_paths = star_dict['pointsWithClusterLabels']
            all_pred_paths = star_dict['clusterPath']
        else:
            all_pred_paths = star_dict['particleCSV']

        for tomo_token in tomo_tokens:
            self.pred_paths[tomo_token] = {}
        for i, csv_path in enumerate(all_pred_paths):
            tomo_token = star_dict['tomoToken'][i]
            mb_token = star_dict['mbToken'][i]
            stack_token = star_dict['stackToken'][i]
            self.pred_paths[tomo_token][stack_token, mb_token] = csv_path

    def initialize_obj_dir(self, star_dict):
        all_tomo_tokens = star_dict['tomoToken']
        all_mb_tokens = star_dict['mbToken']
        all_stack_tokens = star_dict['stackToken']
        if 'objDir' in star_dict.keys():
            self.objPaths = {}
            all_obj_paths = star_dict['objDir']
            for i, obj_path in enumerate(all_obj_paths):
                tomo_token = all_tomo_tokens[i]
                if tomo_token not in self.objPaths.keys():
                    self.objPaths[tomo_token] = {(all_stack_tokens[i], all_mb_tokens[i]): obj_path}
                else:
                    self.objPaths[tomo_token][(all_stack_tokens[i], all_mb_tokens[i])] = obj_path

    def initialize_GTs(self, star_dict):
        all_tomo_tokens = star_dict['tomoToken']
        all_gt_paths = star_dict['gtPath']
        self.gt_paths = {}
        for i, gt_path in enumerate(all_gt_paths):
            tomo_token = all_tomo_tokens[i]
            self.gt_paths[tomo_token] = os.path.join(gt_path, tomo_token, 'as_csv')

    def initialize_tomo_paths_for_bin(self, star_dict):
        all_tomo_tokens = star_dict['tomoToken']
        if self.use_dimi:
            if 'tomoPathDimi' in star_dict.keys():
                all_tomo_paths = star_dict['tomoPathDimi']
            else:
                raise IOError('No dimi path provided!')
        elif self.denoised:
            if 'tomoPathDenoised' in star_dict.keys():
                all_tomo_paths = star_dict['_tomoPathDenoised']
            else:
                raise IOError('No denoised path provided!')
        else:
            all_tomo_paths = star_dict['tomoPath']
        self.tomo_paths_for_bin = {}
        for i, tomo_path in enumerate(all_tomo_paths):
            tomo_token = all_tomo_tokens[i]
            self.tomo_paths_for_bin[tomo_token] = {}
            self.tomo_paths_for_bin[tomo_token][self.consider_bin] = tomo_path

    def initialize_subvol_paths(self, star_dict):
        self.sub_volume_paths = {}
        all_tomo_tokens = star_dict['tomoToken']
        if self.use_dimi:
            rot_key = 'rotDirDimi'
        elif self.denoised:
            rot_key = 'rotDirDenoised'
        else:
            rot_key = 'rotDir'
        if rot_key in star_dict.keys():
            all_subvol_paths = star_dict[rot_key]
            for i, path in enumerate(all_subvol_paths):
                tomo_token = all_tomo_tokens[i]
                self.sub_volume_paths[tomo_token] = path

    def initialize_splits(self, star_dict):
        all_tomo_tokens = star_dict['tomoToken']
        all_mb_tokens = star_dict['mbToken']
        all_stack_tokens = star_dict['stackToken']
        if 'dataSplit' in star_dict.keys():
            self.data_splits = {}
            for i, datasplit in enumerate(star_dict['dataSplit']):
                tomo_token = all_tomo_tokens[i]
                if tomo_token not in self.data_splits.keys():
                    self.data_splits[tomo_token] = {(all_stack_tokens[i], all_mb_tokens[i]): datasplit}
                else:
                    self.data_splits[tomo_token][(all_stack_tokens[i], all_mb_tokens[i])] = datasplit




    def initialize_with_star(self, star_file, is_cluster_center_file):
        self.is_cluster_center_file = is_cluster_center_file
        star_dict = star_utils.read_star_file_as_dict(star_file)

        self.initialize_meta_params(star_dict)
        self.initialize_tokens(star_dict)
        self.initialize_pred_paths(star_dict, is_cluster_center_file)
        self.initialize_obj_dir(star_dict)
        self.initialize_GTs(star_dict)
        self.initialize_tomo_paths_for_bin(star_dict)
        self.initialize_subvol_paths(star_dict)
        self.initialize_splits(star_dict)
        self.project_directory = os.path.dirname(os.path.dirname(os.path.dirname(self.gt_paths[self.tomo_tokens[0]])))

    def denoised_choice(self, dimi, denoised):
        self.use_dimi = False
        self.denoised = False
        if dimi:
            self.use_dimi = True
        if denoised:
            self.use_dimi = False
            self.denoised = True


    def __init__(self, star_file=None, is_cluster_center_file=False, denoised=False, dimi=False):
        self.star_file = star_file
        self.denoised_choice(dimi, denoised)
        self.initialize_with_star(star_file, is_cluster_center_file)


