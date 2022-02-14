from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import utils.star_utils as star_utils
import numpy as np
from utils.parameters import ParameterSettings
from typing import Optional
from scripts.add_labels_and_distances import load_hdf5_files
from config import *
import scripts.rotator as rotator


def add_datasplit_to_star(star_file, train_tokens=None, val_tokens=None, test_tokens=None, random_ratio=(60,20,20)):
    """
    Adds a datasplit column to the star file, indicating which set the current membrane belongs to
    @param train_tokens: dictionary with tomo_tokens as keys. Values are pairs of stack-mb_token. If not specified, will
        be initialized randomly.
    @param val_tokens: dictionary with tomo_tokens as keys. Values are pairs of stack-mb_token. If not specified, will
        be initialized randomly.
    @param test_tokens: dictionary with tomo_tokens as keys. Values are pairs of stack-mb_token. If not specified, will
        be initialized randomly (only in case random ratio is larger than 0).
    @param random_ratio: specified split ratio: training set -- validation set -- test set
    """
    assert np.sum(random_ratio) == 100, 'Sum of splits should always be 100!'
    star_dict = star_utils.read_star_file_as_dict(star_file)
    tomo_tokens = np.unique(star_dict['tomoToken'])
    all_tomo_tokens = star_dict['tomoToken']
    split_array = np.array(['no_choice'] * len(star_dict['tomoToken']))
    if train_tokens is not None:
        for tomo_token in train_tokens.keys():
            for (stack_token, mb_token) in train_tokens[tomo_token]:
                idx = star_utils.find_tomo_mb_idx(tomo_token, mb_token, star_dict, stack_token=stack_token)
                split_array[idx] = 'train'
    if val_tokens is not None:
        for tomo_token in val_tokens.keys():
            for (stack_token, mb_token) in val_tokens[tomo_token]:
                idx = star_utils.find_tomo_mb_idx(tomo_token, mb_token, star_dict, stack_token=stack_token)
                split_array[idx] = 'val'
    if test_tokens is not None:
        for tomo_token in test_tokens.keys():
            for (stack_token, mb_token) in test_tokens[tomo_token]:
                idx = star_utils.find_tomo_mb_idx(tomo_token, mb_token, star_dict, stack_token=stack_token)
                assert idx is not None, 'Could not find corresponding Tomo - Stack - MB combination: ' + tomo_token +\
                                        ' ' + stack_token + ' ' + mb_token
                split_array[idx] = 'test'
    val_flag = val_tokens is not None
    if val_tokens is None and tomo_tokens.shape[0] >= 100. / (random_ratio[1] + 1e-10):
        val_flag = True
        num_tomos_val = int(np.floor(random_ratio[1] / 100 * tomo_tokens.shape[0]))
        tomo_choice_val = np.random.choice(tomo_tokens, num_tomos_val, replace=False)
        split_array[np.isin(all_tomo_tokens, tomo_choice_val)] = 'val'

    test_flag = test_tokens is not None
    if test_tokens is None and tomo_tokens.shape[0] >= 100. / (random_ratio[2] + 1e-10):
        test_flag = True
        num_tomos_test = int(np.floor(random_ratio[2] / 100 * tomo_tokens.shape[0]))
        tomo_choice_test = np.random.choice(tomo_tokens[np.logical_not(np.isin(tomo_tokens, tomo_choice_val))],
                                            num_tomos_test, replace=False)
        split_array[np.isin(all_tomo_tokens, tomo_choice_test)] = 'test'
    remaining_idcs = np.array(np.argwhere(split_array == 'no_choice')).squeeze()
    if not val_flag:
        num_vals = int(np.floor(random_ratio[1] / 100 * split_array.shape[0]))
        if num_vals > 0:
            val_idcs = np.random.choice(remaining_idcs, num_vals, replace=False)
            split_array[val_idcs] = 'val'

    remaining_idcs = np.array(np.argwhere(split_array == 'no_choice')).squeeze()
    if not test_flag:
        num_tests = int(np.floor(random_ratio[2] / 100 * split_array.shape[0]))
        if num_tests > 0:
            test_idcs = np.random.choice(remaining_idcs, num_tests, replace=False)
            split_array[test_idcs] = 'test'

    split_array[split_array == 'no_choice'] = 'train'

    star_dict['dataSplit'] = split_array
    star_utils.write_star_file_from_dict(star_file, star_dict)

def rotate_90_subvol(subvol, times):
    if times == 0:
        return subvol
    subvol_res = np.rot90(subvol, times, (0, 1))
    return subvol_res

def flipx(subvol):
    subvol = np.flip(subvol,0).copy()
    return subvol

def flipy(subvol):
    subvol = np.flip(subvol,1).copy()
    return subvol

def flipz(subvol):
    subvol = np.flip(subvol,2).copy()
    return subvol

def random_noise_subvol(subvol, mean, std):
    noise = np.random.normal(mean, std, subvol.shape).copy()
    subvol = subvol + noise
    return subvol

class MemBrain_dataset(Dataset):
    ##TODO: Implement x-times test-time augmentation
    def __init__(self, star_file, split, part_dists, dist_thres=None, augment=True, normalize=False, test_phase=False,
                 max_dist=None):
        self.star_file = star_file
        self.augment = augment
        self.test_phase = test_phase
        self.settings = ParameterSettings(self.star_file)
        self.split = split
        if self.split == 'train' and ROTATION_AUGMENTATION_DURING_TRAINING:
            self.use_rotation_augmentation = True
        else:
            self.use_rotation_augmentation = False
        self.part_dists = part_dists
        self.__get_subvol_paths__()
        self.__load_h5_data()
        if self.subvolumes is not None and normalize:
            self.subvolumes = self.__scale_subvolumes()
        self.__print__()
        if max_dist is not None:
            self.__cap_distances__(max_dist)


    def __cap_distances__(self, max_dist):
        cap_mask = self.labels > max_dist
        self.labels[cap_mask] = max_dist

    def __print__(self):
        token = ('TRAINING' if self.split == 'train' else 'VALIDATION' if self.split == 'val' else 'TEST')
        print("")
        print("For the " + token + " set, the following combinations were used:")
        print("")
        unique_tomos = np.unique(self.tomo_tokens)
        for tomo in unique_tomos:
            tomo_mask = np.array(self.tomo_tokens) == tomo
            cur_stacks = np.array(self.stack_tokens)[tomo_mask]
            cur_mbs = np.array(self.mb_tokens)[tomo_mask]

            unique_stacks = np.unique(cur_stacks)
            for stack_token in unique_stacks:
                stack_mask = cur_stacks == stack_token
                unique_mbs = np.unique(cur_mbs[stack_mask])
                for mb_token in unique_mbs:
                    print('Tomo:', tomo, 'Stack:', stack_token, 'Membrane:', mb_token)
        print("")


    def __get_subvol_paths__(self):
        for tomo_token, value in self.settings.data_splits.items():
            keep_flag = False
            for _, data_split in value.items():
                if data_split == self.split:
                    keep_flag = True
            if not keep_flag:
                del self.settings.sub_volume_paths[tomo_token]

    def __load_h5_data(self):
        data_dict = load_hdf5_files(self.settings, self.split)
        self.subvolumes = None
        self.positions = None
        self.labels = None
        self.normals = None
        self.angles = None
        self.labels = None
        self.tomo_tokens = None
        self.stack_tokens = None
        self.mb_tokens = None
        for tomo_token in data_dict.keys():
            for bin_token in data_dict[tomo_token].keys():
                for (stack_token, mb_token) in data_dict[tomo_token][bin_token].keys():
                    cur_data = data_dict[tomo_token][bin_token][(stack_token, mb_token)]
                    if self.subvolumes is None:
                        self.subvolumes = cur_data['subvolumes']
                        self.positions = cur_data['positions']
                        self.normals = cur_data['normals']
                        self.angles = cur_data['angles']
                        self.labels = self.__convert_labels(cur_data)
                        self.tomo_tokens = [tomo_token] * cur_data['subvolumes'].shape[0]
                        self.mb_tokens = [mb_token] * cur_data['subvolumes'].shape[0]
                        self.stack_tokens = [stack_token] * cur_data['subvolumes'].shape[0]
                    else:
                        self.subvolumes = np.concatenate((self.subvolumes, cur_data['subvolumes']))
                        self.positions = np.concatenate((self.positions, cur_data['positions']))
                        self.normals = np.concatenate((self.normals, cur_data['normals']))
                        self.angles = np.concatenate((self.angles, cur_data['angles']))
                        self.labels = np.concatenate((self.labels, self.__convert_labels(cur_data)))
                        self.tomo_tokens += [tomo_token] * cur_data['subvolumes'].shape[0]
                        self.mb_tokens += [mb_token] * cur_data['subvolumes'].shape[0]
                        self.stack_tokens += [stack_token] * cur_data['subvolumes'].shape[0]


    def __convert_labels(self, labels):
        all_dists = []
        if isinstance(self.part_dists, list):
            for part_type in self.part_dists:
                if not isinstance(part_type, list):
                    cur_dist = labels['dist_' + part_type]
                else:
                    cur_dists = []
                    for entry in part_type:
                        cur_dists.append(labels['dist_' + entry])
                    cur_dists = np.stack(cur_dists)
                    cur_dist = np.min(cur_dists, axis=0)
                all_dists.append(cur_dist)
            all_dists = np.stack(all_dists, axis=1)
        else:
            all_dists = labels['dist_' + self.part_dists]
        return all_dists


    def __scale_subvolumes(self):
        min = np.amin(self.subvolumes, axis=(1,2,3))
        max = np.amax(self.subvolumes, axis=(1,2,3))
        self.subvolumes -= np.expand_dims(min, (1,2,3))
        self.subvolumes /= (np.expand_dims(max - min, (1,2,3)))

    def __scale_single_subvolume(self, subvol):
        min = np.min(subvol)
        max = np.max(subvol)
        subvol -= min
        subvol /= (max - min)
        return subvol



    def __augment_subvol(self, subvol):
        subvol = rotate_90_subvol(subvol, np.random.randint(0, 4))
        for k, func in enumerate([flipx, flipy, lambda x: random_noise_subvol(x, mean=0, std=0.15)]):
            if np.random.random(1) > 0.5 or k == 2 and np.random.random(1) > 0.5:
                subvol = func(subvol)
        subvol = self.__scale_single_subvolume(subvol)
        return subvol.copy()


    def __rand_rot_augmentation(self, subvol):
        rand_angles = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi),
                       np.random.uniform(-np.pi, np.pi))
        add_comp = BOX_RANGE * 2
        particle_cen = [add_comp] * 3
        svol_cent = np.array(particle_cen)
        rot_subvol = rotator.rotate_vol_for_angles(subvol, rand_angles, svol_cent)
        rot_subvol = rot_subvol[int(particle_cen[0] - BOX_RANGE):int(particle_cen[0] + BOX_RANGE),
                   int(particle_cen[1] - BOX_RANGE):int(particle_cen[1] + BOX_RANGE),
                   int(particle_cen[2] - BOX_RANGE):int(particle_cen[2] + BOX_RANGE)]
        return rot_subvol



    def __getitem__(self, idx):
        subvol = self.subvolumes[idx]
        label = self.labels[idx]
        if not USE_ROTATION_NORMALIZATION:
            if self.use_rotation_augmentation:
                subvol = self.__rand_rot_augmentation(subvol)
            else:
                add_comp = BOX_RANGE * 2
                particle_cen = [add_comp] * 3
                subvol = subvol[int(particle_cen[0] - BOX_RANGE):int(particle_cen[0] + BOX_RANGE),
                             int(particle_cen[1] - BOX_RANGE):int(particle_cen[1] + BOX_RANGE),
                             int(particle_cen[2] - BOX_RANGE):int(particle_cen[2] + BOX_RANGE)]

        if self.augment:
            subvol = self.__augment_subvol(subvol)
        if self.test_phase:
            return np.expand_dims(subvol, 0), label, self.positions[idx], self.tomo_tokens[idx], \
                   self.stack_tokens[idx], self.mb_tokens[idx], self.normals[idx], self.angles[idx]
        return np.expand_dims(subvol, 0), label

    def __len__(self):
        if self.subvolumes is not None:
            return self.subvolumes.shape[0]
        return 0


class MemBrain_datamodule(LightningDataModule):
    def __init__(self, star_file, batch_size, part_dists, dist_thres=None, max_dist=None):
        super().__init__()
        self.star_file = star_file
        self.batch_size = batch_size
        self.dist_thres = dist_thres
        self.part_dists = part_dists
        self.max_dist = max_dist
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train = MemBrain_dataset(self.star_file, 'train', self.part_dists, dist_thres=self.dist_thres,
                                      max_dist=self.max_dist)
        self.val = MemBrain_dataset(self.star_file, 'val', self.part_dists, dist_thres=self.dist_thres,
                                      max_dist=self.max_dist, test_phase=LOG_CLUSTERING_STATS)
        self.test = MemBrain_dataset(self.star_file, 'test', self.part_dists, dist_thres=self.dist_thres,
                                     augment=False, test_phase=True, max_dist=self.max_dist)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, shuffle=False)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

