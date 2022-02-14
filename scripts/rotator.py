import os
import numpy as np
import csv
import torch
import math
import h5py
import time
if not torch.cuda.is_available():
    # import pyto
    pass
from utils import star_utils, data_utils
import multiprocessing as mp
from scipy.ndimage import rotate
from utils.rigid_3d import Rigid3D
from config import *


def compute_all_Euler_angles(in_dir, out_dir, in_del=',', out_del=',', hasHeader=False):
    """
    Given a directory containing .csv files, computes the Euler angles from these .csv files and stores them in out_dir.
    """
    for file in os.listdir(in_dir):
        if file.endswith('.csv'):
            compute_Euler_angles_for_normals(os.path.join(in_dir, file), os.path.join(out_dir, file), in_del=in_del,
                                             out_del=out_del, hasHeader=hasHeader)

def rotate_vol_for_angles(volume, angs, svol_cent):
    """
    Rotate volume around svol_cent with Z-Y-Z Euler angles.
    @param volume: volume to rotate
    @param angs: Euler angles
    @param svol_cent: center to rotate around
    @return: rotated volume
    """
    r3d = Rigid3D()
    r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')
    particle = r3d.transformArray(volume, origin=svol_cent, order=3, prefilter=True)
    return particle


def eulerAnglesToRotationMatrix(theta):
    """
    Computes the rotation matrix for three Euler angles of the shape X-Y-Z
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def compute_all_Euler_angles_for_star(in_star, out_dir, in_del=',', out_del=',', hasHeader=False):
    """
    Computes Euler angles for given normal vectors.
    Output Euler angles are of shape Z-Y-Z
    in_star: path to star file. Should contain normal vector csvs in key "particleCSV"
    out_dir: directory to store output csvs and star file
    in_del: delimiter of csv file
    out_del: delimiter of output csv file
    hasHeader: does the input csv file have a header?
    """
    star_dict = star_utils.read_star_file_as_dict(in_star)
    csv_list = star_dict['particleCSV']
    out_csv_list = []
    for i, csv_file in enumerate(csv_list):
        compute_Euler_angles_for_normals(csv_file, os.path.join(out_dir, os.path.basename(csv_file)), in_del=in_del,
                                         out_del=out_del, hasHeader=hasHeader)
        out_csv_list.append(os.path.join(out_dir, os.path.basename(csv_file)))
    star_dict['particleCSV'] = out_csv_list
    out_star_name = os.path.join(out_dir, os.path.basename(in_star))
    star_utils.write_star_file_from_dict(out_star_name, star_dict)
    return out_star_name


def compute_Euler_angle_for_single_normals_array(normals):
    """
    Use a normal vector to compute corresponding Euler angles of shape Z-Y-Z
    """
    angles = []
    for normal in normals:
        rot, tilt, psi = vect_to_zrelion(-1 * normal)
        angles.append([rot, tilt, psi])
    return angles


def compute_Euler_angles_for_normals(in_csv_path, out_csv_path, in_del=',', out_del=',', hasHeader=False):
    """
    For a csv file containing positions and normals, Z-Y-Z Euler angles are computed and stored into another csv file.
    """
    points = []
    normals = []
    print('Computing Euler Angles for file:', in_csv_path)
    with open(in_csv_path) as in_csv_file:
        csv_reader = csv.reader(in_csv_file, delimiter=in_del)
        for i, row in enumerate(csv_reader):
            if hasHeader and i == 0:
                continue
            points.append(np.array(row[:3], dtype=np.float))
            normals.append(np.array(row[3:6], dtype=np.float))
    angles = compute_Euler_angle_for_single_normals_array(normals)

    with open(out_csv_path, 'w') as out_csv_file:
        csv_writer = csv.writer(out_csv_file)
        for i, point in enumerate(points):
            point = point.tolist()
            normal = normals[i].tolist()
            angle = angles[i]
            csv_writer.writerow(point + normal + angle)


def vect_to_zrelion(normal):
        """
        adapted from Antonio Martinez Sanchez' PySeg:
        https://github.com/anmartinezs/pyseg_system/blob/master/code/pyseg/globals/utils.py

        Computes rotation angles of from an input vector to fit reference [0,0,1] vector having a free Euler angle
        in Relion format
        First Euler angle (Rotation) is assumed 0
        normal: input vector
        Returns: a 2-tuple with the Euler angles in Relion format

        """
        # Normalization
        v_m = np.asarray((normal[0], normal[1], normal[2]), dtype=np.float32)
        try:
            n = v_m / math.sqrt((v_m*v_m).sum())
        except ZeroDivisionError:
            print('WARNING (vect_rotation_ref): vector with module 0 cannot be rotated!')
            return 0., 0., 0.

        # Computing angles in Extrinsic ZYZ system
        alpha = np.arccos(n[2])
        beta = np.arctan2(n[1], n[0])

        # Transform to Relion system (intrinsic ZY'Z'' where rho is free)
        rot, tilt, psi = 0., unroll_angle(math.degrees(alpha), deg=True), \
                         unroll_angle(180.-math.degrees(beta), deg=True)
        return rot, tilt, psi


def unroll_angle(angle, deg=True):
    """
    copied from Antonio Martinez Sanchez' PySeg:
    https://github.com/anmartinezs/pyseg_system/blob/master/code/pyseg/globals/utils.py

    Unroll an angle [-infty, infty] to fit range [-180, 180] (or (-pi, pi) in radians)
    angle: input angle
    deg: if True (defult) the angle is in degrees, otherwise in radians

    """
    fang = float(angle)
    if deg:
        mx_ang, mx_ang2 = 360., 180.
    else:
        mx_ang, mx_ang2 = 2*np.pi, np.pi
    ang_mod, ang_sgn = np.abs(fang), np.sign(fang)
    ur_ang = ang_mod % mx_ang
    if ur_ang > mx_ang2:
        return -1. * ang_sgn * (mx_ang - ur_ang)
    else:
        return ang_sgn * ur_ang


def mp_tomo_split(tomo_tokens, n_pr):
    """
    Distribute tomograms on several processes.
    """
    tomo_lists = []
    for i in range(n_pr):
        tomo_lists.append([])
    for i, tomo_token in enumerate(tomo_tokens):
        tomo_lists[i % n_pr].append(tomo_token)
    return tomo_lists

class Rotator(object):
    def __init__(self, rotation_dir, out_bins, pred_bin, box_range, settings, n_pr, pos_star=None,
                       store_dir=None, store_normals=False, store_angles=False, preprocess_tomograms=True, lp_cutoff=None):
        self.rotation_dir = rotation_dir
        self.out_bins = out_bins
        self.pred_bin = pred_bin
        self.box_ranges = [box_range]
        self.settings = settings
        self.n_pr = n_pr
        self.pos_star = pos_star
        self.store_dir = store_dir
        self.store_normals = store_normals
        self.store_angles = store_angles
        self.preprocess_tomograms = preprocess_tomograms
        self.lp_cutoff = lp_cutoff

    def rotate_all_volumes(self):
        """
        Given sampled coordinates, together with Euler angles, subvolumes are extracted and normalized (i.e. membrane
        aligned with x-y-plane.

        rotation_dir: directory to store the rotated volumes
        out_bins: list containing the binnings of output subvolumes
        pred_bin: binning of input tomogram
        box_ranges: list of sizes of extracted subvolumes (2 * box_range)
        settings: ParameterSettings object, generated by star file
        n_pr: number of processes

        pos_star: If specified, an updated star file is stored into this position.
        store_dir: If specified, sample volumes are stored in the directory to check the outcome subvolumes
        store_normals: If True, normal vectors are added to the output .h5 file
        store_angles: If True, Euler angles are added to the output .h5 file
        """
        print("Start rotation")
        tomo_tokens = self.settings.tomo_tokens
        for tomo_token in tomo_tokens:
            self.rotate_for_one_tomogram(tomo_token)

        out_star_file = os.path.join(self.rotation_dir, os.path.basename(self.settings.star_file))
        add_rotation_dir_to_star(self.settings.star_file, out_star_file, self.rotation_dir)
        if self.pos_star is not None:
            star_dict = star_utils.read_star_file_as_dict(self.pos_star)
            star_utils.write_star_file_from_dict(os.path.join(self.rotation_dir, os.path.basename(self.pos_star)), star_dict)
        return out_star_file


    def rotate_for_one_tomogram(self, tomo_token):
        """
        Computes the rotation for a single tomogram. Description of parameters see "rotate_all_volumes" function.
        Single .h5 files are created for each tomogram.
        """
        out_path = os.path.join(self.rotation_dir, tomo_token + '_all_mbs.h5')
        mb_tokens = self.settings.mb_tokens
        stack_tokens = self.settings.stack_tokens
        pred_paths = self.settings.pred_paths
        tomo_paths_for_bin = self.settings.tomo_paths_for_bin
        with h5py.File(out_path, 'w') as f:
            for nr, out_bin in enumerate(self.out_bins):
                bin_group = f.create_group('bin' + str(out_bin))
                box_range = self.box_ranges[nr]
                tomo_path = tomo_paths_for_bin[tomo_token][out_bin]
                tomo = data_utils.load_tomogram(tomo_path)
                if self.preprocess_tomograms:
                    tomo = preprocess_tomogram(tomo.copy(), lp_cutoff=self.lp_cutoff)
                diff = np.max(tomo) - np.min(tomo)
                tomo = (tomo - np.min(tomo)) / diff
                for mb_nr, mb_token in enumerate(mb_tokens[tomo_token]):
                    if stack_tokens is not None:
                        stack_token = stack_tokens[tomo_token][mb_nr]
                        if (stack_token,mb_token) not in pred_paths[tomo_token].keys():
                            continue
                        csv_path = pred_paths[tomo_token][(stack_token, mb_token)]
                    else:
                        if mb_token not in pred_paths[tomo_token].keys():
                            continue
                        csv_path = pred_paths[tomo_token][mb_token]
                    time_zero = time.time()
                    mb_volumes, mb_positions, mb_normals, mb_angles = \
                        self._rotate_volumes_for_membrane_parallel(csv_path, box_range, out_bin,tomo=tomo)
                    print('This took', time.time() - time_zero, 'seconds.')

                    mb_group = bin_group.create_group(stack_token + mb_token)
                    mb_group.create_dataset('tomo_token', shape=(1,), data=tomo_token)
                    mb_group.create_dataset('mb_token', shape=(1,), data=mb_token)
                    mb_group.create_dataset('stack_token', shape=(1,), data=stack_token)
                    mb_group.create_dataset('bin', shape=(1,), data=out_bin)
                    mb_group.create_dataset('subvolumes', data=mb_volumes)
                    mb_group.create_dataset('positions', data=mb_positions)
                    if self.store_normals:
                        mb_group.create_dataset('normals', data=mb_normals)
                    if self.store_angles:
                        mb_group.create_dataset('angles', data=mb_angles)

    def _rotate_volumes_for_membrane_parallel(self, csv_path, box_range, out_bin, tomo=None):
        pred_scale = self.pred_bin * 1.0 / out_bin
        tomo_token, mb_token = data_utils.get_tomo_and_mb_from_file_name(csv_path, self.settings)
        print("Processing subvolumes for Tomo: ", tomo_token, "  Membrane:", mb_token, "  Bin:", out_bin)

        all_lines = data_utils.get_csv_data(csv_path)
        if self.n_pr > 1:
            parallel_mask = self._get_parallel_line_split(all_lines)
            processes = []
            manager = mp.Manager()
            return_dict = manager.dict()
            for pr_id in range(self.n_pr):
                pr = mp.Process(target=self.rotate_parallel_split,
                                args=(return_dict, all_lines, parallel_mask, pr_id, box_range, tomo,
                                      tomo_token, mb_token, pred_scale))
                pr.start()
                processes.append(pr)
            for pr_id in range(self.n_pr):
                pr = processes[pr_id]
                pr.join()
            time_zero = time.time()
            time_zero = time.time()

            all_volumes = np.concatenate([return_dict[pr_id][0] for pr_id in range(self.n_pr)], 0)
            all_positions = np.concatenate([return_dict[pr_id][1] for pr_id in range(self.n_pr)], 0)
            all_normals = np.concatenate([return_dict[pr_id][2] for pr_id in range(self.n_pr)], 0)
            all_angles = np.concatenate([return_dict[pr_id][3] for pr_id in range(self.n_pr)], 0)
        else:
            all_volumes, all_positions, all_normals, all_angles = self.rotate_parallel_split(None, all_lines, None, 0,
                                                                                             box_range, tomo,
                                                                                             tomo_token, mb_token,
                                                                                             pred_scale)
        return all_volumes, all_positions, all_normals, all_angles


    def _get_parallel_line_split(self, all_lines):
        """
        Distributes all positions of a membrane on multiple processes.
        @param all_lines: positions array
        @type all_lines: numpy array
        @param n_pr: number of processes
        @type n_pr: int
        @return: mask indicating the assigned processes
        """
        if self.n_pr == 1:
            return np.zeros(all_lines.shape[0])
        else:
            out_mask = np.zeros(all_lines.shape[0])
            step = int(all_lines.shape[0] / self.n_pr)
            for i in range(self.n_pr - 1):
                out_mask[i*step:(i+1)*step] = i
            out_mask[(i+1) * step:] = self.n_pr - 1
            return out_mask


    def rotate_parallel_split(self, return_dict, all_lines, line_mask, pr_id, box_range, tomo, tomo_token, mb_token, pred_scale):
        """
        Samples and rotates subvolumes and stores them into the return_dict
        @param return_dict: Multiprocessing dictionary to store sampled subvolumes
        @param all_lines: array containing positions, normals and angles
        @param line_mask: mask indicating the assigned processes
        @param pr_id: id of current process
        @param store_normals: should normals be stored?
        @param box_range: extent of sampled subvolumes (box_range * 2)
        @param tomo: volumetric tomogram
        @param store_dir: if specified, sampled volumes are stored here, in order to verify sampling
        @param pred_scale: scaling factor for positions, indicating whether they need to be multiplied (for binning)
        @param store_angles: should angles be stored?

        """
        print()
        print('Starting process', pr_id, '...')
        print('This process needs to convert', np.sum(line_mask == pr_id), 'subvolumes.')
        print()

        if self.n_pr > 1:
            cur_mask = line_mask == pr_id
            cur_lines = all_lines[cur_mask]
        else:
            cur_lines = all_lines
        if USE_ROTATION_NORMALIZATION:
            all_volumes = np.zeros((0, 2 * box_range, 2 * box_range, 2 * box_range))
        else:
            all_volumes = np.zeros((0, 4 * box_range + 1, 4 * box_range + 1, 4 * box_range + 1))
        all_positions = np.zeros((0, 3))
        all_normals = np.zeros((0, 3))
        all_angles = np.zeros((0, 3))

        particle_list = []
        positions_list = []
        if self.store_angles:
            angles_list = []
        if self.store_normals:
            normals_list = []
        for i, line in enumerate(cur_lines):
            positions = np.expand_dims(np.array([float(line[0]), float(line[1]), float(line[2])]), 0) * pred_scale
            positions = np.squeeze(positions)
            angles = np.array([float(line[6]), float(line[7]), float(line[8])])

            add_component = box_range * 2
            if not self.check_in_range(positions, add_component, tomo.shape):
                continue

            if self.store_angles:
                angles_list.append(angles)
            if self.store_normals:
                normals = np.array([float(line[3]), float(line[4]), float(line[5])])
                normals_list.append(normals)

            volume = tomo[int(round(positions[0]) - add_component): int(round(positions[0]) + add_component) + 1,
                     int(round(positions[1]) - add_component): int(round(positions[1]) + add_component) + 1,
                     int(round(positions[2]) - add_component): int(round(positions[2]) + add_component) + 1]
            if USE_ROTATION_NORMALIZATION:
                coord, angs = positions, angles
                svol_cent = np.array([add_component] * 3)
                particle = rotate_vol_for_angles(volume, angs, svol_cent)
                particle_cen = [add_component] * 3
                particle = particle[int(particle_cen[0] - box_range):int(particle_cen[0] + box_range),
                           int(particle_cen[1] - box_range):int(particle_cen[1] + box_range),
                           int(particle_cen[2] - box_range):int(particle_cen[2] + box_range)]
            else:
                particle = volume
            positions_list.append(positions)
            particle_list.append(particle)
            # if self.store_dir is not None:
            #     filename = os.path.join(self.store_dir, tomo_token + mb_token + '_' + str(i) + '_centervol.mrc')
            #     data_utils.store_tomogram(filename, particle)
        if self.store_angles:
            temp_angles = np.stack(angles_list, axis=0)
            all_angles = np.concatenate((all_angles, temp_angles), 0)
        if self.store_normals:
            temp_normals = np.stack(normals_list, axis=0)
            all_normals = np.concatenate((all_normals, temp_normals), 0)
        temp_volumes = np.stack(particle_list, axis=0)
        temp_positions = np.stack(positions_list, axis=0)
        all_volumes = np.concatenate((all_volumes, temp_volumes), 0)
        all_positions = np.concatenate((all_positions, temp_positions), 0)
        if self.n_pr > 1:
            return_dict[pr_id] = [all_volumes, all_positions, all_normals, all_angles]
        else:
            return all_volumes, all_positions, all_normals, all_angles

    def check_in_range(self, positions, add_component, tomo_shape):
        """
        Checks whether the sampled box around the given positions array is within the range of the tomogram.
        @param positions: positional array (3-dim)
        @param add_component: int specifying the extent of sampled box
        @type add_component: int
        @param tomo_shape: shape of the tomogram
        @type tomo_shape: list of ints (len 3)
        @return: indicator whether the sampled box is valid or not
        @rtype: bool
        """
        valid = True
        for j in range(3):
            if int(round(positions[j]) - add_component) < 0 or int(round(positions[j]) + add_component) + 1 >= \
                    tomo_shape[j]:
                valid = False
        return valid

    # def rotate_vol_for_angles(self, volume, angs, svol_cent):
    #     """
    #     Rotate volume around svol_cent with Z-Y-Z Euler angles.
    #     @param volume: volume to rotate
    #     @param angs: Euler angles
    #     @param svol_cent: center to rotate around
    #     @return: rotated volume
    #     """
    #     r3d = Rigid3D()
    #     r3d.q = r3d.make_r_euler(angles=np.radians(angs), mode='zyz_in_active')
    #     particle = r3d.transformArray(volume, origin=svol_cent, order=3, prefilter=True)
    #     return particle


    def rotate_vol_for_angles_alt(self, volume, angs):
        """
        This method is not working so far.
        """
        particle = rotate(volume, angs[0], axes=(1,0), order=1)
        particle = rotate(particle, angs[1], axes=(2,1), order=1)
        particle = rotate(particle, angs[2], axes=(1,2), order=1)
        return particle


def add_rotation_dir_to_star(in_star_file, out_star_file, rotDir):
    star_dict = star_utils.read_star_file_as_dict(in_star_file)
    tomo_list = star_dict['tomoToken']
    rot_list = []
    for tomo_token in tomo_list:
        rot_path = os.path.join(rotDir, tomo_token + '_all_mbs.h5')
        rot_list.append(rot_path)
    star_token = 'rotDir'
    star_dict[star_token] = rot_list
    star_utils.write_star_file_from_dict(out_star_file, star_dict)


def dir_token_denoised_dimi(denoised, use_dimi):
    if use_dimi:
        dir_token = 'with_dimi'
    elif denoised:
        dir_token = 'denoised'
    else:
        dir_token = 'raw'
    return dir_token

def normalize_tomo(tomo):
    min = np.min(tomo)
    tomo -= min
    max = np.max(tomo)
    tomo /= max
    return tomo


def gaussianLP_of_fft(tomo, cutoff_freq):
    center = (np.array(tomo.shape, dtype=np.float) - 1) / 2
    mg1, mg2, mg3 = np.meshgrid(np.array(range(tomo.shape[0])),np.array(range(tomo.shape[1])),
                                                                 np.array(range(tomo.shape[2])), indexing='ij')
    mg1, mg2, mg3 = np.expand_dims(mg1, 3), np.expand_dims(mg2, 3), np.expand_dims(mg3, 3)
    mg = np.concatenate((mg1, mg2, mg3), 3)
    mg = np.array(mg, dtype=np.float)
    mg -= center
    dist = np.linalg.norm(mg, axis=3)
    dist = dist * dist
    D0 = 1 / cutoff_freq
    dist /= (2 * D0**2)
    dist *= -1
    dist = np.exp(dist)
    tomo *= dist
    return tomo


def gaussianHP_of_fft(tomo, cutoff_freq):
    center = (np.array(tomo.shape, dtype=np.float) - 1) / 2
    mg1, mg2, mg3 = np.meshgrid(np.array(range(tomo.shape[0])), np.array(range(tomo.shape[1])),
                                np.array(range(tomo.shape[2])), indexing='ij')
    mg1, mg2, mg3 = np.expand_dims(mg1, 3), np.expand_dims(mg2, 3), np.expand_dims(mg3, 3)
    mg = np.concatenate((mg1, mg2, mg3), 3)
    mg = np.array(mg, dtype=np.float)
    mg -= center
    dist = np.linalg.norm(mg, axis=3)
    dist = dist * dist
    D0 = 1 / cutoff_freq
    dist /= (2 * D0 ** 2)
    dist *= -1
    dist = 1 - np.exp(dist)
    tomo *= dist
    return tomo


def standardize_tomogram(tomo):
    mean = np.mean(tomo)
    std = np.std(tomo)
    tomo_standardized = (tomo - mean) / std
    return tomo_standardized


def clamp_values_from_std_tomo(tomo):
    pos_mask = tomo > 3.
    neg_mask = tomo < -3.
    tomo[pos_mask] = 3.
    tomo[neg_mask] = -3.
    return tomo


def perform_low_pass_filtering(tomo, cutoff=0.05):
    fft_tomo = np.fft.fftn(tomo)
    fft_tomo = np.fft.fftshift(fft_tomo)
    fft_tomo = gaussianLP_of_fft(fft_tomo, cutoff)
    # fft_tomo = gaussianHP_of_fft(fft_tomo, 0.2)
    fft_tomo_shift = np.fft.ifftshift(fft_tomo)
    tomo2 = np.fft.ifftn(fft_tomo_shift)
    return tomo2


def preprocess_tomogram(tomo, lp_cutoff=None):
    print("Preprocessing.")
    if lp_cutoff is not None:
        print("Performing low-pass filtering. May take a couple of minutes.")
        tomo = perform_low_pass_filtering(tomo, cutoff=lp_cutoff)
    tomo = standardize_tomogram(tomo)
    tomo = clamp_values_from_std_tomo(tomo)
    return tomo

