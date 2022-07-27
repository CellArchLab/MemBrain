import os
import numpy as np
from utils.parameters import ParameterSettings
import csv
import mrcfile
import vtk
import torch

if not torch.cuda.is_available():
    pass
import lxml.etree as ET
import utils.star_utils as star_utils
from config import *
# mb_tokens = None
# tomo_tokens = None
# pred_paths = {}
# gt_paths = {}
# ranges = {}
# tomo_paths_for_bin = {}

# def initialize_for_this_script(in_pred_dir, in_gt_path, prediction_binning, use_dimi=False):
#     global mb_tokens, tomo_tokens, pred_paths, gt_paths, tomo_paths_for_bin, ranges
#     mb_tokens, tomo_tokens = param_init.initialize_tokens(prediction_binning)
#     pred_paths = param_init.initialize_pred_paths(
#         in_pred_dir=in_pred_dir)
#     gt_paths = param_init.initialize_gt_paths(
#         in_gt_path=in_gt_path)
#     tomo_paths_for_bin = param_init.initialize_tomo_paths_for_bin(use_dimi=use_dimi)
#     ranges = param_init.initialize_ranges(prediction_binning)


def get_file_class(file):
    if 'PSII' in file or 'PS2' in file:
        return 'PSII'
    if 'b6f' in file:
        return 'b6f'
    else:
        return 'UK'

def convert_csv_to_xml(in_files, out_file, gt_bin=1, pixel_spacing=3.68):
    root = ET.Element('Session')
    pgs = ET.Element('PointGroups')
    scale = 2 ** (gt_bin - 1) * pixel_spacing
    for file in in_files:
        class_name = get_file_class(file)
        pg = ET.Element('Group', Name=class_name+'_dimer')
        points = get_csv_data(file)
        if points is None:
            continue
        for i in range(points.shape[0]):
            point = np.array(points[i, :], dtype=np.float) * scale
            pentry = ET.Element('Point', ID=str(i), Position=str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]))
            pg.append(pentry)
        pgs.append(pg)
    root.append(pgs)
    with open(out_file, 'w') as out_f:
        out_f.write(ET.tostring(root, pretty_print=True))


def store_pos_ori_dict_in_xml(out_file, gt_dict, ori_dict):
    root = ET.Element('Session')
    pgs = ET.Element('PointGroups')
    for key in gt_dict.keys():
        class_name = key
        pg = ET.Element('Group', Name=class_name+'_dimer')
        points = gt_dict[key]
        orientations = ori_dict[key]
        for i in range(points.shape[0]):
            point = np.array(points[i, :], dtype=np.float)
            orientation = np.array(orientations[i, :], dtype=np.float)
            pentry = ET.Element('Point', ID=str(int(point[3])), Position=str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]), Orientation=str(orientation[0]) + ','+ str(orientation[1]) + ',' + str(orientation[2]))
            pg.append(pentry)
        pgs.append(pg)
    root.append(pgs)
    with open(out_file, 'w') as out_f:
        out_f.write(ET.tostring(root, pretty_print=True))


def mask_list_of_arrays(list_of_arrays, mask):
    new_list = []
    for entry in list_of_arrays:
        new_list.append(entry[mask])
    return tuple(new_list)



def convert_csv_to_membranorama(csv_file, out_file, csv_bin, pixel_spacing):
    csv_data = np.array(get_csv_data(csv_file)[:, :3], dtype=np.float)
    bin_fac = float(csv_bin) / 4
    csv_data *= bin_fac
    # csv_data[:, 2] += (3174 / pixel_spacing)
    # add_right_columns = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    add_right_columns = np.array([0.0])
    add_right_columns = np.tile(add_right_columns, (csv_data.shape[0], 1))
    csv_data = np.concatenate((csv_data * pixel_spacing, csv_data, add_right_columns), axis=1)
    csv_data = np.array(csv_data, dtype=np.float32)
    # header = np.array(['#PositionX', 'PositionY', 'PositionZ', 'VolumeX', 'VolumeY', 'VolumeZ', 'OffsetFromFace', 'm11', 'm21', 'm31', 'm12', 'm22', 'm32', 'm13', 'm23', 'm33'])
    header = np.array(['#PositionX', 'PositionY', 'PositionZ', 'VolumeX', 'VolumeY', 'VolumeZ', 'OffsetFromFace'])
    header = np.expand_dims(header, 0)
    csv_data = np.concatenate((header, csv_data), axis=0)
    store_array_in_csv(out_file, csv_data, out_del='\t')




def save_as_membranorama_xml(out_file, pos_dict, ori_dict=None):
    root = ET.Element('Session')
    pgs = ET.Element('PointGroups')
    for key in pos_dict.keys():
        class_name = key
        pg = ET.Element('Group', Name=class_name+'_dimer')
        points = pos_dict[key]
        if ori_dict is not None:
            oris = ori_dict[key]
        if points is None:
            continue
        for i in range(points.shape[0]):
            point = np.array(points[i, :3], dtype=np.float)
            if ori_dict is None:
                pentry = ET.Element('Point', ID=str(points[i, 3]), Position=str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]))
            else:
                ori = np.array(oris[i], dtype=np.float)
                pentry = ET.Element('Point', ID=str(points[i, 3]), Position=str(point[0]) + ',' + str(point[1]) + ',' + str(point[2]),
                                    Orientation=str(ori[0]) + ',' + str(ori[1]) + ',' + str(ori[2]))

            pg.append(pentry)
        pgs.append(pg)
    root.append(pgs)
    with open(out_file, 'w') as out_f:
        out_f.write(ET.tostring(root, pretty_print=True))


def get_csv_data(csv_path, delimiter=',', with_header=False, return_header=False):
    rows = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows.append(row)
    if len(rows) != 0:
        out_array = np.stack(rows)
    else:
        out_array = np.zeros((0,13))
    if return_header:
        try:
            return np.array(out_array[1:, :], dtype=np.float), out_array[0, :]
        finally:
            return np.array(out_array[1:, :]), out_array[0, :]
    if with_header:
        try:
            return np.array(out_array[1:, :], dtype=np.float)
        finally:
            return np.array(out_array[1:, :])
    return np.array(out_array, dtype=np.float)


def store_np_in_vtp(array, out_path):
    points = vtk.vtkPoints()
    vectors = vtk.vtkDoubleArray()
    vectors.SetNumberOfComponents(3)

    for i in range(array.shape[0]):
        row = array[i]
        coords = row[:3].tolist()
        normal = row[3:6].tolist()
        points.InsertNextPoint(coords[0], coords[1], coords[2])
        vectors.InsertNextTuple([normal[0], normal[1], normal[2]])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(vectors)
    polydata.GetPointData().SetActiveVectors(vectors.GetName())
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        error_msg = 'Error writing the file'
        print(error_msg)


def store_gt_arrays_in_vtp(array_dict, out_path):
    points = vtk.vtkPoints()
    classes = vtk.vtkDoubleArray()
    classes.SetNumberOfComponents(1)
    for j, key in enumerate(array_dict.keys()):
        array = array_dict[key]
        for i in range(array.shape[0]):
            row = array[i]
            coords = row.tolist()
            points.InsertNextPoint(coords[0], coords[1], coords[2])
            classes.InsertNextValue(j)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(classes)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        error_msg = 'Error writing the file'
        print(error_msg)


def check_mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def store_heatmaps_for_dataloader(dataloader, model, out_dir, star_file, consider_bin):
    check_mkdir(out_dir)
    preds = []
    all_labels = []
    all_positions = []
    all_tomos = []
    all_stacks = []
    all_mbs = []
    all_normals = []
    all_angles = []
    device = ('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')
    model = model.to(device)
    for batch in dataloader:
        try:
            vols, labels, positions, tomo_tokens, stack_tokens, mb_tokens, normals, angles = batch
        except:
            raise IOError("Error! Batch information does not contain enough information. If you want to log clustering statistics,"
                  "set LOG_CLUSTERING_STATS to True in config file")
        for entry, all_entries in [(labels, all_labels), (positions, all_positions), (tomo_tokens, all_tomos),
                                   (stack_tokens, all_stacks), (mb_tokens, all_mbs), (normals, all_normals),
                                   (angles, all_angles)]:
            if isinstance(entry, tuple):
                all_entries.append(np.array(entry))
            else:
                all_entries.append(entry.detach())
        batch_pred = model(vols.to(device))
        preds.append(batch_pred.detach().cpu())
    data_dict = {
        'tomo_token': np.concatenate(all_tomos),
        'stack_token': np.concatenate(all_stacks),
        'mb_token': np.concatenate(all_mbs),
        'position': np.concatenate(all_positions),
        'label': np.concatenate(all_labels),
        'pred': np.concatenate(preds),
        'normal': np.concatenate(all_normals),
        'angle': np.concatenate(all_angles)
    }
    consider_bin = consider_bin
    data_dict['position'] *= consider_bin
    out_star = store_pred_results_in_h5(data_dict, out_dir, star_file=star_file)
    return out_star


def store_pred_results_in_h5(data_dict, out_dir, star_file=None):
    if star_file is not None:
        star_dict = star_utils.read_star_file_as_dict(star_file)
        out_dict = {}
    heatmap_paths = []
    unique_tomos = np.unique(data_dict['tomo_token'])
    for unique_tomo in unique_tomos:
        tomo_mask = data_dict['tomo_token'] == unique_tomo
        cur_tomo_dict = data_dict.copy()
        for key in cur_tomo_dict.keys():
            cur_tomo_dict[key] = cur_tomo_dict[key][tomo_mask]
        cur_stack_mbs = np.unique(np.concatenate((np.expand_dims(cur_tomo_dict['stack_token'], axis=1),
                                                  np.expand_dims(cur_tomo_dict['mb_token'], axis=1)), axis=1), axis=0)
        for unique_stack, unique_mb in cur_stack_mbs:
            if star_file is not None:
                cur_idx = star_utils.find_tomo_mb_idx(unique_tomo, unique_mb, star_dict, stack_token=unique_stack)
                star_utils.merge_line_from_star_dict_to_star_dict(cur_idx, star_dict, out_dict)

            out_file = os.path.join(out_dir, unique_tomo + '_' + unique_stack + '_' + unique_mb + '_heatmap.csv')
            heatmap_paths.append(out_file)
            stack_mask = cur_tomo_dict['stack_token'] == unique_stack
            mb_mask = cur_tomo_dict['mb_token'] == unique_mb
            stack_mb_mask = np.logical_and(stack_mask, mb_mask)

            cur_stack_mb_dict = cur_tomo_dict.copy()
            for key in cur_stack_mb_dict:
                cur_stack_mb_dict[key] = cur_stack_mb_dict[key][stack_mb_mask]
            all_data = None
            for key, entry in cur_stack_mb_dict.items():
                if key in ['tomo_token', 'stack_token', 'mb_token']:
                    continue
                if len(entry.shape) == 1:
                    entry = np.expand_dims(entry, 1)
                if all_data is None:
                    all_data = entry
                else:
                    all_data = np.concatenate((all_data, entry), 1)
            header = ['posX', 'posY', 'posZ']
            for entry in TRAINING_PARTICLE_DISTS:
                if not isinstance(entry, list):
                    header.append('labelDist_' + entry)
                else:
                    token = entry[0]
                    for instance in entry[1:]:
                        token += "_" + instance
                    header.append('labelDist_' + token)

            for entry in TRAINING_PARTICLE_DISTS:
                if not isinstance(entry, list):
                    header.append('predDist_' + entry)
                else:
                    token = entry[0]
                    for instance in entry[1:]:
                        token += "_" + instance
                    header.append('predDist_' + token)
            header += ['normalX', 'normalY', 'normalZ', 'anglePhi', 'angleTheta', 'anglePsi']
            store_array_in_csv(out_file, all_data, header=header)
            print(out_file)
            convert_csv_to_vtp(out_file, out_file[:-3] + 'vtp', hasHeader=True)
    out_star_file = os.path.join(out_dir, os.path.basename(star_file))
    star_utils.write_star_file_from_dict(out_star_file, out_dict)
    star_utils.add_or_change_column_to_star_file(out_star_file, 'heatmapDir', heatmap_paths)
    return out_star_file


def convert_csv_to_vtp(in_path, out_path, delimiter=',', hasHeader=False):
    """
    Converts a .csv file to a vtp file containing the points and their respective normal vectors.
    .csv file should contain xyz-coordinates in the first 3 columns and the normal vector (xyz) in the following 3 columns
    :param in_path: path to csv
    :param out_path: path to desired .vtp
    :param delimiter: delimiter of .csv file
    :param hasHeader: boolean, whether .csv file contains header or not
    :return: None
    """
    with open(in_path) as in_file:
        points = vtk.vtkPoints()
        vectors = vtk.vtkDoubleArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("Normal")
        normal_flag = False
        # if hasHeader:
        #     labels = vtk.vtkDoubleArray()
        #     labels.SetNumberOfComponents(1)
        #     labels.SetName("Labels")
        #     preds = vtk.vtkDoubleArray()
        #     preds.SetNumberOfComponents(1)
        #     preds.SetName("Prediction")
        csv_reader = csv.reader(in_file, delimiter=delimiter)
        for i, row in enumerate(csv_reader):
            if hasHeader and i == 0:
                x_col = np.argwhere(np.array(row) == 'posX')[0][0]
                y_col = np.argwhere(np.array(row) == 'posY')[0][0]
                z_col = np.argwhere(np.array(row) == 'posZ')[0][0]
                normX = np.argwhere(np.array(row) == 'normalX')[0][0]
                normY = np.argwhere(np.array(row) == 'normalY')[0][0]
                normZ = np.argwhere(np.array(row) == 'normalZ')[0][0]
                header_labels = np.unique(row)
                header_count = 0
                token_dict = {}
                for header_label in header_labels:
                    if header_label.startswith('labelDist'):
                        header_count += 1
                        token = header_label[10:]
                        token_dict[token] = (np.argwhere(np.array(row) == header_label)[0][0],
                                             np.argwhere(np.array(row) == 'predDist_' + token)[0][0])
                        exec("labels_" + token + " = vtk.vtkDoubleArray()")
                        exec("labels_" + token + ".SetNumberOfComponents(1)")
                        exec("labels_" + token + ".SetName(\"Labels_" + token + "\")")
                        exec("preds_" + token + " = vtk.vtkDoubleArray()")
                        exec("preds_" + token + ".SetNumberOfComponents(1)")
                        exec("preds_" + token + ".SetName(\"Preds_" + token + "\")")

                # label_dist = np.argwhere(np.array(row) == 'labelDist')[0][0]
                # pred_dist = np.argwhere(np.array(row) == 'predDist')[0][0]
                continue
            elif i == 0:
                x_col, y_col, z_col, normX, normY, normZ = 0, 1, 2, 3, 4, 5
            coords = [float(row[x_col]), float(row[y_col]), float(row[z_col])]
            points.InsertNextPoint(coords[0], coords[1], coords[2])
            if len(row) > 3:
                normal = [float(row[normX]), float(row[normY]), float(row[normZ])]
                vectors.InsertNextTuple([normal[0], normal[1], normal[2]])
                normal_flag = True
            if hasHeader:
                for token in token_dict.keys():
                    exec("labels_" + token + ".InsertNextValue(float(row[token_dict[token][0]]))")
                    exec("preds_" + token + ".InsertNextValue(float(row[token_dict[token][1]]))")
                    # preds.InsertNextValue(float(row[pred_dist]))


    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if normal_flag:
        polydata.GetPointData().AddArray(vectors)
        polydata.GetPointData().SetActiveVectors(vectors.GetName())
    if hasHeader:
        for token in token_dict.keys():
            exec("polydata.GetPointData().AddArray(labels_" + token + ")")
            exec("polydata.GetPointData().AddArray(preds_" + token + ")")
            # polydata.GetPointData().AddArray(preds)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        error_msg = 'Error writing the file'
        print(error_msg)


def get_subtomos_from_coords(tomo, coords, box_range, normals=None, return_normals=True):
    add_component = box_range
    subvol_list = []
    for i in range(coords.shape[0]):
        positions = coords[i, :]
        continue_flag = False
        for j in range(3):
            if int(round(positions[j]) - add_component) < 0 or int(round(positions[j]) + add_component) + 1 >= \
                    tomo.shape[j]:
                continue_flag = True
        if continue_flag:
            continue
        cur_subvol = tomo[int(round(positions[0]) - add_component): int(round(positions[0]) + add_component),
                 int(round(positions[1]) - add_component): int(round(positions[1]) + add_component),
                 int(round(positions[2]) - add_component): int(round(positions[2]) + add_component)]
        subvol_list.append(cur_subvol)
    if len(subvol_list) == 0:
        return None, None, None
    subvols = np.stack(subvol_list)
    return subvols, coords, normals


def get_tomo_path_for_tomo_token(in_star, tomo_token, dimi, denoised=False):
    star_dict = star_utils.read_star_file_as_dict(in_star)
    tomo_tokens = star_dict['tomoToken']
    mb_tokens = star_dict['mbToken']
    if dimi:
        tomo_paths = star_dict['tomoPathDimi']
    elif denoised:
        tomo_paths = star_dict['tomoPathDenoised']
    else:
        tomo_paths = star_dict['tomoPath']
    for i in range(len(tomo_tokens)):
        if tomo_tokens[i] == tomo_token:
            return tomo_paths[i]
    raise IOError('No valid combination of tomo token and mb token found!')


def store_in_subvolumes(pos_array, test_labels, test_preds, sample_tomos, slice_dir, settings, in_bin=2, out_bin=4):
    """
    Store positions into subvolumes according to their predicted labels
    :param pos_array: classified positions
    :param test_labels: ground truth labels
    :param test_preds: predicted labels
    :param slice_dir: directory where to store the subvolumes
    :param in_bin: binning of pos_array positions
    :param out_bin: binning of desired subvolumes
    :return: stores positions into file
    """
    assert isinstance(settings, ParameterSettings)
    tomo_tokens = settings.tomo_tokens
    tomo_paths_for_bin = settings.tomo_paths_for_bin

    scale = float(in_bin) / float(out_bin)
    true_pos_mask = ((test_preds > 0.5) * (test_labels > 0.5)) > 0.5
    false_pos_mask = ((test_preds > 0.5) * (test_labels < 0.5)) > 0.5
    true_neg_mask = ((test_preds < 0.5) * (test_labels < 0.5)) > 0.5
    false_neg_mask = ((test_preds < 0.5) * (test_labels > 0.5)) > 0.5

    tomograms = {}
    for tomo_token in tomo_tokens:
        tomo_file_name = tomo_paths_for_bin[tomo_token][out_bin]
        tomogram = load_tomogram(tomo_file_name)
        tomograms[tomo_token] = tomogram

    stem = 'true_pos'
    pred_array_pos = pos_array[true_pos_mask][:, :] * scale
    sample_tomos_pos = sample_tomos[true_pos_mask]
    store_all_positions_as_tomograms(slice_dir, stem, pred_array_pos, tomograms, sample_tomos_pos)
    stem = 'false_pos'
    pred_array_pos = pos_array[false_pos_mask][:, :] * scale
    sample_tomos_pos = sample_tomos[false_pos_mask]
    store_all_positions_as_tomograms(slice_dir, stem, pred_array_pos, tomograms, sample_tomos_pos)
    stem = 'true_neg'
    pred_array_pos = pos_array[true_neg_mask][:, :] * scale
    sample_tomos_pos = sample_tomos[true_neg_mask]
    store_all_positions_as_tomograms(slice_dir, stem, pred_array_pos, tomograms, sample_tomos_pos)
    stem = 'false_neg'
    pred_array_pos = pos_array[false_neg_mask][:, :] * scale
    sample_tomos_pos = sample_tomos[false_neg_mask]
    store_all_positions_as_tomograms(slice_dir, stem, pred_array_pos, tomograms, sample_tomos_pos)


def find_star_idx_for_tomo_mb_combo(star_dict, tomo_token, mb_token, stack_token=None):
    tomo_tokens = star_dict["tomoToken"]
    mb_tokens = star_dict["mbToken"]
    print(tomo_token, mb_token)
    if stack_token is not None:
        stack_tokens = star_dict['stackToken']
    print(stack_token, tomo_token, mb_token)
    for idx, mb in enumerate(mb_tokens):
        if (mb == mb_token or (mb_token == 'M' + mb)) and tomo_tokens[idx] == tomo_token:
            if stack_token is None or stack_tokens[idx] == stack_token:
                return idx
    raise IOError('given star file has no tomo-mb combination as desired')


def get_single_entry_subdict(star_dict, idx):
    out_dict = {}
    for key in star_dict.keys():
        out_dict[key] = star_dict[key][idx]
    return out_dict


def initialize_dict_like(copy_dict, additional_fields=None):
    out_dict = {}
    for key in copy_dict.keys():
        out_dict[key] = []
    if additional_fields is not None:
        if isinstance(additional_fields, str) or (isinstance(additional_fields, list) and isinstance(additional_fields[0], str)):
            if isinstance(additional_fields, str):
                out_dict[additional_fields] = []
            else:
                for entry in additional_fields:
                    out_dict[entry] = []
    return out_dict


def add_single_dict_to_dict(add_dict, full_dict):
    for key in full_dict.keys():
        full_dict[key].append(add_dict[key])
    return full_dict


def store_heatmaps(pos_array, preds, sample_mbs, sample_tomos, gt_labels, in_bin, out_bin, settings, threshold=0.0,
                   out_path='/fs/pool/pool-engel/Lorenz/real_data/all_MBs_V7/for_mb3/', out_star=None, idx=0, stacks=None):
    print('Computing heatmap and pos_preds for threshold:', threshold)
    assert isinstance(settings, ParameterSettings)
    in_star = settings.star_file
    star_dict = star_utils.read_star_file_as_dict(in_star)
    out_dict = initialize_dict_like(star_dict, ['heatmapPath'])
    pos_array_bu = np.squeeze(pos_array)
    preds_bu = preds
    gt_labels_bu = gt_labels
    diff_sample_tomos = np.unique(sample_tomos)
    if stacks is not None:
        # if not np.all(np.array(sample_mbs.shape) == np.array(stacks.shape)):
        #     diff_sample_mbs = np.unique(np.concatenate((sample_mbs, stacks[:, 0]), axis=1), axis=0)
        # else:
        diff_sample_mbs = np.unique(np.concatenate((sample_mbs, stacks), axis=1), axis=0)

    else:
        diff_sample_mbs = np.unique(sample_mbs)

    for sample_tomo in diff_sample_tomos:
        pos_pred_tomo = np.zeros((0, 3))
        neg_pred_tomo = np.zeros((0, 3))
        pos_labels_tomo = np.zeros((0, 3))
        neg_labels_tomo = np.zeros((0, 3))

        for sample_mb in diff_sample_mbs:
            if stacks is not None:
                sample_stack = sample_mb[1]
                sample_mb = sample_mb[0]
                sample_tomo_mask = np.argwhere((sample_tomos[:, 0] == sample_tomo) * (sample_mbs[:, 0] == sample_mb) * (stacks[:,0] == sample_stack))
            else:
                sample_tomo_mask = np.argwhere((sample_tomos == sample_tomo) * (sample_mbs == sample_mb))
            sample_tomo_mask = sample_tomo_mask[:, 0]
            if np.sum(sample_tomo_mask) == 0:
                continue
            if stacks is not None:
                print ('Processing combination: Tomo', sample_tomo, '  Membrane:', sample_mb, 'stack:', sample_stack)
                tomo_mb_star_idx = find_star_idx_for_tomo_mb_combo(star_dict, tomo_token=sample_tomo, mb_token=sample_mb, stack_token=sample_stack)
            else:
                print ('Processing combination: Tomo', sample_tomo, '  Membrane:', sample_mb)
                tomo_mb_star_idx = find_star_idx_for_tomo_mb_combo(star_dict, tomo_token=sample_tomo, mb_token=sample_mb)
            temp_dict = get_single_entry_subdict(star_dict, tomo_mb_star_idx)
            if stacks is not None:
                temp_dict["heatmapPath"] = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_stack' + sample_stack + '_heatmap' + str(idx) + '.csv')
            else:
                temp_dict["heatmapPath"] = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_heatmap' + str(idx) + '.csv')
            # temp_dict["clusterPath"] = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_heatmap' + str(idx) + '.csv')
            out_dict = add_single_dict_to_dict(temp_dict, out_dict)
            print(pos_array_bu.shape)
            pos_array = pos_array_bu[sample_tomo_mask]
            preds = preds_bu[sample_tomo_mask]
            pred_labels = preds > threshold
            gt_labels = gt_labels_bu[sample_tomo_mask]
            scale = in_bin *1.0 / out_bin
            pos_pred_mask = pred_labels > 0.5
            neg_pred_mask = pred_labels < 0.5
            pos_gt_mask = gt_labels > 0.5
            neg_gt_mask = gt_labels < 0.5
            pos_pred = pos_array[pos_pred_mask][:, :] * scale
            neg_pred = pos_array[neg_pred_mask][:, :] * scale
            pos_labels = pos_array[pos_gt_mask][:, :] * scale
            neg_labels = pos_array[neg_gt_mask][:, :] * scale
            np.concatenate((pos_pred_tomo, pos_pred), 0)
            np.concatenate((neg_pred_tomo, neg_pred), 0)
            np.concatenate((pos_labels_tomo, pos_labels), 0)
            np.concatenate((neg_labels_tomo, neg_labels), 0)

            all_points = pos_array[:, :] * scale
            all_points = np.concatenate((all_points, np.expand_dims(preds, 1)), 1)

            # all_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_heatmap.txt')
            # pos_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_pos_pred.txt')
            # neg_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_neg_pred.txt')
            # pos_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_pos_gt.txt')
            # neg_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_neg_gt.txt')

            # store_in_tab_txt_for_membranogram(all_path, all_points)
            # store_in_tab_txt_for_membranogram(pos_path, pos_pred)
            # store_in_tab_txt_for_membranogram(neg_path, neg_pred)
            # store_in_tab_txt_for_membranogram(pos_gt_path, pos_labels)
            # store_in_tab_txt_for_membranogram(neg_gt_path, neg_labels)
            store_in_bin2_slices(out_path, sample_tomo, sample_mb, threshold, all_points, pos_pred, neg_pred, pos_labels, neg_labels, settings, idx=idx, stack=sample_stack)
    if out_star is not None:
        star_utils.write_star_file_from_dict(out_star, out_dict)


def store_in_bin2_slices(out_path, sample_tomo, sample_mb, threshold, all_points, pos_pred, neg_pred, pos_labels, neg_labels, settings, idx=0, stack=None):
    # all_points[:, :-1] *= 2
    # pos_pred *= 2
    # neg_pred *= 2
    # pos_labels *= 2
    # neg_labels *= 2

    # all_points[:, :-1] = change_coords_from_whole_tomo_to_slice(all_points[:, :-1], 2, 2, sample_tomo, sample_mb, settings)
    # pos_pred = change_coords_from_whole_tomo_to_slice(pos_pred, 2, 2, sample_tomo, sample_mb, settings)
    # neg_pred = change_coords_from_whole_tomo_to_slice(neg_pred, 2, 2, sample_tomo, sample_mb, settings)
    # pos_labels = change_coords_from_whole_tomo_to_slice(pos_labels, 2, 2, sample_tomo, sample_mb, settings)
    # neg_labels = change_coords_from_whole_tomo_to_slice(neg_labels, 2, 2, sample_tomo, sample_mb, settings)
    if stack is None:
        all_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_heatmap' + str(idx) + '.csv')
    else:
        all_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_stack' + stack + '_heatmap' + str(idx) + '.csv')

    # pos_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_pos_pred_slice.csv')
    # neg_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_neg_pred_slice.csv')
    # pos_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_pos_gt_slice.csv')
    # neg_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_thres' + str(threshold) + '_neg_gt_slice.csv')

    store_array_in_csv(all_path, all_points)
    # store_array_in_csv(pos_path, pos_pred)
    # store_array_in_csv(neg_path, neg_pred)
    # store_array_in_csv(pos_gt_path, pos_labels)
    # store_array_in_csv(neg_gt_path, neg_labels)


def valid_header(header, data_shape):
    header_flag = True
    if header is not None:
        if isinstance(header, list):
            if len(header) != data_shape[1]:
                header_flag = False
        elif isinstance(header, np.ndarray):
            if header.shape[0] != 1:
                if header.shape[0] != data_shape[1]:
                    header_flag = False
            elif header.shape[1] != data_shape[1]:
                header_flag = False
        else:
            header_flag = False
    else:
        header_flag = False
    return header_flag
#

def store_array_in_csv(out_file, data, out_del=',', header=None):
    with open(out_file, 'w') as out_csv:
        csv_writer = csv.writer(out_csv, delimiter=out_del)
        header_flag = valid_header(header, data.shape)
        if header_flag:
            csv_writer.writerow(header)
        for i in range(data.shape[0]):
            row = data[i]
            csv_writer.writerow(row)


def store_in_csvs(pos_array, pred_labels, sample_mbs, sample_tomos, gt_labels, settings, tomo_nr=1, mb_nr=3, in_bin=2, out_bin=4,
                  out_path='/fs/pool/pool-engel/Lorenz/real_data/all_MBs_V7/for_mb3/', for_membranogram=False, to_pred_slices=False):
    """
    Store predictions into csvs according to their predicted labels
    :param pos_array: positions of analyzed particles
    :param pred_labels: predicted labels
    :param gt_labels: ground truth labels
    :param in_bin: binning of position array
    :param out_bin: desired binning in csvs
    :param out_path: where to store the csvs?
    :return: stores positions into csv
    """

    pos_array_bu = pos_array
    pred_labels_bu = pred_labels
    gt_labels_bu = gt_labels

    diff_sample_tomos = np.unique(sample_tomos)
    diff_sample_mbs = np.unique(sample_mbs)

    for sample_tomo in diff_sample_tomos:
        pos_pred_tomo = np.zeros((0,3))
        neg_pred_tomo = np.zeros((0,3))
        pos_labels_tomo = np.zeros((0,3))
        neg_labels_tomo = np.zeros((0,3))
        tp_tomo = np.zeros((0,3))
        fp_tomo = np.zeros((0,3))
        tn_tomo = np.zeros((0,3))
        fn_tomo = np.zeros((0,3))
        for sample_mb in diff_sample_mbs:
            sample_tomo_mask = np.argwhere((sample_tomos == sample_tomo) * (sample_mbs == sample_mb))
            sample_tomo_mask = sample_tomo_mask[:, 0]
            if np.sum(sample_tomo_mask) == 0:
                print ("Continuing", sample_tomo, sample_mb)
                continue
            pos_array = pos_array_bu[sample_tomo_mask]
            pred_labels = pred_labels_bu[sample_tomo_mask]
            gt_labels = gt_labels_bu[sample_tomo_mask]
            scale = in_bin *1.0 / out_bin
            pos_pred_mask = pred_labels > 0.5
            neg_pred_mask = pred_labels < 0.5
            pos_gt_mask = gt_labels > 0.5
            neg_gt_mask = gt_labels < 0.5
            pos_pred = pos_array[pos_pred_mask][:, 0, :] * scale
            neg_pred = pos_array[neg_pred_mask][:, 0, :] * scale
            pos_labels = pos_array[pos_gt_mask][:, 0, :] * scale
            neg_labels = pos_array[neg_gt_mask][:, 0, :] * scale
            tp = pos_array[pos_pred_mask * pos_gt_mask][:, 0, :] * scale
            fp = pos_array[pos_pred_mask * neg_gt_mask][:, 0, :] * scale
            tn = pos_array[neg_pred_mask * neg_gt_mask][:, 0, :] * scale
            fn = pos_array[neg_pred_mask * pos_gt_mask][:, 0, :] * scale
            np.concatenate((pos_pred_tomo, pos_pred), 0)
            np.concatenate((neg_pred_tomo, neg_pred), 0)
            np.concatenate((pos_labels_tomo, pos_labels), 0)
            np.concatenate((neg_labels_tomo, neg_labels), 0)
            np.concatenate((tp_tomo, tp), 0)
            np.concatenate((fp_tomo, fp), 0)
            np.concatenate((tn_tomo, tn), 0)
            np.concatenate((fn_tomo, fn), 0)


            if to_pred_slices and not for_membranogram:
                pos_pred = change_coords_from_whole_tomo_to_slice(pos_pred, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                neg_pred = change_coords_from_whole_tomo_to_slice(neg_pred, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                pos_labels = change_coords_from_whole_tomo_to_slice(pos_labels, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                neg_labels = change_coords_from_whole_tomo_to_slice(neg_labels, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                tp = change_coords_from_whole_tomo_to_slice(tp, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                fp = change_coords_from_whole_tomo_to_slice(fp, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                tn = change_coords_from_whole_tomo_to_slice(tn, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)
                fn = change_coords_from_whole_tomo_to_slice(fn, 4, 2, tomo_token=sample_tomo, mb_token=sample_mb, settings=settings)

            if not for_membranogram:
                pos_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_pos_pred.csv')
                neg_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_neg_pred.csv')
                pos_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_pos_gt.csv')
                neg_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_neg_gt.csv')
                tp_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_tp.csv')
                fp_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_fp.csv')
                tn_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_tn.csv')
                fn_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_fn.csv')
                store_particle_positions_in_csv(pos_path, pos_pred)
                store_particle_positions_in_csv(neg_path, neg_pred)
                store_particle_positions_in_csv(pos_gt_path, pos_labels)
                store_particle_positions_in_csv(neg_gt_path, neg_labels)
                store_particle_positions_in_csv(tp_path, tp)
                store_particle_positions_in_csv(fp_path, fp)
                store_particle_positions_in_csv(tn_path, tn)
                store_particle_positions_in_csv(fn_path, fn)
            else:
                pos_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_pos_pred.txt')
                neg_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_neg_pred.txt')
                pos_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_pos_gt.txt')
                neg_gt_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_neg_gt.txt')
                tp_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_tp.txt')
                fp_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_fp.txt')
                tn_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_tn.txt')
                fn_path = os.path.join(out_path, sample_tomo + '_m' + sample_mb + '_fn.txt')

                store_in_tab_txt_for_membranogram(pos_path, pos_pred)
                store_in_tab_txt_for_membranogram(neg_path, neg_pred)
                store_in_tab_txt_for_membranogram(pos_gt_path, pos_labels)
                store_in_tab_txt_for_membranogram(neg_gt_path, neg_labels)
                store_in_tab_txt_for_membranogram(tp_path, tp)
                store_in_tab_txt_for_membranogram(fp_path, fp)
                store_in_tab_txt_for_membranogram(tn_path, tn)
                store_in_tab_txt_for_membranogram(fn_path, fn)

        if not for_membranogram:
            pos_path = os.path.join(out_path, sample_tomo +  '_pos_pred.csv')
            neg_path = os.path.join(out_path, sample_tomo + '_neg_pred.csv')
            pos_gt_path = os.path.join(out_path, sample_tomo + '_pos_gt.csv')
            neg_gt_path = os.path.join(out_path, sample_tomo + '_neg_gt.csv')
            tp_path = os.path.join(out_path, sample_tomo + '_tp.csv')
            fp_path = os.path.join(out_path, sample_tomo + '_fp.csv')
            tn_path = os.path.join(out_path, sample_tomo + '_tn.csv')
            fn_path = os.path.join(out_path, sample_tomo + '_fn.csv')

            # store_particle_positions_in_csv(pos_path, pos_pred_tomo)
            # store_particle_positions_in_csv(neg_path, neg_pred_tomo)
            # store_particle_positions_in_csv(pos_gt_path, pos_labels_tomo)
            # store_particle_positions_in_csv(neg_gt_path, neg_labels_tomo)
            # store_particle_positions_in_csv(tp_path, tp_tomo)
            # store_particle_positions_in_csv(fp_path, fp_tomo)
            # store_particle_positions_in_csv(tn_path, tn_tomo)
            # store_particle_positions_in_csv(fn_path, fn_tomo)
        else:
            pos_path = os.path.join(out_path, sample_tomo  + '_pos_pred.txt')
            neg_path = os.path.join(out_path, sample_tomo  + '_neg_pred.txt')
            pos_gt_path = os.path.join(out_path, sample_tomo  + '_pos_gt.txt')
            neg_gt_path = os.path.join(out_path, sample_tomo + '_neg_gt.txt')
            tp_path = os.path.join(out_path, sample_tomo + '_tp.txt')
            fp_path = os.path.join(out_path, sample_tomo + '_fp.txt')
            tn_path = os.path.join(out_path, sample_tomo + '_tn.txt')
            fn_path = os.path.join(out_path, sample_tomo + '_fn.txt')

            store_in_tab_txt_for_membranogram(pos_path, pos_pred_tomo)
            store_in_tab_txt_for_membranogram(neg_path, neg_pred_tomo)
            store_in_tab_txt_for_membranogram(pos_gt_path, pos_labels_tomo)
            store_in_tab_txt_for_membranogram(neg_gt_path, neg_labels_tomo)
            store_in_tab_txt_for_membranogram(tp_path, tp_tomo)
            store_in_tab_txt_for_membranogram(fp_path, fp_tomo)
            store_in_tab_txt_for_membranogram(tn_path, tn_tomo)
            store_in_tab_txt_for_membranogram(fn_path, fn_tomo)


def change_coords_from_whole_tomo_to_slice(positions, in_bin, out_bin, tomo_token, mb_token, settings, coords=None):
    assert isinstance(settings, ParameterSettings)
    ranges = settings.ranges
    scale = in_bin * 1.0 / out_bin
    # coords_name = 'bin_' + str(out_bin) + '_coordinates_mb' + str(mb_nr)
    # exec('coords = ' + coords_name)
    coords = ranges['bin' + str(out_bin)][tomo_token][mb_token]
    temp_positions = positions * scale
    temp_positions[:, 0] -= coords['x_range'][0]
    temp_positions[:, 1] -= coords['y_range'][0]
    temp_positions[:, 2] -= coords['z_range'][0]
    return temp_positions


def change_coords_from_slice_to_whole_tomo(positions, in_bin, out_bin, tomo_token, mb_token, settings, coords=None):
    assert isinstance(settings, ParameterSettings)
    ranges = settings.ranges
    scale = in_bin * 1.0 / out_bin
    # coords_name = 'bin_' + str(out_bin) + '_coordinates_mb' + str(mb_nr)
    # exec('coords = ' + coords_name)
    coords = ranges['bin' + str(in_bin)][tomo_token][mb_token]
    temp_positions = positions
    temp_positions[:, 0] += coords['x_range'][0]
    temp_positions[:, 1] += coords['y_range'][0]
    temp_positions[:, 2] += coords['z_range'][0]
    temp_positions = temp_positions * scale
    return temp_positions


def convert_from_txt_to_csv(in_file, out_file, delimiter_in='\t', delimiter_out=','):
    with open(out_file, 'w') as csv_out_file:
        with open(in_file) as csv_in_file:
            csv_reader = csv.reader(csv_in_file, delimiter=delimiter_in)
            csv_writer = csv.writer(csv_out_file, delimiter=delimiter_out)
            for row in csv_reader:
                csv_writer.writerow(row)


def load_tomogram(filename, return_header=False, verbose=True):
    """
    Loads data and transposes s.t. we have data in the form x,y,z
    :param filename:
    :return:
    """
    if verbose:
        print("Loading tomogram:", filename)

    with mrcfile.open(filename, permissive=True) as mrc:
        data = np.array(mrc.data)
        data = np.transpose(data, (2,1,0))
        cella = mrc.header.cella
        cellb = mrc.header.cellb
        origin = mrc.header.origin
        pixel_spacing = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        header_dict = {
            'cella': cella,
            'cellb': cellb,
            'origin': origin,
            'pixel_spacing': pixel_spacing
        }
        if return_header:
            return data, header_dict
    return data


def store_tomogram(filename, tomogram, header_dict=None):
    if tomogram.dtype != np.int8:
        tomogram = np.array(tomogram, dtype=np.float32)
    tomogram = np.transpose(tomogram, (2,1,0))
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(tomogram)
        if header_dict is not None:
            mrc.header.cella = header_dict['cella']
            mrc.header.cellb = header_dict['cellb']
            mrc.header.origin = header_dict['origin']


def store_all_positions_as_tomograms(dir_name, stem, pos_array, tomograms, sample_tomos, slice_size=26):
    out_coords = np.zeros(3)
    slice_add = slice_size * 2 + 1
    for i in range(pos_array.shape[0]):
        if i % 10 == 0:
            print ("Storing prediction number ", i, " / ", pos_array.shape[0], " of ", stem)
        if i == 20:
            break
        file_name = os.path.join(dir_name, stem + "_" + str(i) + ".mrc")
        cur_pos = pos_array[i]
        tomogram = tomograms[sample_tomos[i][0]]
        tomo_shape = tomogram.shape
        out_coords[0] = int(np.round(np.min((np.max((0, cur_pos[0] - slice_size)), tomo_shape[0] - 1 - slice_size))))
        out_coords[1] = int(np.round(np.min((np.max((0, cur_pos[1] - slice_size)), tomo_shape[1] - 1 - slice_size))))
        out_coords[2] = int(np.round(np.min((np.max((0, cur_pos[2] - slice_size)), tomo_shape[2] - 1 - slice_size))))
        tomo_slice = tomogram[int(out_coords[0]):(int(out_coords[0])+slice_add),
                              int(out_coords[1]):(int(out_coords[1])+slice_add),
                              int(out_coords[2]):(int(out_coords[2])+slice_add)]
        tomo_slice = np.transpose(tomo_slice, (2, 1, 0))
        store_tomogram(file_name, tomo_slice)


def raise_to_whole_tomogram(prediction, tomo_token, mb_token, settings, in_bin=2, out_bin=4, coords=None):
    """
    Adds the minimal value of the area-of-interest-slice to each coordinate
    :param prediction: numpy array containing the coordinates
    :param in_bin: which binning is used in detection?
    :param out_bin: which binning should be used for training the neural network?
    :param coords: if specified, the used coordinates can be set manually (should be a dict like the standard coord
            dicts)
    :return: new coordinates (numpy array)
    """
    assert isinstance(settings, ParameterSettings)
    ranges = settings.ranges
    temp_res = np.zeros_like(prediction)
    if coords is None:
        # coord_name = 'bin_' + str(in_bin) + '_coordinates_mb' + str(mb_nr)
        # exec ('coords = ' + coord_name)
        coords = ranges['bin' + str(in_bin)][tomo_token][mb_token]
    for i in range(prediction.shape[0]):
        temp_res[i, 0] = prediction[i, 0] + coords['x_range'][0]
        temp_res[i, 1] = prediction[i, 1] + coords['y_range'][0]
        temp_res[i, 2] = prediction[i, 2] + coords['z_range'][0]
    scale = in_bin * 1.0 / out_bin
    res = temp_res * scale
    return res


def store_particle_positions_in_csv(store_path, positions, out_del=','):
    with open(store_path, 'w') as csv_out_file:
        csv_writer = csv.writer(csv_out_file, delimiter=out_del)
        for i in range(positions.shape[0]):
            csv_writer.writerow(positions[i])


def store_in_tab_txt_for_membranogram(store_path, bin4_positions, out_del='\t'):
    print (bin4_positions.shape)
    if bin4_positions.shape[0] == 0:
        return
    print (store_path)
    with open(store_path, 'w') as csv_out_file:
        csv_writer = csv.writer(csv_out_file, delimiter=out_del)
        if bin4_positions.shape[1] == 3:
            csv_writer.writerow(['#PositionX', 'PositionY', 'PositionZ', 'VolumeX', 'VolumeY', 'VolumeZ'])
        elif bin4_positions.shape[1] == 4:
            csv_writer.writerow(['#PositionX', 'PositionY', 'PositionZ', 'VolumeX', 'VolumeY', 'VolumeZ', 'Pred_Score'])
        for i in range(bin4_positions.shape[0]):
            temp_pos = np.array(bin4_positions[i][0:3])
            # temp_pos *= 14.08
            # temp_pos[2] += 3173.74
            new_row = np.concatenate((temp_pos, bin4_positions[i]))
            csv_writer.writerow(new_row)


def change_delimiter_of_file(in_path, out_path, in_del, out_del):
    with open(in_path) as in_file:
        csv_reader = csv.reader(in_file, delimiter=in_del)
        rows = []
        for row in csv_reader:
            rows.append(row)
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter=out_del)
        for row in rows:
            csv_writer.writerow(row)


def visualize_points_on_segmentation(seg_path, seg_out_path, csv_file, box_range=1):
    """
    Loads a segmentation, adds the detected points in it (by a box of range box_range) and stores it
    :param seg_path: path to segmentation file
    :param seg_out_path: path to output file
    :param csv_file: path to position file
    :param box_range: range of box surrounding detected positions
    :return:
    """
    tomo, header_dict = load_tomogram(seg_path, return_header=True)
    tomo = np.array((tomo == 1), dtype=np.int8)
    with open(csv_file) as csv_thing:
        csv_reader = csv.reader(csv_thing)
        for row in csv_reader:
            coords = np.array(row, dtype=np.float)
            coords[0] = int(round(coords[0]))
            coords[1] = int(round(coords[1]))
            coords[2] = int(round(coords[2]))
            coords = np.array(coords[:3], dtype=np.int)
            tomo[coords[0] - box_range: coords[0] + box_range,
                coords[1] - box_range: coords[1] + box_range,
                coords[2] - box_range: coords[2] + box_range] = 2

    tomo = np.transpose(tomo, (2, 1, 0))
    store_tomogram(seg_out_path, tomo, header_dict)



def convert_euler(angles, init, final):
    """
    Converts Euler angles from the initial to the final Euler angle
    convention (mode).

    The following Euler angle conventions can be used: 'zxz_ex_active'
    'zxz_in_active', 'zyz_ex_active', 'zyz_in_active', 'zxz_ex_passive',
    'zxz_in_passive', 'zyz_ex_passive' and 'zyz_in_passive', where 'ex'
    and 'in' denote extrinsic and intrinsic transformations. These are
    defined in the standard way, namely:

    Active transformations rotate points (vectors) in space, while
    passive rotate the coordinate system and not points.

    A ZXZ transformation is composed of a rotation around z-axis,
    rotation around x-axis and another around z-axis. In a ZYZ
    transformation the second transformation is around y-axis.

    In case of extrinsic transformation, the axes corresponding to
    the three rotations are fixed. On the contrary, the (second and
    the third) rotations are carried around the rotated axes.

    The three Euler angles are given as (phi, theta, psi). in the
    order the rotations are applied.

    Note that the resulting angles are not necessarily in the proper
    range (phi and psi 0 - 2 pi and theta 0 - pi). However they can
    be used to make correct rotation matrices (using make_r_euler()).

    Arguments:
      - angles: (phi, theta, psi), where rotation by phi is applied first
      and by psi last
      - init: Euler angles convention (mode) in which arg. angles are
      specified
      - final: Euler angles convention (mode) to which arg. angles
      should be converted

    Returns Euler angles (phi, theta, psi) in the final Euler angle
    convention (mode).
    """

    # parse angles
    phi, theta, psi = np.asarray(angles, dtype=float)

    # check modes to avoid infinite recursions
    modes = [
        'x',
        'zxz_ex_active', 'zxz_in_active', 'zyz_ex_active', 'zyz_in_active',
        'zxz_ex_passive', 'zxz_in_passive', 'zyz_ex_passive',
        'zyz_in_passive']
    if init not in modes:
        raise ValueError(
            "Argument init: " + init + " was not understood. Defined "
                                       "values are given in the following list: " + str(modes))
    if final not in modes:
        raise ValueError(
            "Argument final: " + final + " was not understood. Defined "
                                         "values are given in the following list: " + str(modes))

        # initial mode zxz_ex_active
    if (init == 'x') or (init == 'zxz_ex_active'):

        # final mode active
        if (final == 'x') or (final == 'zxz_ex_active'):
            result = (phi, theta, psi)
        elif (final == 'zxz_in_active'):
            result = (psi, theta, phi)
        elif final == 'zyz_ex_active':
            result = (phi + np.pi / 2, theta, psi - np.pi / 2)
        elif final == 'zyz_in_active':
            result = (psi + np.pi / 2, theta, phi - np.pi / 2)



    # final mode zxz_ex_active
    elif (final == 'x') or (final == 'zxz_ex_active'):

        # initial mode active
        if (init == 'x') or (init == 'zxz_ex_active'):
            result = (phi, theta, psi)
        elif (init == 'zxz_in_active'):
            result = (psi, theta, phi)
        elif init == 'zyz_ex_active':
            result = (phi - np.pi / 2, theta, psi + np.pi / 2)
        elif init == 'zyz_in_active':
            result = (psi - np.pi / 2, theta, phi + np.pi / 2)



    # all other cases convert via 'zxz_ex_active'
    else:
        intermediate = convert_euler(
            angles=angles, init=init, final='zxz_ex_active')
        result = convert_euler(
            angles=intermediate, init='zxz_ex_active', final=final)
    return result


def get_stack_from_file_name(file_name, settings):
    assert isinstance(settings, ParameterSettings)
    tomo_tokens = settings.tomo_tokens
    stack_tokens = settings.stack_tokens
    out_stack = None

    for tomo_token in tomo_tokens:
        if not any(tomo_token + token in file_name for token in ['_', '/']) or any([tomo_token + token + str(a) in file_name for a in range(10) for token in ['_', '/']]):
            continue
        for stack_token in stack_tokens[tomo_token]:
            if (stack_token.startswith('S') and (
                    stack_token + 'M' in file_name or stack_token + '.' in file_name or stack_token + '_' in file_name or stack_token + '/' in file_name)):  # or tomo_token + '_' + mb_token in file_name:
                out_stack = stack_token
                break
    return out_stack


def get_tomo_and_mb_from_file_name(file_name, settings):
    assert isinstance(settings, ParameterSettings)
    tomo_tokens = settings.tomo_tokens
    mb_tokens = settings.mb_tokens
    out_tomo = None
    out_mb = None
    for tomo_token in tomo_tokens:
        if not any(tomo_token + token in file_name for token in ['_', '/']) or any([tomo_token + token + str(a) in file_name for a in range(10) for token in ['_', '/']]):
            continue
        for mb_token in mb_tokens[tomo_token]:
            if 'M' + mb_token + "_" in file_name or 'M' + mb_token + "." in file_name or  'm' + mb_token + '_' in file_name or \
                    (mb_token.startswith('M') and (mb_token + '_' in file_name or mb_token + '.' in file_name)):# or tomo_token + '_' + mb_token in file_name:
                out_tomo = tomo_token
                out_mb = mb_token
                break
    return out_tomo, out_mb


def give_files_with_ending(path, ending):
    all_files = os.listdir(path)
    out_files = []
    for file in all_files:
        if file.endswith(ending):
            out_files.append(os.path.join(path, file))
    return out_files
