from utils import data_utils
from scripts.add_labels_and_distances import read_GT_data_membranorama_xml
from utils.parameters import ParameterSettings
import os


def xml_to_vtp_for_file(xml_file, out_dir, settings):
    pos_dict = read_GT_data_membranorama_xml(xml_file, settings=settings)
    for key in pos_dict.keys():
        pos_dict[key] = pos_dict[key] * 4
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.vtp')
    data_utils.store_gt_arrays_in_vtp(pos_dict, out_path)


def xml_to_vtp_for_star(in_star):
    settings = ParameterSettings(in_star)
    for tomo_key in settings.gt_paths.keys():
        out_dir = os.path.join(os.path.dirname(settings.gt_paths[tomo_key]), 'as_vtp')
        for gt_file in os.listdir(settings.gt_paths[tomo_key]):
            xml_file = os.path.join(settings.gt_paths[tomo_key], gt_file)
            if not os.path.isfile(xml_file):
                continue
            xml_to_vtp_for_file(xml_file, out_dir, settings)


initial_star = '/fs/pool/pool-engel/Lorenz/pipeline_new_spinach/heatmaps_2nd_round/new_spinach.star'
xml_to_vtp_for_star(initial_star)