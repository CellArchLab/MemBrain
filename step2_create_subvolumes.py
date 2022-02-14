import os
from scripts.rotator import Rotator
from scripts.data_loading import add_datasplit_to_star
from scripts.add_labels_and_distances import add_labels_and_distances
from utils.parameters import ParameterSettings
from config import *



def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    out_star_name = os.path.join(os.path.join(project_directory, 'positions', 'normals_corrected_with_euler'),
                                 PROJECT_NAME + '_with_inner_outer.star')
    settings = ParameterSettings(out_star_name)#
    rotator = Rotator(os.path.join(project_directory, 'rotated_volumes'), out_bins=[4], pred_bin=4,
                                       box_range=BOX_RANGE, settings=settings, n_pr=N_PR_ROTATION,
                                       store_dir=os.path.join(project_directory, 'rotated_volumes', 'raw'),
                                       store_normals=True, store_angles=True, preprocess_tomograms=True, lp_cutoff=LP_CUTOFF)
    out_star_name = rotator.rotate_all_volumes()
    out_star_name = os.path.join(os.path.join(project_directory, 'rotated_volumes'), os.path.basename(out_star_name))

    add_labels_and_distances(out_star_name, project_directory, prot_tokens=PROT_TOKENS, prot_shapes=PROT_SHAPES,
                             particle_orientations=True, membranorama_xmls=True)
    add_datasplit_to_star(out_star_name, val_tokens=VAL_TOKENS, train_tokens=TRAIN_TOKENS, test_tokens=TEST_TOKENS)


if __name__ == '__main__':
    main()