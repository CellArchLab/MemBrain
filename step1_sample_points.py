import os
from scripts import create_initial_stars
from utils import inspect_segmentations
from scripts.sample_points_on_seg import sample_points_on_seg
from scripts.normal_voting import normal_voting_for_star
from scripts.rotator import compute_all_Euler_angles_for_star
from utils.pipeline_structure import pipeline_structure
from config import *


def main():
    #TODO(!!): Fix segmentation fusion for stacks with multiple names (will be overwritten otherwise)
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    pipeline_structure(project_directory, TOMO_DIR)

    out_star = os.path.join(project_directory, 'initial_stars', PROJECT_NAME + '.star')
    out_star2 = os.path.join(project_directory,'initial_stars', PROJECT_NAME + '_with_inner_outer.star')
    gt_out_dir = os.path.join(project_directory, 'gt_coords')

    create_initial_stars.create_initial_stars(TOMO_DIR, out_star, gt_out_dir, binning=TOMO_BINNING, with_dimi=False,
                                              with_denoised=True, pixel_spacing_bin1=PIXEL_SPACING_BIN1,
                                              unbinned_offset_Z=UNBINNED_OFFSET_Z)
    inspect_segmentations.fuse_segmentations_together(out_star, os.path.join(project_directory, 'temp_files/'))
    inspect_segmentations.inspect_segmentation_before(out_star, out_star2, os.path.join(project_directory, 'temp_files/'))
    normals_star = sample_points_on_seg(project_directory, out_star2, os.path.join(project_directory, 'positions', 'sampled'), max_mb_dist=MAX_DIST_FROM_MEMBRANE)
    normals_corrected_star = normal_voting_for_star(normals_star, os.path.join(project_directory, 'positions', 'normals_corrected'), npr=N_PR_NORMALVOTING)
    compute_all_Euler_angles_for_star(normals_corrected_star, os.path.join(project_directory, 'positions', 'normals_corrected_with_euler'))


if __name__ == '__main__':
    main()