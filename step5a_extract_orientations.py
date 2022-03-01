from config import *
from scripts import particle_orientation
from utils.parameters import ParameterSettings
import os



def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    pos_star_file = os.path.join(os.path.join(project_directory, 'cluster_centers', 'plain'),
                                 PROJECT_NAME + '_with_inner_outer_bw' + str(float(CLUSTER_BANDWIDTHS[0])) + '.star')
    cluster_dir = os.path.join(os.path.join(project_directory, 'cluster_centers'))
    out_dir = os.path.join(cluster_dir, 'with_orientation')
    settings = ParameterSettings(pos_star_file, is_cluster_center_file=True)
    out_star_file = particle_orientation.compute_all_orientations(cluster_dir, settings, out_dir, visualize=False)
    particle_orientation.quantify_orientations(out_star_file)


if __name__ == '__main__':
    main()