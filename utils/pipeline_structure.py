import os


def check_mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def pipeline_structure(main_directory, tomo_dir=None):
    dirs = []
    dirs.append('cluster_centers')
    dirs.append('cluster_centers/plain')
    dirs.append('cluster_centers/with_orientation')
    dirs.append('cluster_centers/classified')
    dirs.append('gt_coords') # add folder for each tomo
    dirs.append('heatmaps')
    dirs.append('initial_stars')
    dirs.append('mics')
    dirs.append('models')
    dirs.append('positions')
    dirs.append('positions/sampled')
    dirs.append('positions/normals_corrected')
    dirs.append('positions/normals_corrected_with_euler')
    dirs.append('rotated_volumes')
    dirs.append('rotated_volumes/raw')
    dirs.append('rotated_volumes/denoised')
    dirs.append('segs')
    dirs.append('subtomogram_averaging')
    dirs.append('temp_files')
    if tomo_dir is not None:
        for tomo_folder in os.listdir(tomo_dir):
            if os.path.isdir(os.path.join(tomo_dir, tomo_folder)):
                dirs.append(os.path.join('gt_coords', tomo_folder, 'as_xml'))
                dirs.append(os.path.join('gt_coords', tomo_folder, 'as_csv'))
                dirs.append(os.path.join('gt_coords', tomo_folder, 'as_vtp'))
    check_mkdir(main_directory)
    for entry in dirs:
        check_mkdir(os.path.join(main_directory, entry))