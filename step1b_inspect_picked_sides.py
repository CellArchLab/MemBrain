from config import *
import os
from utils import inspect_segmentations


def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    mics_dir = os.path.join(project_directory, 'mics')
    segs_dir = os.path.join(project_directory, 'segs')
    temp_folder = os.path.join(project_directory, 'temp_files')
    inspect_segmentations.inspect_segmentation_after(mics_dir, segs_dir, temp_folder)


if __name__ == '__main__':
    main()
