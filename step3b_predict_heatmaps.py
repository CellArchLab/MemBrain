import os
import argparse
from scripts.data_loading import MemBrain_datamodule
from scripts.trainer import MemBrainer
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("ckpt", type=str, help="Path to trained model checkpoint.")
args = parser.parse_args()

def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    out_star_name = os.path.join(os.path.join(project_directory, 'rotated_volumes'),
                                 PROJECT_NAME + '_with_inner_outer.star')
    heatmap_out_dir = os.path.join(project_directory, 'heatmaps')

    dm = MemBrain_datamodule(out_star_name, BATCH_SIZE, part_dists=TRAINING_PARTICLE_DISTS, max_dist=MAX_PARTICLE_DISTANCE)
    trainer = MemBrainer(box_range=BOX_RANGE, dm=dm, project_dir=project_directory, star_file=out_star_name,
                         ckpt=args.ckpt, part_dists=TRAINING_PARTICLE_DISTS)
    trainer.predict(heatmap_out_dir, star_file=out_star_name)


if __name__ == '__main__':
    main()