import os
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint. Can be used to continue training")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
parser.add_argument("--loss_fn", type=str, default=None, help="Which loss function to use? Huber / MSE / L1")
parser.add_argument("--run_token", type=str, default=None, help="Optional token for multiple parallel runs")
args = parser.parse_args()
if args.lr is not None:
    config.LEARNING_RATE = args.lr
if args.weight_decay is not None:
    config.WEIGHT_DECAY = args.weight_decay
if args.loss_fn is not None:
    assert args.loss_fn in ['MSE', 'Huber', 'L1']
    config.LOSS_FN = args.loss_fn
if args.run_token is not None:
    config.RUN_TOKEN = args.run_token

from scripts.data_loading import MemBrain_datamodule
from scripts.trainer import MemBrainer
from config import *
import random
import numpy as np



def main():
    #TODO: add choice for max distance during training ( not up to 30 or so)
    random.seed(999)
    np.random.seed(999)
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    out_star_name = os.path.join(os.path.join(project_directory, 'rotated_volumes'),
                                 PROJECT_NAME + '_with_inner_outer.star')
    dm = MemBrain_datamodule(out_star_name, BATCH_SIZE, part_dists=TRAINING_PARTICLE_DISTS, max_dist=MAX_PARTICLE_DISTANCE)
    trainer = MemBrainer(box_range=BOX_RANGE, dm=dm, project_dir=project_directory, star_file=out_star_name, part_dists=TRAINING_PARTICLE_DISTS,
                         ckpt=args.ckpt, max_epochs=MAX_EPOCHS)
    trainer.train()


if __name__ == '__main__':
    main()