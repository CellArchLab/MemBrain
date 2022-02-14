from config import *
from scripts.clustering import MeanShift_clustering
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--eval", type=bool, default=False, help="Should the cluster evaluation metrics be computed? Only"
                                                             "possible if there is ground truth available.")
args = parser.parse_args()


def main():
    project_directory = os.path.join(PROJECT_DIRECTORY, PROJECT_NAME)
    heatmap_star = os.path.join(project_directory, 'heatmaps', PROJECT_NAME + '_with_inner_outer.star')
    ms = MeanShift_clustering(heatmap_star, os.path.join(project_directory, 'cluster_centers', 'plain'))
    for i, bandwidth in enumerate(CLUSTER_BANDWIDTHS):
        if RECLUSTER_FLAG:
            cluster_star = ms.start_clustering(bandwidth=bandwidth, recluster_thres=RECLUSTER_THRES[i],
                                               recluster_bw=RECLUSTER_BANDWIDTHS[i])
        else:
            cluster_star = ms.start_clustering(bandwidth=bandwidth)
        if args.eval:
            ms.evaluate_clustering(cluster_star, PROT_TOKENS, bandwidth=bandwidth, store_mb_wise=True)
    if args.eval:
        ms.store_metrics()


if __name__ == '__main__':
    main()