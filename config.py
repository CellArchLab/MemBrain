## General project settings

PROJECT_NAME = 'sample_pipeline'
PROJECT_DIRECTORY = '/fs/pool/pool-engel/Lorenz/MemBrain_testing/pipeline' # within this folder, the pipeline folder structure will be created
TOMO_DIR = '/fs/pool/pool-engel/Lorenz/MemBrain_testing/code/MemBrain/toy_data/tomograms' # path of directory containing the data (Tomo1, Tomo2, ...)
PIXEL_SPACING_BIN1 = 13.68
UNBINNED_OFFSET_Z = 3174.  # This should normally be 0. For toy data, it is 3174.
TOMO_BINNING = 4
USE_GPU = True
TEST_ALL = False  # Should all membranes be assigned to the test set?
GT_AVAIL = True  # Is ground truth data available?

## Protein details

MEMBRANORAMA_XMLS = True # This has no usage in current version (tuning)
USE_GT_ORIENTATIONS = True
# Protein tokens:
# For each protein you want to detect, specify a dictionary entry. The list of names corresponds to possible namings in the membranorama
# file. E.g. 'PSII': ['PSII', 'PS2'] means that for protein 'PSII', both 'PSII' and 'PS2' tokens are accepted.
PROT_TOKENS = {'PSII': ['PSII', 'PS2'],
                   'PSI': ['PSI_', 'PS1'],
                   'b6f': ['b6f', 'bf6'],
                   'UK': ['UK', 'unknown']}

# If you want to use the particle shapes for training, please specify the paths to the structures used to map into
# the membranorama views for generating the ground truth.
# In case you have clikced a particle without specific shape, you can also use the string "sphereX" where X corresponds to
# the desired sphere radius.
PSII_PARTICLE = '/fs/pool/pool-engel/Lorenz/4Lorenz/structures/Chlamy_C2_14A.mrc'
B6F_PARTICLE = '/fs/pool/pool-engel/Lorenz/4Lorenz/structures/Cyt b6f_14A_center.mrc'
UK_PARTICLE = 'sphere12'  # "sphereX" tells the program to use a sphere with radius X voxels as a particle model
PROT_SHAPES = {'PSII': PSII_PARTICLE, 'b6f': B6F_PARTICLE, 'UK': UK_PARTICLE} # keys should correspond to keys of PROT_TOKENS


## Efficiency details
N_PR_NORMALVOTING = 4 # number of processes used for normal voting
N_PR_ROTATION = 1 # number of processes used for rotating subvolumes; only recommended for small subvolumes and many sampled points --> mostly 1 is enough!

## Preprocessing details
USE_ROTATION_NORMALIZATION = True  # experimental & might not work if set to False!
ROTATION_AUGMENTATION_DURING_TRAINING = False # This enables the rotational augmentation around random angles for all axes
                                            # If deactivated, still rotational augmentation around z-axis is performed
BOX_RANGE = 6 # size of subvolumes sampled (cube of length BOX_RANGE*2)
LP_CUTOFF = None    # cutoff value for low-pass filtering of tomogram before extracting subvolumes
                    # (can increase generalizability, but takes some time)
                    # If LP_CUTOFF = None, no low-pass filtering is performed. Otherwise should be in the range 0.0 - 0.25


## Training settings

# Specify the tokens used for training, validation and test sets.
# Should be a dictionary with tomogram tokens corresponding to keys. For each key, specify a list, where each entry of the
# list specifies a certain membrane via (stack token, membrane token)
# If tokens are set to NONE, splits are automatically generated using the splits (70, 15, 15)
# CAUTION: This may lead to different training results and biases, as at least the test set should be fixed.
TRAIN_TOKENS = {'Tomo1': [('S1', 'M1')]}
VAL_TOKENS = {'Tomo1': [('S2', 'M15')]}
TEST_TOKENS = {'Tomo1': [('S4', 'M8')]}
if TEST_ALL:
    TRAIN_TOKENS = None
    VAL_TOKENS = None
    TEST_TOKENS = None

# Specify which distances should be used for training. Should be a list with entries:
#       - either a protein token --> distances to only this protein class are computed
#       - a list of protein tokens --> minimal distances to any of the protein tokens in the list are computed
# For each entry in the list, the network generates a separate output, so you can have multiple heatmaps for multiple
# particle classes.
TRAINING_PARTICLE_DISTS = [['PSII', 'UK'], 'b6f']
TRAINING_PARTICLE_DISTS = [['PSII', 'UK']]

LOG_CLUSTERING_STATS = True # Flag whether or not to log clustering statistics during training
                            # Currently only works for single output heatmaps (i.e. 1 predicted protein class)
                            # If more than one cluster bandwidths are specified, only the first will be used
LOG_CLUSTERING_EVERY_NTH_EPOCH = 10  # Every nth epoch, clustering will be performed to validate training (only applies if
                                    # LOG_CLUSTERING_STATS is True

BATCH_SIZE = 512
MAX_EPOCHS = 10
MAX_PARTICLE_DISTANCE = 7. # all distances above this value will be capped
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-3
LOSS_FN = 'MSE' # 'Huber', 'L1'   ## Choose between MSE, Huber or L1 loss
RUN_TOKEN = None


## Clustering settings
CLUSTER_BANDWIDTHS = [18, 23, 28] # Multiple values can be provided, if F1 scores for different bandwidth should be
                                    # calculated and compared
RECLUSTER_FLAG = True  # Should the clusters be re-clustered if they are too large?
RECLUSTER_THRES = [78, 78, 78]  # If a cluster has a diameters longer than these values, they will be reclustered with
                                # a smaller bandwidth. Needs to be of the same shape es CLUSTER_BANDWIDTHS
RECLUSTER_BANDWIDTHS = [13, 18, 24]  # Bandwidths for reclustering. Needs to be of the same shape es CLUSTER_BANDWIDTHS

## Evaluation settings
PROT_TOKENS_PRED = ['PSII', 'UK']  # Tokens of proteins that can be considered GT for predicted positions (if pred. position close to one of them, it is a hit)
PROT_TOKENS_GT = ['PSII']  # Tokens for proteins that should be hit. If a predicted position is close to them, this GT position is hit


## Expert settings
MAX_DIST_FROM_MEMBRANE = 15     # maximum distance for points sampled from membrane segmentation. Higher value increases
                                # robustness of normals, but increases computing efforts.
SHRINK_THRES = 118 #118

