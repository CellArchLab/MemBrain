import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
import os
from config import *
from utils.data_utils import store_heatmaps_for_dataloader
from scripts.clustering import MeanShift_clustering


class MemBrain_model(LightningModule):
    def __init__(self, box_range, part_dists, lr, settings=None):
        super().__init__()
        self.box_range = box_range
        self.conv1 = nn.Conv3d(1, 32, (3, 3, 3), stride=1, padding=0, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros')
        self.conv1b = nn.Conv3d(32, 32, (3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True,
                                      padding_mode='zeros')
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), stride=1, padding=0, dilation=1, groups=1, bias=True,
                                     padding_mode='zeros')
        self.conv2b = nn.Conv3d(32, 32, (3, 3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True,
                                      padding_mode='zeros')
        self.mlp_fac = int((self.box_range * 2 - 4) ** 3 / 2)
        self.batchnorm1 = torch.nn.BatchNorm3d(32)
        self.batchnorm2 = torch.nn.BatchNorm3d(32)
        self.batchnorm3 = torch.nn.BatchNorm3d(32)
        self.batchnorm4 = torch.nn.BatchNorm3d(32)
        self.mlp1 = torch.nn.Linear(self.mlp_fac * 64, len(part_dists))
        self.lr = lr
        self.loss_fn = (F.mse_loss if LOSS_FN == 'MSE' else F.huber_loss if LOSS_FN == 'Huber' else F.l1_loss)
        self.settings = settings
        self.save_hyperparameters("part_dists", "lr")
        self.save_hyperparameters({'max_epochs': MAX_EPOCHS,
                                   'weight_decay': WEIGHT_DECAY,
                                   'loss_fn': LOSS_FN,
                                   'max_particle_distance': MAX_PARTICLE_DISTANCE,
                                   'batch_size': BATCH_SIZE,
                                   'cluster_bandwidth': CLUSTER_BANDWIDTHS[0],
                                   'recluster_thres': RECLUSTER_THRES[0],
                                   'recluster_bw': RECLUSTER_BANDWIDTHS[0],
                                   'normalized_volumes': USE_ROTATION_NORMALIZATION})
        self.best_f1 = 0.

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = x.view(-1, self.mlp_fac * 64)
        x = self.mlp1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("Train_Loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("hp/Val_Loss", loss, on_step=False, on_epoch=True)
        self.log("Val_Loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """
        If clustering stats during training is activated, this will be performed here (for activating, check config file)
        """

        if LOG_CLUSTERING_STATS and self.current_epoch > 0 and \
                self.current_epoch % LOG_CLUSTERING_EVERY_NTH_EPOCH == 0:
            val_data_loader = self.val_dataloader()

            out_star = store_heatmaps_for_dataloader(val_data_loader, self, out_dir=os.path.join(PROJECT_DIRECTORY, PROJECT_NAME, 'temp_files', 'heatmaps'),
                                          star_file=self.settings.star_file, consider_bin=self.settings.consider_bin)
            ms = MeanShift_clustering(out_star, os.path.join(PROJECT_DIRECTORY, PROJECT_NAME, 'temp_files', 'cluster_centers'))
            try:
                cluster_star = ms.start_clustering(CLUSTER_BANDWIDTHS[0], recluster_thres=RECLUSTER_THRES[0], recluster_bw=RECLUSTER_BANDWIDTHS[0])
                ms.evaluate_clustering(cluster_star, PROT_TOKENS, bandwidth=CLUSTER_BANDWIDTHS[0], store_mb_wise=True, distance_thres=15)
                gt_hits = ms.all_metrics['confusion_matrix'][0][0]
                gt_misses = ms.all_metrics['confusion_matrix'][0][1]
                pred_hits = ms.all_metrics['confusion_matrix'][0][2]
                pred_misses = ms.all_metrics['confusion_matrix'][0][3]
                prec = pred_hits / (pred_hits + pred_misses)
                rec = gt_hits / (gt_hits + gt_misses)
                f1 = 2 * prec * rec / (prec + rec)
                if f1 > self.best_f1:
                    self.log("hp/Val_precision", prec, on_epoch=True)
                    self.log("hp/Val_recall", rec, on_epoch=True)
                    self.log("hp/Val_F1", f1, on_epoch=True)
                    self.log("Val_F1", f1, on_epoch=True)
            except:
                self.log("hp/Val_precision", -0.1, on_epoch=True)
                self.log("hp/Val_recall", -0.1, on_epoch=True)
                self.log("hp/Val_F1", -0.1, on_epoch=True)
                self.log("Val_F1", -0.1, on_epoch=True)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'hp/Val_F1': -0.1, 'hp/Val_precision': -0.1, 'hp/Val_recall': -0.1, 'hp/Val_Loss': 15.})
        self.log("Val_F1", -0.1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)