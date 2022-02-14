from scripts.MemBrain_model import MemBrain_model
from pytorch_lightning import Trainer
import numpy as np
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import star_utils, data_utils
from utils.parameters import ParameterSettings
from config import *
from utils.data_utils import store_heatmaps_for_dataloader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping





class MemBrainer():
    def __init__(self, box_range, dm, project_dir, star_file, part_dists, ckpt=None, max_epochs=100):
        self.box_range = box_range
        self.project_dir = project_dir
        self.star_file = star_file
        self.settings = ParameterSettings(self.star_file)
        self.max_epochs = max_epochs
        if ckpt is not None:
            self.model = MemBrain_model.load_from_checkpoint(ckpt, box_range=self.box_range, part_dists=part_dists,
                                                             lr=LEARNING_RATE, settings=self.settings)
        else:
            self.model = MemBrain_model(box_range=self.box_range, part_dists=part_dists, lr=LEARNING_RATE, settings=self.settings)
        self.ckpt_dir = os.path.join(self.project_dir, '../lightning_logs')
        self.dm = dm



    def train(self):
        checkpoint_callback = ModelCheckpoint(monitor='hp/Val_Loss', save_last=True,
                                              filename="loss_track_{epoch}-{Val_Loss:.2f}-{Val_F1:.2f}")
        checkpoint_callback2 = ModelCheckpoint(monitor='Val_F1', save_last=True,
                                               filename="f1_track_{epoch}-{Val_Loss:.2f}-{Val_F1:.2f}", mode='max')
        # checkpoint_callback = ModelCheckpoint(monitor='hp/Val_Loss', save_last=True, filename="{epoch}-{Val_Loss:.2f}-{Val_F1:.2f}")
        early_stop_callback = EarlyStopping(monitor='hp/Val_Loss', min_delta=0.0, patience=80, mode='min')
        tb_logger = pl_loggers.TensorBoardLogger(self.ckpt_dir, default_hp_metric=False, name="lightning_logs")
        trainer = Trainer(max_epochs=self.max_epochs, callbacks=[checkpoint_callback, checkpoint_callback2, early_stop_callback], default_root_dir=self.ckpt_dir,
                          logger=tb_logger, gpus=(1 if USE_GPU else 0))
        trainer.fit(self.model, self.dm)

    def predict(self, out_dir, star_file=None):
        test_dl = self.dm.test_dataloader()
        out_star = store_heatmaps_for_dataloader(test_dl, self.model, out_dir, star_file, self.settings.consider_bin)
        return out_star



