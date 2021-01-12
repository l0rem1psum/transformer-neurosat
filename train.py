import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ProgressBar

from model import NeuroSAT
from utils import CnfDataModule

sys.path.append("utils")

run_dir = os.path.join('run', str(int(time.time())))
os.makedirs(run_dir)

model = NeuroSAT(128, 3, 3, 16)

class LitProgressBar(ProgressBar):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.main_progress_bar.update(batch[1].shape[0])

logger = TensorBoardLogger(
    save_dir=run_dir,
    version=1,
    log_graph=True,
    name="lightning_logs"
)
trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=50,
    logger=logger,
    log_every_n_steps=1,
    default_root_dir=run_dir,
    gpus=1,
    callbacks=[LitProgressBar()]
)

datamodule = CnfDataModule("data", n_pairs=1000, one=True, max_nodes_per_batch=2000)
trainer.fit(model, datamodule=datamodule)
