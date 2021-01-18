import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model import NeuroSAT
from utils import CnfDataModule, BatchAwareProgressBar

sys.path.append("utils")

run_dir = os.path.join('run', str(int(time.time())))
os.makedirs(run_dir)

model = NeuroSAT(128, 3, 3, 16)

logger = TensorBoardLogger(
    save_dir=run_dir,
    version=1,
    log_graph=True,
    name="lightning_logs"
)

progress_bar = BatchAwareProgressBar()

trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=3,
    logger=logger,
    log_every_n_steps=1,
    default_root_dir=run_dir,
    val_check_interval=1.0,
    # gpus=1,
    callbacks=[progress_bar]
)

datamodule = CnfDataModule("data", n_pairs=100, one=True, max_nodes_per_batch=2000)
trainer.fit(model, datamodule=datamodule)
