import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model import NeuroSAT
from utils.data_module import SATDataModule

sys.path.append("utils")

run_dir = os.path.join('run', str(int(time.time())))
os.makedirs(run_dir)

data_module = SATDataModule(
    "data",
    100,
    5,
    10,
    0.3,
    0.4,
    60000,
    0
)

model = NeuroSAT(128, 3, 3, 16)

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
    default_root_dir=run_dir
)

trainer.fit(model, datamodule=data_module)
