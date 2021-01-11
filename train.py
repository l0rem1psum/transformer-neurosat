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
    "data/23a9bcc9/d5ddbb15",
    50000,
    10,
    40,
    0.3,
    0.4,
    20000,
    0
)
print(data_module.get_uuid())

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
    default_root_dir=run_dir,
    gpus=1
)

trainer.fit(model, datamodule=data_module)
