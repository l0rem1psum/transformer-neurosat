import os
import sys
from argparse import Namespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from model import LightningNeuroSAT
from utils import SatProblemDataSet

sys.path.append("utils")

model = LightningNeuroSAT(128, 3, 3, 16)

logger = TensorBoardLogger(
    save_dir=os.getcwd(),
    version=1,
    log_graph=True,
    name="lightning_logs"
)
trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=20,
    logger=logger
)

dm_opts = Namespace(
    run_dir="run"
)

ds = SatProblemDataSet("data/pickle/train/sr5-10")
dl = torch.utils.data.DataLoader(ds, num_workers=4)

trainer.fit(model, train_dataloader=dl)
