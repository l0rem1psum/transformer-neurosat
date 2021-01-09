import os
import sys
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model import LightningNeuroSAT
from utils.data_module import SATDataModule

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
    max_epochs=50,
    logger=logger,
    log_every_n_steps=1
)

dm_opts = Namespace(
    run_dir="run"
)

# ds = SatProblemDataSet('data/1610177328/pickle/train')
# dl = torch.utils.data.DataLoader(ds, num_workers=4)

data_module = SATDataModule(
        100,
        5,
        10,
        0.3,
        0.4,
        "data",
        60000,
        0
    )

trainer.fit(model, datamodule=data_module)
