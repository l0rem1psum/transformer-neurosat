import os
import sys
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from model import NeuroSAT, SimpleAttentionSat
from utils import CnfDataModule, BatchAwareProgressBar, KSatDataModule

sys.path.append("utils")

run_dir = os.path.join('run', str(int(time.time())))
os.makedirs(run_dir)

model = NeuroSAT(128, 3, 3, 8)
# model = SimpleAttentionSat(128)

logger = TensorBoardLogger(
    save_dir=run_dir,
    log_graph=True,
    name="lightning_logs"
)

progress_bar = BatchAwareProgressBar()

trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=1000,
    logger=logger,
    log_every_n_steps=1,
    default_root_dir=run_dir,
    val_check_interval=1.0,
    num_sanity_val_steps=0,
    # gpus=1,
    callbacks=[progress_bar],
    # overfit_batches=100,
    track_grad_norm=2,
)

datamodule = CnfDataModule("data", n_pairs=2000, one=True, max_nodes_per_batch=10000, min_n=10, max_n=15)
# datamodule = KSatDataModule()
# datamodule.prepare_data()
# datamodule.setup()
trainer.fit(model, datamodule=datamodule)

# TODO: Save run config

result = trainer.test()
print(result)
