from pytorch_lightning.callbacks import ProgressBar


class BatchAwareProgressBar(ProgressBar):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.main_progress_bar.update(batch[1].shape[0])
