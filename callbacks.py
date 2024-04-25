import wandb
import pytorch_lightning as pl
from .models.LitOCRXL import LitOCRXL


class ExampleLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.num_samples = num_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module: LitOCRXL):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        output = pl_module(val_imgs)
        val_labels = pl_module.tokenizer.batch_decode(val_labels.cpu().numpy())
        preds = pl_module.tokenizer.batch_decode(output.cpu().numpy())
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(val_imgs, preds, val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )
