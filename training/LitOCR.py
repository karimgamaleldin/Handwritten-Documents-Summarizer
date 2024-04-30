import torch
import wandb
import pytorch_lightning as pl
from torch import nn, optim
from ..models.vision_encoder import VisionEncoder
from torchmetrics.text import CharErrorRate
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitOCR(pl.LightningModule):
    """
    MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        decoder,
        tokenizer,
        *,
        lr=5e-4,
        context_length=15,
        max_len=100,
        schedule_sample=True,
        in_dims=(1, 128, 512),
        **kwargs,
    ):
        super(LitOCR, self).__init__(**kwargs)
        self.lr = lr
        self.context_length = context_length
        self.vision_encoder = vision_encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.metric = CharErrorRate()
        self.max_len = (
            max_len  # In case of infinite loop so initalize to have the max value
        )
        self.automatic_optimization = False
        self.schedule_sample = schedule_sample
        self.schedule_sample_prob = 0.95
        self.in_dims = in_dims
        self.save_hyperparameters(ignore=["metric"])

    def forward(self, imgs):
        """
        Forward pass for the OCR
        """
        enc_output = self.vision_encoder(imgs)
        output = torch.zeros((imgs.shape[0], 1), dtype=torch.long).to(imgs.device)
        output[:, 0] = self.tokenizer.token_to_id("[SOS]")
        print(output.shape)
        while (
            output[:, -1][0] != self.tokenizer.token_to_id("[EOS]")
            and output.shape[1] < self.max_len
        ):
            if output.shape[1] >= self.context_length:
                input = output[:, -self.context_length :]
            else:
                input = torch.zeros(
                    (output.shape[0], self.context_length), dtype=torch.long
                ).to(imgs.device)
                input[:, : output.shape[1]] = output
            dec_output = self.decoder(input, enc_output)
            dec_output = dec_output.argmax(dim=-1).unsqueeze(-1)
            output = torch.cat([output, dec_output], dim=-1)
        return output

    def training_step(self, batch, batch_idx):
        """
        Training step for the OCR Model
        """
        imgs, transcriptions = batch
        outputs, loss_arr = self.teacher(imgs, transcriptions, train=True)
        loss_mean = torch.stack(loss_arr).mean()
        return loss_mean

    def on_train_epoch_end(self, outputs=None):
        sch = self.lr_schedulers()
        val_loss = self.trainer.callback_metrics["val_loss"]
        sch.step(val_loss)
        print(f"Learning rate stepped with val_loss: {val_loss}")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the OCR Model
        """
        imgs, transcriptions = batch
        outputs, loss_arr = self.teacher(imgs, transcriptions)
        loss_mean = torch.stack(loss_arr).mean()
        return loss_mean

    # def on_validation_epoch_end(self, validation_outputs=None):
    #   dummy_input = torch.zeros((1, *self.in_dims), dtype=torch.float32).to(self.device)
    #   model_filename = f'model_{self.current_epoch}.onnx'
    #   torch.onnx.export(self, dummy_input, model_filename)
    #   wandb.save(model_filename)

    def configure_optimizers(self):
        """
        Configure the optimizer for the OCR Model
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def teacher(self, imgs, transcriptions, train=False):
        """
        Teacher forcing for the OCR model
        """
        transcriptions = transcriptions.long()
        start_id, pad_id, eos_id = (
            self.tokenizer.token_to_id("[SOS]"),
            self.tokenizer.token_to_id("[PAD]"),
            self.tokenizer.token_to_id("[EOS]"),
        )
        batch_size = imgs.size(0)
        dec_input = torch.full(
            (batch_size, self.context_length),
            pad_id,
            device=self.device,
            dtype=torch.long,
        )
        dec_input[:, 0] = start_id
        loss_arr = []
        outputs = torch.full(
            (batch_size, 1), start_id, device=self.device, dtype=torch.long
        )
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for i in range(1, transcriptions.size(1)):
            # Forward pass
            enc_output = self.vision_encoder(imgs)  # inside to use the updated weights
            out_idx = i - 1 if i < self.context_length else -1
            dec_output = self.decoder(
                dec_input.detach(), enc_output.detach(), out_idx=out_idx
            )
            dec_out_max = dec_output.argmax(-1).unsqueeze(-1)
            outputs = torch.cat([outputs, dec_out_max], dim=-1)

            # Backward pass
            label = transcriptions[:, i]
            loss = self.criterion(dec_output, label)
            cer = self.metric(
                self.tokenizer.batch_decode(outputs.cpu().numpy()),
                self.tokenizer.batch_decode(transcriptions.cpu().numpy()),
            )
            if train:
                if torch.isnan(loss):
                    print("NAN Loss")
                    print("Loss: ", loss)
                    print(torch.any(torch.isnan(dec_output)))
                    print("Dec_out_max: ", dec_output.shape)
                    print(torch.isnan(label))
                    print("Label: ", label.shape)
                    raise ValueError("NAN Loss")
                loss_arr.append(loss)
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log("train_cer", cer, on_step=True, on_epoch=True, prog_bar=True)
                optim = self.optimizers()
                optim.zero_grad()
                self.manual_backward(loss)
                self.clip_gradients(
                    optim, gradient_clip_val=3.0, gradient_clip_algorithm="norm"
                )
                optim.step()
            else:
                loss_arr.append(loss)
                self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("val_cer", cer, on_step=False, on_epoch=True, prog_bar=True)

            # Updating our sequences

            new_label = self.teacher_or_sample(label, dec_out_max, train)
            if i < self.context_length:
                dec_input[:, i] = new_label.squeeze(-1)
            else:
                dec_input = torch.cat([dec_input[:, 1:], new_label], dim=-1)

        return outputs, loss_arr

    def teacher_or_sample(self, label, dec_out_max, train=False):
        """
        Obtains the next label for the OCR model
        """
        if train:
            return dec_out_max.detach().long()

        label = label.unsqueeze(-1)
        if not self.schedule_sample:
            return label
        if self.schedule_sample and torch.rand(1) < self.schedule_sample_prob:
            return label
        dec_out_max = torch.where(
            dec_out_max == self.tokenizer.token_to_id("[PAD]"), label, dec_out_max
        )  # Teacher forcing if pad is predicted
        return dec_out_max.detach().long()
