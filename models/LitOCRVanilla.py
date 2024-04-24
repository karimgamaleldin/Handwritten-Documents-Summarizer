import torch 
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.nn import functional as F
from .vision_encoder import VisionEncoder
from torchmetrics.text import CharErrorRate
from .vanilla_decoder import VanillaDecoder

class LitOCRVanilla(pl.LightningModule):
  '''
  MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
  '''

  def __init__(self, vision_configs, decoder_configs, tokenizer,*, lr=5e-4, context_length=15, metric=CharErrorRate(), max_len=100, **kwargs):
    super(LitOCRVanilla, self).__init__(**kwargs)
    self.lr = lr
    self.context_length = context_length
    self.vision_encoder = VisionEncoder(**vision_configs)
    self.decoder = VanillaDecoder(**decoder_configs, pad_id=tokenizer.token_to_id('[PAD]'))
    self.criterion = nn.CrossEntropyLoss()
    self.tokenizer = tokenizer
    self.metric = metric
    self.max_len = max_len # In case of infinite loop so initalize to have the max value
    self.automatic_optimization = False
    self.save_hyperparameters(ignore=['metric'])  

  def forward(self, imgs):
    '''
    Forward pass for the OCR
    '''
    enc_output = self.vision_encoder(imgs)
    output = torch.zeros((imgs.shape[0], 1), dtype=torch.long).to(imgs.device)
    output[:, 0] = self.tokenizer.token_to_id('[SOS]')
    print(output.shape)
    while output[:, -1][0] != self.tokenizer.token_to_id('[EOS]') and output.shape[1] < self.max_len:
      if output.shape[1] >= self.context_length:
        input = output[:, -self.context_length:]
      else:
        input = torch.zeros((output.shape[0], self.context_length), dtype=torch.long).to(imgs.device)
        input[:, :output.shape[1]] = output
      dec_output = self.decoder(input, enc_output)
      dec_output = dec_output.argmax(dim=-1).unsqueeze(-1)
      output = torch.cat([output, dec_output], dim=-1)
    return output


  def training_step(self, batch, batch_idx):
    '''
    Training step for the OCR Model
    '''
    imgs, transcriptions = batch
    outputs, loss_arr = self.teacher(imgs, transcriptions, train=True)
    loss_mean = torch.stack(loss_arr).mean()
    return loss_mean


  def validation_step(self, batch, batch_idx):
    '''
    Validation step for the OCR Model
    '''
    imgs, transcriptions = batch
    outputs, loss_arr = self.teacher(imgs, transcriptions)
    loss_mean = torch.stack(loss_arr).mean()
    return loss_mean

  def configure_optimizers(self):
    '''
    Configure the optimizer for the OCR Model
    '''
    return optim.Adam(self.parameters(), lr=self.lr)
  

  def teacher(self, imgs, transcriptions, train=False):
        '''
        Teacher forcing for the OCR model
        '''
        transcriptions = transcriptions.long()
        start_id, pad_id, eos_id = self.tokenizer.token_to_id('[SOS]'), self.tokenizer.token_to_id('[PAD]'), self.tokenizer.token_to_id('[EOS]')
        batch_size = imgs.size(0)
        enc_output = self.vision_encoder(imgs)
        dec_input = torch.full((batch_size, self.context_length), pad_id, device=self.device, dtype=torch.long)
        dec_input[:, 0] = start_id
        loss_arr = []
        outputs = torch.full((batch_size, 1), start_id, device=self.device, dtype=torch.long)
        for i in range(1, transcriptions.size(1)):
            # Forward pass
            out_idx = i - 1 if i < self.context_length else -1 
            dec_output = self.decoder(dec_input.detach(), enc_output.detach(), out_idx=out_idx)
            dec_out_max = dec_output.argmax(-1).unsqueeze(-1)
            outputs = torch.cat([outputs, dec_out_max], dim=-1)

            # Backward pass 
            label = transcriptions[:, i]
            loss = self.criterion(dec_output, label)
            cer = self.metric(self.tokenizer.batch_decode(outputs.cpu().numpy()), self.tokenizer.batch_decode(transcriptions.cpu().numpy()))
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
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('train_cer', cer, on_step=True, on_epoch=True, prog_bar=True)
                optim = self.optimizers()
                optim.zero_grad() 
                self.manual_backward(loss)
                self.clip_gradients(optim, gradient_clip_val=3.0, gradient_clip_algorithm='norm')
                optim.step()
            else:
                loss_arr.append(loss)
                self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('val_cer', cer, on_step=True, on_epoch=True, prog_bar=True)
            
            # Updating our sequences
            # scheduled_sampling =  label.unsqueeze(1) if torch.rand(1) < self.teacher_forcing_ratio else dec_out_max.detach().long()
            scheduled_sampling =  label.unsqueeze(-1) 
            if i < self.context_length:
                dec_input[: , i] = scheduled_sampling.squeeze(-1)
            else:
                dec_input = torch.cat([dec_input[:, 1:], scheduled_sampling], dim=-1)

        return outputs, loss_arr
