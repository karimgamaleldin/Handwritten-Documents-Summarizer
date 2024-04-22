import torch 
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.nn import functional as F
from .Transformer_XL import TransformerXL
from .vision_encoder import VisionEncoder
from torchmetrics.text import CharErrorRate

class LitOCRXL(pl.LightningModule):
    '''
    MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
    '''
    def __init__(self, vision_configs, decoder_configs, tokenizer,*, lr=5e-4, context_length=15, eos_id=3, pad_id=0, metric_teacher_force='val_cer', minimize_metric=False, teacher_forcing_ratio=0.95, metric=CharErrorRate(), **kwargs):
        super(LitOCRXL, self).__init__(**kwargs)
        self.lr = lr
        self.context_length = context_length
        self.eos_id = eos_id
        self.vision_encoder = VisionEncoder(**vision_configs)
        self.decoder = TransformerXL(**decoder_configs)
        self.criterion = nn.CrossEntropyLoss()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        self.metric_teacher_force = metric_teacher_force
        self.minimize_metric = minimize_metric
        self.tokenizer = tokenizer
        self.prev_metric = float('inf') if minimize_metric else float('-inf')
        self.metric = metric
        self.automatic_optimization = False
        self.save_hyperparameters() 

    def setup(self, stage=None):
        self.decoder.set_device(self.device) # to set the device for the decoder tensors created in the forward pass

    def forward(self, imgs):
        '''
        Forward pass for the OCR model
        '''
        # Vision Encoder
        enc_output = self.vision_encoder(imgs)
        # Decoder
        output = self.greedy_search(enc_output)
        return output
    
    def training_step(self, batch, batch_idx):
        '''
        Training step for the OCR Model
        '''
        imgs, transcriptions = batch
        outputs, loss_arr = self.teacher(imgs, transcriptions, train=True)
        return None 
    
    def validation_step(self, batch, batch_idx):
        '''
        Validation step for the OCR Model
        '''
        imgs, transcriptions = batch
        outputs, loss_arr = self.teacher(imgs, transcriptions)
        return loss_arr
    
    def teacher(self, imgs, transcriptions, train=False):
        '''
        Teacher forcing for the OCR model
        '''
        transcriptions = transcriptions.long()
        start_id, pad_id, eos_id = self.tokenizer.token_to_id('[SOS]'), self.tokenizer.token_to_id('[PAD]'), self.tokenizer.token_to_id('[EOS]')
        batch_size = imgs.size(0)
        enc_output = self.vision_encoder(imgs)
        dec_input = torch.full((batch_size, 1), pad_id, device=self.device, dtype=torch.long)
        dec_input[:, 0] = start_id
        loss_arr = []
        mem=None
        outputs = torch.full((batch_size, 1), start_id, device=self.device, dtype=torch.long)
        for i in range(1, transcriptions.size(1)):
            # Forward pass
            out_idx = i - 1 if i < self.context_length else -1 
            dec_output, mem = self.decoder(dec_input.detach(), enc_output.detach(), mem=mem, out_idx=out_idx)
            dec_out_max = dec_output.argmax(-1).unsqueeze(1)
            outputs = torch.cat([outputs, dec_out_max], dim=-1)

            # Backward pass 
            label = transcriptions[:, i]
            loss = self.criterion(dec_output, label)
            if train:
                loss_arr.append(loss)
                optim = self.optimizers()
                optim.zero_grad() 
                self.manual_backward(loss)
                optim.step()
            else:
                loss_arr.append(loss)
            
            # Updating our sequences
            scheduled_sampling =  label.unsqueeze(1) if torch.rand(1) < self.teacher_forcing_ratio else dec_out_max.detach().long()
            if i < self.context_length:
                dec_input = torch.cat([dec_input, scheduled_sampling], dim=-1)
            else:
                dec_input = torch.cat([dec_input[:, 1:], scheduled_sampling], dim=-1)

        return outputs, loss_arr
    

    def greedy_search(self, enc_output):
        '''
        Greedy search for the OCR model
        '''
        start_token, pad_id, eos_id = self.tokenizer.token_to_id('[SOS]'), self.tokenizer.token_to_id('[PAD]'), self.tokenizer.token_to_id('[EOS]')
        batch_size = enc_output.size(0)
        dec_input = torch.full((batch_size, 1), self.pad_id, device=self.device, dtype=torch.long)
        dec_input[:, 0] = start_token
        mem=None
        outputs = torch.full((batch_size, 1), start_token, device=self.device, dtype=torch.long)
        for i in range(1, self.context_length):
            dec_output, mem = self.decoder(dec_input, enc_output, mem=mem, out_idx=i-1)
            dec_out_max = dec_output.argmax(-1).unsqueeze(1)
            outputs = torch.cat([outputs, dec_out_max], dim=-1)
            dec_input = torch.cat([dec_input, dec_out_max], dim=-1)
        return outputs
    
    def configure_optimizers(self):
        '''
        Configure the optimizer for the OCR model
        '''
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer