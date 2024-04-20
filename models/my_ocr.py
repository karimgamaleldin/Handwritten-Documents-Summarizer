import torch 
import math
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn import functional as F
from .Transformer_XL import TransformerXL
from .vision_encoder import VisionEncoder

class MyOCR(pl.LightningModule):
    '''
    MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
    '''
    def __init__(self, vision_configs, decoder_configs, lr=5e-4, context_length=15, eos_id=3, pad_id=0, metric_teacher_force='val_accuracy', minimize_metric=False):
        super(MyOCR, self).__init__()
        self.lr = lr
        self.context_length = context_length
        self.eos_id = eos_id
        self.vision_encoder = VisionEncoder(**vision_configs)
        self.decoder = TransformerXL(**decoder_configs, pad_id=pad_id)
        self.criterion = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        self.teacher_forcing_ratio = 0.95
        self.initial_teacher_forcing_ratio = 0.95
        self.metric_teacher_force = metric_teacher_force
        self.minimize_metric = minimize_metric
        self.prev_metric = float('inf') if minimize_metric else float('-inf')
        self.save_hyperparameters() 
        

    def forward(self, batch, batch_idx, teacher_forcing=True, predict=False, backward=None):
        # Checks
        if backward is not None and not teacher_forcing:
            raise ValueError('Cannot pass backward function when teacher_forcing is False')
        if teacher_forcing and predict:
            raise ValueError('Cannot pass teacher_forcing and predict as True')
        if  predict and backward is not None:
            raise ValueError('Cannot pass backward function when predict is False')
        
        if not predict: # Used for training by the trainer
            img, seq = batch
            enc_output = self.vision_encoder(img)
            if teacher_forcing:
                out, loss_arr = self.training_step(seq, enc_output, backward=backward)
            else:
                out, loss_arr = self.greedy_search(seq, enc_output)
            return out, loss_arr
        else: # Used for inference
            enc_output = self.vision_encoder(batch)
            out, _ = self.greedy_search(batch, enc_output)
            return out

    def training_step(self, batch, batch_idx):
        def backward(outputs, targets, optimizer: optim.Optimizer=self.optimizers()):
            optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            self.manual_backward(loss)
            optimizer.step()
            return loss
        outputs, loss_arr = self.forward(batch, batch_idx, teacher_forcing=True, predict=False, backward=backward)

        # logs
        return loss_arr
    
    def on_validation_epoch_end(self):
        current_metric = self.trainer.callback_metrics[self.metric_teacher_force]
        if (self.minimize_metric and current_metric < self.prev_metric) or (not self.minimize_metric and current_metric > self.prev_metric):
            self.teacher_forcing_ratio = max(self.initial_teacher_forcing_ratio * self.teacher_forcing_ratio, 0.1) # Decay teacher forcing ratio
            self.prev_metric = current_metric


    def test_step(self, batch, batch_idx):
        outputs, loss_arr = self(batch, batch_idx, teacher_forcing=False, predict=False)
        # logs
        return loss_arr
    
    def validation_step(self, batch, batch_idx):
        outputs, loss_arr = self(batch, batch_idx, teacher_forcing=True, predict=False)
        # logs
        return loss_arr

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def teacher_forcing(self, seq, enc_output, step_fn=None):
        inp = torch.zeros(seq.size(0), self.context_length, dtype=torch.long)
        out, mem = inp, None
        inp[:, 0] = 2 
        outputs = torch.zeros(seq.size(0), 1, dtype=torch.long)
        outputs[:, 0] = 2
        loss_arr = []
        outputs.append(out.clone())
        for i in range(1, seq.size(1)):
            label = seq[:, i]
            # Forward pass
            out_idx = i - 1 if i < self.context_length else -1
            out, mem = self.decoder(out, enc_output, mem=mem, out_idx=out_idx)
            outputs = torch.cat((outputs, out.argmax(dim=-1).unsqueeze(1)), dim=1)
            # Backward pass
            if step_fn is not None:
                loss = step_fn(out, label)
                loss_arr.append(loss.item())
            else:
                loss_arr.append(self.criterion(out, label).item())
            # Update input
            scheduled_sample = label if torch.rand(1) < self.teacher_forcing_ratio else out.argmax(dim=-1)
            if i < self.context_length:
                inp[:, i] = scheduled_sample
            else:
                inp = torch.cat((inp[:, 1:], scheduled_sample.unsqueeze(1)), dim=1)
        return outputs, loss_arr              
            
    def greedy_search(self, seq, enc_output):
        inp = torch.zeros(seq.size(0), self.context_length, dtype=torch.long)
        out, mem = inp, None
        inp[:, 0] = 2
        loss_arr = []
        outputs = torch.zeros(seq.size(0), 1, dtype=torch.long)
        outputs[:, 0] = 2
        outputs.append(out.clone())
        for i in range(1, seq.size(1)):
            label = seq[:, i]
            # Forward pass
            out_idx = i - 1 if i < self.context_length else -1
            out, mem = self.decoder(out, enc_output, mem=mem, out_idx=out_idx)
            outputs = torch.cat((outputs, out.argmax(dim=-1).unsqueeze(1)), dim=1)
            loss_arr.append(self.criterion(out, label).item())
            # Update input
            if i < self.context_length:
                inp[:, i] = out.argmax(dim=-1)
            else:
                inp = torch.cat((inp[:, 1:], out.argmax(dim=-1).unsqueeze(1)), dim=1)
        return outputs, loss_arr

