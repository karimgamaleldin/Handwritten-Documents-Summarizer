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
        self.automatic_optimization = False
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.initial_teacher_forcing_ratio = teacher_forcing_ratio
        self.metric_teacher_force = metric_teacher_force
        self.minimize_metric = minimize_metric
        self.tokenizer = tokenizer
        self.prev_metric = float('inf') if minimize_metric else float('-inf')
        self.metric = metric
        self.save_hyperparameters() 

    def setup(self, stage=None):
        self.decoder.set_device(self.device) # to set the device for the decoder tensors created in the forward pass

    def forward(self, batch, teacher_forcing=False, predict=True, backward=None):
        # Checks
        if backward is not None and not teacher_forcing:
            raise ValueError('Cannot pass backward function when teacher_forcing is False')
        if teacher_forcing and predict:
            raise ValueError('Cannot pass teacher_forcing and predict as True')
        if  predict and backward is not None:
            raise ValueError('Cannot pass backward function when predict is False')
        
        if not predict: # Used for training by the trainer
            print(teacher_forcing, predict, backward)
            print('Batch tupleeeeeee: ', batch[0].size(), batch[1].size(), len(batch))
            print('~~~~~~~~~~~~~~~~~~~~~~~~')
            print(batch)
            img, seq = batch
            enc_output = self.vision_encoder(img)
            if teacher_forcing:
                out, loss_arr = self.teacher_forcing(seq, enc_output, step_fn=backward)
            else:
                out, loss_arr = self.greedy_search(seq, enc_output)
            return out, loss_arr
        else: # Used for inference
            img = batch
            print(img.shape)
            enc_output = self.vision_encoder(img)
            out, _ = self.greedy_search(enc_output=enc_output)
            return out

    def training_step(self, batch, batch_idx):
        print('training')
        def backward(outputs, targets, optimizer: optim.Optimizer=self.optimizers()):
            optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            self.manual_backward(loss)
            optimizer.step()
            return loss
        outputs, loss_arr = self.forward(batch, teacher_forcing=True, predict=False, backward=backward)
        # logs
        return loss_arr
    
    def on_train_epoch_end(self):
        # Do something
        pass
    
    def on_validation_epoch_end(self, validation_step_outputs):
        current_metric = self.trainer.callback_metrics[self.metric_teacher_force]
        if (self.minimize_metric and current_metric < self.prev_metric) or (not self.minimize_metric and current_metric > self.prev_metric):
            self.teacher_forcing_ratio = max(self.initial_teacher_forcing_ratio * self.teacher_forcing_ratio, 0.1) # Decay teacher forcing ratio
            self.prev_metric = current_metric

        # Do something
        # model_filename = f'model_{str(self.global_step).zfill(5)}.onnx'
        # torch.onnx.export(self, _, model_filename) # wrong
        # wandb.save(model_filename)

        # # Log flattened logits
        # flattened_logits = torch.cat([logits.flatten() for logits in self.logits], dim=0)
        # self.logger.experiment.log({'valid/flattened_logits': wandb.Histogram(flattened_logits), 'global_step':self.global_step})

    def test_step(self, batch, batch_idx):
        print('test')
        outputs, loss_arr = self.forward(batch, teacher_forcing=False, predict=False)
        # logs
        return loss_arr
    
    def validation_step(self, batch, batch_idx):
        print('validation')
        outputs, loss_arr = self.forward(batch, teacher_forcing=True, predict=False)
        # logs
        return loss_arr
    
    def predict_step(self, batch, batch_idx):
        print('predict')
        outputs = self.forward(batch, batch_idx, teacher_forcing=False, predict=True)
        return outputs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def teacher_forcing(self, seq, enc_output, step_fn=None):
        '''
        inp --> input to the decoder
        mem --> memory for the decoder
        seq --> target sequence
        outputs --> predicted sequence
        loss_arr --> list of losses for each timestep
        '''
        inp, mem, seq = torch.zeros(seq.size(0), self.context_length, device=self.device), None, seq.long()
        inp[:, 0] = 2 
        outputs = torch.zeros(seq.size(0), 1, dtype=torch.long, device=self.device)
        outputs[:, 0] = 2
        loss_arr = []
        for i in range(1, seq.size(1)):
            label = seq[:, i]
            # Forward pass
            print(inp.size(), enc_output.size())
            out_idx = i - 1 if i < self.context_length else -1
            out, mem = self.decoder(inp, enc_output, mem=mem, out_idx=out_idx)
            out_max = out.argmax(dim=-1)
            outputs = torch.cat((outputs, out_max.unsqueeze(1)), dim=1)
            print('out_size:', out.size(), 'label_size:', label.size())
            # Backward pass
            if step_fn is not None:
                loss = step_fn(out, label)
                # self.log('train/loss', loss, on_step=True, on_epoch=True)
                # self.log(f'train/{self.metric_teacher_force}', self.metric(outputs, seq), on_step=True, on_epoch=True)  
                # loss_arr.append(loss.item())
            else:
                loss = self.criterion(out, label)
                # self.log('val/loss', loss, on_step=False, on_epoch=True)
                # self.log(f'val/{self.metric_teacher_force}', self.metric(outputs, seq), on_step=False, on_epoch=True)
                # loss_arr.append(loss.item())
            # Update input
            scheduled_sample = label if torch.rand(1) < self.teacher_forcing_ratio else out_max.detach().long()
            if i < self.context_length:
                inp[:, i] = scheduled_sample
            else:
                inp = torch.cat((inp[:, 1:], scheduled_sample.unsqueeze(1)), dim=1)
        return outputs, loss_arr              
            
    def greedy_search(self, seq=None, enc_output=None):
        batch = enc_output.size(0)
        inp, mem, seq = torch.zeros(batch, self.context_length, device=self.device).long(), None, seq.long() if seq is not None else None
        inp[:, 0] = 2
        loss_arr = []
        outputs = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        outputs[:, 0] = 2
        if seq is not None:
            for i in range(1, seq.size(1)):
                # Forward pass
                out_idx = i - 1 if i < self.context_length else -1
                out, mem = self.decoder(inp, enc_output, mem=mem, out_idx=out_idx)
                out_max = out.argmax(dim=-1).long()
                outputs = torch.cat((outputs, out_max.unsqueeze(1)), dim=1)
                label = seq[:, i]
                # loss_arr.append(self.criterion(out, label).item())
                # Update input
                if i < self.context_length:
                    inp[:, i] = out_max
                else:
                    inp = torch.cat((inp[:, 1:], out_max.unsqueeze(1)), dim=1)
        else:
            i = 1
            # todo: fix the problem of a batch running at the same time.
            while True:
                print(inp.size(), enc_output.size())
                out, mem = self.decoder(inp, enc_output, mem=mem)
                out_max = out.argmax(dim=-1).long()
                outputs = torch.cat((outputs, out_max.unsqueeze(1)), dim=1)
                if out_max.item() == self.eos_id:
                    break
                if i < self.context_length:
                    inp[:, i] = out_max
                else:
                    inp = torch.cat((inp[:, 1:], out_max.unsqueeze(1)), dim=1)
                i += 1
                
        return outputs, loss_arr

