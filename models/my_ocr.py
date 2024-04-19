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
    def __init__(self, vision_configs, decoder_configs, lr=5e-4, context_length=15, eos_id=3):
        super(MyOCR, self).__init__()
        self.lr = lr
        self.context_length = context_length
        self.eos_id = eos_id
        self.vision_encoder = VisionEncoder(**vision_configs)
        self.decoder = TransformerXL(**decoder_configs)
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters() 

    def forward(self, batch, batch_idx, greedy=True):
        img, seq = batch
        enc_output = self.vision_encoder(img)
        if greedy:
            return self.greedy_search(seq, enc_output)
        else:
            return self.beam_search(seq, enc_output)
        pass

    def training_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def teacher_forcing(self, seq, enc_output):
        inp = torch.zeros(seq.size(0), self.context_length, dtype=torch.long)
        inp[:, 0] = 2 
        for i in range(1, seq.size(1)):
            label = seq[:, i]
            # Forward pass
            out = self.decoder(inp, enc_output)
            # loss = self.criterion(out, label)
            
            # Get non finished sequences
            eos_idx = (label != self.eos_id)

            # Filter out finished sequences
            if eos_idx.any():
                inp, seq, label, enc_output = inp[eos_idx], seq[eos_idx], label[eos_idx], enc_output[eos_idx]
                if inp.size(0) == 0:
                    break

            # Update input
            if i < self.context_length:
                inp[:, i] = label
            else:
                inp = torch.cat([inp[:, 1:], label[:, None]], dim=1)
            
            


    def greedy_search(self, seq, enc_output):
        pass

    def beam_search(self, seq, enc_output):
        pass