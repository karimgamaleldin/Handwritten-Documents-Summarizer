import torch 
import math
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn import functional as F
from .Transformer_XL import Transformer_XL

class MyOCR(pl.LightningModule):
    '''
    MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
    '''
    def __init__(self, vision_configs, decoder_configs, lr=5e-4):
        super(MyOCR, self).__init__()
        self.lr = lr
        pass

    def forward(self):
        pass

    def training_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def teacher_forcing(self):
        pass

    def greedy_search(self):
        pass

    def beam_search(self):
        pass