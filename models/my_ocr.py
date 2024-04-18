import torch 
import math
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn import functional as F
from .Transformer_XL import Transformer_XL
from .VisionEncoder import VisionEncoder

class MyOCR(pl.LightningModule):
    '''
    MyOCR Module for Optical Character Recognition (OCR) using the VisionEncoder (inspired by ViT) and Decoder (inspired by GPT2) modules
    '''
    def __init__(self, vision_configs, decoder_configs, lr=5e-4):
        super(MyOCR, self).__init__()
        self.lr = lr
        self.vision_encoder = VisionEncoder(**vision_configs)
        self.decoder = Transformer_XL(**decoder_configs)
        self.fc = nn.Linear(decoder_configs['d_model'], decoder_configs['vocab_size'])
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch, batch_idx):
        img, seq = batch
        enc_output = self.vision_encoder(img)
        
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