import torch
import math
from torch import nn
import torch.nn.functional as F
from .PositionalEncoding import PositionalEncoding

class ConvEmbed(nn.Module):
    '''
    Embedding layer for image data using convaolutional layers then project it to the embed_dim

    It does non-overlapping patch extraction using Conv2d layer with kernel_size=patch_size and stride=patch_size
    '''
    def __init__(self, patch_size=16, in_chans=1, out_channels=32, embed_dim=64, stride=16, padding=2):
        super(ConvEmbed, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.linear = nn.LazyLinear(out_features=embed_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return F.gelu(x)

class VisionEncoder(nn.Module):
    '''
    A VisionEncoder module inspired by the Vision Transformer (ViT) model: https://arxiv.org/abs/2010.11929, but with a Convolutional Embedding layer
    
    We patch the image into non-overlapping patches and project them into a lower dimensional space using a Conv2d layer.
    '''
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        maximum_position_encoding: int,
        patch_size: int = 16,
        patch_stride: int = 16,
        patch_padding: int = 0,
        in_chans: int = 1,
        out_channels: int = 32,
        dropout: float = 0.1,
    ):
        super(VisionEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout = dropout

        self.positional_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.maximum_position_encoding)
        self.embedding = ConvEmbed(patch_size=patch_size, in_chans=in_chans, out_channels=out_channels, embed_dim=d_model, stride=patch_stride, padding=patch_padding)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dff,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return x