from torch import nn
import torch.nn.functional as F
from .PositionalEncoding import PositionalEncoding

class DenseBlock(nn.Module):
    def __init__(self):
        pass 

    def forward(self, x):
        pass

class VGGBlock(nn.Module):
    '''
    A VGGBlock class to be used for the ConvEmbed layer in the VisionEncoder module
    '''
    def __init__(self, kernel_size=3, in_channels=1, out_channels=32, stride=1, padding=1, num_convs=4):
        super(VGGBlock, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_convs = num_convs

        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        for _ in range(1, self.num_convs):
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.vgg_block(x)


class ConvEmbed(nn.Module):
    '''
    Embedding layer for image data using convaolutional layers then project it to the embed_dim

    It does non-overlapping patch extraction using Conv2d layer with kernel_size=patch_size and stride=patch_size
    '''
    def __init__(self, patch_size=8, in_chans=1, out_channels=32, seq_len=30, embed_dim=32, stride=8, padding=2, block_type='vgg'):
        super(ConvEmbed, self).__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding
        self.block_type = block_type
        self.seq_len = seq_len

        self.block = VGGBlock(kernel_size=3, in_channels=in_chans, out_channels=out_channels, stride=1, padding=1, num_convs=8)

        self.conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=seq_len,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.linear = nn.LazyLinear(out_features=embed_dim)
    
    def forward(self, x):
        x = self.block(x)
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
        block_type = 'vgg'
    ):
        super(VisionEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout = dropout
        self.block_type = block_type

        self.positional_encoding = PositionalEncoding(d_model=self.d_model, max_len=self.maximum_position_encoding)
        self.embedding = ConvEmbed(patch_size=patch_size, in_chans=in_chans, out_channels=out_channels, embed_dim=d_model, stride=patch_stride, padding=patch_padding, seq_len=maximum_position_encoding, block_type=block_type)
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