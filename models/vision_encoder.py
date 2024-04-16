import torch.nn as nn
import torch.nn.functional as F
from patch_embedding import PatchEmbedding
from positional_encoding import PositionalEncoding


class VisionEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        maximum_position_encoding: int,
        dropout: float = 0.1,
    ):
        super(VisionEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout = dropout

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model, max_len=self.maximum_position_encoding
        )
        self.embedding = PatchEmbedding()
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
