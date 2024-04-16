import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        target_vocab_size: int,
        maximum_position_encoding: int,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout = dropout

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model, max_len=self.maximum_position_encoding
        )
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            num_heads,
            dff,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, enc_output):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x, enc_output)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x
