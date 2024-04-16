import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    '''
    Decoder model for the transformer model used to generate text from the encoded image data

    It is inspired by the Transformer-XL model
    '''
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

        self.positional_encoding = AdaptivePositionalEncoding(d_model=self.d_model, max_len=self.maximum_position_encoding)
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.decoders = []
        for _ in range(num_layers):
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model,
                num_heads,
                dff,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(self.decoder_layer, 1)
            self.decoders.append(self.decoder)
        self.fc = nn.Linear(d_model, target_vocab_size)

    def forward(self, x, hidden_past_arr, enc_output):
        # Masks
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        pad_mask = x == 0
        # forward
        x = self.embedding(x)
        x = self.positional_encoding(x)
        hidden_states = [x] # store hidden states for next call
        for hidden_past, decoder in zip(hidden_past_arr, self.decoders):
            concat = torch.cat([hidden_past, x], dim=1)
            x = decoder(concat, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
            hidden_states.append(x)
        x = self.fc(x)
        return x, hidden_states


class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        pass
