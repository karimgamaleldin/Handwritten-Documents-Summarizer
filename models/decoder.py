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


class RecurrenceAttention(nn.Module):
    '''
    Recurrence Attention module for the decoder model
    
    A class that implements the recurrence attention mechanism in the Transformer-XL model
    '''
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(RecurrenceAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x, enc_output, hidden_past, tgt_mask, tgt_key_padding_mask):
        # Calculate the query, key and value
        h_telda = torch.concat([hidden_past.detach(), x], dim=1) # detach hidden_past to prevent backpropagation
        qkv = self.qkv(x) # produces the query, key & value qkv shape: (batch_size, seq_len, 3 * embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # split qkv into q, k, v shape: (batch_size, seq_len, embed_dim)
        q = q.reshape(1, -1, self.num_heads, self.embed_dim // self.num_heads)
        k = k.reshape(1, -1, self.num_heads, self.embed_dim // self.num_heads)
        v = v.reshape(1, -1, self.num_heads, self.embed_dim // self.num_heads)

        q = q.permute(1, 0, 2, 3)
        k = k.permute(1, 0, 2, 3)
        v = v.permute(1, 0, 2, 3)

        q, k, v = torch.matmul(x, q.T), torch.matmul(h_telda, k.T), torch.matmul(h_telda, v.T)

        x = self.scaled_dot_product_attention(q, k, v, tgt_mask, tgt_key_padding_mask)
        x = self.fc(x)
        x = self.dropout(x)
        return x

    def scaled_dot_product_attention(self, q, k, v, mask, padding_mask):
        pass