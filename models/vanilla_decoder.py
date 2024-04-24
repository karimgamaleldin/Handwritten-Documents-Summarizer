import torch 
from torch import nn
from .PositionalEncoding import PositionalEncoding

class VanillaDecoder(nn.Module):
  def __init__(self, vocab_size, embed_size, dff, n_layers, dropout=0.1, num_heads=4, maximum_position_encoding=15, tie_weights=True, pad_id=0):
    super(VanillaDecoder, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.dff = dff
    self.n_layers = n_layers
    self.dropout = dropout
    self.tie_weights = tie_weights
    self.pad_id = pad_id
    self.maximum_position_encoding = maximum_position_encoding

    self.embed = nn.Embedding(vocab_size, embed_size)
    self.pos_enc = PositionalEncoding(embed_size, maximum_position_encoding)
    self.layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=dff, dropout=dropout, activation='gelu', batch_first=True, layer_norm_eps=1e-4)
    self.decoder = nn.TransformerDecoder(self.layer, num_layers=n_layers)
    self.fc = nn.Linear(embed_size, vocab_size)
    if tie_weights:
      self.fc.weight = self.embed.weight


  def forward(self, x, enc_output, out_idx=-1):
    tgt_key_padding_mask = torch.where(x == self.pad_id, float('-inf'), float(0.0))
    x = self.embed(x)
    x = self.pos_enc(x)
    tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1)
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf')).to(x.device)
    out = self.decoder(x, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    out = out[:, out_idx]
    out = self.fc(out)
    return out
    