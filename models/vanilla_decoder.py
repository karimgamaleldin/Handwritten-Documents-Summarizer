import torch 
from torch import nn
from torch.optim import Adam
from torchmetrics.text import CharErrorRate
from .PositionalEncoding import PositionalEncoding

class VanillaDecoder(nn.Module):
  def __init__(self, vocab_size, embed_size, dff, n_layers, dropout=0.5, num_heads=8, seq_len=15, tie_weights=True):
    super(VanillaDecoder, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.dff = dff
    self.n_layers = n_layers
    self.dropout = dropout
    self.tie_weights = tie_weights

    self.embed = nn.Embedding(vocab_size, embed_size)
    self.pos_enc = PositionalEncoding(embed_size, seq_len)
    self.layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=dff, dropout=dropout, activation='gelu', batch_first=True)
    self.decoder = nn.TransformerDecoder(self.layer, num_layers=n_layers)
    self.fc = nn.Linear(embed_size, vocab_size)
    if tie_weights:
      self.fc.weight = self.embed.weight


  def forward(self, x, enc_output, out_idx=-1):
    tgt_key_padding_mask = (x == 0)
    tgt_key_padding_mask = tgt_key_padding_mask.to(x.device)
    x = self.embed(x)
    x = self.pos_enc(x)
    tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1)) * float('-inf'), diagonal=1).to(x.device)
    x = self.decoder(x, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    x = x[:, out_idx]
    x = self.fc(x)
    return x
    