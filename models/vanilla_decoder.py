import torch 
from torch import nn
from torch.optim import Adam
from torchmetrics.text import CharErrorRate
from .PositionalEncoding import PositionalEncoding

class VanillaDecoder(nn.Module):
  def __init__(self, vocab_size, embed_size, dff, n_layers, dropout=0.1, num_heads=4, seq_len=15, tie_weights=True, pad_id=0):
    super(VanillaDecoder, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.dff = dff
    self.n_layers = n_layers
    self.dropout = dropout
    self.tie_weights = tie_weights
    self.pad_id = pad_id

    self.embed = nn.Embedding(vocab_size, embed_size)
    self.pos_enc = PositionalEncoding(embed_size, seq_len)
    self.layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=dff, dropout=dropout, activation='gelu', batch_first=True, layer_norm_eps=1e-4)
    self.decoder = nn.TransformerDecoder(self.layer, num_layers=n_layers)
    self.fc = nn.Linear(embed_size, vocab_size)
    if tie_weights:
      self.fc.weight = self.embed.weight


  def forward(self, x, enc_output, out_idx=-1):
    inp = x
    if torch.isnan(x).any():
      print("NAN in Decoder input", x.max(), x.min(), x.mean())
    tgt_key_padding_mask = torch.where(x == self.pad_id, float('-inf'), float(0.0))
    x = self.embed(x)
    if torch.isnan(x).any():
      print("NAN in Decoder embedding", x.max(), x.min(), x.mean())
    x = self.pos_enc(x)
    if torch.isnan(x).any():
      print("NAN in Decoder pos_enc", x.max(), x.min(), x.mean())
    tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1)
    tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf')).to(x.device)
    # print(tgt_key_padding_mask)
    # print(tgt_mask)
    # raise Exception
    if torch.isnan(tgt_mask).any():
      print("NAN in Decoder tgt_mask", tgt_mask.max(), tgt_mask.min(), tgt_mask.mean())
    if torch.isnan(tgt_key_padding_mask).any():
      print("NAN in Decoder tgt_key_padding_mask", tgt_key_padding_mask.max(), tgt_key_padding_mask.min(), tgt_key_padding_mask.mean())
    out = self.decoder(x, enc_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    if torch.isnan(out).any():
      print("NAN in Decoder decoder", x.max(), x.min(), x.mean())
      print("NAN in Decoder decoder", out.max(), out.min(), out.mean())
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
      print(tgt_key_padding_mask)
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
      print(tgt_mask)
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
      print(x)
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
      print(inp)
    out = out[:, out_idx]
    if torch.isnan(out).any():
      print("NAN in Decoder out_idx", out.max(), out.min(), out.mean())
    out = self.fc(out)
    if torch.isnan(out).any():
      print("NAN in Decoder fc", out.max(), out.min(), out.mean())
    return out
    