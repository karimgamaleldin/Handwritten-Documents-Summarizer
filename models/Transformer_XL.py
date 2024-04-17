import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding

class RecurrenceAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(RecurrenceAttention, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)
    self.fc = nn.Linear(self.embed_dim, self.embed_dim)
    self.dropout = nn.Dropout(self.dropout)

  def forward(self, x, h_past, tgt_mask, pad_mask):
    '''
    Shapes:
    x: (batch_size, seq_len, embed_dim)
    h_past: (batch_size, seq_len, embed_dim)
    tgt_mask: (seq_len, seq_len)
    pad_mask: (batch_size, seq_len)
    '''
    batch_size, _ , _ = x.size()
    # Getting the query, key and value
    h_telda = torch.cat([h_past.detach(), x], dim=1) # Calculate the concatenation of the hidden states, Shape (batch_size, seq_len, 2 * embed_dim)
    qkv = self.qkv(x) # Calculate the query, key and value, Shape (batch_size, seq_len, 3 * embed_dim)
    q, k, v = torch.chunk(qkv, 3, dim=-1) # Split qkv into q, k, v, Shape (batch_size, seq_len, embed_dim)
    
    q, k, v = self.split_heads(q, batch_size), self.split_heads(k, batch_size), self.split_heads(v, batch_size) # Split the query, key and value into heads, Shape (batch_size, num_heads, seq_len, embed_dim // num_heads)
    q, k, v = torch.matmul(x, q.T), torch.matmul(h_telda, k.T), torch.matmul(h_telda, v.T) # Calculate the query, key and value

    x = self.scaled_dot_product_attention(q, k, v, tgt_mask, pad_mask) # Calculate the scaled dot product attention
    x = self.fc(x)
    return x, q, k, v
  
  def split_heads(self, x, batch_size):
    x = x.reshape(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads)
    return x.permute(0, 2, 1, 3)
  
  def scaled_dot_product_attention(self, q, k, v, mask, padding_mask):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)) / (q.size(-1) ** 0.5) # Calculate the scaled dot product attention
    if mask is not None:
      qk = qk.masked_fill(mask == 0, float('-inf')) # Apply the mask to the scaled dot product attention
    qk = F.softmax(qk, dim=-1)
    if padding_mask is not None:
      qk = qk.masked_fill(padding_mask == 0, 0)
    output = torch.matmul(qk, v)
    return output

    

class AdaptiveEmbedding(nn.Module):
  def __init__(self):
    pass 

  def forward(self):
    pass

class PointwiseFeedForward(nn.Module):
  def __init__(self):
    pass 

  def forward(self):
    pass

class Transformer_Layer(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout=0.1):
    super(Transformer_Layer, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout

    self.multi_head_attention = RecurrenceAttention(self.embed_dim, self.num_heads)
    self.adaptive_positional_encoding = AdaptiveEmbedding()
    self.layer_norm1 = nn.LayerNorm(self.embed_dim)
    self.point_ff = PointwiseFeedForward()


  def forward(self, x):
    x = self.multi_head_attention(x)
    x = self.adaptive_positional_encoding(x)
    x = self.layer_norm1(x)
    x = self.point_ff(x)
    return x

class Transformer_XL(nn.Module):
  def __init__(self, n_tokens, n_layers, n_heads, d_model, d_head, d_inner, tie_weights=True, dropout=0.1):
    super(Transformer_XL, self).__init__()
    self.n_tokens = n_tokens
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.d_model = d_model 

    self.adap_embed = AdaptiveEmbedding(n_tokens, d_model, d_head, d_inner)
    self.dropout = nn.Dropout(dropout)
    
    self.layers = nn.ModuleList()
    for _ in range(n_layers):
      self.layers.append(Transformer_Layer(d_model, n_heads, dropout))

    self.fc = nn.Linear(d_model, n_tokens)
    if tie_weights: # ties the embedding with the weights of the final layer
      self.fc.weight = self.adap_embed.embeddings.weight

    self.r_emb = nn.Parameter(pass)
    self.r_w_bias = nn.Parameter(pass)
    self.r_r_bias = nn.Parameter(pass)

  def forward(self):
    pass 


