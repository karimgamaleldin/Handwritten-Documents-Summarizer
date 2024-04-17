import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
  def __init__(self, d):
    super(PositionalEmbedding, self).__init__()
    self.d = d
    inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d)) # Calculate the inverse frequency to be used in the positional encoding
    self.register_buffer('inv_freq', inv_freq)

  def forward(self, pos):
    sinusoid_inp = torch.einsum('i,j->ij', pos, self.inv_freq) # multiply each position i in the pos array with each position j in the inv_freq array
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return pos_emb

class RecurrenceAttention(nn.Module):
  def __init__(self, d_model, d_inner, num_heads, dropout=0.1, pre_norm=False):
    super(RecurrenceAttention, self).__init__()
    self.d_model = d_model
    self.d_inner = d_inner
    self.num_heads = num_heads
    self.d_head = d_model // num_heads

    self.q = nn.Linear(d_model, d_inner * num_heads, bias=False) # to project the query to the inner dimension
    self.kv = nn.Linear(d_model, d_inner * num_heads * 2, bias=False) # to project the key and value to the inner dimension

    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(self.d_inner * self.num_heads, d_model) # to project the output of the multi-head attention to the original dimension
    self.layer_norm = nn.LayerNorm(self.d_model) 
    self.pre_norm = pre_norm

  def forward(self, x, pos_emb, u, v, tgt_mask=None, mem=None):
    '''
    Shapes:
    x: (batch_size, seq_len, embed_dim)
    pos_emb: (cur_seq + prev_seq, d_in)
    u: (num_heads, d_head)
    v: (num_heads, d_head)
    tgt_mask: (batch_size, seq_len, seq_len)
    mem: (batch_size, prev_seq, embed_dim)
    '''
    prev_seq = mem.size(1) if mem is not None else 0
    batch_size, cur_seq = x.size(0), x.size(1)
    
    # Get the weight matrices for the query, key and value
    h_telda = torch.cat([mem, x], dim=1) if mem is not None else x
    q_tfmd, kv = self.q(x), self.kv(h_telda) # Calculate the query and key-value matrices
    k_tfmd, v_tfmd = torch.chunk(kv, 2, dim=-1) # Split the key-value matrix into key and value matrices # (b, seq_len + prev_seq, d_inner * num_heads)
    q_tfmd, k_tfmd, v_tfmd = q_tfmd.view(batch_size, cur_seq, self.num_heads, self.d_head), k_tfmd.view(batch_size, cur_seq + prev_seq, self.num_heads, self.d_head), v_tfmd.view(batch_size, cur_seq + prev_seq, self.num_heads, self.d_head)

    # Calculate the attention
    content_attn = torch.einsum('bihd,bjhd->bijh', q_tfmd + u, k_tfmd) # Calculate the content-based attention
    pos_emb = pos_emb.view(cur_seq + prev_seq, self.num_heads, self.d_head)
    pos_attn = torch.einsum('bihd,jhd->bijh', q_tfmd + v, pos_emb) # Calculate the position-based attention
    position_attn = self._rel_shift(pos_attn) # Shift the position-based attention
    attn = content_attn + position_attn # Combine the content-based and position-based attention

    if tgt_mask is not None:
      attn = attn.masked_fill(tgt_mask == 0, float('-inf'))
    attn = attn / (self.d_head ** 0.5)
    attn = F.softmax(attn, dim=1)
    attn = self.drop(attn)
    weightd_val = torch.einsum('bijh,bjhd->bihd', attn, v_tfmd).contiguous().view(batch_size, cur_seq, -1)

    if not self.pre_norm:
      return self.layer_norm(x + self.fc(weightd_val))
    else:
      return self.fc(self.layer_norm(x + weightd_val))


  def _rel_shift(self, x):
    '''
    The shift operation used to calculate the relative position encoding
    '''
    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
    x = x_padded[1:].view_as(x)
    return x
    

  def _split_heads(self, x: torch.Tensor, batch_size: int):
    x = x.view(batch_size, -1, self.num_heads, self.d_head)
    return x.permute(0, 2, 1, 3)
  
  def _scaled_dot_product_attention(self, q, k, v, mask, padding_mask):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)) / (q.size(-1) ** 0.5) # Calculate the scaled dot product attention
    if mask is not None:
      qk = qk.masked_fill(mask == 0, float('-inf')) # Apply the mask to the scaled dot product attention
    qk = F.softmax(qk, dim=-1)
    if padding_mask is not None:
      qk = qk.masked_fill(padding_mask == 0, 0)
    output = torch.matmul(qk, v)
    return output

class TransformerDecoderLayerRelative(nn.Module):
  def __init__(self, d_model, num_heads, dff, d_inner, dropout=0.1, pre_norm=True, activation='gelu'):
    super(TransformerDecoderLayerRelative, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.dff = dff
    self.dropout = dropout
    self.pre_norm = pre_norm
    self.d_inner = d_inner

    self.multi_head_attention = RecurrenceAttention(d_model, d_inner, num_heads)
    fc1 = nn.Linear(d_model, dff)
    activ = nn.GELU() if activation == 'gelu' else nn.ReLU()
    dropout1 = nn.Dropout(dropout)
    fc2 = nn.Linear(dff, d_model)
    dropout2 = nn.Dropout(dropout)

    self.net = nn.Sequential(fc1, activ, dropout1, fc2, dropout2)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x, pos_emb, u, v, mem=None, tgt_mask=None):
    output = self.multi_head_attention(x, pos_emb, u, v, mem=mem, tgt_mask=tgt_mask)
    if self.pre_norm:
      output = self.layer_norm(output)
      return self.net(output) + x 
    else:
      output = self.net(output) + output
      return self.layer_norm(output)

class TransformerXL(nn.Module):
  def __init__(self, vocab_size, n_layers, n_heads, d_model, d_inner, dff, seq_len, tie_weights=True, dropout=0.1):
    super(TransformerXL, self).__init__()
    self.n_tokens = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_inner = d_inner
    self.dff = dff
    self.seq_len = seq_len
    assert d_model % n_heads == 0, 'd_model should be divisible by n_heads'
    self.d_head = d_model // n_heads

    self.embed = nn.Embedding(vocab_size, d_model)
    self.pos_emb = PositionalEmbedding(d_model)

    self.dropout = nn.Dropout(dropout)
    self.layers = nn.ModuleList()
    for _ in range(n_layers):
      self.layers.append(TransformerDecoderLayerRelative(d_model, n_heads, dff, d_inner, dropout=dropout))

    self.fc = nn.Linear(d_model, vocab_size)
    if tie_weights: # ties the embedding with the weights of the final layer
      self.fc.weight = self.embed.weight

    # Learnable parameters for the relative positional encoding
    self.u = nn.Parameter(torch.Tensor(n_layers, n_heads, self.d_head))
    self.v = nn.Parameter(torch.Tensor(n_layers, n_heads, self.d_head))
    

  def forward(self, x):
    pass
    # out = self.embed(x) * (self.d_model ** 0.5) # Calculate the embedding
    # out = self.dropout(out) # Apply dropout to the output
    # pos_idxs = torch.arange()

    # new_mem = [out] # Initialize the new memory
    # for i, layer in enumerate(self.layers): # loop on all layers
    #   u, v = self.u[i], self.v[i]
    #   out = layer(out, )

    # out = self.dropout(out)
    # return out, new_mem

