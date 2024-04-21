import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
  def __init__(self, d, **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.d = d
    inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d)) # Calculate the inverse frequency to be used in the positional encoding
    self.register_buffer('inv_freq', inv_freq)

  def forward(self, pos):
    sinusoid_inp = torch.einsum('i,j->ij', pos, self.inv_freq) # multiply each position i in the pos array with each position j in the inv_freq array
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    return pos_emb

class RecurrenceAttention(nn.Module):
  def __init__(self, d_model, d_inner, num_heads, dropout=0.1, pre_norm=False, **kwargs):
    super(RecurrenceAttention, self).__init__(**kwargs)
    self.d_model = d_model
    self.d_inner = d_inner # the inner dimension of the query, key and value matrices
    self.num_heads = num_heads
    self.d_head = d_inner // num_heads
    
    self.q = nn.Linear(d_model, d_inner, bias=False) # to project the query to the inner dimension
    self.kv = nn.Linear(d_model, d_inner * 2, bias=False) # to project the key and value to the inner dimension

    self.drop = nn.Dropout(dropout)
    self.fc = nn.Linear(self.d_inner, d_model) # to project the output of the multi-head attention to the original dimension
    self.layer_norm = nn.LayerNorm(self.d_model) 
    self.pre_norm = pre_norm

  def forward(self, x, pos_emb, u, v, tgt_mask=None, pad_mask=None, mem=None):
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
    print('prev_seq: ', mem.size() if mem is not None else None)
    print('cur_seq: ', x.size() if x is not None else None)
    
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

    if tgt_mask is not None: # Apply the mask to the attention shape: (batch_size, cur_seq, cur_seq + prev_seq, num_heads)
      print('tgt_mask: ', tgt_mask.size(), attn.size())
      attn = attn.masked_fill(tgt_mask, float('-inf')) # Apply the mask to the attention

    if pad_mask is not None:
      attn = attn.masked_fill(pad_mask, float('-inf'))

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
    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), dtype=x.dtype, device=x.device)
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
  
class CrossAttention(nn.Module):
  def __init__(self, d_model, d_inner, num_heads, dropout=0.1, **kwargs):
    super(CrossAttention, self).__init__(**kwargs)
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_head = d_inner // num_heads
    self.dropout = nn.Dropout(dropout)

    self.q = nn.LazyLinear(d_inner, bias=False)
    self.kv = nn.LazyLinear(d_inner * 2, bias=False)
    self.fc = nn.Linear(d_inner, d_model)

    self.layer_norm = nn.LayerNorm(d_model)


  def forward(self, dec, enc):
    batch = dec.size(0)
    dec_len, enc_len = dec.size(1), enc.size(1)

    q, kv = self.q(dec), self.kv(enc)
    k, v = torch.chunk(kv, 2, dim=-1)
    q = q.view(-1, dec_len, self.num_heads, self.d_head)
    k = k.view(-1, enc_len, self.num_heads, self.d_head)
    v = v.view(-1, enc_len, self.num_heads, self.d_head)

    # Calculate the attention
    attn = torch.einsum('bihd,bjhd->bijh', q, k)
    attn = attn / (self.d_head ** 0.5)
    attn = F.softmax(attn, dim=1)

    attn = self.dropout(attn)
    output = torch.einsum('bijh,bjhd->bihd', attn, v).contiguous().view(batch, dec_len, -1)
    return self.layer_norm(dec + self.fc(output))

class TransformerDecoderLayerRelative(nn.Module):
  def __init__(self, d_model, num_heads, dff, d_inner, dropout=0.1, pre_norm=True, activation='gelu', **kwargs):
    super(TransformerDecoderLayerRelative, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.d_model = d_model
    self.dff = dff
    self.dropout = dropout
    self.pre_norm = pre_norm
    self.d_inner = d_inner

    self.multi_head_attention = RecurrenceAttention(d_model, d_inner, num_heads)
    self.cross_attention = CrossAttention(d_model, d_inner, num_heads)
    fc1 = nn.Linear(d_model, dff)
    activ = nn.GELU() if activation == 'gelu' else nn.ReLU()
    dropout1 = nn.Dropout(dropout)
    fc2 = nn.Linear(dff, d_model)
    dropout2 = nn.Dropout(dropout)

    self.net = nn.Sequential(fc1, activ, dropout1, fc2, dropout2)
    self.layer_norm_1 = nn.LayerNorm(d_model)
    self.layer_norm_2 = nn.LayerNorm(d_model)
    self.layer_norm_3 = nn.LayerNorm(d_model)

  def forward(self, x, enc_output, pos_emb, u, v, mem=None, tgt_mask=None, pad_mask=None):
    if not self.pre_norm:
      out = self.multi_head_attention(x, pos_emb, u, v, tgt_mask=tgt_mask, pad_mask=pad_mask, mem=mem)
      out = self.layer_norm_1(x + out)

      out2 = self.cross_attention(out, enc_output)
      out2 = self.layer_norm_2(out + out2)

      out3 = self.net(out2)
      out3 = self.layer_norm_3(out2 + out3)
    else:
      out = self.layer_norm_1(x)
      out = self.multi_head_attention(out, pos_emb, u, v, tgt_mask=tgt_mask, pad_mask=pad_mask, mem=mem)
      out = x + out

      out2 = self.layer_norm_2(out)
      out2 = self.cross_attention(out2, enc_output)
      out2 = out + out2

      out3 = self.layer_norm_3(out2)
      out3 = self.net(out3)
      out3 = out2 + out3

    return out3
  

class CustomEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model, **kwargs):
    super(CustomEmbedding, self).__init__(**kwargs)
    self.embed = nn.Embedding(vocab_size, d_model)
    self.d_model = d_model
    
  def forward(self, x):
    x = x.long()
    return self.embed(x) * (self.d_model ** 0.5)



class TransformerXL(nn.Module):
  def __init__(self, vocab_size, n_layers, n_heads, d_model, d_inner, dff, seq_len, tie_weights=True, dropout=0.1, pad_id=0, **kwargs):
    super(TransformerXL, self).__init__(**kwargs)
    self.n_tokens = vocab_size
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.d_model = d_model
    self.d_inner = d_inner
    self.dff = dff
    self.seq_len = seq_len
    assert d_inner % n_heads == 0, 'd_inner should be divisible by n_heads'
    self.d_head = d_inner // n_heads
    self.pad_id = pad_id

    self.embed = CustomEmbedding(vocab_size, d_model)
    self.pos_emb = PositionalEmbedding(d_inner)

    self.dropout = nn.Dropout(dropout)
    self.layers = nn.ModuleList()
    for _ in range(n_layers):
      self.layers.append(TransformerDecoderLayerRelative(d_model, n_heads, dff, d_inner, dropout=dropout))

    self.fc = nn.Linear(d_model, vocab_size)
    if tie_weights: # ties the embedding with the weights of the final layer
      self.fc.weight = self.embed.embed.weight

    # Learnable parameters for the relative positional encoding
    self.u = nn.Parameter(torch.Tensor(n_layers, n_heads, self.d_head))
    self.v = nn.Parameter(torch.Tensor(n_layers, n_heads, self.d_head))

    

  def forward(self, x, enc_output, mem=None, out_idx=-1): # out_idx is the index of the output layer to return, because of padding
    batch_size, seq_len = x.size(0), x.size(1)
    prev_seq = mem[0].size(1) if mem is not None else 0

    # attn mask
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len + prev_seq)[...,None], diagonal=1 + prev_seq).bool().to(self.device)
    pad_mask = (x == self.pad_id).unsqueeze(-1).unsqueeze(-1).bool().to(self.device)

    word_emb = self.embed(x)
    pos_idxs = torch.arange(seq_len + prev_seq - 1, -1, -1.0, dtype=torch.float).to(self.device)
    pos_emb = self.pos_emb(pos_idxs)

    # forward through the layers
    new_mem = [word_emb]
    out = word_emb
    for i, layer in enumerate(self.layers):
      lay_mem = None if mem is None else mem[i].detach()
      print('lay_mem: ', lay_mem.size() if lay_mem is not None else None)
      u, v = self.u[i], self.v[i]
      out = layer(out, enc_output, pos_emb, u, v, mem=lay_mem, tgt_mask=tgt_mask, pad_mask=pad_mask)
      print('out: ', out.size())
      new_mem.append(out)
    
    logits = self.fc(self.dropout(out[:, out_idx]))

    return logits, new_mem
  
  def set_device(self, device):
    self.device = device
    self.pos_emb.to(device)
    self.u.to(device)
    self.v.to(device)