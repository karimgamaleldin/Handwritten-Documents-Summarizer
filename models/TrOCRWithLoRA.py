import torch
import torch.nn as nn
from transformers import TrOCRConfig, VisionEncoderDecoderModel


class LoRALayer(nn.Module):
    def __init__(self, rank, lin, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.lin = lin
        
        self.A = nn.Parameter(torch.randn(self.lin.in_features, self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, self.lin.out_features))


    def forward(self, x):
        h = self.lin(x)
        h += x @ (self.A @ self.B) * self.alpha
        return h
        


class LoRATransformerLayer(nn.Module):
    def __init__(self, layer, rank, decoder=True):
        super(LoRATransformerLayer, self).__init__()
        self.layer = layer
        
        if decoder:
            # Self_attention
            self.layer.self_attn.k_proj = LoRALayer(rank, self.layer.self_attn.k_proj)
            self.layer.self_attn.v_proj = LoRALayer(rank, self.layer.self_attn.v_proj)
            self.layer.self_attn.q_proj = LoRALayer(rank, self.layer.self_attn.q_proj)
            self.layer.self_attn.out_proj = LoRALayer(rank, self.layer.self_attn.out_proj)

            # Cross_attention
            self.layer.encoder_attn.k_proj = LoRALayer(rank, self.layer.encoder_attn.k_proj)
            self.layer.encoder_attn.v_proj = LoRALayer(rank, self.layer.encoder_attn.v_proj)
            self.layer.encoder_attn.q_proj = LoRALayer(rank, self.layer.encoder_attn.q_proj)
            self.layer.encoder_attn.out_proj = LoRALayer(rank, self.layer.encoder_attn.out_proj)
        else:
            self.layer.attention.attention.query = LoRALayer(rank, self.layer.attention.attention.query)
            self.layer.attention.attention.key = LoRALayer(rank, self.layer.attention.attention.key)
            self.layer.attention.attention.value = LoRALayer(rank, self.layer.attention.attention.value)
            self.layer.attention.output.dense = LoRALayer(rank, self.layer.attention.output.dense)

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class TrOCRWithLoRA(nn.Module):
    def __init__(self, model='microsoft/trocr-base-stage1', lora_rank=8, processor=None):
        super(TrOCRWithLoRA, self).__init__()
        # Replace the decoder layers with LoRA layers
        self.processor = processor
        self.pre_trained = self.get_pretrained(model)
        
        for name, param in self.named_parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.pre_trained.decoder.model.decoder.layers):
            self.pre_trained.decoder.model.decoder.layers[i] = LoRATransformerLayer(
                layer, lora_rank
            )
            
        for i, layer in enumerate(self.pre_trained.encoder.encoder.layer):
            self.pre_trained.encoder.encoder.layer[i] = LoRATransformerLayer(
                layer, lora_rank, decoder=False
            )
            
                
    def forward(self, img, decoder_input):
        ret = self.pre_trained(img, decoder_input)
        return ret
    
    def generate(self, img):
        return self.pre_trained.generate(img)
    
    def get_pretrained(self, model='microsoft/trocr-base-stage1'):
        model = VisionEncoderDecoderModel.from_pretrained(model)
        model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        model.config.max_length = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
        return model
        