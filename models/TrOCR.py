import torch
import torch.nn as nn
from transformers import TrOCRConfig, VisionEncoderDecoderModel


class LoRALayer(nn.Module):
    def __init__(self, config, rank, lin, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.config = config
        self.alpha = alpha
        self.lin = lin

        self.A = nn.Parameter(torch.randn(self.config.d_model, rank))
        self.B = nn.Parameter(torch.randn(rank, self.config.d_model))

    def forward(self, x):
        h = self.lin(x)
        h += x @ (self.A @ self.B) * self.alpha
        


class LoRATransformerLayer(nn.Module):
    def __init__(self, layer, config, rank):
        super(LoRATransformerLayer, self).__init__()
        self.layer = layer
        
        # Self_attention
        self.layer.self_attn.k_proj = LoRALayer(config, rank, self.layer.self_attn.k_proj)
        self.layer.self_attn.v_proj = LoRALayer(config, rank, self.layer.self_attn.v_proj)
        self.layer.self_attn.q_proj = LoRALayer(config, rank, self.layer.self_attn.q_proj)
    
        # Cross_attention
        self.layer.encoder_attn.k_proj = LoRALayer(config, rank, self.layer.encoder_attn.k_proj)
        self.layer.encoder_attn.v_proj = LoRALayer(config, rank, self.layer.encoder_attn.v_proj)
        self.layer.encoder_attn.q_proj = LoRALayer(config, rank, self.layer.encoder_attn.q_proj)

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


class TrOCRWithLoRA(nn.Module):
    def __init__(self, model='microsoft/trocr-base-stage1', config=TrOCRConfig(), lora_rank=32, processor=None):
        super(TrOCRWithLoRA, self).__init__()
        # Replace the decoder layers with LoRA layers
        self.pre_trained = self.get_pretrained(model)
        self.config = config
        self.processor = processor
        for i, layer in enumerate(self.pre_trained.decoder.model.decoder.layers):
            self.pre_trained.decoder.model.decoder.layers[i] = LoRATransformerLayer(
                layer, self.config, lora_rank
            )

        for name, param in self.named_parameters():
            if "attn" not in name:
                param.requires_grad = False
                
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
        model.config.eos_token_id = self.tokenizer.sep_token_id
        model.config.max_length = 64
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
        return model
        