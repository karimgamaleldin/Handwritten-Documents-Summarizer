import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, GenerationConfig
from PIL import Image

class Recognizer:
  def __init__(self, model='microsoft/trocr-base-handwritten', processor='microsoft/trocr-base-handwritten'):
    self.processor_name = processor
    self.model_name = model
    self.processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(processor)
    self.model = VisionEncoderDecoderModel.from_pretrained(model)
    self.set_model_configs()


  def generate(self, img):
    inputs = self.processor(img, return_tensors="pt")
    outputs = self.model.generate(**inputs)
    predicted_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text
  
  def set_model_configs(self, gen_configs=None):
    if gen_configs is None:
      generation_config = GenerationConfig(
      max_length=64,
      early_stopping=True,
      num_beams=3,
      length_penalty=2.0,
      no_repeat_ngram_size=3  
      )
    
    self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
    self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
    self.model.config.vocab_size = self.model.config.decoder.vocab_size
    self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
  


