from transformers import VisionEncoderDecoderModel, TrOCRProcessor, GenerationConfig

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
  
  def set_model_configs(self):
    self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
    self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
    self.model.config.vocab_size = self.model.config.decoder.vocab_size
    self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
  


