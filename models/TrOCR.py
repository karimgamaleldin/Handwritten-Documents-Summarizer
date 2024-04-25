from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
from PIL import Image

path = 'data/train/a01-000u/a01-000u-00.png'
img = Image.open(path).convert("RGB")

# create the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") # contains the image processor and the image tokenizer
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

pixel_values = processor(img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)
plt.imshow(pixel_values[0].permute(1, 2, 0))
plt.show()
