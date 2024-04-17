# OCR-Summarizer 👀

## Motivation
This projects aim to create an Optical Character Recognizer and Summarizer for hand-written text training on the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

## Descriptions 🖊️
This projects takes inspirations from well-known papers to implement, create and deploy an multi-modal OCR-Summarizer model.

1. OCR encoder - A transformer-based vision model inspired by [ViT](https://arxiv.org/abs/2010.11929) architecture and [CvT](https://arxiv.org/abs/2103.15808v1) (implemented from scratch using pytorch)
2. OCR decoder - A transformer-based text model inspired by [Transformer-XL](https://arxiv.org/abs/1901.02860) architecture (implemented from scratch using pytorch)
3. Text summarizer - a pretrained large language model (LLM) obtained from [HuggingFace models](https://huggingface.co/models?pipeline_tag=summarization&sort=trending)

The deployed version of the project can be found on [OCR-Summarizer]()

## Tech Stack 💻
- Data pre-processing: NumPy, OpenCV, Pytorch, Albumentations, HuggingFace's Tokenizers
- Model development: Pytorch, Pytorch lightning, HuggingFace's Transformers
- Training: Kaggle
- Experiment Tracking: Weights & Biases
## Credits

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808v1)
