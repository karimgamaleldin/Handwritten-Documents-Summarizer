import torch
import wandb
import argparse
from configs import *
from IAM import IAM
from models.LitOCR import LitOCR
from models.vision_encoder import VisionEncoder
from models.vanilla_decoder import VanillaDecoder
from callbacks import ExampleLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tokenizer.my_tokenizer import MyTokenizer

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a model')
  parser.add_argument('project_name', type=str, help='Project for wandb logging')
  args = parser.parse_args()
  project_name = args.project_name

  wandb_logger = WandbLogger(project=project_name)


  # Data module
  print('Preparing data...')
  iam = IAM('data/train', 'data/test', tokenizer_path=TOKENIZER_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
  iam.prepare_data()
  iam.setup()
  val_samples = next(iter(iam.val_dataloader()))
  print(len(val_samples))
  print(val_samples[0].shape)
  print(iam.train_dataset[0][0].shape)

  # Model 
  print('Creating model...')
  vision_encoder = VisionEncoder(**VISION_CONFIGS)
  print('Maximum position encoding:', vision_encoder.maximum_position_encoding)
  vanilla_decoder = VanillaDecoder(**VANILLA_DECODER_CONFIGS)
  tokenizer = MyTokenizer(TOKENIZER_PATH)
  model = LitOCR(vision_encoder, vanilla_decoder, tokenizer=tokenizer, lr=LEARNING_RATE, context_length=CONTEXT_LENGTH, schedule_sample=False)
  # Callbacks
  print('Creating callbacks...')
  callbacks = []

  example_logger = ExampleLogger(val_samples, num_samples=BATCH_SIZE)
  callbacks.append(example_logger)

  model_checkpoint = ModelCheckpoint(monitor='val_cer', mode='min', save_top_k=5, save_last=True, dirpath='checkpoints/model_1',  filename='model-{epoch:02d}-{val_cer:.2f}', verbose=True)
  callbacks.append(model_checkpoint)

  print('Training model...')
  # Trainer
  cuda = 'gpu' if torch.cuda.is_available() else 'cpu'
  print(f'Using {cuda}...')
  print(f'Dropout: {model.decoder.dropout}') 
  trainer = Trainer(callbacks=callbacks, max_epochs=EPOCHS, deterministic=DETERMINISTIC, accelerator=cuda, devices=1, precision='32-true', logger=wandb_logger)

  # Train
  trainer.fit(model, iam)
  
  wandb.finish()



