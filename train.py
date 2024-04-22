import torch
import wandb
import argparse
from configs import *
import pytorch_lightning
from IAM import IAM
from models.LitOCRXL import LitOCRXL
from callbacks import ExampleLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from tokenizer.my_tokenizer import MyTokenizer


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a model')
  parser.add_argument('project_name', type=str, help='Project for wandb logging')
  args = parser.parse_args()
  project_name = args.project_name

  # wandb_logger = WandbLogger(project=project_name)


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
  tokenizer = MyTokenizer(TOKENIZER_PATH)
  model = LitOCRXL(VISION_CONFIGS, DECODER_CONFIGS, tokenizer=tokenizer, lr=LEARNING_RATE, context_length=CONTEXT_LENGTH, metric_teacher_force=METRIC_TEACHER_FORCE, minimize_metric=MINIMIZE_METRIC, teacher_forcing_ratio=TEACHER_FORCING_RATIO, metric=METRIC)

  # Callbacks
  print('Creating callbacks...')
  callbacks = []
  # callbacks.append(ExampleLogger(val_samples, num_samples=BATCH_SIZE))

  print('Training model...')
  # Trainer
  cuda = 'gpu' if torch.cuda.is_available() else 'cpu'
  print(f'Using {cuda}...')
  trainer = Trainer(callbacks=callbacks, max_epochs=EPOCHS, deterministic=DETERMINISTIC, accelerator=cuda, devices=1, precision='16-mixed')

  # Train
  trainer.fit(model, iam)
  
  wandb.finish()



