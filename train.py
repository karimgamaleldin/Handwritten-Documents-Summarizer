import torch
import wandb
import argparse
from configs import *
import pytorch_lightning
from IAM import IAM
from models.my_ocr import LitOCR
from callbacks import ExampleLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a model')
  parser.add_argument('project_name', type=str, help='Project for wandb logging')
  args = parser.parse_args()
  project_name = args.project_name

  wandb_logger = WandbLogger(project=project_name)

  # Data module
  iam = IAM('data/train', 'data/test', tokenizer_path=TOKENIZER_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
  iam.prepare_data()
  iam.setup()
  val_samples = next(iter(iam.val_dataloader()))

  # Model 
  model = LitOCR(VISION_CONFIGS, DECODER_CONFIGS, lr=LEARNING_RATE, context_length=CONTEXT_LENGTH, eos_id=EOS_ID, pad_id=PAD_ID, metric_teacher_force=METRIC_TEACHER_FORCE, minimize_metric=MINIMIZE_METRIC, teacher_forcing_ratio=TEACHER_FORCING_RATIO, tokenizer_path=TOKENIZER_PATH, metric=METRIC)

  # Callbacks
  callbacks = []
  callbacks.append(ExampleLogger(val_samples, num_samples=BATCH_SIZE))


  # Trainer
  cuda = -1 if torch.cuda.is_available() else 0
  trainer = Trainer(logger=wandb_logger, callbacks=callbacks, max_epochs=EPOCHS, deterministic=DETERMINISTIC, gpus=cuda, )

  # Train
  trainer.fit(model, iam)

  # Test
  trainer.test(datamodule=iam, 
               ckpt_path=None) # To use last saved model
  
  wandb.finish()



