import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
import shutil


class IAM(pl.LightningDataModule):
  def __init__(self):
    pass 

  def prepare_data(self):
    pass 

  def setup(self, stage=None):
    pass 

  def train_dataloader(self):
    pass 

  def val_dataloader(self):
    pass 

  def test_dataloader(self):
    pass 

  def distribute_lines(self):
    '''
    Distribute the dataset into train and test sets, merging the train and validation sets.
    '''
    if not os.path.exists('data/train'):
      os.makedirs('data/train')
      self.transfer_line_img(['trainset', 'validationset1', 'validationset2'], 'data/train')
      self.transfer_transcription(['trainset', 'validationset1', 'validationset2'], 'data/train')
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    if not os.path.exists('data/test'):
      os.makedirs('data/test')
      self.transfer_line_img(['testset'], 'data/test')
      self.transfer_transcription(['testset'], 'data/test')

  def transfer_line_img(self, txt_files, path):
    '''
    Get the data in the split from the txt files and save them in the path.
    '''
    count = 0
    for txt in txt_files:
      print(f'Getting images from {txt}')
      with open(f'data/{txt}.txt') as f:
        lines = f.readlines()
        for l in tqdm(lines):
          line = l.rstrip('\n').split('-')
          form_prefix = line[0] # a01, a02, etc.
          form_id = f'{form_prefix}-{line[1]}' # a01-000, a01-001, etc.
          img_idx = line[2] # 0, 1, 2, etc.
          src_file = f'data/lines/{form_prefix}/{form_id}/{form_id}-{img_idx}.png'
          dest_folder = f'{path}/{form_id}'
          os.makedirs(dest_folder, exist_ok=True) # create the folder if it doesn't exist
          shutil.copy(src_file, dest_folder)
          count += 1
    print(f'Copied {count} files.')

  def transfer_transcription(self, txt_files, path):
    '''
    Get the data in the split from the txt files and save them in the path.
    '''
    count = 0
    lines_path = 'data/ascii/lines.txt'
    for txt in txt_files:
      print(f'Getting transcriptions for {txt}')
      with open(lines_path) as f:
        lines = f.readlines()
        while lines[0].startswith('#'):
          lines.pop(0)
        for l in tqdm(lines):
          line = l.rstrip('\n').split(' ')
          transcription = line[-1].replace('|', ' ')
          # print(' ', transcription)
          id = line[0].split('-')
          form_prefix = id[0] # a01, a02, etc.
          form_id = f'{form_prefix}-{id[1]}' # a01-000, a01-001, etc.
          img_idx = id[2] # 0, 1, 2, etc.

          # Check if this image is in this set
          image_path = f'{path}/{form_id}/{form_id}-{img_idx}.png'
          dest_path = f'{path}/{form_id}/{form_id}-{img_idx}.txt'
          if not os.path.exists(image_path) or os.path.exists(dest_path):
            continue

          with open(dest_path, 'w') as f:
            f.write(transcription)
          count += 1

    print(f'Copied {count} transcriptions.')


iam = IAM()
iam.distribute_lines()
