import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize
import shutil 
from my_tokenizer import MyTokenizer

'''
The Datamodule class containing the lighting data module and the torch dataset for the IAM dataset.
'''
class IAM(pl.LightningDataModule):
  def __init__(self, train_path:str, test_path:str, tokenizer_path:str = None, *, distribute_data: bool = False, batch_size: int = 32, num_workers: int = 4):
    self.train_path = train_path
    self.test_path = test_path
    self.distribute_data = distribute_data
    self.batch_size = batch_size
    self.num_workers = num_workers 
    self.tokenizer_path = tokenizer_path

  def prepare_data(self):
    '''
    Loads the dataset to disk
    '''
    if self.distribute_data:
      self.distribute_lines() 

  def setup(self, stage=None):
    '''
    Split the dataset into train and test sets and initialize datasets
    '''
    if stage == 'fit' or stage is None:
      # load train dataset
      self.train_dataset = IAMDataset(self.train_path)
      if self.tokenizer_path is None:
        self.tokenizer = MyTokenizer()
        self.tokenizer.train(self.train_dataset.transcriptions)
      else:
        self.tokenizer = MyTokenizer(self.tokenizer_path)
      
      self.train_dataset.set_tokenizer(self.tokenizer)
    if stage == 'test' or stage is None: 
      self.test_dataset = IAMDataset(self.test_path)
      self.test_dataset.set_tokenizer(self.tokenizer)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True) 


  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers) 

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

class IAMDataset(Dataset):
  def __init__(self, path:str, transform=None):
    self.path = path 
    self.transform = transform
    self.get_data_from_dirs()

  def __len__(self):
    return len(self.imgs) 

  def __getitem__(self, idx):
    img = self.load_image(self.imgs[idx])
    transcription = np.array(self.tokenizer.encode(self.transcriptions[idx]).ids, dtype=np.int32)
    return img, transcription
  
  def set_tokenizer(self, tokenizer):
    self.tokenizer = tokenizer

  def get_data_from_dirs(self):
    '''
    Get the datasets in an array format from the dirs.
    '''
    print('Getting data from dirs...')
    self.imgs = []
    self.transcriptions = []
    self.ids = []
    for dir in os.listdir(self.path):
      path = os.path.join(self.path, dir).replace('\\', '/')
      for img_path in os.listdir(path):
        if img_path.endswith('.png'):
          transcription_path = os.path.join(path, img_path.replace('.png', '.txt')).replace('\\', '/')
          img_path = os.path.join(path, img_path).replace('\\', '/')
          self.imgs.append(img_path)
          with open(transcription_path) as f:
            transcription = f.read()
            self.transcriptions.append(transcription)

  def load_image(self, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img
  
iam = IAMDataset('data/train')
sent = iam.transcriptions

max = 0
min = 93
for s in sent:
  if len(s) > max:
    max = len(s)
  if len(s) < min:
    min = len(s)

print(max, min)
iam.set_tokenizer(MyTokenizer(path='tokenizer.json'))
print(iam[0][0].dtype, iam[0][1].dtype)

print(type(iam[0][0]))
print(type(iam[0][1]))