import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from .tokenizer.my_tokenizer import MyTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .custom_augmentation import Erosion, Dilation

"""
The Datamodule class containing the lighting data module and the torch dataset for the IAM dataset.
"""


class IAM(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        tokenizer_path: str = None,
        *,
        distribute_data: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        transform=None,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.distribute_data = distribute_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer_path = tokenizer_path
        if transform is None:
            self.transform = A.Compose(
                [
                    A.Rotate(limit=10, p=0.5),
                    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                    Dilation(p=0.5),
                    Erosion(p=0.5),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = transform

    def prepare_data(self):
        """
        Loads the dataset to disk
        """
        if self.distribute_data:
            self.distribute_lines()

        self.train_dataset = IAMDataset(self.train_path, transform=self.transform)
        if self.tokenizer_path is None:
            self.tokenizer = MyTokenizer()
            self.tokenizer.train(self.train_dataset.transcriptions)
        else:
            self.tokenizer = MyTokenizer(self.tokenizer_path)

    def setup(self, stage=None):
        """
        Split the dataset into train and test sets and initialize datasets
        """
        if stage == "fit" or stage is None:
            self.train_dataset = IAMDataset(self.train_path, transform=self.transform)
            self.train_dataset.set_tokenizer(self.tokenizer)
        if stage == "test" or stage is None:
            self.test_dataset = IAMDataset(self.test_path)
            self.test_dataset.set_tokenizer(self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def distribute_lines(self):
        """
        Distribute the dataset into train and test sets, merging the train and validation sets.
        """
        if not os.path.exists("data/train"):
            os.makedirs("data/train")
            self.transfer_line_img(
                ["trainset", "validationset1", "validationset2"], "data/train"
            )
            self.transfer_transcription(
                ["trainset", "validationset1", "validationset2"], "data/train"
            )
            print(
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            )

        if not os.path.exists("data/test"):
            os.makedirs("data/test")
            self.transfer_line_img(["testset"], "data/test")
            self.transfer_transcription(["testset"], "data/test")

    def transfer_line_img(self, txt_files, path):
        """
        Get the data in the split from the txt files and save them in the path.
        """
        count = 0
        for txt in txt_files:
            print(f"Getting images from {txt}")
            with open(f"data/{txt}.txt") as f:
                lines = f.readlines()
                for l in tqdm(lines):
                    line = l.rstrip("\n").split("-")
                    form_prefix = line[0]  # a01, a02, etc.
                    form_id = f"{form_prefix}-{line[1]}"  # a01-000, a01-001, etc.
                    img_idx = line[2]  # 0, 1, 2, etc.
                    src_file = (
                        f"data/lines/{form_prefix}/{form_id}/{form_id}-{img_idx}.png"
                    )
                    dest_dir = f"{path}/{form_id}"
                    dest_file = f"{dest_dir}/{form_id}-{img_idx}.png"
                    os.makedirs(dest_dir, exist_ok=True)
                    self.load_save_image(src_file, dest_file)
                    count += 1
        print(f"Copied {count} files.")

    def transfer_transcription(self, txt_files, path):
        """
        Get the data in the split from the txt files and save them in the path.
        """
        count = 0
        lines_path = "data/ascii/lines.txt"
        for txt in txt_files:
            print(f"Getting transcriptions for {txt}")
            with open(lines_path) as f:
                lines = f.readlines()
                while lines[0].startswith("#"):
                    lines.pop(0)
                for l in tqdm(lines):
                    line = l.rstrip("\n").split(" ")
                    transcription = line[-1].replace("|", " ")
                    # print(' ', transcription)
                    id = line[0].split("-")
                    form_prefix = id[0]  # a01, a02, etc.
                    form_id = f"{form_prefix}-{id[1]}"  # a01-000, a01-001, etc.
                    img_idx = id[2]  # 0, 1, 2, etc.

                    # Check if this image is in this set
                    image_path = f"{path}/{form_id}/{form_id}-{img_idx}.png"
                    dest_path = f"{path}/{form_id}/{form_id}-{img_idx}.txt"
                    if not os.path.exists(image_path) or os.path.exists(dest_path):
                        continue

                    with open(dest_path, "w") as f:
                        f.write(transcription)
                    count += 1

        print(f"Copied {count} transcriptions.")

    def load_save_image(self, src, dest):
        img = cv2.imread(src)
        img = cv2.resize(img, (384, 384))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(dest, binary)


class IAMDataset(Dataset):
    def __init__(self, path: str, transform=None, context_size: int = 100):
        self.path = path
        self.transform = transform
        self.context_size = context_size
        self.get_data_from_dirs()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        img = np.expand_dims(img, axis=-1)
        if self.transform:
            img = self.transform(image=img)["image"]

        transcription = np.array(
            self.tokenizer.encode(self.transcriptions[idx]).ids, dtype=np.int32
        )
        return img, self.pad_transcription(transcription)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_data_from_dirs(self):
        """
        Get the datasets in an array format from the dirs.
        """
        print("Getting data from dirs...")
        self.imgs = []
        self.transcriptions = []
        self.ids = []
        for dir in os.listdir(self.path):
            path = os.path.join(self.path, dir).replace("\\", "/")
            for img_path in os.listdir(path):
                if img_path.endswith(".png"):
                    transcription_path = os.path.join(
                        path, img_path.replace(".png", ".txt")
                    ).replace("\\", "/")
                    img_path = os.path.join(path, img_path).replace("\\", "/")
                    self.imgs.append(img_path)
                    with open(transcription_path) as f:
                        transcription = f.read()
                        self.transcriptions.append(transcription)

    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        return img

    def pad_transcription(self, transcription):
        """
        Pad the transcription to the context size.
        """
        if len(transcription) < self.context_size:
            zeros = np.zeros(self.context_size - len(transcription), dtype=np.int32)
            return np.concatenate((transcription, zeros))


# iam = IAM("data/train", "data/test", distribute_data=True)
# iam.prepare_data()

# iam = IAMDataset('data/train')
# sent = iam.transcriptions

# max = 0
# min = 93
# for s in sent:
#   if len(s) > max:
#     max = len(s)
#   if len(s) < min:
#     min = len(s)

# print(max, min)
# iam.set_tokenizer(MyTokenizer(path='tokenizer.json'))
# print(iam[0][0].dtype, iam[0][1].dtype)
# print(iam[0][0].shape, iam[0][1].shape)
# print(iam[0][1])

# print(type(iam[0][0]))
# print(type(iam[0][1]))
