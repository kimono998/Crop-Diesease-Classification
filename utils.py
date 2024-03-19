from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from sklearn.model_selection import train_test_split

import os

def clean_ds(IMAGE_LABEL_PATH, IMG_PATH):
    df = pd.read_csv(IMAGE_LABEL_PATH)
    df_img_list = df['image_id'].tolist()
    img_list = os.listdir(IMG_PATH)


    missing = []
    for i, n in enumerate(df_img_list):
        if n not in img_list:
            missing.append(i)


    new_df = df.drop(index=missing).reset_index(drop=True)
    new_df.to_csv('data/Data/reindexed_train.csv', index = False)

def resize_img(input_path, img_list, output_path, target_size):


    for im in tqdm(img_list, desc = 'resizing images'):
        img = Image.open(f'{input_path}/{im}')
        downsized = img.resize(target_size, Image.ANTIALIAS)
        downsized.save(f'{output_path}/{im}')
        img.close()


def to_grayscale(input_path, img_list, output_path):

    for im in tqdm(img_list, desc='converting to grayscale'):
        img = Image.open(f'{input_path}/{im}')
        grayscale = img.convert('L')
        grayscale.save(f'{output_path}/{im}')
        img.close()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def split_dataset(img_dir, csv_path, bs):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    random_seed = 42
    dataset = CustomDataset(img_dir=img_dir, csv_file=csv_path, transform=transform)
    num_data = len(dataset)
    indices = list(range(num_data))

    train_i, test_i = train_test_split(indices, test_size=0.2, random_state=random_seed, stratify=dataset.df['label'])
    train_i, val_i = train_test_split(train_i, test_size=0.25, random_state=random_seed, stratify=dataset.df.iloc[train_i]['label'])

    train_sampler = SubsetRandomSampler(train_i)
    val_sampler = SubsetRandomSampler(val_i)
    test_sampler = SubsetRandomSampler(test_i)

    train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=bs, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=bs, sampler=test_sampler)

    return train_loader, val_loader, test_loader



