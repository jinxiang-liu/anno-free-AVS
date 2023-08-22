import os
from wave import _wave_params
import torch
import torch.nn as nn

import warnings
warnings.simplefilter("ignore", UserWarning)

from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as audio_T
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('..')
# from configs.avsb_config import cfg
from torchvggish import vggish_input

import pdb
import ipdb
from tqdm import tqdm


def load_mask_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        # return img_tensor
    return img_tensor, img_PIL



def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', args=None):
        super(S4Dataset, self).__init__()
        self.split = split
        self.args = args
        self.mask_num = 1 if self.split == 'train' else 5
        
        # Using part of the training dataset
        if (self.split == 'train') and args.trainset_shuffle:
            df_all = pd.read_csv(args.config['AVSBenchS4']['ANNO_CSV_SHUFFLE'], sep=',')
            self.df_split = df_all[df_all['split'] == split]
            len_set = (len(self.df_split)) * args.trainset_ratio
            len_set = int(len_set)
            self.df_split = self.df_split[:len_set]

        else:
            df_all = pd.read_csv(args.config['AVSBenchS4']['ANNO_CSV'], sep=',')
            self.df_split = df_all[df_all['split'] == split]
            self.df_split = self.df_split  #[:2]

        if args.local_rank == 0:
            print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.args.inp_size, self.args.inp_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.args.inp_size, self.args.inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        self.AmplitudeToDB = audio_T.AmplitudeToDB()


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(self.args.config['AVSBenchS4']['DIR_IMG'], self.split, category, video_name)
        mask_base_path = os.path.join(self.args.config['AVSBenchS4']['DIR_MASK'], self.split, category, video_name)
        audio_lm_path = os.path.join(self.args.config['AVSBenchS4']['DIR_AUDIO_LOG_MEL'], self.split, category, video_name + '.pkl')
        audio_log_mel = load_audio_lm(audio_lm_path)           # torch.Tensor: [5,1,96,64]

        imgs, masks, audios, img_PILs = [], [], [], []
        for img_id in range(1, self.mask_num + 1):
            img, img_PIL = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
            img_PILs.append(np.array(img_PIL))
            audio = audio_log_mel[img_id-1]
            audios.append(audio)
            
        for mask_id in range(1, self.mask_num + 1):
            mask = load_mask_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        
        imgs_tensor = torch.stack(imgs, dim=0)     # torch.Size([5, 3, 224, 224])
        masks_tensor = torch.stack(masks, dim=0)    # torch.Size([5, 1, 224, 224])
        spectrogram = torch.stack(audios, dim=0)

        return imgs_tensor, spectrogram, masks_tensor, category, video_name

        

    def __len__(self):
        return len(self.df_split)



