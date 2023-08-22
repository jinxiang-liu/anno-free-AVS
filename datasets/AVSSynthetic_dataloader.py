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
import random
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('..')

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


class SyntheticDataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train', args=None):
        super(SyntheticDataset, self).__init__()
        split = "validate" if split == 'val' else split
        self.split = split 
        self.args = args
        self.mask_num = 1   

        ann_file = args.config['AVSSynthetic']['ANNO_FILE']
        self.image_dir = args.config['AVSSynthetic']['IMAGE_DIR']
        self.audio_dir = args.config['AVSSynthetic']['AUDIO_DIR']
        self.mask_dir = args.config['AVSSynthetic']['MASK_DIR']
        VGGSound_dict_path = args.config['AVSSynthetic']['VGGSOUND_DICT_PATH']
        
        with open(VGGSound_dict_path,'rb') as f:
            self.vggsound_dict = pickle.load(f)
        
        
        df_split_all = pd.read_csv(ann_file)

        self.df_split = df_split_all[df_split_all['split']== split] 

        # self.df_split = self.df_split[:20]
        
        if args.local_rank == 0:
            print("{} images are used for {}".format(len(self.df_split),  self.split))
        
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
        one_item = self.df_split.iloc[index]

        if self.split == 'train':
            inst_id, img_id, _, _ , cls_name, cls_id, datatset, _,_ = one_item
            audio_name = None
        elif self.split in ['test','validate']:
            inst_id, img_id, _, _, cls_name, cls_id, datatset, _, audio_name = one_item

        cls_id = int(cls_id)
        category = cls_name
        video_name = img_id

        img_path = os.path.join(self.image_dir, img_id +'.jpg')
        mask_path = os.path.join(self.mask_dir,  inst_id + '.png')

        if audio_name is None:
            audio_name = random.choice(self.vggsound_dict[cls_id])

        audio_path = os.path.join(self.audio_dir, audio_name + '.wav')

        img_tensor, _ = load_image_in_PIL_to_Tensor(img_path, transform=self.img_transform)
        masks_tensor, _ = load_image_in_PIL_to_Tensor(mask_path, transform=self.mask_transform, mode='1')
        audio_log_mel = vggish_input.wavfile_to_examples(audio_path) 
        spectrogram = audio_log_mel[1].clone().detach().unsqueeze(dim=0) 
        imgs_tensor = img_tensor.unsqueeze(dim=0)
        masks_tensor = masks_tensor.unsqueeze(dim=0)

        return imgs_tensor, spectrogram, masks_tensor, category, video_name

        

    def __len__(self):
        return len(self.df_split)



