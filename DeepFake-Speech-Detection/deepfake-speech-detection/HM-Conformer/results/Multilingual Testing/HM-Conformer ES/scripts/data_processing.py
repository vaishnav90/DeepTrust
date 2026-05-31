"""
Data Processing Module for HM-Conformer

This module provides TrainSet and TestSet classes that work with both:
- OLD: ASVspoof2021_DF_LA dataset (via DF_Item objects)
- NEW: MultilingualDataset (via DF_Item objects)

Both datasets provide DF_Item objects with:
- path: path to audio file
- label: 0 for real, 1 for fake
- attack_type: model/speaker name or attack type
- is_fake: boolean indicating if sample is fake

The TrainSet and TestSet classes are compatible with both datasets.
"""

import sys
import random
import numpy as np
import soundfile as sf

from torch.utils.data import Dataset

from egg_exp.data.augmentation import *

class TrainSet(Dataset):
    def __init__(self, items, crop_size, DA_p, DA_list, DA_params):
        self.items = items
        self.crop_size = crop_size
        self.DA_p = DA_p
        self.DA_list = DA_list
        self.sr = 16000
        self.DA = {}
        
        for da in DA_list:
            if da == 'MUS':
                self.DA['MUS'] = Musan(
                    DA_params['MUS']['path']
                )
                self.category = ['noise','speech','music']
            elif da == 'RIR':
                self.DA['RIR'] = RIRReverberation(
                    DA_params['RIR']['path']
                )
            else:
                raise ValueError(f'There is no data augmentation: {da}')
        
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        wav = rand_crop_read(item.path, self.crop_size)
        
        for da in self.DA_list:
            p = random.random()
            if p < self.DA_p:
                if da == 'MUS':
                    category = random.choice(self.category)
                    wav = self.DA[da](wav, category)
                else:
                    wav = self.DA[da](wav)
        
        return wav, item.label
    
class TestSet(Dataset):
    def __init__(self, items, crop_size):
        self.items = items
        self.crop_size = crop_size
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # sample
        item = self.items[index]
        
        # read wav
        wav, _ = sf.read(item.path)
        if wav.shape[0] < self.crop_size:
            shortage = self.crop_size - wav.shape[0]
            wav = np.pad(wav, (0, shortage), 'wrap')
        else:
            wav = wav[:self.crop_size]

        return wav, item.label