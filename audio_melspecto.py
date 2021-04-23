import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchaudio
import torchaudio.transforms as transforms
from torchvision import models

from clearml import Task

## connect ClearML agent -----
task = Task.init(project_name="VR Mental Health Clinic", 
    take_name="Audio classification using Mel spectogram")
config_dict = {"num_epochs": 6, "batch_size": 8, "drop_out": 0.25,
    "base_lr": 0.005, "num_mel_filters": 32, "resample_freq": 22050}
config_dict = task.connect(config_dict)

## dataset class -----
class CookieAudioDataset(Dataset):
    def __init__(self, meta_csv_path, return_audio=False):
        self.audio_file_paths = []
        self.labels = []
        self.num_mel_filters = config_dict.get("num_mel_filters")
        self.resample_freq = config_dict.get("resample_freq")
        self.return_audio = return_audio

        meta_data = pd.read_csv(meta_csv_path)
        for i in range(0, len(meta_data)):
            self.audio_file_paths.append(meta_data.iloc[i,0])
            self.labels.append(meta_data.iloc[i,1])

    def __getitem__(self, idx):
        # read data and resample
        sound_data, sample_rate = torchaudio.load(self.audio_file_paths[idx])
        if self.resample_freq != 0: # no resample is resample_freq=0
            resampler = transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.resample_freq)
            sound_data = resampler(sound_data)

        # mono channel
        sound_data = torch.mean(sound_data, dim=0, keepdim=True)
        
        # spectrogram
        mel_spectro_transform = transforms.MelSpectrogram(
            sample_rate=self.resample_freq, 
            n_mels=self.num_mel_filters)
        decibel_transform = transforms.AmplitudeToDB()

        mel_spectro = mel_spectro_transform(sound_data)
        mel_spectro_db = decibel_transform(mel_spectro)


    def __len__(self):
        return len(self.audio_file_paths)
