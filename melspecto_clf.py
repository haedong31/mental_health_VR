import torch
from torch.utils.data import Dataset, DataLoader

import torchaudio

import matplotlib.pyplot as plt

from clearml import Task

# connect ClearML agent
task = Task.init(project_name="VR Mental Health Clinic", 
    take_name="Audio classification using Mel spectogram")
config_dict = {"num_of_epochs": 6, "batch_size": 8, "drop_out": 0.25,
    "base_lr": 0.005, "num_of_mel_filters": 64, "resample_freq": 22050}
config_dict = task.connect(config_dict)
print(config_dict)

sound_data, sample_rate = torchaudio.load("./001-0.wav")
resample_transform = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=22050)
audio_mono = torch.mean(resample_transform(sound_data), dim=0, keepdim=True)

plt.figure()
plt.plot(audio_mono[0,:])

# dataset class
class CookieDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    