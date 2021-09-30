import torch
import torchaudio

import numpy as np

# + colab={} colab_type="code" id="cZ0wb9EN5i9f"
class SpeechDataset:
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16):
        super().__init__()
        # list of files

        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        self.sample_rate = 48000
        
        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.len_ = len(self.noisy_files)
        
        # fixed len
        self.max_len = 165000

    def continue_init(self, sample_frames):
        self.max_len = sample_frames

    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return torch.squeeze(waveform).numpy()
  
    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])

        
        return x_clean, x_noisy
