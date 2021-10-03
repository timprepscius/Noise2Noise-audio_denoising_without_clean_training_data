import torch
import numpy as np

def log_(*args):
    # print(*args)
    pass

def stft_torch(data, window, step):
    log_(f"stft data min_max {np.min(data)} {np.max(data)}")

    result = torch.stft(
        input=torch.from_numpy(data), 
        n_fft=window, 
        hop_length=step, 
        normalized=True,
        return_complex=True
    )

    return result

def istft_torch(data, window, step):
    log_(f"stft data shape {data.shape}")

    result = torch.istft(
        input=data, 
        n_fft=window, 
        hop_length=step, 
        normalized=True,
        # return_complex=True
    )

    return result

def stft_scipy(data):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
    pass

def stft_compute(data, window, step):
    return stft_torch(data, window, step)

def stft_inverse(data, window, step):
    return istft_torch(data, window, step)

class STFT:
    def __init__(self, window, step):
        self.window = window
        self.step = step

    def compute(self, data):
        return stft_compute(data, self.window, self.step)

    def inverse(self, data):
        return stft_inverse(data, self.window, self.step)

    def to_magnitude(self, complex):
        # magnitude = torch.unsqueeze(torch.abs(complex), axis=-1)
        magnitude = torch.abs(complex)
        # magnitude = torch.unsqueeze(magnitude, axis=1)
        return magnitude

    def to_phase(self, complex):
        phase = torch.angle(complex)
        # phase = torch.unsqueeze(torch.angle(complex), axis=-1)
        return phase        

    def to_complex(self, magnitude, phase):
        complex = magnitude * torch.exp(1j * phase)
        log_(f"complex.shape {complex.shape}")
        return complex        

    def to_complex_zero_phase(self, v):
        i = torch.zeros_like(v)
        return self.to_complex(v, i)
