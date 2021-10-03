import torch
import numpy as np

from stft import STFT

# + colab={} colab_type="code" id="cZ0wb9EN5i9f"

def log_(*args):
    # print(*args)
    pass

class AudioToSTFT:

    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, generator, image_size=(64, 2049)):
        super().__init__()

        self.generator = generator
        sample_rate = generator.sample_rate
        self.fft_window_size = (sample_rate * 64) // 1000 # 64 ms
        self.fft_window_step = (sample_rate * 16) // 1000 # 16 ms
        self.image_size = image_size

        self.stft = STFT(self.fft_window_size, self.fft_window_step)

        strangeness = self.stft.compute(np.zeros((self.fft_window_size))).shape[1]
        self.sample_size_frames = self.fft_window_size + (self.fft_window_step * (self.image_size[0] - strangeness))
        generator.continue_init(self.sample_size_frames)

        check = self.stft.compute(np.zeros((self.sample_size_frames))).shape[1]
        assert(check == self.image_size[0])
        

    def __len__(self):
        return self.generator.__len__()
      
    def __getitem__(self, index):
        source_, target_ = self.generator.__getitem__(index)

        # padding/cutting
        source = self._prepare_sample(source_)
        target = self._prepare_sample(target_)
        
        # Short-time Fourier transform
        source_stft = self.stft.compute(source)
        target_stft = self.stft.compute(target)

        return source_stft, target_stft   

    def _prepare_sample(self, waveform):
        length = self.sample_size_frames
        current_length = min(waveform.shape[0], length)
        
        output = np.zeros((length), dtype='float32')
        output[-current_length:] = waveform[:current_length]
        output = output.reshape((1, output.shape[0]))

        log_(f"output min_max {np.min(output)} {np.max(output)}")
        
        return output
