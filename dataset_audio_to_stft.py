import torch
import numpy as np

# + colab={} colab_type="code" id="cZ0wb9EN5i9f"

def log_(s):
    print(s)

class AudioToSTFT:

    def stft_torch(self, data):
        log_(f"stft data min_max {np.min(data)} {np.max(data)}")

        result = torch.stft(
            input=torch.from_numpy(data), 
            n_fft=self.fft_window_size, 
            hop_length=self.fft_window_step, 
            normalized=True,
            return_complex=True
        )

        result = result.numpy()


        magnitude = np.expand_dims(np.abs(result), axis=-1)
        phase = np.expand_dims(np.angle(result), axis=-1)

        log_(f"stft result shape {result.shape} magnitude {magnitude.shape} min_max {np.min(magnitude)} {np.max(magnitude)}")

        return magnitude, phase

    def istft_torch(self, data):
        log_(f"stft data shape {data.shape}")

        result = torch.istft(
            input=data, 
            n_fft=self.fft_window_size, 
            hop_length=self.fft_window_step, 
            normalized=True
        )

        return result.numpy()

    def stft_scipy(self, data):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
        log_(f"stft data min_max {np.min(data)} {np.max(data)}")

        result = torch.stft(
            input=torch.from_numpy(data), 
            n_fft=self.fft_window_size, 
            hop_length=self.fft_window_step, 
            normalized=True
        )

        result_np = result.numpy()
        log_(f"stft result min_max {np.min(result_np)} {np.max(result_np)}")

        return result

    def stft_tf(self, data):
        log_(f"stft data min_max {np.min(data)} {np.max(data)}")

        result = torch.stft(
            input=torch.from_numpy(data), 
            n_fft=self.fft_window_size, 
            hop_length=self.fft_window_step, 
            normalized=True
        )

        result_np = result.numpy()
        log_(f"stft result min_max {np.min(result_np)} {np.max(result_np)}")

        return result    

    def stft(self, data):
        return self.stft_torch(data)

    def to_complex(self, real, imaginary):
        complexed = np.concatenate([real, imaginary], axis=-1)
        log_(f"complexed.shape {complexed.shape}")
        return complexed

    def istft(self, data):
        return self.istft_torch(data)
        
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, generator, image_size=(64, 2049), complex=True):
        super().__init__()

        self.generator = generator
        sample_rate = generator.sample_rate
        self.fft_window_size = (sample_rate * 64) // 1000 # 64 ms
        self.fft_window_step = (sample_rate * 16) // 1000 # 16 ms
        self.image_size = image_size
        self.complex = complex

        strangeness = self.stft(np.zeros((self.fft_window_size)))[0].shape[1]
        self.sample_size_frames = self.fft_window_size + (self.fft_window_step * (self.image_size[0] - strangeness))
        generator.continue_init(self.sample_size_frames)

        check = self.stft(np.zeros((self.sample_size_frames)))[0].shape[1]
        assert(check == self.image_size[0])

    def __len__(self):
        return self.generator.__len__()
      
    def __getitem__(self, index):
        source_, target_ = self.generator.__getitem__(index)

        # padding/cutting
        source = self._prepare_sample(source_)
        target = self._prepare_sample(target_)
        
        # Short-time Fourier transform
        source_stft, source_phase = self.stft(source)
        target_stft, target_phase = self.stft(target)

        if self.complex:
            source_stft = self.to_complex(source_stft, source_phase)
            target_stft = self.to_complex(target_stft, target_phase)
        else:
            source_stft = np.squeeze(source_stft, axis=-1)
            target_stft = np.squeeze(target_stft, axis=-1)
        
        return torch.from_numpy(source_stft), torch.from_numpy(target_stft)
        
    def _prepare_sample(self, waveform):
        length = self.sample_size_frames
        current_length = min(waveform.shape[0], length)
        
        output = np.zeros((length), dtype='float32')
        output[-current_length:] = waveform[:current_length]
        output = output.reshape((1, output.shape[0]))

        log_(f"output min_max {np.min(output)} {np.max(output)}")
        
        return output

