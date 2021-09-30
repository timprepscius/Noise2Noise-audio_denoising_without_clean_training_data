import datasets
import dataset_noise_models
import display

import matplotlib.pyplot as plt

PREFIX_DIR = "/Users/tprepscius/Projects/denoise/madhavmk3/Noise2Noise-audio_denoising_without_clean_training_data"
# PREFIX_DIR = "./"
CLEAN_INPUT_DIR = f"{PREFIX_DIR}/Datasets/clean"
NOISE_INPUT_DIR = f"{PREFIX_DIR}/Datasets/noise"

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

dataset_tjp = datasets.getDataset("tjp-0", { 
    "clean_dir": CLEAN_INPUT_DIR, 
    "noise_dir": NOISE_INPUT_DIR, 
    "image_size": (215, 2049),
    "fft_window": N_FFT, "fft_step": HOP_LENGTH,
    "source_noise_model": dataset_noise_models.additive_noise_model,
    "target_noise_model": dataset_noise_models.clean_noise_model,
    "snr": (0.5, 1.5),
    "override_length": 100
});

u, v = dataset_tjp[0]

fig, axes = plt.subplots(2, 2, figsize=(10, 12))

display.plot(u.numpy(), axes[0][0])
display.plot(v.numpy(), axes[0][1])

noise_class = 0
NOISY_DIR = 'Datasets/US_Class'+str(noise_class)+'_Train_Input'
NOISY_DIR = 'Datasets/clean_trainset_28spk_wav'
CLEAN_DIR = 'Datasets/clean_trainset_28spk_wav'

dataset_mad = datasets.getDataset("madhavmk", { 
    "source_dir": NOISY_DIR, 
    "target_dir": CLEAN_DIR, 
    "fft_window": N_FFT, 
    "fft_step": HOP_LENGTH 
});

w, x = dataset_mad[0]

display.plot(w.numpy(), axes[1][0])
display.plot(x.numpy(), axes[1][1])


plt.show()