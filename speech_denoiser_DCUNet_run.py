# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="9oX5DLS3COS5"
# # Speech Denoising without Clean Training Data: a Noise2Noise Approach #
# -

# ### Enter the noise type you want to train the model to denoise. The test and train dataset must already be generated beforehand. ###

# ### white : additive_gaussian_noise ###
# ### 0 : air_conditioner ###
# ### 1 : car_horn ###
# ### 2 : children_playing ###
# ### 3 : dog_bark ###
# ### 4 : drilling ###
# ### 5 : engine_idling ###
# ### 6 : gun_shot ###
# ### 7 : jackhammer ###
# ### 8 : siren ###
# ### 9 : street_music ###
#

noise_class = "0" 

# ### Specify the type of training you want to employ: either "Noise2Noise" or "Noise2Clean"  ###

training_type =  "Noise2Noise" 

# + [markdown] colab_type="text" id="FiXUioRtCOS6"
# ### Import of libraries ###

# +

# + colab={} colab_type="code" id="D0bWtt2J5i9F"
import time
import pickle
import warnings
import gc
import copy
from pathlib import Path

import noise_addition_utils

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

import datasets
import dataset_noise_models
import model_dcu20


# %matplotlib inline

# not everything is smooth in sklearn, to conveniently output images in colab
# we will ignore warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# +
np.random.seed(999)
torch.manual_seed(999)

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# + [markdown] colab_type="text" id="-xnlgTsICOTA"
# ### Checking whether the GPU is available ###

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="byUnPtQ25i9O" outputId="4b6d9f85-e5b6-4cb5-c3db-2f48321ed391"
# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
       
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

# + colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="qacrfNwA6vw_" outputId="e1183ab5-4f39-478a-b00a-74fe5edfb511"
# !nvidia-smi

# + [markdown] colab={} colab_type="code" id="_N4AFJANDcBG"
# ### Set Audio backend as Soundfile for windows and Sox for Linux ###

# + colab={} colab_type="code" id="_N4AFJANDcBG"
torchaudio.set_audio_backend("soundfile")
print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

# + [markdown] colab_type="text" id="IWvlIABzCOTM"
# ### The sampling frequency and the selected values for the Short-time Fourier transform. ###

# + colab={} colab_type="code" id="8ngFJtPj5i9V"
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

# + [markdown] colab_type="text" id="3G5CqR3-COT8"
# ### Loss function ###

# + colab={} colab_type="code" id="J71ny6expQeW"
from pesq import pesq
from scipy import interpolate

PREFIX_DIR = "/Users/tprepscius/Projects/denoise/madhavmk3/Noise2Noise-audio_denoising_without_clean_training_data"
# PREFIX_DIR = "./"
model_weights_path = f"{PREFIX_DIR}/transfer/dc20_model_9.pth"
input_path = f"{PREFIX_DIR}/Samples/Sample_Test_Input"

clean_dir = f"{PREFIX_DIR}/Datasets/clean"
noise_dir = f"{PREFIX_DIR}/Datasets/noise"

complex = False

dcunet20 = model_dcu20.DCUnet20(complex).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())

checkpoint = torch.load(
    model_weights_path,
    map_location=torch.device('cpu')
)

dcunet20.load_state_dict(checkpoint)

# #### Enter the index of the file in the Test Set folder to Denoise and evaluate metrics waveforms (Indexing starts from 0) ####

dataset = datasets.getDataset("tjp-0", { 
    "clean_dir": clean_dir, 
    "noise_dir": noise_dir, 
    "image_size": (215, 2049),
    "source_noise_model": dataset_noise_models.additive_noise_model,
    "target_noise_model": dataset_noise_models.clean_noise_model,
    "snr": (-20, 10.0),
    "complex": complex,
    "randomize": False
    # "override_length": 2
});

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# +
dcunet20.eval()

sounds = []
noise_cleans = []

def to_sound(magnitude, phase):
    sounded = dataset.data.istft_using_phase(
        magnitude, 
        torch.from_numpy(phase)
    )
    return sounded[0].view(-1).detach().cpu().numpy()


total = len(loader)
saveIncremental = True
for i, (source, target) in enumerate(dataset):
    phase = dataset.data.source_phase_for(i)

    # source_ = source.cuda()
    source_ = torch.unsqueeze(source, 0)

    stft = dcunet20(source_)
    output = to_sound(stft, phase)
    sounds.append(output)

    noise_cleans.append(to_sound(source, phase))
    noise_cleans.append(output)

    print(f"[{i}/{total}] sound.min_max {np.min(output)} {np.max(output)}\r")

    if saveIncremental:
        sound = np.concatenate(sounds)
        noise_clean = np.concatenate(noise_cleans)

        noise_addition_utils.save_audio_file(
            np_array=sound,file_path=Path(f"{PREFIX_DIR}/Samples/output.wav"), 
            sample_rate=SAMPLE_RATE, 
            bit_precision=16
        )

        noise_addition_utils.save_audio_file(
            np_array=noise_clean,file_path=Path(f"{PREFIX_DIR}/Samples/noise_clean.wav"), 
            sample_rate=SAMPLE_RATE, 
            bit_precision=16
        )

        noise_addition_utils.save_audio_file(
            np_array=output,file_path=Path(f"{PREFIX_DIR}/Samples/output-{i}.wav"), 
            sample_rate=SAMPLE_RATE, 
            bit_precision=16
        )

        noise_addition_utils.save_audio_file(
            np_array=to_sound(source, phase),file_path=Path(f"{PREFIX_DIR}/Samples/source-{i}.wav"), 
            sample_rate=SAMPLE_RATE, 
            bit_precision=16
        )

        noise_addition_utils.save_audio_file(
            np_array=to_sound(target, phase),file_path=Path(f"{PREFIX_DIR}/Samples/target-{i}.wav"), 
            sample_rate=SAMPLE_RATE, 
            bit_precision=16
        )


print();
print("writing");

sound = np.concatenate(sounds)

noise_addition_utils.save_audio_file(
    np_array=sound,file_path=Path(f"{PREFIX_DIR}/Samples/output.wav"), 
    sample_rate=SAMPLE_RATE, 
    bit_precision=16
)

print("done")