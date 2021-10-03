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
from pathlib import Path
from datetime import datetime
now = datetime.now()
date_string = now.strftime("%Y%m%d_%H%M%S")

import os
output_path = f"output/"
os.makedirs(output_path,exist_ok=True)

import sys
import time
import pickle
import warnings
import gc
import copy

import noise_addition_utils

from metrics import AudioMetrics
from metrics import AudioMetrics2

import numpy as np
import torch
import torch.nn as nn
import torchaudio

import datasets
import dataset_noise_models
import model_dcu20

from stft import STFT

from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from pypesq import pesq
from IPython.display import clear_output

def log_(*args):
    # print(*args)
    pass

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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


# + [markdown] colab_type="text" id="x8suREWkb5Se"
# ### The declaration of datasets and dataloaders ###

# test_dataset = datasets.getDataset("speech", { "source_dir": TRAIN_INPUT_DIR, "target_dir": TRAIN_TARGET_DIR, "fft_window": N_FFT, "fft_step": HOP_LENGTH });
# train_dataset = datasets.getDataset("speech", { "source_dir": TEST_NOISY_DIR, "target_dir": TEST_CLEAN_DIR, "fft_window": N_FFT, "fft_step": HOP_LENGTH });


from pesq import pesq
from scipy import interpolate

def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original

def wsdr_fn(x, y_pred, y_true, eps=1e-8):
    y_pred = y_pred.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

def preprocess(x, y):
    log_(f"preprocess x.shape {x.shape}, y.shape {y.shape}")
    x_, y_ = stft.to_magnitude(x), stft.to_magnitude(y)
    return x_, y_

def postprocess(x, y, z):
    log_(f"postprocess x.shape {x.shape}, y.shape {y.shape}, z.shape {z.shape}")

    x = torch.squeeze(x, axis=1)
    x_ = stft.inverse(x)

    y = torch.squeeze(y, axis=1)
    y_ = stft.inverse(y)

    z_ = stft.inverse(stft.to_complex(z, stft.to_phase(x)))

    return x_, y_, z_

def getMetricsonLoader(stft, loader, net):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    metric_names = {}
    overall_metrics = {}
    total = len(loader)
    for i, (x_, y_) in enumerate(loader):
        x_, y_ = x_.to(DEVICE), y_.to(DEVICE)
        x, y = preprocess(x_, y_)

        # get the output from the model
        predicted = net(x)

        # calculate loss

        x__, y__, predicted__ = postprocess(x_, y_, predicted)

        x_est_np = predicted__.view(-1).detach().cpu().numpy()
        x_clean_np = y__.view(-1).detach().cpu().numpy()
            
        metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)
        
        ref_wb = resample(x_clean_np, 48000, 16000)
        deg_wb = resample(x_est_np, 48000, 16000)
        pesq_wb = 0.0
        try:
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
        except:
            pass
        
        ref_nb = resample(x_clean_np, 48000, 8000)
        deg_nb = resample(x_est_np, 48000, 8000)
        pesq_nb = 0.0
        try:
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')
        except:
            pass

        #print(new_scores)
        #print(metrics.PESQ, metrics.STOI)

        overall_metrics.setdefault("pesq_wb", []).append(pesq_wb)
        overall_metrics.setdefault("pesq_nb", []).append(pesq_nb)
        overall_metrics.setdefault("SNR", []).append(metrics.SNR)
        overall_metrics.setdefault("SSNR", []).append(metrics.SSNR)
        overall_metrics.setdefault("STOI", []).append(metrics.STOI)

        print(f"[{i}/{total}]\r")

    print()
    print("Sample metrics computed")
    results = {}
    for k, v in overall_metrics.items():
        temp = {}
        temp["Mean"] = np.mean(v)
        temp["STD"] = np.std(v)
        temp["Min"] = min(v)
        temp["Max"] = max(v)
        results[k] = temp
        
    print("Averages computed")
    print(results)

    return results


def train_epoch(net, train_loader, loss_fn, stft, optimizer):
    net.train()
    train_ep_loss = 0.
    total = len(train_loader)
    # loss_fn = nn.MSELoss()

    for i, (x_, y_) in enumerate(train_loader):

        x_, y_ = x_.to(DEVICE), y_.to(DEVICE)
        x, y = preprocess(x_, y_)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        predicted = net(x)

        # calculate loss

        x__, y__, predicted__ = postprocess(x_, y_, predicted)

        # loss = loss_fn(istft_fn, noisy_x, pred_x, clean_x)
        loss = loss_fn(x__, predicted__, y__)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item() 

        sys.stdout.write('\r')
        sys.stdout.write(f"[{i}/{total}]")
        sys.stdout.flush()

    train_ep_loss /= total

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


def test_epoch(net, test_loader, loss_fn, stft):
    net.eval()
    test_ep_loss = 0.
    '''
    total = len(test_loader)
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 

    test_ep_loss /= total
    '''
    
    #print("Actual compute done...testing now")

    # this doesn't do anything   
    test_metrics = getMetricsonLoader(stft, test_loader, net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, test_metrics


def train(net, train_loader, test_loader, loss_fn, stft, optimizer, scheduler, epochs):
    
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):
        train_loss = train_epoch(net, train_loader, loss_fn, stft, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")
        
        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, stft)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        #print("skipping testing cuz peak autism idk")
        
        with open(output_path + f"/{date_string}_results.txt","a") as f:
            f.write("Epoch :"+str(e+1) + "\n" + str(testmet))
            f.write("\n")
        
        print("OPed to txt")
        
        torch.save(net.state_dict(), output_path +f"/{date_string}_dc20_model_"+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), output_path +f"/{date_string}_dc20_opt_"+str(e+1)+'.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        #print("Epoch: {}/{}...".format(e+1, epochs),
        #              "Loss: {:.6f}...".format(train_loss),
        #              "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss




# --------------------------------
# --------------------------------
# --------------------------------
# --------------------------------


gc.collect()
torch.cuda.empty_cache()

# --------------------------------
# --------------------------------
# --------------------------------
# --------------------------------

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

# PREFIX_DIR = "/Users/tprepscius/Projects/denoise/madhavmk3/Noise2Noise-audio_denoising_without_clean_training_data"
PREFIX_DIR = "./"
CLEAN_INPUT_DIR = f"{PREFIX_DIR}/Datasets/clean"
NOISE_INPUT_DIR = f"{PREFIX_DIR}/Datasets/noise"

complex = False
test_dataset = datasets.getDataset("tjp-0", { 
    "clean_dir": CLEAN_INPUT_DIR, 
    "noise_dir": NOISE_INPUT_DIR, 
    "image_size": (215, 2049),
    "fft_window": N_FFT, "fft_step": HOP_LENGTH,
    "source_noise_model": dataset_noise_models.additive_noise_model,
    "target_noise_model": dataset_noise_models.clean_noise_model,
    "snr": (-20.0, 10.0),
    "override_length": 32,
    # "override_length": 2,
    "randomize": True,
});

train_dataset = datasets.getDataset("tjp-0", { 
    "clean_dir": CLEAN_INPUT_DIR, 
    "noise_dir": NOISE_INPUT_DIR, 
    "image_size": (215, 2049),
    "source_noise_model": dataset_noise_models.additive_noise_model,
    "target_noise_model": dataset_noise_models.clean_noise_model,
    "snr": (-20.0, 10.0),
    "randomize": True,
    # "override_length": 2,
    # "override_length": 32
});

stft = train_dataset.data.stft
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


dcunet20 = model_dcu20.DCUnet20(complex=complex).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# +

print(dcunet20)
print(f"using input shape: {train_dataset[0][0].shape}")

# +
# specify paths and uncomment to resume training from a given point
# model_checkpoint = torch.load(path_to_model)
# opt_checkpoint = torch.load(path_to_opt)
# dcunet20.load_state_dict(model_checkpoint)
# optimizer.load_state_dict(opt_checkpoint)

# + colab={} colab_type="code" id="ppXkJUsY55vI"
num_epochs = 2048
loss_fn = wsdr_fn
train_losses, test_losses = train(dcunet20, train_loader, test_loader, loss_fn, stft, optimizer, scheduler, num_epochs)
# -

print("done")
