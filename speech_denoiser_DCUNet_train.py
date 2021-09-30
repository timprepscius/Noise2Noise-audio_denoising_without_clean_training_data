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

if noise_class == "white": 
    TRAIN_INPUT_DIR = Path('Datasets/WhiteNoise_Train_Input')

    if training_type == "Noise2Noise":
        TRAIN_TARGET_DIR = Path('Datasets/WhiteNoise_Train_Output')
    elif training_type == "Noise2Clean":
        TRAIN_TARGET_DIR = Path('Datasets/clean_trainset_28spk_wav')
    else:
        raise Exception("Enter valid training type")

    TEST_NOISY_DIR = Path('Datasets/WhiteNoise_Test_Input')
    TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav') 
    
else:
    TRAIN_INPUT_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Train_Input')

    if training_type == "Noise2Noise":
        TRAIN_TARGET_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Train_Output')
    elif training_type == "Noise2Clean":
        TRAIN_TARGET_DIR = Path('Datasets/clean_trainset_28spk_wav')
    else:
        raise Exception("Enter valid training type")

    TEST_NOISY_DIR = Path('Datasets/US_Class'+str(noise_class)+'_Test_Input')
    TEST_CLEAN_DIR = Path('Datasets/clean_testset_wav') 
# -

from datetime import datetime
now = datetime.now()
date_string = now.strftime("%Y%m%d_%H%M%S")

import os
output_path = f"output/{date_string}"
os.makedirs(output_path,exist_ok=True)

import sys

# + colab={} colab_type="code" id="D0bWtt2J5i9F"
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

from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from pypesq import pesq
from IPython.display import clear_output


def log_(*args):
    # print(*args)
    pass

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


# + [markdown] colab_type="text" id="x8suREWkb5Se"
# ### The declaration of datasets and dataloaders ###

# test_dataset = datasets.getDataset("speech", { "source_dir": TRAIN_INPUT_DIR, "target_dir": TRAIN_TARGET_DIR, "fft_window": N_FFT, "fft_step": HOP_LENGTH });
# train_dataset = datasets.getDataset("speech", { "source_dir": TEST_NOISY_DIR, "target_dir": TEST_CLEAN_DIR, "fft_window": N_FFT, "fft_step": HOP_LENGTH });

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
    "snr": (0.5, 1.5),
    "override_length": 8*16,
    "complex": complex
});

train_dataset = datasets.getDataset("tjp-0", { 
    "clean_dir": CLEAN_INPUT_DIR, 
    "noise_dir": NOISE_INPUT_DIR, 
    "image_size": (215, 2049),
    "source_noise_model": dataset_noise_models.additive_noise_model,
    "target_noise_model": dataset_noise_models.clean_noise_model,
    "snr": (0.5, 1.5),
    "complex": complex,
    # "override_length": 2,
    # "override_length": 32
});

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)


# + [markdown] colab_type="text" id="3G5CqR3-COT8"
# ### Loss function ###

# + colab={} colab_type="code" id="J71ny6expQeW"
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

def istft_complex(v):
    v = torch.squeeze(v, 1)
    v = torch.istft(v, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    return v

def istft_magnitude(v):
    v = torch.unsqueeze(v, -1)
    i = torch.zeros_like(v)
    w = torch.cat([v, i], axis=-1)
    return istft_complex(w)

istft_fn = None
if complex:
    istft_fn = istft_complex
else:
    istft_fn = istft_magnitude

def wsdr_fn(istft_func, x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    log_(f"y_true_.shape {y_true_.shape}")
    # y_true_ = torch.squeeze(y_true_, 1)
    # y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    y_true = istft_func(y_true_)

    log_(f"x_.shape {x_.shape}")
    # x_ = torch.squeeze(x_, 1)
    # x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    x = istft_func(x_)

    log_(f"y_pred_.shape {y_pred_.shape}")
    # y_pred_ = torch.squeeze(y_pred_, 1)
    # y_pred = torch.istft(y_pred_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    y_pred = istft_func(y_pred_)


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

def getMetricsonLoader(istft_fn, loader, net):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    metric_names = {}
    overall_metrics = {}
    total = len(loader)
    for i, data in enumerate(loader):
        noisy = data[0]
        clean = data[1]

        x_est_np = istft_fn(noisy).view(-1).detach().cpu().numpy()
        x_clean_np = istft_fn(clean).view(-1).detach().cpu().numpy()
            
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
    print("Metrics on test data",addon)
    print(results)

    return results


# + [markdown] colab_type="text" id="kYvWz_6jRZ3e"
# ### Description of the training of epochs. ###

# + colab={} colab_type="code" id="VukJTCGIZ8ZU"
def train_epoch(net, train_loader, loss_fn, istft_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0
    total = len(train_loader)

    for noisy_x, clean_x in train_loader:

        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(istft_fn, noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item() 
        counter += 1

        sys.stdout.write('\r')
        sys.stdout.write(f"[{counter}/{total}]")
        sys.stdout.flush()

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


# + [markdown] colab_type="text" id="zYCJWaTMRgS3"
# ### Description of the validation of epochs ###

# + colab={} colab_type="code" id="-JKrTMpPhw19"
def test_epoch(net, test_loader, loss_fn, istft_fn):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    '''
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter
    '''
    
    #print("Actual compute done...testing now")
    
    testmet = getMetricsonLoader(istft_fn, test_loader, net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, testmet


# + [markdown] colab_type="text" id="879vq_uBRm_2"
# ### To understand whether the network is being trained or not, we will output a train and test loss. ###

# + colab={} colab_type="code" id="I4gdVmhRr1Qi"
def train(net, train_loader, test_loader, loss_fn, istft_fn, optimizer, scheduler, epochs):
    
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        # first evaluating for comparison
        
        if e == 0 and training_type=="Noise2Clean":
            print("Pre-training evaluation")
            #with torch.no_grad():
            #    test_loss,testmet = test_epoch(net, test_loader, loss_fn,use_net=False)
            #print("Had to load model.. checking if deets match")
            testmet = getMetricsonLoader(istft_fn, test_loader, net, False)    # again, modified cuz im loading
            #test_losses.append(test_loss)
            #print("Loss before training:{:.6f}".format(test_loss))
        
            with open(output_path + "/results.txt","w+") as f:
                f.write("Initial : \n")
                f.write(str(testmet))
                f.write("\n")
        
         
        train_loss = train_epoch(net, train_loader, loss_fn, istft_fn, optimizer)
        test_loss = 0
        scheduler.step()
        print("Saving model....")
        
        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, loss_fn, istft_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        #print("skipping testing cuz peak autism idk")
        
        with open(output_path + "/results.txt","a") as f:
            f.write("Epoch :"+str(e+1) + "\n" + str(testmet))
            f.write("\n")
        
        print("OPed to txt")
        
        torch.save(net.state_dict(), output_path +'/dc20_model_'+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), output_path +'/dc20_opt_'+str(e+1)+'.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        #print("Epoch: {}/{}...".format(e+1, epochs),
        #              "Loss: {:.6f}...".format(train_loss),
        #              "Test Loss: {:.6f}".format(test_loss))
    return train_loss, test_loss


# + [markdown] colab_type="text" id="HO3p2zrOcn_z"
# ### 20 Layer DCUNet Model ###

# + colab={} colab_type="code" id="j8XCrVIg5i-K"

# + [markdown] colab_type="text" id="G6bIE9iOj8pq"
# ## Training New Model ##

# + colab={} colab_type="code" id="QyBc1awQkI-D"
# # clear cache
gc.collect()
torch.cuda.empty_cache()

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
train_losses, test_losses = train(dcunet20, train_loader, test_loader, loss_fn, istft_fn, optimizer, scheduler, num_epochs)
# -

print("done")
