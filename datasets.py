import torch
from torch.utils.data import Dataset, DataLoader

from dataset_speech_original import SpeechDataset
from dataset_audio import *
from dataset_audio_to_stft import *

class DatasetWrapper(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, index):
        return self.data.__getitem__(index)

def getDataset(type, options):
    if type == "original":

        source_files = sorted(list(options["source_dir"].rglob('*.wav')))
        target_files = sorted(list(options["target_dir"].rglob('*.wav')))

        return DatasetWrapper (
            AudioToSTFT(
                SpeechDataset(
                    source_files, target_files,
                    options["fft_window"], options["fft_step"]
                )
            )
        )

    if type == "tjp-0":
        override_length = None
        if "override_length" in options:
            override_length = options["override_length"]

        return DatasetWrapper(
            AudioToSTFT(
                AudioGeneratorNoisyAndClean(
                    AudioGenerator(options["clean_dir"]),
                    AudioGenerator(options["noise_dir"]),
                    options["source_noise_model"],
                    options["target_noise_model"],
                    options["snr"],
                    override_length=override_length,
                ),
                image_size = options["image_size"]
            )
        )
