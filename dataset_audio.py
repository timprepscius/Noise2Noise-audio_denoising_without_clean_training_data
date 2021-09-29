from pathlib import Path
import random
import numpy as np

import matplotlib.pyplot as plt
import struct

import sys
import os

import pydub
import util

import pickle as dumper

default_sample_rate = 48000
default_fft_window_size = (default_sample_rate * 64) // 1000
default_fft_window_step = (default_sample_rate * 16) // 1000
default_image_size = (64, 384)
default_sample_size_frames = default_fft_window_size + (default_fft_window_step * (default_image_size[0] - 1))

def log_(s):
    pass

class AudioGenerator:
    def __init__(self, source_dir, verbose=0):
        super().__init__()

        audio_suffixes = (".wav", ".mp3", ".mp4", ".m4a", ".flac")

        self.override_len = None
        self.verbose = verbose
        self.bin_sample_rate = 48000
        self.sample_rate = 48000

        self.paths = util.find_files([source_dir], audio_suffixes, self.verbose)

        self.audio_info = {}
        self.cache_audio_info = False

        if self.cache_audio_info:
            try:
                self.audio_info = dumper.load( open( "audio_info.npy", "rb" ))
            except:
                pass

        self.preprocess_audio_files(self.paths)

    def continue_init(self, sample_size_frames):
        self.sample_size_frames = sample_size_frames
        self.picks = self.generate_sample_list(self.paths)

    def info(self):
        return {
            "len(picks)": len(self.picks),
            "len(paths)": len(self.paths),
        }

    def preprocess_audio_files(self, paths):

        if self.verbose > 0:
            print(f"preprocess_audio_files audio_info for {len(paths)} paths")

        written = 0
        for audio_path in paths:
            if audio_path not in self.audio_info:

                bin_file_name = f"{audio_path}.npy"

                try:
                    frame_count = os.path.getsize(bin_file_name) // 2
                    self.audio_info[audio_path] = (bin_file_name, frame_count)

                    if self.verbose > 0:
                        print(f"found {bin_file_name} {frame_count}")

                except:
                    try:
                        if self.verbose > 0:
                            print(f"generating {bin_file_name}")

                        audio_ = pydub.AudioSegment.from_file(file=audio_path)
                        rate = audio_.frame_rate
                        channels = audio_.channels

                        if channels > 1:
                            audio_ = audio_.split_to_mono()[0]

                        if audio_.sample_width != 2:
                            audio_ = audio_.set_sample_width(2)

                        if audio_.frame_rate != self.bin_sample_rate:
                            audio_ = audio_.set_frame_rate(self.bin_sample_rate)

                        audio = np.asarray(audio_.get_array_of_samples())

                        if isinstance(audio[0], float):
                            audio = np.asarray(audio * 32768.0, 'int16')
                        else:
                            audio = np.asarray(audio, 'int16')

                        frame_count = len(audio)
                        temp_bin_file_name = f"{bin_file_name}.temp"
                        audio.tofile(temp_bin_file_name)
                        os.rename(temp_bin_file_name, bin_file_name)
                        
                        self.audio_info[audio_path] = (bin_file_name, frame_count)
                        print(f"{audio_path} {bin_file_name} {frame_count}")

                    except:
                        if self.verbose > 0:
                            print(f"failed to process {audio_path}")
                        pass

                written += 1

                if self.cache_audio_info:
                    if written % 32 == 0:
                        dumper.dump(self.audio_info, open( "audio_info.npy", "wb") )

        if self.cache_audio_info:
            dumper.dump(self.audio_info, open( "audio_info.npy", "wb") )

    def generate_sample_list(self, paths):

        picks = []
        for path_index, path in enumerate(paths):
            if path in self.audio_info:
                bin_path, audio_length = self.audio_info[path]

                n = self.sample_size_frames
                chunks = [(path_index, i) for i in range((audio_length + n - 1) // n )]
                picks.extend(chunks)

        random.shuffle(picks)

        return picks

    def getIndex(self, paths, picks, index_):

        index = index_ % len(picks)
        # print(f"{index_} {index}")

        path_index, sample_index = picks[index]
        path = paths[path_index]

        # print(path)

        bin_path, audio_length = self.audio_info[path]
        # print(audio_length)

        sample_start = sample_index * self.sample_size_frames
        sample_end = sample_start + self.sample_size_frames

        # print(f"{sample_start} {sample_end}")

        if sample_end >= audio_length:
            sample_end = audio_length - 1


        # print(f"{sample_start} {sample_end}")

        sample_quantity = sample_end - sample_start

        with open(bin_path, "rb") as f:
            f.seek(sample_start)
            audio = np.fromfile(f, dtype="int16", count=sample_quantity)

        # print(audio, sample_start, sample_end)

        audio = np.asarray(audio, 'float32')
        audio = audio / 32768.0

        if self.bin_sample_rate != self.sample_rate:
            audio = tfio.audio.resample(audio, self.bin_sample_rate, self.sample_rate)

        return audio


    def set_override_len(self, length):
        self.override_len = length

    def __len__(self):
        if self.override_len is not None:
            return self.override_len
            
        return len(self.picks)

    def __getitem__(self, index):
        return self.getIndex(self.paths, self.picks, index)


class AudioGeneratorNoisyAndClean:
    def __init__(self, clean, noise, source_noise_model, target_noise_model, snr, override_length=None):
        super().__init__()

        self.clean = clean
        self.noise = noise
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.sample_rate = clean.sample_rate
        self.snr = snr
        self.override_length = override_length

    def continue_init(self, sample_size_frames):
        self.sample_size_frames = sample_size_frames

        self.clean.continue_init(sample_size_frames)
        self.noise.continue_init(sample_size_frames)

    def __len__(self):
        v = max([len(self.clean), len(self.noise)])
        if self.override_length is not None:
            return min(v, self.override_length)

        return v

    def __getitem__(self, index):
        clean = self.clean[index]
        noise = self.noise[index]

        source = self.source_noise_model(self.sample_size_frames, clean, noise, self.snr)
        target = self.source_noise_model(self.sample_size_frames, clean, noise, self.snr)

        return source, target
