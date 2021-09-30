import random
import numpy as np

default_sample_rate = 48000
default_fft_window_size = (default_sample_rate * 64) // 1000
default_fft_window_step = (default_sample_rate * 16) // 1000
default_image_size = (64, 384)
default_sample_size_frames = default_fft_window_size + (default_fft_window_step * (default_image_size[0] - 1))

def log_(*args):
    # print(*args)
    pass

def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)

def average_db(sample):
    watts = sample ** 2
    avg_watts = np.mean(watts) if len(watts) > 0 else 0.0
    avg_db = 10 * np.log10(avg_watts) if avg_watts > 0.0 else 0.0
    return avg_db

def additive_noise_model(samples, clean_, noise_, snr):
    clean = np.zeros((samples), dtype='float32')
    clean[0:clean_.shape[0]] = clean_
    clean_avg_db = average_db(clean)

    noise = np.zeros((samples), dtype='float32')
    noise[0:noise_.shape[0]] = noise_
    noise_avg_db = average_db(noise)

    if snr is not None:
        snr_ = random.uniform(snr[0], snr[1])
        log_(f"snr_ {snr_}")

        added_noise_avg_db = clean_avg_db + snr_
        db_change = added_noise_avg_db - noise_avg_db
        log_(f"clean_avg_db {clean_avg_db} snr {snr} added_noise_avg_db {added_noise_avg_db}")
        log_(f"added_noise_avg_db {added_noise_avg_db} noise_avg_db {noise_avg_db} db_change {db_change}")

        noise_add = noise * db_to_float(db_change)
        noise_add_avg_db = average_db(noise_add)
        log_(f"noise_add_avg_db {noise_add_avg_db}")

        noise = noise_add

    combined = clean + noise

    combined_avg_db = average_db(combined)
    log_(f"combined_avg_db {combined_avg_db}")

    # combined = combined * db_to_float(clean_avg_db - combined_avg_db)
    # adjusted_combined_avg_db = average_db(combined)
    # log_(f"adjusted_combined_avg_db {adjusted_combined_avg_db}")

    combined = np.clip(combined, -1.0, 1.0)

    combined_avg_db = average_db(combined)
    # print(f"combined_avg_db {combined_avg_db}")

    return combined

def clean_noise_model(samples, clean_, noise_, snr):
    clean = np.zeros((samples), dtype='float32')
    clean[0:clean_.shape[0]] = clean_
    clean_avg_db = average_db(clean)

    clean = np.clip(clean, -1.0, 1.0)
    return clean