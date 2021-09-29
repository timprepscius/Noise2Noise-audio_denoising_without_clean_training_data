import cv2
import tensorflow as tf
import numpy as np
import pydub
import os

def log_(s=""):
    # print(s)
    pass

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def find_files_r(dirs=[], extensions=[], files=[]):
    new_dirs = []
    for d in dirs:
        try:
            new_dirs += [ os.path.join(d, f) for f in os.listdir(d) ]
        except OSError:
            extension = os.path.splitext(d)[-1].lower()
            if extension in extensions:
                files.append(str(d))
    
    if new_dirs:
        return find_files_r(new_dirs, extensions, files)
    else:
        return files

def find_files(dirs, extensions=[], verbose=0):
    result = find_files_r(dirs, extensions, [])

    if verbose > 0:
        print(f"find files {dirs} -> {len(result)}")

    return result

def image_adjust_size(image, shape, rows=True, cols=True):
    shape_cmp = shape + (1,)

    if shape_cmp == image.shape:
        log_(f"resize_trunc {shape_cmp} == {image.shape}")
        return image
        
    log_(f"resize_trunc {shape_cmp} != {image.shape}")

    if rows:
        if image.shape[0] < shape[0]:
            log_(f"image.shape[0] < shape[0]")
            zeros = np.zeros((shape[0] - image.shape[0], shape[1], 1))
            image = np.append(image, zeros, axis=0)
            log_(image.shape)
        elif image.shape[0] > shape[0]:
            log_(f"image.shape[0] > shape[0]")
            image = image[:shape[0],:,:]
            log_(image.shape)

    if cols:
        if shape[1] != image.shape[1]:
            image = tf.image.resize(image, (image.shape[0], shape[1]))

    log_(f"image_adjust_size final {image.shape}")

    return image

def to_conv_image(image, shape, rows=True, cols=True):
    return image_adjust_size(image, shape, rows, cols)

def to_original_image(image, original_image, rows=True, cols=True):
    shape = (original_image.shape[0], original_image.shape[1])
    return image_adjust_size(image, shape, rows, cols)

def normalize_image(image):
    print(f"normalize_image min {np.min(image)} max {np.max(image)}")
    return image / np.max(image)

def normalize_sound(sound):
    print(f"normalize_sound min {np.min(sound)} max {np.max(sound)}")
    return sound / np.max(sound)

def write_image(filename, image):
    cv2.imwrite(filename, np.asarray(np.clip(image, -1.0, 1.0) * 32768.0 + 32768.0, 'uint16'))

def complex_image_to_image(image):
    log_(f"image.shape {image.shape}")

    image_real, image_imaginary = tf.split(image, num_or_size_splits=2, axis=-1)
    log_(f"image_real.shape {image_real.shape}")
    log_(f"image_real.min_max {np.min(image_real)} {np.max(image_real)}")

    image_cast = tf.squeeze(image_real)
    log_(f"image_cast.shape {image_cast.shape}")

    image_result = tf.cast(image_cast, dtype=tf.float32)
    log_(f"image_result.shape {image_result.shape}")

    return image_result

def real_image_to_image(image):
    return image

def image_to_image(image):
    return real_image_to_image(image)

def sound_to_real_image(sound, frame_length, frame_step):
    log_(f"sound.min_max {np.min(sound)} {np.max(sound)}")

    image_ = tf.signal.stft(sound, frame_length=frame_length, frame_step=frame_step)
    real = tf.cast(image_, dtype=tf.float32)

    # https://pytorch.org/docs/stable/generated/torch.stft.html
    normalizer = frame_length ** -0.5
    real = tf.abs(real * normalizer)

    log_(f"real.shape {image_.shape}")

    return tf.expand_dims(real, -1)

def sound_to_complex_image(sound, frame_length, frame_step):
    log_(f"sound.min_max {np.min(sound)} {np.max(sound)}")

    image_ = tf.signal.stft(sound, frame_length=frame_length, frame_step=frame_step)

    # https://pytorch.org/docs/stable/generated/torch.stft.html
    # normalizer = frame_length ** -0.5
    # image_ = image_ * normalizer

    log_(f"image_.shape {image_.shape}")

    extended_bin = image_[..., None]
    log_(f"extended_bin.shape {extended_bin.shape}")
    complexed = tf.concat([tf.math.real(extended_bin), tf.math.imag(extended_bin)], axis=-1)

    log_(f"complexed.shape {complexed.shape}")

    complexed_ = tf.cast(complexed, dtype=tf.float32)
    return complexed_

def sound_to_image(sound, frame_length, frame_step):
    return sound_to_real_image(sound, frame_length, frame_step)

def complex_image_to_sound(image, frame_length, frame_step):
    image_real, image_imaginary = tf.split(image, num_or_size_splits=2, axis=-1)

    image_cast = tf.squeeze(tf.complex(image_real, image_imaginary))
    sound = tf.signal.inverse_stft(image_cast, frame_length=frame_length, frame_step=frame_step)

    sound = sound / np.max(sound)
    return sound

def real_image_to_sound(image_real, frame_length, frame_step):
    image_cast = tf.squeeze(tf.cast(image_real, tf.complex64))
    sound = tf.signal.inverse_stft(image_cast, frame_length=frame_length, frame_step=frame_step)

    sound = sound / np.max(sound)
    return sound    

def image_to_sound(image, frame_length, frame_step):
    return real_image_to_sound(image, frame_length, frame_step)

def write_sound(filename, sound):
    sound_ = np.clip(sound, -1.0, 1.0)
    sound__ = np.asarray(sound_ * 32768.0, 'int16')

    segment = pydub.AudioSegment(
        sound__.tobytes(), 
        frame_rate=48000,
        sample_width=2, 
        channels=1
    )
    segment.export(filename, format="wav");