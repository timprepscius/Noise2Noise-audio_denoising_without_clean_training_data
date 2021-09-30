import numpy as np

def log_(s=""):
    print(s)
    pass

def complex_image_to_image(image):
    log_(f"image.shape {image.shape}")

    image_real, image_imaginary = np.split(image, indices_or_sections=2, axis=-1)
    log_(f"image_real.shape {image_real.shape}")
    log_(f"image_real.min_max {np.min(image_real)} {np.max(image_real)}")

    log_(f"image_imaginary.shape {image_imaginary.shape}")
    log_(f"image_imaginary.min_max {np.min(image_imaginary)} {np.max(image_imaginary)}")

    image_cast = np.squeeze(image_real)
    log_(f"image_cast.shape {image_cast.shape}")

    image_result = image_cast.astype(np.float32)
    log_(f"image_result.shape {image_result.shape}")

    return image_result

def plot(image, ax):
    image = complex_image_to_image(image)

    log_spec = np.log(image.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(image), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

