from PIL import Image
from typing import Union
import jax
import jax.numpy as jnp

# import skimage.io
import numpy as np


# Internally we represent images and volumes as numpy arrays of floating point data between 0.0 and
# 1.0.

# For images, I consider the bottom left pixel to be (0, 0), and the first index of the array to be
# the x coordinate. This is not the convention used by PIL, so we have to do axis swapping to
# convert between the two representations.


def array_to_image(array):
    """
    Creates a PIL image from a numpy array.

    Pixel (0, 0) in the numpy array becomes the bottom left pixel. Pixel (1, 0) is one to the right
    of the bottom left pixel.
    """

    if array.dtype == np.uint8:
        pass
    elif array.dtype in (np.float16, np.float32, np.float64):
        # Convert to uint8.
        array = np.clip(np.array(array), 0.0, 1.0)
        array = (255 * array).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {array.dtype}")

    # Convert to PIL's expected format (pixel (0, 0) is top left, and (1, 0) is one pixel below the
    # top left).
    array = np.flip(np.swapaxes(np.array(array), 0, 1), 0)

    n_channels = array.shape[2] if len(array.shape) == 3 else 1
    if n_channels == 1:
        # 8 bit greyscale
        format = "L"
    elif n_channels == 3:
        format = "RGB"
    elif n_channels == 4:
        format = "RGBA"
    else:
        raise ValueError(f"Unsupported number of channels: {n_channels}")

    return Image.fromarray(array, format)


def write_array_as_image(array, filename):
    image = array_to_image(array)
    image.save(filename)


def write_array_as_u16_tiff(array: Union[np.ndarray, jax.Array], filename):
    """
    This method assumes you've already converted your data to uint16.
    """

    assert array.ndim == 2
    if array.dtype not in (np.uint16, jnp.uint16):
        raise ValueError(f"Unsupported dtype: {array.dtype}")

    image = Image.fromarray(np.asarray(array), "I;16")
    image.save(filename)
