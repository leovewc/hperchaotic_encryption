import numpy as np

def combine_channels_to_image(grayscale_channel, alpha_channel):

    if grayscale_channel.shape != alpha_channel.shape:
        raise ValueError("Grayscale and alpha channels must have the same dimensions.")

    encrypted_image = np.stack((grayscale_channel, alpha_channel), axis=-1)

    return encrypted_image

from PIL import Image

def save_image(array, file_path):


    image = Image.fromarray((array * 255).astype(np.uint8), 'LA')
    image.save(file_path)



