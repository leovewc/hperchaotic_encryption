import numpy as np
import cv2
import hashlib
import matplotlib.pyplot as plt
import numpy as np


def permute_image_channel(channel, index_sequence):

    flat_index_sequence = index_sequence.flatten()
    #检查
    if len(flat_index_sequence) != channel.size:
        raise ValueError(
            f"Index sequence length {len(flat_index_sequence)} does not match the number of elements in channel {channel.size}")

    flat_channel = channel.flatten()

    flat_index_sequence = flat_index_sequence.astype(int)

    permuted_flat = flat_channel[flat_index_sequence]
    #jc
    if permuted_flat.size != channel.size:
        raise ValueError(f"Cannot reshape array of size {permuted_flat.size} into shape {channel.shape}")

    #  2D
    permuted_channel = permuted_flat.reshape(channel.shape)

    return permuted_channel


def display_images(original, permuted, title):

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray', interpolation='none')
    plt.title('Original ' + title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(permuted, cmap='gray', interpolation='none')
    plt.title('Permuted ' + title)
    plt.axis('off')

    plt.show()