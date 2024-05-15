import numpy as np
import cv2
import hashlib
import matplotlib.pyplot as plt
from PIL import Image
import sort as s
import Zhiluan as Zhiluan
import zuhe as zuhe
import kuosan as kuosan
import s_kuosan as s_kuosan

# Define functions directly from sort.py
def sort_chaotic_sequence(X1, M, N):
    X1_matrix = X1.reshape((M, N))
    I1 = np.zeros_like(X1_matrix)
    X1_prime = np.zeros_like(X1_matrix, dtype=int)
    for i in range(M):
        indices = np.argsort(X1_matrix[i, :])
        I1[i, :] = X1_matrix[i, indices]
        X1_prime[i, :] = indices
    return I1, X1_prime

def permute_and_sort_sequences(X, X_prime, M, N):
    X_matrix = X.reshape((M, N))
    permuted_matrix = np.zeros_like(X_matrix)
    sorted_matrix = np.zeros_like(X_matrix)
    new_indices = np.zeros_like(X_matrix, dtype=int)
    for i in range(M):
        permuted_matrix[i, :] = X_matrix[i, X_prime[i, :]]
    for i in range(M):
        indices = np.argsort(permuted_matrix[i, :])
        sorted_matrix[i, :] = permuted_matrix[i, indices]
        new_indices[i, :] = indices
    return sorted_matrix, new_indices

# Define functions directly from Zhiluan.py
def permute_image_channel(channel, index_sequence):
    flat_index_sequence = index_sequence.flatten()
    flat_channel = channel.flatten()
    permuted_flat = flat_channel[flat_index_sequence]
    return permuted_flat.reshape(channel.shape)

# Define functions directly from kuosan.py
def convert_to_binary_with_floor(X):
    floored_X = np.floor(X).astype(int)
    binary_string = [format(int(x), '08b') for x in floored_X]
    return binary_string

# Define functions directly from s_kuosan.py
def binary_to_decimal(binary_sequence,M,N):
    valid_binary_sequence = [b for b in binary_sequence if isinstance(b, str) and all(c in '01' for c in b)]
    print(f"Valid binary sequence length: {len(valid_binary_sequence)}")
    if len(valid_binary_sequence) != M * N:
        raise ValueError(f"Expected length: {M * N}, but got: {len(valid_binary_sequence)}")
    return np.array([int(b, 2) for b in valid_binary_sequence])



def reshape_to_2d(channel, M, N):
    if len(channel) != M * N:
        raise ValueError('channel must be equal to M*N')
    return np.reshape(channel, (M, N))

def s_shape_diffusion_ni(Q):
    M, N = Q.shape
    Q_prime = []
    for j in range(M):
        if j % 2 == 0:
            for i in range(N - 1, -1, -1):
                Q_prime.append(Q[j, i])
        else:
            for i in range(N):
                Q_prime.append(Q[j, i])
    return np.array(Q_prime)

def separate_channels(image):
    grayscale_channel = image[:, :, 0]
    alpha_channel = image[:, :, 1]
    return grayscale_channel, alpha_channel

# Define functions directly from zuhe.py
def combine_channels_to_image(grayscale_channel, alpha_channel):
    if grayscale_channel.shape != alpha_channel.shape:
        raise ValueError("Grayscale and alpha channels must have the same dimensions.")
    encrypted_image = np.stack((grayscale_channel, alpha_channel), axis=-1)
    return encrypted_image

def save_image(array, file_path):
    image = Image.fromarray(array, 'LA')  # 'LA' for Luminance-Alpha
    image.save(file_path)
def create_grayscale_alpha_image(P0, P1):
    if P0.shape != P1.shape:
        raise ValueError("Both images must have the same dimensions.")
    GA_image = np.zeros((P0.shape[0], P0.shape[1], 2), dtype=np.uint8)
    GA_image[:, :, 0] = P0  # Grayscale channel
    GA_image[:, :, 1] = P1  # Alpha channel
    return GA_image

def generate_hash_key(GA_image):
    PG, PA = GA_image[:, :, 0], GA_image[:, :, 1]
    PC = np.concatenate((PG.flatten(), PA.flatten()))
    hash_obj = hashlib.sha512(PC.tobytes())
    return hash_obj.digest()

def generate_initial_key_stream(hash_digest):
    ci = [int.from_bytes(hash_digest[i:i+1], 'big') for i in range(64)]
    x_keys = [ci[i*10] ^ ci[i*10+1] ^ ci[i*10+2] ^ ci[i*10+3] for i in range(5)]
    lambda_y = np.mod(ci[50] ^ ci[51] ^ ci[52] ^ ci[53], 6.35) - 4.0
    return x_keys, lambda_y
# Define the decryption process
def runge_kutta(x, h, lambda_y, params):
    a, b, c, d, e, f, g = params
    max_val = 1e7

    def derivs(x):
        x = np.tanh(x / max_val) * max_val
        x1, x2, x3, x4, x5 = x
        dx1dt = (a + lambda_y) * (x2 - x1) + x2 * x3 * x4
        dx2dt = (b + lambda_y) * (x1 + x2) + x5 - x1 * x3 * x4
        dx3dt = -(c + lambda_y) * x2 - d * x3 - e * x4 + x1 * x2 * x4
        dx4dt = -f * x4 + x1 * x2 * x3
        dx5dt = -(g + lambda_y) * (x1 + x2)
        return np.array([dx1dt, dx2dt, dx3dt, dx4dt, dx5dt])

    k1 = h * derivs(x)
    k2 = h * derivs(x + 0.5 * k1)
    k3 = h * derivs(x + 0.5 * k2)
    k4 = h * derivs(x + k3)
    x_next = x + (k1 + 2 * (k2 + k3) + k4) / 6

    return np.clip(x_next, -max_val, max_val)

def regenerate_chaotic_sequences(x_keys, lambda_y, M, N):
    num_steps = M * N + 500
    h = 0.01
    params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x = np.array(x_keys)
    X1, X2, X3, X4, X5 = np.zeros(M * N), np.zeros(M * N), np.zeros(M * N), np.zeros(M * N), np.zeros(M * N)
    for i in range(num_steps):
        x = runge_kutta(x, h, lambda_y, params)
        if i >= 500:
            idx = i - 500
            X1[idx], X2[idx], X3[idx], X4[idx], X5[idx] = x
    return {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5}


def decrypt_image(encrypted_image_path, initial_image_paths, M, N):
    encrypted_image = np.array(Image.open(encrypted_image_path))

    # Separate the combined image into grayscale and alpha channels
    CG, CA = separate_channels(encrypted_image)

    # Load the initial images
    P0 = cv2.imread(initial_image_paths[0], cv2.IMREAD_GRAYSCALE)
    P1 = cv2.imread(initial_image_paths[1], cv2.IMREAD_GRAYSCALE)

    # Create the GA image and generate the hash key and initial key stream
    GA_image = create_grayscale_alpha_image(P0, P1)
    hash_digest = generate_hash_key(GA_image)
    x_keys, lambda_y = generate_initial_key_stream(hash_digest)

    # Regenerate the chaotic sequences
    chaotic_sequences = regenerate_chaotic_sequences(x_keys, lambda_y, M, N)




    # Reverse the encryption process
    # Step 1: Separate the permuted image into grayscale and alpha channels
    B3, B4 = separate_channels(encrypted_image)

    #print(f"B3 content: {B3[:10]}")  # 只打印前 10 个元素进行调试
    #print(f"B4 content: {B4[:10]}")
    #print(f"B3 content: {B3.flatten()[:10]}")  # 只打印前 10 个元素进行调试
    #print(f"B4 content: {B4.flatten()[:10]}")

    # Step 2: Convert binary sequences to decimal
    B1_decimal = binary_to_decimal(B3.flatten(),M,N)
    B2_decimal = binary_to_decimal(B4.flatten(),M,N)

    #print(f"B1_decimal length: {len(B1_decimal)}")
    #print(f"B2_decimal length: {len(B2_decimal)}")

    # Step 3: Reshape to 2D matrices
    B1_prime = reshape_to_2d(B1_decimal, M, N)
    B2_prime = reshape_to_2d(B2_decimal, M, N)

    # Step 4: Reverse the diffusion process
    B1_middman = s_shape_diffusion_ni(B1_prime)
    B2_middman = s_shape_diffusion_ni(B2_prime)

    # Step 5: Reverse permutation based on index sequences
    I2, X2_prime = permute_and_sort_sequences(chaotic_sequences['X2'], chaotic_sequences['X1_prime'], M, N)
    I3, X3_prime = permute_and_sort_sequences(chaotic_sequences['X3'], chaotic_sequences['X1_prime'], M, N)
    PG_permuted = permute_image_channel(B1_middman, I2)
    PA_permuted = permute_image_channel(B2_middman, I3)

    # Step 6: Combine the permuted channels into the original image
    original_image = combine_channels_to_image(PG_permuted, PA_permuted)

    return original_image