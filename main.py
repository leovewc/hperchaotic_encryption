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
import jiemi as jiemi
def create_grayscale_alpha_image(P0, P1):
    if P0.shape != P1.shape:
        raise ValueError("Both images must have the same dimensions.")
    GA_image = np.zeros((P0.shape[0], P0.shape[1], 2), dtype=np.uint8)
    GA_image[:, :, 0] = P0  # Grayscale
    GA_image[:, :, 1] = P1  # Alpha
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

def runge_kutta(x, h, lambda_y, params):
    a, b, c, d, e, f, g = params
    max_val = 1e7# 这个大小刚好不至于爆表

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


def plot_data(X1, X2):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(X1, label='X1')
    plt.title('Time Series of X1')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(X2, label='X2')
    plt.title('Time Series of X2')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    P0 = cv2.imread('images1.jpeg', cv2.IMREAD_GRAYSCALE)
    P1 = cv2.imread('images2.jpeg', cv2.IMREAD_GRAYSCALE)

    try:
        GA_image = create_grayscale_alpha_image(P0, P1)
        PG, PA = GA_image[:, :, 0], GA_image[:, :, 1]
        hash_digest = generate_hash_key(GA_image)
        x_keys, lambda_y = generate_initial_key_stream(hash_digest)


        M, N = GA_image.shape[:2]
        sequence_length = M * N

        print("Initial Key Stream:", x_keys)
        print("Lambda:", lambda_y)


        params = (30, 10, 15.7, 5, 2.5, 4.45, 38.5)
        h = 0.00001
        x = np.array(x_keys, dtype=np.float64)  # Initial
        n_steps = 500 + sequence_length

        # Initial
        X1, X2, X3, X4, X5 = (np.zeros(sequence_length) for _ in range(5))

        for i in range(n_steps):
            x = runge_kutta(x, h, lambda_y, params)
            # 取501步
            if i >= 500:
                idx = i - 500  # Index
                X1[idx], X2[idx], X3[idx], X4[idx], X5[idx] = x

        # 检查
        print("Sample from X1:", X1[:5])  # 前5个
        print("Sample from X2:", X2[:5])
        print("Sample from X3:", X3[:5])
        print("Sample from X4:", X4[:5])
        print("Sample from X5:", X5[:5])

    except Exception as e:
        print("An error occurred:", e)

    ##plot_data(X1, X2)

    print("Length of X1:", len(X1))
    print("Expected Length (MxN):", M * N)


    I1, X1_prime = s.sort_chaotic_sequence(X1, M, N)

    # Output results for verification
    print("Sample from sorted matrix I1:", I1[:1])  # 检查
    print("Sample from index sequence matrix X'1:", X1_prime[:1])  # 检查


    I2, X2_prime = s.permute_and_sort_sequences(X2, X1_prime, M, N)
    I3, X3_prime = s.permute_and_sort_sequences(X3, X1_prime, M, N)

    print("Sorted matrix I2:", I2)
    print("New index sequence matrix X''2:", X2_prime)
    print("Sorted matrix I3:", I3)
    print("New index sequence matrix X''3:", X3_prime)

    PG_permuted = Zhiluan.permute_image_channel(PG, I2)
    PA_permuted = Zhiluan.permute_image_channel(PA, I3)

    #Zhiluan.display_images(PG, PG_permuted, 'Grayscale Channel')
    #Zhiluan.display_images(PA, PA_permuted, 'Alpha Channel')

    image_C = zuhe.combine_channels_to_image(PG_permuted, PA_permuted)
    zuhe.save_image(image_C, 'put.png')

    X4_prime = kuosan.convert_to_binary_with_floor(X4)
    X5_prime = kuosan.convert_to_binary_with_floor(X5)

    #print(image_C.shape)   #检查
    #重新把密文图像分解为灰度通道和alpha通道
    CG, CA = image_C[:, :, 0], image_C[:, :, 1]

    #引用s变换
    T_image = s_kuosan.process_channels(CG, CA, X4, X4_prime, X5, X5_prime)

    zuhe.save_image(T_image, 'T.png')

    print(f"T_image shape: {T_image.shape}")
    print(f"T_image content (first 10 elements): {T_image.flatten()[:10]}")

    #decrypted_image = jiemi.decrypt_image('T.png', ['images1.jpeg', 'images2.jpeg'], 8, 6328)
    #Image.fromarray(decrypted_image).save('decrypted_image.png')
if __name__ == "__main__":
    main()

