import numpy as np
import cv2
import hashlib

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


def main():
    P0 = cv2.imread('.venv/images1.jpeg', cv2.IMREAD_GRAYSCALE)
    P1 = cv2.imread('.venv/images2.jpeg', cv2.IMREAD_GRAYSCALE)

    try:
        GA_image = create_grayscale_alpha_image(P0, P1)
        hash_digest = generate_hash_key(GA_image)
        x_keys, lambda_y = generate_initial_key_stream(hash_digest)

        print("Initial Key Stream:", x_keys)
        print("Lambda:", lambda_y)
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
