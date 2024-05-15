import numpy as np
import cv2
import hashlib

def create_grayscale_alpha_image(P0, P1):

    if P0.shape != P1.shape:
        raise ValueError("Both images must have the same dimensions.")
    # RGBA
    height, width = P0.shape
    GA_image = np.zeros((height, width, 2), dtype=np.uint8)
    GA_image[:, :, 0] = P0
    GA_image[:, :, 1] = P1
    return GA_image

def generate_hash_key(PG, PA):

    # 将PG和PA转换为一维数组并拼接
    PC = np.concatenate((PG.flatten(), PA.flatten()))
    # SHA-512
    hash_obj = hashlib.sha512(PC.tobytes())
    hash_digest = hash_obj.digest()
    # 转为64个十进制数
    ci = [int.from_bytes(hash_digest[i:i+1], 'big') for i in range(64)]
    return ci

def generate_initial_key_stream(ci):

    x = np.zeros(5)
    for i in range(5):
        # 每四个值进行一次异或操作
        x[i] = ci[10*i] ^ ci[10*i+1] ^ ci[10*i+2] ^ ci[10*i+3]
    lambda_y = np.mod(ci[50] ^ ci[51] ^ ci[52] ^ ci[53], 6.35) - 4.0
    return x, lambda_y

# 示例使用
# 读取图像
P0 = cv2.imread('test1.jpeg', cv2.IMREAD_GRAYSCALE)
P1 = cv2.imread('test1.jpeg', cv2.IMREAD_GRAYSCALE)

# 创建 Grayscale-Alpha 图像
GA_image = create_grayscale_alpha_image(P0, P1)
PG = GA_image[:, :, 0]
PA = GA_image[:, :, 1]

# 生成哈希密钥
ci = generate_hash_key(PG, PA)

# 生成初始密钥流
initial_key_stream, lambda_y = generate_initial_key_stream(ci)

print("Initial Key Stream:", initial_key_stream)
print("Lambda:", lambda_y)
