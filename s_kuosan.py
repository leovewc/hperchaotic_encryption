import numpy as np
import matplotlib.pyplot as plt
def s_box_diffusion_shun(channel):
    M, N = channel.shape
    channel_prime = []
    for j in range(M):
        if j % 2 == 0:
            for i in range(N):
                channel_prime.append(channel[j, i])
        else:
            for i in range(N - 1, -1, -1):
                channel_prime.append(channel[j, i])


    return np.array(channel_prime)


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
#def decimal_to_binary(channel):
    # Convert decimal matrix to binary strings
    #return [''.join(f'{int(pixel):08b}' for pixel in row) for row in channel]

def decimal_to_binary(decimal_sequence):

    binary_sequence = [format(int(num), '08b') for num in decimal_sequence]
    return binary_sequence

def binary_to_decimal(binary_sequence):
    return np.array([int(b, 2) for b in binary_sequence])

def reshape_to_2d(channel, M,N):
    if(len(channel) !=M*N):
        raise ValueError('channel must be equal to M*N')
    return np.reshape(channel, (M, N))


def combine_grayscale_and_alpha(grayscale_matrix, alpha_matrix):

    if grayscale_matrix.shape != alpha_matrix.shape:
        raise ValueError("Grayscale and alpha matrices must have the same dimensions.")

    encrypted_image = np.stack((grayscale_matrix, alpha_matrix), axis=-1)

    return encrypted_image

def process_channels(C_G, C_A, X4, X4_prime, X5, X5_prime):
    # 分别进行s顺向扩散
    C_G_middman = s_box_diffusion_shun(C_G)
    C_A_middman = s_box_diffusion_shun(C_A)

    # 再进行十进制到二进制转换
    C_G_prime = decimal_to_binary(C_G_middman)
    C_A_prime = decimal_to_binary(C_A_middman)

    print(f"C_G_prime (first 10 elements): {C_G_prime[:10]}")
    print(f"C_A_prime (first 10 elements): {C_A_prime[:10]}")

    #根据公式进行异或运算
    M, N = len(C_G_prime) // 8, 8  #  M*8N
    length = M * N

    # 转整数
    C_G_prime_int = [int(b, 2) for b in C_G_prime]
    C_A_prime_int = [int(b, 2) for b in C_A_prime]
    X4_prime_int = [int(x, 2) for x in X4_prime if isinstance(x, str) and all(c in '01' for c in x)]


    B1 = np.zeros(length, dtype=int)
    B2 = np.zeros(length, dtype=int)


    sum_X4_mod_2 = int(np.floor(np.sum(X4) % 2))
    B1[0] = sum_X4_mod_2 ^ C_G_prime_int[0] ^ X4_prime_int[0]
    B2[0] = sum_X4_mod_2 ^ C_A_prime_int[0] ^ X4_prime_int[0]


    for i in range(1, length):
        B1[i] = C_G_prime_int[i - 1] ^ B1[i - 1] ^ X4_prime_int[i % len(X4_prime_int)]
        B2[i] = C_A_prime_int[i - 1] ^ B2[i - 1] ^ X4_prime_int[i % len(X4_prime_int)]


    #B1_binary = [format(num, '08b') for num in B1]
    #B2_binary = [format(num, '08b') for num in B2]

    B1_binary = [format(int(b), '08b') for b in B1.flatten()]
    B2_binary = [format(int(b), '08b') for b in B2.flatten()]

    print(f"B1_binary (first 10 elements): {B1_binary[:10]}")
    print(f"B2_binary (first 10 elements): {B2_binary[:10]}")

    #转成十进制
    B1_decimal = binary_to_decimal(B1_binary)
    B2_decimal = binary_to_decimal(B2_binary)

    #转成二维矩阵
    B1_prime = reshape_to_2d(B1_decimal,M,N)
    B2_prime = reshape_to_2d(B2_decimal,M,N)

    #逆向s型扩散
    B1_middman = s_shape_diffusion_ni(B1_prime)
    B2_middman = s_shape_diffusion_ni(B2_prime)

    #转成二进制
    B1_prime_prime = decimal_to_binary(B1_middman)
    B2_prime_prime = decimal_to_binary(B2_middman)

    print(f"B1_prime_prime (first 10 elements): {B1_prime_prime[:10]}")
    print(f"B2_prime_prime (first 10 elements): {B2_prime_prime[:10]}")

    M, N = len(B1_prime_prime) // 8, 8  # M*8N
    length = M * N

    # 转整数
    B1_prime_prime_int = [int(b, 2) for b in B1_prime_prime]
    B2_prime_prime_int = [int(b, 2) for b in B2_prime_prime]
    X5_prime_int = [int(x, 2) for x in X5_prime]  # X'4 is a binary string list


    B3 = np.zeros(length, dtype=int)
    B4 = np.zeros(length, dtype=int)

   #异或
    sum_X5_mod_2 = int(np.floor(np.sum(X5) % 2))
    B3[0] = sum_X5_mod_2 ^ B1_prime_prime_int[0] ^ X5_prime_int[0]
    B4[0] = sum_X5_mod_2 ^ B2_prime_prime_int[0] ^ X5_prime_int[0]


    for i in range(1, length):
        B3[i] = B1_prime_prime_int[i - 1] ^ B1[i - 1] ^ X5_prime_int[i % len(X5_prime_int)]
        B4[i] = B2_prime_prime_int[i - 1] ^ B2[i - 1] ^ X5_prime_int[i % len(X5_prime_int)]

    # 转二进制
    B3_binary = [format(num, '08b') for num in B1]
    B4_binary = [format(num, '08b') for num in B2]

    # 转成十进制
    B3_decimal = binary_to_decimal(B3_binary)
    B4_decimal = binary_to_decimal(B4_binary)

    # 转成二维矩阵
    B3_prime = reshape_to_2d(B3_decimal, M, N)
    B4_prime = reshape_to_2d(B4_decimal, M, N)

    print(f"B3_prime shape: {B3_prime.shape}")
    print(f"B3_prime content (first 10 elements): {B3_prime.flatten()[:10]}")
    print(f"B4_prime shape: {B4_prime.shape}")
    print(f"B4_prime content (first 10 elements): {B4_prime.flatten()[:10]}")

    T_image = combine_grayscale_and_alpha(B3_prime, B4_prime)
    return T_image




