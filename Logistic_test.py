import cv2
import numpy as np
from PIL import Image

def logistic(Img, x, u,times):
    M = Img.shape[0]
    N = Img.shape[1]
    for i in range(1, times):
        x = u * x * (1 - x)
    array = np.zeros(M * N)
    array[1] = x
    for i in range(1, M * N - 1):
        array[i + 1] = u * array[i] * (1 - array[i])
    array = np.array(array * 255, dtype='uint8')
    code = np.reshape(array, (M, N))
    xor = Img ^ code
    v = xor
    return v

# 0<x<1
x = 0.1
# 3.5699456...<u<=4
u = 4
times = 500

Img = cv2.imread('test1.jpeg')
Img = Img[:, :, [2, 1, 0]]
(r, g, b) = cv2.split(Img)
R = logistic(r, x, u, times)
G = logistic(g, x, u, times)
B = logistic(b, x, u, times)
merged = np.ones(Img.shape, dtype=np.uint8)
merged[:, :, 2] = B
merged[:, :, 1] = G
merged[:, :, 0] = R

Img = Image.fromarray(merged)
Img.save("C:/Users/李坤/PycharmProjects/D2" + '\\log1.jpeg')