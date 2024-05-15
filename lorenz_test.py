import numpy as np
import matplotlib.pyplot as plt

def lorenz(t, x, a, b, c, r):
    """超混沌 Lorenz 系统的微分方程"""
    dx = a * (x[1] - x[0]) + x[3]
    dy = x[0] * c - x[1] - x[0] * x[2]
    dz = x[0] * x[1] - b * x[2]
    dw = -x[1] * x[2] + r * x[3]
    return np.array([dx, dy, dz, dw])



# 参数设置
a = 10
b = 8/3
c = 28
r = -1
x0 = np.array([0.1, 0, 0, 0])
dt = 0.01
num_steps = 10000

# 使用四阶龙格-库塔法离散化 Lorenz 系统
t, x = runge_kutta_lorenz(a, b, c, r, x0, dt, num_steps)

# 绘制轨迹
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.set_title('super Lorenz System - runge_kutta_lorenz')
ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax = fig.add_subplot(122, projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 3], lw=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('w')

plt.show()

