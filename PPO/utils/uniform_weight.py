# author by 蒋权
import numpy as np
from itertools import combinations
from scipy.special import comb


def init_weight(weight_size, objective, low_bound=0.0):
    h1 = 1
    while comb(h1 + objective - 1, objective - 1) <= weight_size:
        h1 = h1 + 1
    h1 = h1 - 1
    # 从H+M-1个隔板中选取m-1个,将H+M划分成m份
    x = list(combinations(range(1, h1 + objective), objective - 1))
    weight_vector = np.array(list(combinations(range(1, h1 + objective), objective - 1)))
    # 得到每一份的大小 第一份的大小 = 隔板大小 - 0
    # 中间的大小 = 下一隔板的大小 - 上一隔板的大小
    # 最后一份的大小 = H+M - 最后隔板的大小
    weight_vector = np.hstack((weight_vector, h1 + objective + np.zeros((weight_vector.shape[0], 1)))) - \
                    np.hstack((np.zeros((weight_vector.shape[0], 1)), weight_vector))
    # 实际大小 = (每份大小-1) / H
    weight_vector = (weight_vector - 1) / h1
    if h1 < objective:
        h2 = 0
        while comb(h2 + objective - 1, objective - 1) + comb(h1 + objective - 1, objective - 1) < weight_size:
            h2 = h2 + 1
        h2 = h2 - 1
        if h2 > 0:
            temp = np.array(list(combinations(range(1, h2 + objective), objective - 1)))
            temp = np.hstack((weight_vector, h2 + objective + np.zeros((temp.shape[0], 1)))) - \
                   np.hstack((np.zeros((temp.shape[0], 1)), weight_vector))
            temp = (temp - 1) / h2
            weight_vector = np.vstack((weight_vector, temp))
    weight_size = weight_vector.shape[0]
    flag = np.ones(weight_size).astype(bool)
    for i in range(weight_size):
        w = weight_vector[i]
        if len(np.where(w < low_bound)[0]) != 0:
            flag[i] = False
    weight_vector = weight_vector[flag]
    weight_size = weight_vector.shape[0]
    return weight_vector, weight_size


def cweight(weight, env, sa=None, ra=None, mode=0):
    if len(weight.shape) != 2:
        raise Exception(f'weight shape should be 2, but got {weight.shape}, weight = {weight}')
    wsize = weight.shape[0]
    w = weight[np.random.randint(low=0, high=wsize)]
    if mode == 0:
        env.w1 = w.tolist()
        env.w2 = w.tolist()
        sa.cweight(w)
        ra.cweight(w)
    elif mode == 1:
        env.w1 = w.tolist()
        sa.cweight(w)
    elif mode == 2:
        env.w2 = w.tolist()
        ra.cweight(w)
    else:
        raise Exception(f'mode = {mode} is not in 0, 1, 2')
    return w


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    # 110个权重向量,剔除不满足约束向量的大小为36
    weight, size = init_weight(110, 3, 0.1)
    print(size)
    for i in range(size):
        w = weight[i]
        ax.plot([0, w[0]], [0, w[1]], [0, w[2]], linewidth=2, c='r')
        ax.plot(w[0], w[1], w[2], '-o')
    plt.pause(100)