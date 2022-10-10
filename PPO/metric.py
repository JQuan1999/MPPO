import math
import json
import numpy as np
import copy
import os

from utils.utils import save_result


def get_ps(P):
    function_value = copy.deepcopy(P)
    size, objective = function_value.shape
    index = []
    for i in range(size):
        diff = np.tile(function_value[i], (size, 1)) - function_value
        less = np.sum(diff < 0, axis=1).reshape(size, )
        big = np.sum(diff > 0, axis=1).reshape(size, )
        equal = np.sum(diff == 0, axis=1).reshape(size, )
        equal_not_zero = (equal != objective)
        less_equal_zero = (less == 0)
        # 被其他个体支配的索引
        dominated_index = less_equal_zero * equal_not_zero
        # 支配其他个体的索引
        # dominate_index = less_equal_zero * big_equal_zero
        if np.sum(dominated_index) == 0:
            index.append(i)
    return P[index]


def GD(A, P):
    size = A.shape[0]
    dis = 0
    for i in range(size):
        s = A[i]
        diff = np.tile(s, (P.shape[0], 1)) - P
        dis += np.sum(diff ** 2, axis=1).min()
    gd = math.sqrt(dis) / size
    return gd


def IGD(P, A):
    size = P.shape[0]
    dis = 0
    for i in range(size):
        s = P[i]
        diff = np.tile(s, (A.shape[0], 1)) - A
        dis += np.sum(diff ** 2, axis=1).min()
    igd = math.sqrt(dis) / size
    return igd


def spread(A, P):
    sizeA = A.shape[0]
    objs = A.shape[1]
    d_AA = np.zeros(sizeA)
    flag = np.ones(sizeA).astype(bool)
    for i in range(sizeA):
        s = A[i]
        flag[i] = False
        diff = np.tile(s, (sizeA - 1, 1)) - A[flag]
        flag[i] = True
        d_AA[i] = math.sqrt(np.sum(diff ** 2, axis=1).min())
    ave_d = d_AA.mean()

    dAP = 0
    for o in range(objs):
        singleA = A[:, o]
        singleP = P[:, o]
        amax = singleA.max()
        amin = singleA.min()
        pmax = singleP.max()
        pmin = singleP.min()
        dAP += max(amax - pmin, pmax - amin)

    v = np.abs((d_AA - ave_d)).sum()
    deta = (dAP + v) / (dAP + ave_d * sizeA)
    return deta


def metric():
    agent_log = './log/eval/multi-agent.json'
    random_log = './log/eval/multi-random.json'
    with open(agent_log, 'r') as f:
        a_result = json.load(f)
    with open(random_log, 'r') as f:
        r_result = json.load(f)
    test_data = './data/test'
    keys = ['/'.join([test_data, insdir]) for insdir in os.listdir(test_data)]
    agent_key = "agent"
    random_key = "random"

    result = {agent_key: {}, random_key: {}}
    result_path = './log/pareto'
    for key in keys:
        a_value = np.array(a_result[key])
        r_value = np.array(r_result[key])
        a_value[:, 0] *= -1
        r_value[:, 0] *= -1
        P = np.concatenate((a_value, r_value), axis=0)
        v_max = P.max(axis=0)
        v_min = P.min(axis=0)
        P = ((P - np.tile(v_min, (P.shape[0], 1))) / (np.tile(v_max, (P.shape[0], 1)) - np.tile(v_min, (P.shape[0], 1))))
        a_value = P[:a_value.shape[0]]
        r_value = P[a_value.shape[0]:]
        P = get_ps(P)
        gd_a = GD(a_value, P)
        gd_r = GD(r_value, P)
        igd_a = IGD(P, a_value)
        igd_r = IGD(P, r_value)
        spread_a = spread(a_value, P)
        spread_r = spread(r_value, P)
        result[agent_key][key] = [gd_a, igd_a, spread_a]
        result[random_key][key] = [gd_r, igd_r, spread_r]
        print(f'key = {key},agent metric = {[gd_a, igd_a, spread_a]}')
        print(f'key = {key},random metric = {[gd_r, igd_r, spread_r]}')
    save_result(result, result_path)
    print('end')


if __name__ == "__main__":
    metric()