import math
import json
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from utils.config import config
from mpl_toolkits.mplot3d import axes3d
from utils.utils import save_result
from env import color

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文


def param_experiment_metric():
    args = config()
    args.test_data = "./data/test4/j40_m20_n60"
    args.log_dir = "./log/param_experiment"
    metric(args)


def param_metric_anaylize():
    args = config()

    args.metric_result_dir += "/02-21-16-25.json"
    with open(args.metric_result_dir, 'r') as f:
        result = json.load(f)
    test_data = "j40_m20_n60"
    metric_v = []
    for key in result:
        metric_v.append(result[key][test_data]["metric"])
    metric_v = np.array(metric_v)

    max_v = metric_v.max(axis=0).reshape(1, -1)
    min_v = metric_v.min(axis=0).reshape(1, -1)

    norm_v = (metric_v - min_v) / (max_v - min_v)
    ave_v = np.sum(norm_v, axis=1) / 3
    print(ave_v)


def show_trend():
    values = [[0.53637, 0.28400, 0.26718], [0.51174, 0.25214, 0.32366], [0.44566, 0.27612, 0.36577]]
    ticks = [[5, 10, 15], [64, 128, 256], [5, 10, 15]]
    fig, axes = plt.subplots(1, 3, sharey=True)
    axes[0].set_ylabel("Fmean")
    axes[0].plot(ticks[0], values[0])
    axes[0].set_xticks(ticks[0])
    axes[0].set_title("Epoch")

    axes[1].plot(ticks[1], values[1])
    axes[1].set_xticks(ticks[1])
    axes[1].set_title("Batchsize")

    axes[2].plot(ticks[2], values[2])
    axes[2].set_xticks(ticks[2])
    axes[2].set_title("Step")

    plt.show()


def get_ps(P):
    function_value = copy.deepcopy(P)
    size, objective = function_value.shape
    index = []
    for i in range(size):
        diff = np.tile(function_value[i], (size, 1)) - function_value
        less = np.sum(diff < 0, axis=1).reshape(size, )
        equal = np.sum(diff == 0, axis=1).reshape(size, )
        dominated_index = ((less == 0) & (equal != objective))
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


def metric(args):
    # 获取最后一个文件名,根据最后一个文件名判断是对所有规模的调度问题进行对比还是只对比单个
    last_dir = args.test_data.split('/')[-1]
    if last_dir[:4] == "test":
        keys = os.listdir(args.test_data)
    else:
        keys = [last_dir]
    # 获取对比实验数据
    log_dir = args.log_dir
    files = os.listdir(log_dir)
    log_list = ['/'.join([log_dir, file]) for file in files]

    # 对比算法的名称
    # algo_name = [file.split('-')[-1].split('.')[0] for file in files]
    algo_name = files

    # 保存评价指标结果
    result = {}
    for name in algo_name:
        result[name] = {}

    # key为测试问题规模大小:m=j10_m10_n30邓
    for key in keys:
        objs = []
        times = []
        # log_list记录了所有要对比的实验结果, log记录一条实验结果
        for log in log_list:
            with open(log, 'r') as f:
                r = json.load(f)
                objs.append(np.array(r[key]["result"]))
                times.append(r[key]["time"])

        P = np.concatenate([obj for obj in objs], axis=0)
        v_max = P.max(axis=0)
        v_min = P.min(axis=0)
        # 归一化
        P = ((P - np.tile(v_min, (P.shape[0], 1))) / (
                np.tile(v_max, (P.shape[0], 1)) - np.tile(v_min, (P.shape[0], 1))))
        norm_objs = []
        begin = 0
        # norm_objs记录归一化后的目标函数值
        for obj in objs:
            norm_objs.append(P[begin:(begin + obj.shape[0])])
            begin = begin + obj.shape[0]
        P = get_ps(P)

        # 计算评价指标
        for i, norm_obj in enumerate(norm_objs):
            gd = GD(norm_obj, P)
            igd = IGD(P, norm_obj)
            spr = spread(norm_obj, P)
            t = times[i]
            algo = algo_name[i]
            result[algo][key] = {}
            result[algo][key]["metric"] = [gd, igd, spr]
            result[algo][key]["time"] = t
            print(f'key = {key}, algorithm = {algo} metric value = {gd, igd, spr} time = {t}')
        print('--------------------------------------')
    save_result(result, args.metric_result_dir)
    print('end')


def print_metric():
    log_dir = './log/compared_tradition'
    log_files = os.listdir(log_dir)
    algo_name = [log.split('-')[-1].split('.')[0] for log in log_files]

    test_dir = './data/test4'
    test_datas = os.listdir(test_dir)

    metric_file = './log/pareto/10-26-17-39.json'
    with open(metric_file, 'r') as f:
        result = json.load(f)

    for data in test_datas:
        metrics = []
        for algo in algo_name:
            m = result[algo][data]["metric"]
            metrics.append(m)
        metrics = np.array(metrics)
        min_index = np.argmin(metrics[:, 0], axis=0)
        print(f'For metric gd and data {data}, the best algo is {algo_name[min_index]}'
              f' the value is {metrics[min_index].tolist()}')
        min_index = np.argmin(metrics[:, 1], axis=0)
        print(f'For metric igd and data {data}, the best algo is {algo_name[min_index]}'
              f' the value is {metrics[min_index].tolist()}')
        min_index = np.argmin(metrics[:, 2], axis=0)
        print(f'For metric spread and data {data}, the best algo is {algo_name[min_index]}'
              f' the value is {metrics[min_index].tolist()}')
        print('-------------------------------')

    for data in test_datas:
        times = []
        for algo in algo_name:
            t = result[algo][data]["time"]
            times.append(t)
        print(f'For test data {data}, the run time of {algo_name} is {times}')


def show_pareto():
    test_data = './data/test4'
    keys = os.listdir(test_data)[5:7]
    # key = "j30_m20_n40"
    log_dir = 'log/compared_rule'
    files = os.listdir(log_dir)
    log_list = ['/'.join([log_dir, file]) for file in files]
    algo_name = [file.split('-')[-1].split('.')[0] for file in files]

    new_name = []
    for i, algo in enumerate(algo_name):
        if i == 0:
            new_name.append("MPPO")
        elif i == 1:
            new_name.append("R-R")
        elif 1 < i < 6:
            new_name.append(algo+"-R")
        else:
            new_name.append("R-"+algo)
    print(new_name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
    axes = axes.reshape(-1, )
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    c = color
    c = np.array(c).reshape(-1, 3)
    for index, key in enumerate(keys):
        for i, log in enumerate(log_list):
            with open(log, 'r') as f:
                r = json.load(f)
                obj = np.array(r[key]["result"])
            obj = get_ps(obj)
            axes[index].scatter3D(obj[:, 0], obj[:, 1], obj[:, 2], c=c[i].reshape(1, -1), marker=mark[i], label=new_name[i], s=10)
            axes[index].legend(fontsize=10, ncol=3)
            axes[index].tick_params(labelsize=8)
            axes[index].set_xlabel('完工时间', fontsize=10)
            axes[index].set_ylabel('延迟时间', fontsize=10)
            axes[index].set_zlabel('能耗', fontsize=10)
            axes[index].set_title(key, fontsize=10)
        axes[index].ticklabel_format(axis='both', style='sci', scilimits=[-1, 2])
    plt.show()


if __name__ == "__main__":
    show_trend()