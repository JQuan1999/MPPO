import math
import json
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from utils.config import config
from utils.utils import save_result
from env import color

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文


def param_experiment_metric():
    args = config()
    args.test_data = "./data/test4/j40_m20_n60"
    args.agent_eval = "./log/param_experiment"
    metric(args)


def param_metric_anaylize():
    args = config()

    args.metric_result += "./log/pareto/02-21-16-25.json"
    with open(args.metric_result, 'r') as f:
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
    # values 分别代表三个参数在不同取值下的fmean值
    # 需要根据param_metric_anaylize输出的结果进行计算此部分为手动计算
    # [0.80989862 0.31065899 0.48857733 0.5346363  0.21804045 0.54934482 0.39070418 0.22775922 0.18309056]

    values = [[0.53637, 0.434, 0.26718], [0.51174, 0.25214, 0.407], [0.52899, 0.34279, 0.36577]]
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


def _metric(args):
    # 获取最后一个文件名,根据最后一个文件名判断是对所有规模的调度问题进行对比还是只对比单个
    last_dir = args.test_data.split('/')[-1]
    if last_dir[:4] == "test":
        keys = os.listdir(args.test_data)
    else:
        keys = [last_dir]
    # 获取对比实验数据
    # 便于处理将agent和r-r的结果都放在rule_eval文件夹下
    log_dir = args.rule_eval
    files = os.listdir(log_dir)
    log_list = ['/'.join([log_dir, file]) for file in files]

    # 对比算法的名称
    algo_name = [file.split('-')[-1].split('.')[0] for file in files]
    # algo_name = files

    # 保存评价指标结果
    result = {}
    for name in algo_name:
        result[name] = {}

    # key为测试问题规模大小:m=j10_m10_n30等
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
        # norm_objs记录对比方法的目标函数归一化后的目标函数值
        for obj in objs:
            norm_objs.append(P[begin:(begin + obj.shape[0])])
            begin = begin + obj.shape[0]
        P = get_ps(P) # 得到对比方法组成的pareto最优解

        # 对每个对比方法 计算评价指标
        for i, norm_obj in enumerate(norm_objs):
            gd = GD(norm_obj, P)
            igd = IGD(P, norm_obj)
            spr = spread(norm_obj, P)
            t = times[i]
            algo = algo_name[i]
            result[algo][key] = {}
            result[algo][key]["metric"] = [gd, igd, spr] # 记录algo算法在key规模的调度问题上的评价指标值
            result[algo][key]["time"] = t
            print(f'key = {key}, algorithm = {algo} metric value = {gd, igd, spr} time = {t}')
        print('--------------------------------------')
    save_result(result, args.metric_result)
    print('end')


def _print_metric(args):
    # 便于处理将agent和r-r的结果都放在rule_eval文件夹下
    log_dir = args.rule_eval
    log_files = os.listdir(log_dir)
    algo_name = [log.split('-')[-1].split('.')[0] for log in log_files]

    test_dir = args.test_data
    test_datas = os.listdir(test_dir)
    test_datas = test_datas[-1:] + test_datas[0:-1] # 重新排序 因为最小规模的测试问题j7_m5_n10在最后一个
    metric_file = args.metric_result
    with open(metric_file, 'r') as f:
        result = json.load(f)

    # 依次对每个规模的动态调度问题计算3个指标中最好的算法
    def print_best():
        for data in test_datas:
            metrics = []
            for algo in algo_name:
                m = result[algo][data]["metric"]
                metrics.append(m)
            # 将所有algo的结果拼接到一起
            metrics = np.array(metrics)
            # 第0列是gd值
            min_index = np.argmin(metrics[:, 0], axis=0)
            print(f'For metric gd and data {data}, the best algo is {algo_name[min_index]}'
                  f' the value is {metrics[min_index].tolist()}')
            # 第1列是igd值
            min_index = np.argmin(metrics[:, 1], axis=0)
            print(f'For metric igd and data {data}, the best algo is {algo_name[min_index]}'
                  f' the value is {metrics[min_index].tolist()}')
            # 第2列是spread值
            min_index = np.argmin(metrics[:, 2], axis=0)
            print(f'For metric spread and data {data}, the best algo is {algo_name[min_index]}'
                  f' the value is {metrics[min_index].tolist()}')
            print('-------------------------------')

    # 输出cpu时间
    def print_time():
        for data in test_datas:
            times = []
            for algo in algo_name:
                t = result[algo][data]["time"]
                times.append(t)
            print(f'For test data {data}, the run time of {algo_name} is {times}')

    # 分别以每个评价指标 统计对比算法在每个测试问题上的评价指标值
    def print_obj():
        row = len(test_datas)
        col = len(algo_name)
        # 输出表格第一行的第一列
        print("\t", end="")
        for algo in algo_name:
            print(algo, end="\t")
        print()
        for data in test_datas:
            print(data)
        # 输出表格数值
        for obj in range(args.objective):
            if obj == 0:
                print('For GD')
            elif obj == 1:
                print("For IGD")
            else:
                print("For spread")
            obj_v = []
            for data in test_datas:
                for algo in algo_name:
                    m = result[algo][data]["metric"]
                    obj_v.append(round(m[obj], 3))
            obj_v = np.array(obj_v).reshape((row, col))
            print(obj_v)
    print_best()


def _show_pareto(args):
    test_datas = os.listdir(args.test_data)
    # listdir获得的文件排序乱了 需要重排
    test_datas = test_datas[-1:] + test_datas[:-1]
    # key = "j30_m20_n40"
    files = os.listdir(args.rule_eval)
    log_list = ['/'.join([args.rule_eval, file]) for file in files]
    algo_name = [file.split('-')[-1].split('.')[0] for file in files]

    new_name = []
    # 为了便于处理algo_name只保存了agent, FIFO, SQT等单个规则的关键词 因此需要加上R
    for i, algo in enumerate(algo_name):
        if algo == "agent":
            new_name.append("MPPO")
        elif algo == "random":
            new_name.append("R-R")
        else:
            new_name.append(f"{algo}-R")
    print(new_name)
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    c = color
    c = np.array(c).reshape(-1, 3)

    # 一次绘制一个图
    def print_signle():
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': '3d'})
        # key保存了不同规模的调度问题
        # 依次对不同的调度规模问题画出对比算法的pareto前沿
        for index, data in enumerate(test_datas):
            for log_index, log in enumerate(log_list):
                with open(log, 'r') as f:
                    r = json.load(f)
                    obj = np.array(r[data]["result"])
                obj = get_ps(obj)  # 计算pareto前沿
                ax.scatter3D(obj[:, 0], obj[:, 1], obj[:, 2], c=c[i].reshape(1, -1), marker=mark[log_index],
                             label=new_name[log_index], s=10)
                ax.legend(fontsize=10, ncol=3)
                ax.tick_params(labelsize=8)
                ax.set_xlabel('完工时间', fontsize=10)
                ax.set_ylabel('加权延期时间', fontsize=10)
                ax.set_zlabel('加工能耗', fontsize=10)
                ax.set_title(data, fontsize=10)
            ax.ticklabel_format(axis='both', style='sci', scilimits=[-1, 2])
            plt.pause(10)
            plt.cla()

    # 每次画两个规模的图1行2列
    def print_pair():
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})
        # 按两个一组分组
        pair_test_datas = []
        for index in range(0, len(test_datas), 2):
            pair_test_datas.append([test_datas[index], test_datas[index+1]])

        for pair_data in pair_test_datas:
            for index, data in enumerate(pair_data):
                for algo_index, log in enumerate(log_list):
                    with open(log, 'r') as f:
                        r = json.load(f)
                        obj = np.array(r[data]["result"])
                    obj = get_ps(obj)  # 计算pareto前沿
                    axes[index].scatter3D(obj[:, 0], obj[:, 1], obj[:, 2], c=c[algo_index].reshape(1, -1), marker=mark[algo_index],
                                          label=new_name[algo_index], s=10)
                    axes[index].legend(fontsize=10, ncol=3)
                    axes[index].tick_params(labelsize=8)
                    axes[index].set_xlabel('完工时间', fontsize=10)
                    axes[index].set_ylabel('加权延期时间', fontsize=10)
                    axes[index].set_zlabel('加工能耗', fontsize=10)
                    axes[index].set_title(data, fontsize=10)
                    axes[index].ticklabel_format(axis='both', style='sci', scilimits=[-1, 2])
            plt.pause(10)
            axes[0].cla()
            axes[1].cla()

    print_pair()
    print('---------end---------')


def metric():
    args = config()
    args.test_data = './data/test4'
    args.rule_eval = "./log/eval/rule/03-25-20-02/"
    args.metric_result = "./log/pareto/" # 保存metric的文件夹
    _metric(args)


def print_metric():
    args = config()
    args.test_data = './data/test4'
    args.rule_eval = "./log/eval/rule/03-25-20-02/" # 所有对比算法的推理结果
    args.metric_result = "./log/pareto/03-25-20-26.json" # 对比算法计算出来的metric值
    _print_metric(args)


def show_pareto():
    args = config()
    args.test_data = './data/test4'
    args.rule_eval = "./log/eval/rule/03-25-20-02/"  # 所有对比算法的推理结果
    args.metric_result = "./log/pareto/03-25-20-26.json"  # 对比算法计算出来的metric值
    _show_pareto(args)


if __name__ == "__main__":
    show_pareto()