import json
import os
import argparse
import time
import numpy as np
import torch

from env import PPO_ENV
from agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import cweight
from utils.config import config
from utils.utils import get_data, save_result, get_ckpt

np.random.seed(1)


def param_experiment_eval():
    args = config()
    Epoch_Bacth_Step = [[5, 64, 5],
                        [5, 128, 10],
                        [5, 256, 15],
                        [10, 64, 10],
                        [10, 128, 15],
                        [10, 256, 5],
                        [15, 64, 15],
                        [15, 128, 5],
                        [15, 256, 10]]
    ckpt_path = "./param/param_experiment/ckpt"
    args.test_data = "./data/test4/j40_m20_n60"
    # 保存评估结果的文件夹
    logdir = "./log/param_experiment"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for param in Epoch_Bacth_Step:
        # 设置参数路径
        name = "E{}_B{}_S{}".format(param[0], param[1], param[2])
        dirname = "/".join([ckpt_path, name])
        args.sa_ckpt = dirname + "/" + "sa"
        args.ra_ckpt = dirname + "/" + "sa"
        # 保存评估结果的文件路径
        args.log_dir = logdir + "/" + name
        # 参考向量文件
        args.weight_path = dirname + "/" + "weight.npy"
        eval_(args)

    print('eval end!!!')


def eval_(args):
    print(args)
    # 获取最后一个文件名,根据最后一个文件名判断是对所有规模的调度问题进行对比还是只对比单个
    last_dir = args.test_data.split('/')[-1]
    if last_dir[:4] == "test":
        test_datas = ['/'.join([args.test_data, insdir, 't0.json']) for insdir in os.listdir(args.test_data)]
    else:
        test_datas = ['/'.join([args.test_data, 't0.json'])]

    # 加载参考向量和每个参考向量对应的sa和ra的ckpt文件
    weight = np.load(args.weight_path)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    sa_ckpt = get_ckpt(args.sa_ckpt)
    ra_ckpt = get_ckpt(args.ra_ckpt)

    result = {}
    for index, data in enumerate(test_datas):
        objs = np.zeros((weight.shape[0], args.objective))
        data_name = data.split('/')[-2]
        begin = time.time()
        for i in range(weight.shape[0]):
            sa.load(sa_ckpt[i])
            ra.load(ra_ckpt[i])
            env = PPO_ENV(data, args)
            w = weight[i].reshape(1, -1)
            cweight(w, env, ra, sa)
            sa_state, mach_index1, t = env.reset(ra)
            done2 = False
            a1 = []
            a2 = []
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state)
                # a1.append(sa_action)
                job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

                if not done2 and env.jobs[job_index].not_finish():
                    ra_state = env.get_ra_state(job_index)
                    ra_action = ra.choose_action(ra_state)
                    # a2.append(ra_action)
                    ra_reward, done2 = env.ra_step(job_index, ra_action, t)

                # env.render()
                sa_state_, mach_index1, t = env.step(ra, t)
                sa_state = sa_state_
                if done1:
                    break
            env.render(t=100, key=data_name)
            obj = env.cal_objective()
            objs[i] = obj
            print(f'data {data_name} weight {weight[i].reshape(-1, ).tolist()} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
            # print(f'a1 = {a1}\n a2 = {a2}')
        t = time.time() - begin
        print(f'----------time = {t}------------')
        result[data_name] = {}
        result[data_name]["time"] = t
        result[data_name]["result"] = objs.tolist()
    save_result(result, args.log_dir)
    print('end')


if __name__ == '__main__':
    args = config()
    args.test_data = './data/test4/j7_m5_n10'
    args.sa_ckpt = './param/pareto_weight/11-02-16-11/sa'
    args.ra_ckpt = './param/pareto_weight/11-02-16-11/ra'
    args.weight_path = './param/pareto_weight/11-02-16-11/weight.npy'
    prefix = time.strftime('%m-%d-%H-%M')
    name = prefix + "-multi-agent.json"
    args.log_dir = args.log_dir + name
    eval_(args)
    # param_experiment_eval()