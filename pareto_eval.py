import json
import os
import argparse
import shutil
import time
import numpy as np
import torch

from utils.env import PPO_ENV
from utils.agent import Sequence_Agent, Route_Agent
from utils.uniform_weight import cweight
from utils.config import config
from utils.utils import get_data, save_result, get_ckpt

np.random.seed(1)
torch.manual_seed(1)


# 推理
def agent_eval():
    args = config()

    # 设置具体的参数
    # 例如
    # args.sa_ckpt = ...

    eval_(args)


# 不同的参数组合推理
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
    # 设置args的参数
    # 例如 args.test_data = './data/test'
    args.test_data = "./data/test/j40_m20_n60" # 参数组合测试数据集
    savedir = "./log/param_experiment" # 保存评估结果的文件夹
    for i, param in enumerate(Epoch_Bacth_Step):
        # 设置参数路径
        name = "E{}_B{}_S{}".format(param[0], param[1], param[2]) # 参数组合的名称
        ckptdir = "/".join([args.param_comb_ckpt, name])
        assert os.path.exists(ckptdir), f'dirname: {ckptdir} is not existed'

        args.sa_ckpt_path = ckptdir + "/" + "sa" # sa参数路径
        args.ra_ckpt_path = ckptdir + "/" + "ra" # ra参数路径
        args.weight_path = ckptdir + "/" + "weight.npy" # 权重向量
        args.agent_eval_savedir = savedir
        filename = name + ".json"
        if i == 0:
            eval_(args, True, filename)
        else:
            eval_(args, False, filename)


def eval_(args, clear=True, filename="multi-agent.json"):
    print(args)
    # 获取最后一个文件名,根据最后一个文件名判断是对所有规模的调度问题进行对比还是只对比单个
    last_dir = args.test_data.split('/')[-1]
    if last_dir[:4] == "test":
        test_datas = ['/'.join([args.test_data, insdir, 't0.json']) for insdir in os.listdir(args.test_data)]
    else:
        test_datas = ['/'.join([args.test_data, 't0.json'])]

    if not os.path.exists(args.agent_eval_savedir):
        os.makedirs(args.agent_eval_savedir)
    elif clear:
        shutil.rmtree(args.agent_eval_savedir)  # 清空历史结果
        os.makedirs(args.agent_eval_savedir)

    # 加载参考向量和每个参考向量对应的sa和ra的ckpt文件
    weight = np.load(args.weight_path)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)
    sa_ckpt = get_ckpt(args.sa_ckpt_path)
    ra_ckpt = get_ckpt(args.ra_ckpt_path)

    result = {}
    for index, data in enumerate(test_datas):
        objs = np.zeros((weight.shape[0], args.objective))
        data_name = data.split('/')[-2]
        begin = time.time()
        for i in range(weight.shape[0]):
            sa.load(sa_ckpt[i]) # load第i个参考向量对应的sa ckpt
            ra.load(ra_ckpt[i]) # load第i个参考向量对应的ra ckpt
            env = PPO_ENV(data, args) # 创建env环境
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
            # env.render(t=10, key=data_name)
            obj = env.cal_objective()
            objs[i] = obj
            print(f'data {data_name} weight {weight[i].reshape(-1, ).tolist()} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
            # print(f'a1 = {a1}\n a2 = {a2}')
        t = time.time() - begin
        result[data_name] = {}
        result[data_name]["time"] = t
        result[data_name]["result"] = objs.tolist()
    save_result(result, args.agent_eval_savedir, filename)
    print('----------end------------')


if __name__ == '__main__':
    args = config()
    eval_(args)