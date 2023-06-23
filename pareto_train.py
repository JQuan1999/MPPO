import numpy as np
import os
import time
import torch
import shutil
from utils.agent import Sequence_Agent, Route_Agent
from utils.env import PPO_ENV
from utils.config import config
from utils.uniform_weight import init_weight, cweight
from utils.utils import get_data, get_weight

np.random.seed(1)
torch.manual_seed(1)

# 9种参数组合对比试验
# id  Epoch   BatchSize   Step
# 1    1(5)    1(64)      1(5)
# 2    1(5)    2(128)     2(10)
# 3    1(5)    3(256)     3(15)
# 4    2(10)   1(64)      2(10)
# 5    2(10)   2(128)     3(15)
# 6    2(10)   3(256)     1(5)
# 7    3(15)   1(64)      3(15)
# 8    3(15)   2(128)     1(5)
# 9    3(15)   3(256)     2(10)
def param_experiment_train():
    args = config()
    Epoch_Bacth_Step = [
        [5, 64, 5],
        [5, 128, 10],
        [5, 256, 15],
        [10, 64, 10],
        [10, 128, 15],
        [10, 256, 5],
        [15, 64, 15],
        [15, 128, 5],
        [15, 256, 10]]
    if not os.path.exists(args.param_comb_ckpt):
        os.makedirs(args.param_comb_ckpt)

    for param_comb in Epoch_Bacth_Step:
        args.epochs = param_comb[0]
        args.batch_size = param_comb[1]
        args.a_update_step = param_comb[2]
        args.c_update_step = param_comb[2]
        name = "E{}_B{}_S{}".format(param_comb[0], param_comb[1], param_comb[2])
        dirname = '/'.join([args.param_comb_ckpt, name]) # 参数组合ckpt保存文件夹
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            os.makedirs(dirname)
        args.sa_ckpt_path = '/'.join([dirname, "sa"]) # sa ckpt文件路径
        args.ra_ckpt_path = '/'.join([dirname, "ra"]) # ra ckpt文件路径
        train(args)
    print('param combinations train end')


def train(args):
    if not os.path.exists(args.sa_ckpt_path):
        os.makedirs(args.sa_ckpt_path)
    if not os.path.exists(args.ra_ckpt_path):
        os.makedirs(args.ra_ckpt_path)
    train_data = get_data(args.train_data)
    train_data_size = len(train_data)

    weight, size = get_weight(args)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)

    for i in range(size):
        w = weight[size - 1 - i].reshape(1, -1)
        for epoch in range(args.epochs):
            step = 0
            done2 = False
            inst = train_data[(i * args.epochs + epoch) % train_data_size]
            env = PPO_ENV(inst, args)
            cweight(w, env, sa, ra)
            sa_state, mach_index1, t = env.reset(ra)
            a1 = []
            a2 = []
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
                a1.append(sa_action)
                job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

                if not done2 and env.jobs[job_index].not_finish():
                    ra_state = env.get_ra_state(job_index)
                    ra_action = ra.choose_action(ra_state)
                    a2.append(ra_action)
                    ra_reward, done2 = env.ra_step(job_index, ra_action, t)
                    ra.store(ra_state, ra_action, ra_reward, done2)

                sa_state_, mach_index1, t = env.step(ra, t)
                sa.buffer.store(sa_state, sa_action, sa_reward, sa_state_, done1)
                if sa.buffer.cnt == args.batch_size:
                    sa.learn(sa_state_, done1)
                elif done1 and sa.buffer.cnt != 0:
                    sa.learn(sa_state_, done1)

                if ra.buffer.cnt == args.batch_size:
                    ra.learn(ra_state, done2)
                elif done2 and ra.buffer.cnt != 0:
                    ra.learn(ra_state, done2)

                sa_state = sa_state_
                step += 1

                if done1:
                    break
            # print("sa_action = ", a1)
            # print("ra_action = ", a2)
            obj = env.cal_objective()
            inst = inst.split('/')[-1]
            print(
                f'inst {inst}, weight{w.reshape(-1, ).tolist()}, epoch{epoch} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        sa.save(weight=w)
        ra.save(weight=w)
    print('--------end-------')


if __name__ == "__main__":
    # 在训练之前设置好训练参数
    param_experiment_train()
