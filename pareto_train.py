import numpy as np
import os
import time

from agent import Sequence_Agent, Route_Agent
from env import PPO_ENV
from utils.config import config
from utils.uniform_weight import init_weight, cweight
from utils.utils import get_data, get_weight


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
    ckpt_path = "./param/param_experiment/ckpt"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    args.sa_ckpt_path = ckpt_path
    args.ra_ckpt_path = ckpt_path
    for param_comb in Epoch_Bacth_Step:
        args.epochs = param_comb[0]
        args.batch_size = param_comb[1]
        args.a_update_step = param_comb[2]
        args.c_update_step = param_comb[2]
        name = "E{}_B{}_S{}".format(args.epochs, args.batch_size, param_comb[0])
        dirname = '/'.join([ckpt_path, name])
        if os.path.exists(dirname):
            os.removedirs(dirname)
        args.sa_ckpt_path = '/'.join([dirname, "sa"])
        args.ra_ckpt_path = '/'.join([dirname, "ra"])
        train(args)
    print('train end')


def train(args):
    args.train_data = "./data/train2/j30_m20_n40"
    train_data = get_data(args.train_data)
    train_data_size = len(train_data)

    # date = time.strftime('%m-%d-%H-%M')
    # args.sa_ckpt_path = '/'.join([args.sa_ckpt_path, date, 'sa'])
    # args.ra_ckpt_path = '/'.join([args.ra_ckpt_path, date, 'ra'])
    weight, size = get_weight(args)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)

    for i in range(size):
        w = weight[size - 1 - i].reshape(1, -1)
        for epoch in range(args.epochs):
            step = 0
            done2 = False
            inst = train_data[(i * args.epochs + epoch) % train_data_size]
            # inst = train_data[0]
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
                elif done1 and ra.buffer.cnt != 0:
                    ra.learn(ra_state, done2)

                sa_state = sa_state_
                step += 1

                if done1:
                    break
            # print(env.cnt.tolist())
            print("sa_action = ", a1)
            print("ra_action = ", a2)
            obj = env.cal_objective()
            inst = inst.split('/')[-1]
            print(
                f'inst {inst}, weight{w.reshape(-1, ).tolist()}, epoch{epoch} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        sa.save(weight=w)
        ra.save(weight=w)
    # sa.show_loss()
    # ra.show_loss()


if __name__ == "__main__":
    param_experiment_train()
