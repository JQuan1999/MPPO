import numpy as np
import os
import time

from agent import Sequence_Agent, Route_Agent
from env import PPO_ENV
from utils.config import config
from utils.uniform_weight import init_weight, cweight


def get_data(path):
    files = os.listdir(path)
    train_data = []
    for file in files[:]:
        dir_path = '/'.join([path, file])
        if os.path.exists(dir_path):
            json_f = os.listdir(dir_path)
            for jf in json_f:
                jf_path = '/'.join([dir_path, jf])
                train_data.append(jf_path)
    return train_data


def get_weight(args):
    weight, size = init_weight(args.weight_size, args.objective, low_bound=0.1)
    dir_name = os.path.dirname(args.sa_ckpt_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    weight_path = dir_name + '/' + 'weight.npy'
    np.save(weight_path, weight)
    return weight, size


def train():
    args = config()
    print(args)
    train_data = get_data(args.train_data)
    train_data_size = len(train_data)

    date = time.strftime('%m-%d-%H-%M')
    args.sa_ckpt_path = '/'.join([args.sa_ckpt_path, date, 'sa'])
    args.ra_ckpt_path = '/'.join([args.ra_ckpt_path, date, 'ra'])
    weight, size = get_weight(args)
    sa = Sequence_Agent(args)
    ra = Route_Agent(args)

    for i in range(size):
        w = weight[i].reshape(1, -1)
        for epoch in range(args.epochs):
            step = 0
            done2 = False
            inst = train_data[(i * args.epochs + epoch) % train_data_size]
            env = PPO_ENV(inst, args)
            cweight(w, env, sa, ra)
            sa_state, mach_index1, t = env.reset(ra)
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t, ra)

                sa_action = sa.choose_action(sa_state)  # 在mach的候选buffer中选择1个工件进行加工
                job_index, sa_reward, done1, end = env.sa_step(mach_index1, sa_action, t)

                if not done2 and env.jobs[job_index].not_finish():
                    ra_state = env.get_ra_state(job_index)
                    ra_action = ra.choose_action(ra_state)
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
            obj = env.cal_objective()
            inst = inst.split('/')[-1]
            print(
                f'inst {inst}, weight{w.reshape(-1, ).tolist()}, epoch{epoch} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj2 = {obj[2]}')
        sa.save(weight=w)
        ra.save(weight=w)
    # sa.show_loss()
    # ra.show_loss()


if __name__ == "__main__":
    train()
