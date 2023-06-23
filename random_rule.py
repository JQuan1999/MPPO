# author by 蒋权
import os
import shutil
import time
import numpy as np
import torch

from utils.config import config
from utils.rule_env import RULE_ENV
from utils.utils import save_result


np.random.seed(1)
torch.manual_seed(1)


def _random_eval(sa_action, ra_action, args, savedir, result_file):
    assert type(sa_action) is list, f"sa action {sa_action} is not a list"
    assert type(ra_action) is list, f"ra action {ra_action} is not a list"

    def get_comb_action():
        ac1 = sa_action[np.random.randint(len(sa_action))]
        ac2 = ra_action[np.random.randint(len(ra_action))]
        return ac1, ac2

    weight = np.load(args.weight_path)
    size = weight.shape[0]
    # 获取最后一个文件名,根据最后一个文件名判断是对所有规模的调度问题进行对比还是只对比单个
    last_dir = args.test_data.split('/')[-1]
    if last_dir[:4] == "test":
        test_datas = ['/'.join([args.test_data, insdir, 't0.json']) for insdir in os.listdir(args.test_data)]
    else:
        test_datas = ['/'.join([args.test_data, 't0.json'])]
    result = {}

    for index, data in enumerate(test_datas):
        objs = np.zeros((size, args.objective))
        data_name = data.split('/')[-2] # 测试集规模
        begin = time.time()
        for i in range(size):
            ac = get_comb_action()
            env = RULE_ENV(ac, data, args)
            done2 = False
            mach_index1, t = env.reset()
            while True:
                if env.check_njob_arrival(t):
                    job_index = env.njob_insert()
                    env.njob_route(job_index, t)

                job_index, r, done1, end = env.sa_step(mach_index1, ac[0], t)

                if not done2 and env.jobs[job_index].not_finish():
                    r, done2 = env.ra_step(job_index, ac[1], t)

                mach_index1, t = env.step(None, t)
                if done1:
                    break
                # 更新调度规则
                ac = get_comb_action()
                env.sequence_action = ac[0]
                env.route_action = ac[1]
            # env.render(t=10, key=data_name)
            obj = env.cal_objective()
            objs[i] = obj
            print(f'data {data_name} | obj1 = {obj[0]}, obj2 = {obj[1]}, obj3 = {obj[2]}')
        t = time.time() - begin
        result[data_name] = {}
        result[data_name]["time"] = t
        result[data_name]["result"] = objs.tolist()
    save_result(result, savedir, result_file)
    print('----------end------------')


# 部分随机组合
def part_random():
    args = config()

    # 设置具体的参数
    # 例如
    # args.test_data = ...
    args.test_data = './data/test'
    # 清空args.rule_eval文件夹
    if not os.path.exists(args.rule_eval_savedir):
        os.makedirs(args.rule_eval_savedir)
    else:
        shutil.rmtree(args.rule_eval_savedir)
        os.makedirs(args.rule_eval_savedir)

    sa_action = [i for i in range(args.sa_action_space)]
    ra_action = [i for i in range(args.sa_action_space)]
    sa_action_name = ['FIFO', 'MS', 'EDD', 'CR']
    ra_action_name = ['SPT', 'SEC', 'EA', 'SQT']
    for i in range(2):
        # 第1次sa_action + 随机ra_action
        # 第2次ra_action + 随机sa_action
        if i == 0:
            choose_action = sa_action
        else:
            choose_action = ra_action
        for c_a in choose_action:
            if i == 0:
                action_name = sa_action_name[c_a]
            else:
                action_name = ra_action_name[c_a]
            # 保存的文件名
            result_file = f"random-{action_name}.json"
            if i == 0:
                _random_eval([c_a], ra_action, args, args.rule_eval_savedir, result_file)
            else:
                _random_eval(sa_action, [c_a], args, args.rule_eval_savedir, result_file)


# 完全随机组合
def random_eval():
    args = config()
    # 设置具体的参数
    # 例如
    # args.test_data = ...
    args.test_data = './data/test'
    # 清空args.rand_eval文件夹
    if not os.path.exists(args.randrule_eval_savedir):
        os.makedirs(args.randrule_eval_savedir)
    else:
        shutil.rmtree(args.randrule_eval_savedir)
        os.makedirs(args.randrule_eval_savedir)

    sa_action = [i for i in range(args.sa_action_space)]
    ra_action = [i for i in range(args.sa_action_space)]
    _random_eval(sa_action, ra_action, args, args.randrule_eval_savedir, "multi-random.json")


if __name__ == '__main__':
    part_random()
    random_eval()
