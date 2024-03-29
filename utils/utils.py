import json
import os
import time
import numpy as np
from .uniform_weight import init_weight
from .generator_data import NoIndentEncoder


def save_result(obj_record, dpath, filename=None):
    assert os.path.exists(dpath), f'path:{dpath} is not existed'
    assert os.path.isdir(dpath), f'path"{dpath} is not a dir'
    result = json.dumps(obj_record, indent=2, sort_keys=True, cls=NoIndentEncoder)
    if filename is None:
        filename = time.strftime('%m-%d-%H-%M') + '.json'
    path = '/'.join([dpath, filename])
    with open(path, 'w') as f:
        f.write(result)


def get_data(dpath):
    """
        dpath: 数据集的路径 可以为文件夹或文件
    """
    if os.path.isdir(dpath):
        files = os.listdir(dpath)
        test_data = []
        for file in files[:]:
            file_path = '/'.join([dpath, file])
            test_data.append(file_path)
        return test_data
    else:
        return [dpath]


def get_ckpt(dpath):
    dir_name = os.listdir(dpath)
    ckpt_path = []
    for i in range(len(dir_name)):
        dir_path = '/'.join([dpath, dir_name[i]])
        ckpt_path.append(['/'.join([dir_path, 'actor.pkl']), '/'.join([dir_path, 'critic.pkl'])])
    return ckpt_path


def get_weight(args):
    weight, size = init_weight(args.weight_size, args.objective, low_bound=0.1)
    dir_name = os.path.dirname(args.sa_ckpt_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    weight_path = dir_name + '/' + 'weight.npy'
    np.save(weight_path, weight)
    return weight, size