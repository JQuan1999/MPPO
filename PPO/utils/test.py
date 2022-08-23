import json
import numpy as np
import torch
import matplotlib.pyplot as plt

def test1():
    xs = np.random.randint(low=0,high=10,size=10)
    print(xs)
    nx = np.sort(xs)
    print(nx)


def test2():
    eps = 1e-5
    # x = [batch,channel,h,w]
    x = np.random.random((2,3,2,2))
    print(x)
    mean = np.mean(x,axis=(0,2,3),keepdims=True)
    print(mean.shape)
    var = np.var(x,axis=(0,2,3),keepdims=True)
    print(var.shape)
    print(mean)
    print(var)
    x_norm = (x - mean) / np.sqrt(var + eps)
    print(x_norm)


def test3():
    eps=1e-5
    x = np.random.random((2,3,2,2))
    print(x)
    mean = np.mean(x, axis=(1,2,3),keepdims=True)
    print("mean = {}, mean.shape = {}".format(mean, mean.shape))
    var = np.var(x, axis=(1,2,3),keepdims=True)
    print("var = {}, var.shape = {}".format(var, var.shape))
    x_norm = (x - mean) / np.sqrt(var + eps)
    print(x_norm)


def test4():
    x = torch.randn(2,2,3).float()
    print(x)
    norm = torch.nn.BatchNorm1d(x.shape[1],eps=1e-5,affine=False)
    x_norm1 = norm(x)
    print(x_norm1)
    mean = torch.mean(x,dim=(0,2),keepdim=True)
    print(mean,mean.shape)
    var = torch.var(x,dim=(0,2),keepdim=True)
    print(var,var.shape)
    norm_x = (x - mean) / torch.sqrt(var + 1e-5)
    print(norm_x)


def test5(key):
    if key == "job":
        print(key)
    elif key == "new_job":
        print(key+1)
    else:
        ex = Exception("key error")
        raise ex


def test6():
    x = np.arange(5).tolist()
    h = 5
    c = ['r', 'y', 'black', 'blue', 'orange']
    plt.bar(x, height=h, color=c)
    plt.show()


def test7():
    x = np.arange(5)
    x = np.insert(x, 5, 10)
    print(x)


def test8():
    x = np.arange(4).reshape(2,2)
    y = np.arange(2)
    z = np.dot(x, y)
    print(z)


def test9():
    pass
    # t = self.jobs[job_index].pre_start
    # # route agent的特征长度为8
    # ava_t = []
    # # use_ratio = []
    # for m in range(len(self.machines)):
    #     ava_t.append(self.machines[m].ava_t)
    #     # use_ratio.append(self.machines[m].use_ratio[-1])
    # ava_t = np.array(ava_t)
    # mean_ava_t = ava_t.mean()
    # min_ava_t = ava_t.min()
    # ava_t_state = np.array([mean_ava_t, min_ava_t])
    # if ava_t_state.max() == 0:
    #     ava_t_state = np.zeros(2).tolist()
    # else:
    #     ava_t_state = (ava_t_state / ava_t_state.max()).tolist()
    # # use_ratio = np.array(use_ratio)
    # # ava_use_ratio = use_ratio.mean()
    #
    # pt = np.array(self.jobs[job_index].pre_op.pt)
    # ave_pt = pt.mean()
    # min_pt = pt.min()
    # slack = self.jobs[job_index].get_slack_time(t)
    # tardiness = self.jobs[job_index].get_tardiness(t)
    # pt_state = np.array([ave_pt, min_pt, slack, tardiness])
    # pt_state = (pt_state / pt_state.max()).tolist()
    #
    # ect = np.array(self.jobs[job_index].pre_op.ect)
    # mean_ect = ect.mean()
    # min_ect = ect.min()
    # ect_state = np.array([mean_ect, min_ect])
    # ect_state = (ect_state / ect_state.max()).tolist()
    #
    # ra_state = ava_t_state + pt_state + ect_state
    # ra_state = self.w + ra_state
    # return ra_state


def get_sa_state():
    pass
    # # sequence agent的state 长度为13
    # # buffer_job = len(self.machines[mach_index].op_buffer)
    # if len(self.machines[mach_index].buffer_op) == 0:
    #     return np.zeros(self.sa_state_dim)
    # sum_pt, mean_pt, min_pt, job_inds = self.machines[mach_index].get_pt_state()
    # left_pt = np.zeros(len(job_inds))
    # ddt = np.zeros(len(job_inds))
    # slack = np.zeros(len(job_inds))
    # tardiness = np.zeros(len(job_inds))
    # for j in range(len(job_inds)):
    #     job_index = job_inds[j]
    #     left_pt[j] = self.jobs[job_index].get_remain_pt()
    #     # ddt[j] = self.jobs[j].ddt
    #     slack[j] = self.jobs[job_index].get_slack_time(t)
    #     tardiness[j] = self.jobs[job_index].get_tardiness(t)
    #
    # sa_state = np.array([sum_pt, mean_pt, min_pt,
    #                      left_pt.sum(), left_pt.mean(), left_pt.min(),
    #                      slack.mean(), slack.min(),
    #                      tardiness.max(), tardiness.mean()])
    # norm_sa_state = (sa_state / sa_state.max()).tolist()
    # norm_sa_state = self.w + norm_sa_state
    # return norm_sa_state


def test10():
    x = np.linspace(-2*np.pi, 2*np.pi, 400)
    y1 = np.sin(x)
    y2 = np.cos(x)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
    ax1.plot(x, y1)
    ax1.set_title('sin')
    ax2.plot(x, y2)
    ax2.set_title('cos')
    plt.pause(10)


def test11():
    x = torch.tensor([[1, 2, np.nan], [np.nan, np.nan, 3]])
    if torch.isnan(x).any():
        print('x has nan')
    a = torch.where(torch.isnan(x), torch.zeros_like(x), x)
    print(a.numpy().tolist())


def test12():
    x = torch.randn(2, 3)
    linear = torch.nn.Linear(3, 3)
    x = linear(x)
    temp = x.detach().numpy()
    softmax = torch.nn.Softmax(dim=1)
    x = softmax(x)
    print(x)
    print(temp)


def test13():
    import time
    f = time.strftime('%Y-%m-%d-%H:%M:%S')
    print(f)


if __name__ == '__main__':
    test13()