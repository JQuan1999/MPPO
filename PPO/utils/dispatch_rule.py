# author by 蒋权
import numpy as np


def FIFO(jobs, t):
    ava_t = np.zeros(len(jobs))
    for i in range(len(ava_t)):
        ava_t[i] = jobs[i].pre_start
    index = np.argmin(ava_t)
    return index


# 最短松弛时间优先
# minimum slack time
def DS(jobs, t):
    slack_time = np.zeros(len(jobs))
    for i in range(len(jobs)):
        slack_time[i] = jobs[i].get_slack_time(t)
        if slack_time[i] < 0:
            slack_time[i] = jobs[i].u_degree * slack_time[i]
    index = np.argmin(slack_time)
    return index


# 最早预期时间优先
# earliest due-date
def EDD(jobs, t):
    ddt = np.array([j.due_date for j in jobs.tolist()])
    index = np.argmin(ddt)
    return index


# 关键率
# critical ration
# ttd = due_date - arrival
# 如果工件没拖延 min cr = ttd / sum(remain)
# 否则 max cr = over_time / remain
def CR(jobs, t):
    cr = np.zeros(len(jobs))
    tardy = np.array([True if j.get_tardiness(t) != 0 else False for j in jobs.tolist()])
    if tardy.sum() != 0:
        for i in range(len(jobs)):
            over_time = jobs[i].get_tardiness(t)
            remain = jobs[i].get_remain_pt()
            cr[i] = over_time / remain
        index = np.argmax(cr)
        return index
    else:
        for i in range(len(jobs)):
            ttd = jobs[i].due_date - jobs[i].arrival_t
            remain = jobs[i].get_remain_pt()
            cr[i] = ttd / remain
        index = np.argmin(cr)
        return index


# 最短剩余时间优先
# shortest remaining processing time
def SRPT(jobs, t):
    remain_pt = np.array([job.get_remain_pt() for job in jobs.tolist()])
    index = np.argmin(remain_pt)
    return index


def job_dispatch(action, jobs, t):
    if len(jobs) == 1:
        return 0
    else:
        rules = [FIFO, DS, EDD, CR, SRPT]
        index = rules[action](jobs, t)
        return index


# 加工时间最短
# short processing machine
def SPT(machines, op):
    pt = np.zeros(len(machines))
    for m in range(len(machines)):
        pt[m] = op.get_pt(machines[m].mach_index)
    index = np.argmin(pt)
    return index


# 加工能耗最小
# short energy cost machine
def SECM(machines, op):
    ect = np.zeros(len(machines))
    for m in range(len(machines)):
        ect[m] = op.get_ect(machines[m].mach_index)
    index = np.argmin(ect)
    return index


# 可用时间最早
# earliest available machine
def EAM(machines, op):
    ava_t = np.zeros(len(machines))
    for mach_index in range(len(machines)):
        ava_t[mach_index] = machines[mach_index].ava_t
    index = np.argmin(ava_t)
    return index


# 最小队列时间
# shortest queue time
def SQT(machines, op):
    queue_t = np.zeros(len(machines))
    for mach_index in range(len(machines)):
        est_ava = machines[mach_index].get_estimate_ava_time()
        queue_t[mach_index] = est_ava - machines[mach_index].ava_t
    index = np.argmin(queue_t)
    return index


def machine_dispatch(machines, op, machine_action):
    machine_rules = [SPT, SECM, EAM, SQT]
    index = machine_rules[machine_action](machines, op)
    return index
