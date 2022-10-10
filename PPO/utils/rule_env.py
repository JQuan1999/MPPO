# author by 蒋权
import numpy as np
import sys
sys.path.append('..')

from env import PPO_ENV


class RULE_ENV(PPO_ENV):
    def __init__(self, ac, instance, args, t=5000):
        super(RULE_ENV, self).__init__(instance, args, t)
        self.sequence_action = ac[0]
        self.route_action = ac[1]

    def cal_schedule_time(self, ra=None):
        init = False
        choose = None
        t = None
        for i in range(len(self.machines)):
            m = self.machines[i]
            if len(m.buffer_op) == 0:
                continue
            if not init:
                choose = i
                t = m.ava_t
                init = True
            else:
                if m.ava_t < t:
                    choose = i
                    t = m.ava_t
        if choose is None:  # 工件的buffer都为空 切换到下一个新工件到达时间点
            if len(self.new_jobs) == 0:
                raise Exception('no job will arrival, simulation should be terminal')
            t = self.new_arrival_t[self.new_job_cnt]
            job_index = self.njob_insert()
            choose = self.njob_route(job_index, t, ra)
        return choose, t

    def njob_route(self, job_index, t, ra=None):
        mach_index = self.route_rule(job_index, self.route_action)
        _, done = self.ra_step(job_index, self.route_action, t)
        return mach_index

    def step(self, ra, t):
        if not self.sa_done(t):
            mach_index, t = self.cal_schedule_time(ra)
        else:
            mach_index = None
            t = None
        return mach_index, t

    def reset(self, ra=None, t=0):
        job_index = np.arange(self.job_num)
        np.random.shuffle(job_index)
        for i in range(self.job_num):
            j = job_index[i]
            self.ra_step(j, self.route_action, t)
        mach_index, t = self.cal_schedule_time(ra)
        return mach_index, t
