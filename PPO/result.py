# author by 蒋权
import json

import numpy as np
import matplotlib.pyplot as plt


def show_objective1(obj_values, rule, data, obj):
    plt.figure()
    xtick = ['Agent'] + rule + ['Random']
    if obj == 0:
        plt.title('USE_RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
    if obj == 2:
        plt.title('TOTAL TARDINESS')
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    xline_space = np.linspace(0, 12, num=12)
    plt.xticks(xline_space, xtick, fontsize=8)
    plt.yticks(range(len(xtick)+1))
    plt.ylim(0, 15)
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'gold', 'chartreuse', 'aquamarine', 'lightseagreen', 'cornflowerblue']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X']
    for i in range(obj_values.shape[0]):
        if obj == 0:
            index = np.argsort(obj_values[i])
        else:
            index = np.argsort(-obj_values[i])
        rank = np.arange(1, obj_values.shape[1]+1)[::-1]
        sort_rank = np.zeros(obj_values.shape[1])
        sort_rank[index] = rank
        plt.plot(xline_space, sort_rank, marker=mark[i], color=color[i], label=data[i])
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.show()


def show_objective2(obj_values, rule, data, obj):
    plt.figure()
    xtick = ['Agent'] + rule + ['Random']
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    if obj == 0:
        plt.ylim(0, 1)
        plt.title('USE_RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
    if obj == 2:
        plt.title('TOTAL TARDINESS')
    xline_space = np.linspace(0, 12, num=12)
    plt.xticks(xline_space, xtick, fontsize=8)
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'gold', 'chartreuse', 'aquamarine',
             'lightseagreen', 'cornflowerblue']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', 'x']
    for i in range(obj_values.shape[0]):
        plt.plot(xline_space, obj_values[i, :], marker=mark[i], color=color[i], label=data[i])
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.show()


def show_result(agent_log, rule_log, randm_log, test_data):
    with open(agent_log, 'r') as f:
        alog = json.load(f)
    with open(rule_log, 'r') as f:
        rlog = json.load(f)
    with open(randm_log, 'r') as f:
        rmlog = json.load(f)
    rule = [key for key in rlog]
    for o in range(3):
        obj_values = np.zeros((len(test_data), len(rule)+2))
        for i in range(test_data.shape[0]):
            obj_values[i][0] = alog[test_data[i]][o]
            for r in range(len(rule)):
                obj_values[i][r+1] = rlog[rule[r]][test_data[i]][o]
            obj_values[i][-1] = rmlog[test_data[i]][o]
        show_objective2(obj_values, rule, test_data, o)
    print('end')


if __name__ == '__main__':
    agent_log = './log/eval/08-31-23-02-agent.json'
    rule_log = './log/eval/08-30-19-10-rule.json'
    randm_log = './log/eval/08-30-19-16-randRule.json'
    test_data = np.array(['j10_m10_n20', 'j10_m10_n50', 'j20_m20_n50', 'j20_m20_n100', 'j30_m30_n50', 'j30_m30_n100'])
    show_result(agent_log, rule_log, randm_log, test_data)