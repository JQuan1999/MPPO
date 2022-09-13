# author by 蒋权
import json

import numpy as np
import matplotlib.pyplot as plt


def show_objective1(obj_values, rule, data, obj):
    plt.figure()
    xtick = ['Agent'] + rule + ['Ran\ndom']
    if obj == 0:
        plt.title('USE RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
    if obj == 2:
        plt.title('TOTAL TARDINESS')
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    xline_space = np.linspace(0, len(rule)+2, num=len(rule)+2)
    plt.xticks(xline_space, xtick, fontsize=6)
    plt.xlabel('Rule')
    plt.ylabel('Rank')
    plt.yticks(range(len(xtick)+5))
    plt.ylim(0, len(rule)+8)
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'wheat', 'lightsteelblue', 'slategray', 'olive']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    for i in range(obj_values.shape[0]):
        if obj == 0:
            index = np.argsort(obj_values[i])
        else:
            index = np.argsort(-obj_values[i])
        rank = np.arange(1, obj_values.shape[1]+1)[::-1]
        sort_rank = np.zeros(obj_values.shape[1])
        sort_rank[index] = rank
        plt.scatter(xline_space, sort_rank, marker=mark[i], color=color[i], label=data[i])
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.show()


def show_objective2(obj_values, rule, data, obj):
    plt.figure()
    xtick = ['Agent'] + rule + ['Ran\ndom']
    for i in range(len(xtick)):
        xtick[i] = '\n'.join(xtick[i].split('_'))
    if obj == 0:
        plt.ylim(0, 1)
        plt.title('USE RATIO')
    if obj == 1:
        plt.title('TOTAL ECOST')
        plt.ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        plt.ylim(0, obj_values.max()*1.21)
    if obj == 2:
        plt.title('TOTAL TARDINESS')
        plt.ticklabel_format(axis='y', style='sci', scilimits=[-1, 2])
        plt.ylim(0, obj_values.max() * 1.21)
    xline_space = np.linspace(0, len(rule)+2, num=len(rule)+2)
    plt.xticks(xline_space, xtick, fontsize=6)
    plt.xlabel('Rule')
    plt.ylabel('Value')
    color = ['lightgray', 'lightcoral', 'tomato', 'sienna', 'darkorange', 'wheat', 'lightsteelblue', 'slategray', 'olive']
    mark = ['o', 'v', '^', 's', 'p', 'x', 'd', 'X', '*', '+']
    for i in range(obj_values.shape[0]):
        plt.plot(xline_space, obj_values[i, :], marker=mark[i], color=color[i], label=data[i])
    plt.legend(loc='upper right', fontsize=8, ncol=3)
    plt.show()


def show_result(agent_log, rule_log, randm_log, test_data):
    with open(agent_log, 'r') as f:
        alog = json.load(f)
    with open(rule_log, 'r') as f:
        rlog = json.load(f)
    with open(randm_log, 'r') as f:
        rmlog = json.load(f)
    rule = np.array([key for key in rlog]).tolist()
    # rule = np.random.choice(rule, size=20, replace=False).tolist()
    for o in range(3):
        obj_values = np.zeros((len(test_data), len(rule)+2))
        for i in range(test_data.shape[0]):
            obj_values[i][0] = alog[test_data[i]][o]
            for r in range(len(rule)):
                obj_values[i][r+1] = rlog[rule[r]][test_data[i]][o]
            obj_values[i][-1] = rmlog[test_data[i]][o]
        show_objective1(obj_values, rule, test_data, o)
        show_objective2(obj_values, rule, test_data, o)
    print('end')


if __name__ == '__main__':
    agent_log = './log/eval/09-13-12-45-agent.json'
    rule_log = './log/eval/09-04-14-19-rule.json'
    randm_log = './log/eval/09-04-13-24-randRule.json'
    test_data = np.array(['j10_m10_n20', 'j10_m10_n50', 'j10_m10_n100',
                          'j20_m20_n20', 'j20_m20_n50', 'j20_m20_n100',
                          'j30_m30_n20', 'j30_m30_n50', 'j30_m30_n100'])
    show_result(agent_log, rule_log, randm_log, test_data)