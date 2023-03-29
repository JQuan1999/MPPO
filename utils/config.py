import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15, help='every weight learn epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='learning batch size')
    parser.add_argument('--a_update_step', type=int, default=10, help='actor learning step')
    parser.add_argument('--c_update_step', type=int, default=10, help='critic learning step')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount1 reward')
    parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
    parser.add_argument('--weight_size', type=int, default=100, help='sample weight')
    parser.add_argument('--objective', type=int, default=3, help='objective size')
    parser.add_argument('--sa_state_dim', type=int, default=12, help='sequence agent state dim')
    parser.add_argument('--ra_state_dim', type=int, default=12, help='route agent state dim')
    parser.add_argument('--sa_action_space', type=int, default=4, help='sequence agent action space')
    parser.add_argument('--ra_action_space', type=int, default=4, help='route agent action space')

    parser.add_argument('--agent_eval', type=str, default='./log/eval/agent/', help='eval result saved path')
    parser.add_argument('--rule_eval', type=str, default='./log/eval/rule/', help='rule combination eval result saved path')
    parser.add_argument('--rand_rule_eval', type=str, default='./log/eval/rule/', help='rule combination eval result saved path')
    parser.add_argument('--param_comb_eval', type=str, default='./log/eval/param_comb/', help='config params combination eval result saved path')

    parser.add_argument('--metric_result', type=str, default='', help='the save path of pareto metric result')

    parser.add_argument('--train_data', type=str, default='./data/train')
    parser.add_argument('--test_data', type=str, default='./data/test')
    parser.add_argument('--sa_ckpt_path', type=str, default='./param/pareto_weight', help='path to save sa ckpt')
    parser.add_argument('--ra_ckpt_path', type=str, default='./param/pareto_weight', help='path to save ra ckpt')
    parser.add_argument('--weight_path', type=str, default='', help='weight path')
    args = parser.parse_args()
    return args