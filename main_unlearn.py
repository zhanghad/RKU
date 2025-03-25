import argparse
import time
import pickle
import torch
import torch.nn as nn
from approach.approach_unlearning import unlearn_extra_parser
from approach.approach_unlearning import perform_all_unlearn_method
from lib.logger import print_unlearn_test_result, recursive_load_pkl_to_dict
from lib.unlearn_metric import all_unlearn_metirc
from lib.utils import seed_everything
from lib.unlearn_config import get_unlearn_config, get_unlearn_index_map
from lib.logger import ExpLogger
from lib.dataset import get_dataset
from lib.network import CLNN
from approach.base import Server

def unlearn_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--exp-name', default='debug', type=str)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset', default='CIFAR100', type=str, help="TinyImageNet|CIFAR100")
    parser.add_argument('--batch-size', default=32, type=int, required=False)
    parser.add_argument('--num-workers', default=0, type=int, required=False)

    parser.add_argument('--unlearn-config-path', default='', type=str)
    # config/unlearn_config_class_1.json
    parser.add_argument('--saved-info-path', default='', type=str)

    return parser.parse_known_args()


def load_pretrained_info(args):

    s_time_load_info = time.time()
    all_saved_info = {}
    recursive_load_pkl_to_dict(args.saved_info_path, all_saved_info)
    print(f"load pretrained info Time[{time.time()-s_time_load_info:.2f}s]")

    server = all_saved_info['server']
    all_model_dict = all_saved_info['all_model_dict']
    unlearn_related_info = all_saved_info['unlearn_related_info']
    unlearn_related_info['server'] = server

    # convert str key to int
    global_weights = {}
    for task_id_str in unlearn_related_info["global_weights"].keys():
        if int(task_id_str) not in global_weights:
            global_weights[int(task_id_str)] = {}
        for round_id_str in unlearn_related_info["global_weights"][task_id_str].keys():
            global_weights[int(task_id_str)][int(round_id_str)] = unlearn_related_info["global_weights"][task_id_str][round_id_str]
    unlearn_related_info["global_weights"] = global_weights

    client_dict = {}
    for client_id_str in unlearn_related_info["client_dict"].keys():
        client_dict[int(client_id_str)] = unlearn_related_info["client_dict"][client_id_str]
    unlearn_related_info["client_dict"] = client_dict

    client_weights = {}
    for task_id_str in unlearn_related_info["client_weights"].keys():
        if int(task_id_str) not in client_weights:
            client_weights[int(task_id_str)] = {}
        for round_id_str in unlearn_related_info["client_weights"][task_id_str].keys():
            if int(round_id_str) not in client_weights[int(task_id_str)]:
                client_weights[int(task_id_str)][int(round_id_str)] = {}
            for client_id_str in unlearn_related_info["client_weights"][task_id_str][round_id_str].keys():
                client_weights[int(task_id_str)][int(round_id_str)][int(client_id_str)] = unlearn_related_info["client_weights"][task_id_str][round_id_str][client_id_str]
    unlearn_related_info["client_weights"] = client_weights

    return server, all_model_dict, unlearn_related_info


def edit_origin_model(model: nn.Module, stop_task_id):
    assert isinstance(model, CLNN)
    # clear classifiers
    model.classifiers = model.classifiers[:stop_task_id+1]


if __name__ == '__main__':

    args, extra_args = unlearn_args_parser()
    # get unlearning related args
    unlearn_args, extra_args = unlearn_extra_parser(extra_args)

    # init logger
    full_exp_name = f"unlearn_{args.dataset}_{args.exp_name}"
    logger = ExpLogger(args.results_dir, full_exp_name)

    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    args.device = device

    # load pretrained info
    server, all_model_dict, unlearn_related_info = load_pretrained_info(args)
    assert isinstance(server, Server), "server is not Server class"

    # get dataset
    train_dataset, test_dataset = get_dataset(server.args)

    # reassign logger for server
    server.logger = logger
    server.train_dataset = train_dataset
    server.test_dataset = test_dataset

    # log args
    all_args = {**(args.__dict__),**(unlearn_args.__dict__)}
    logger.log_args(all_args)
    all_args = argparse.Namespace(**all_args)
    unlearn_related_info["args"] = all_args

    # get unlearn_info
    seed_everything(seed=args.seed)
    unlearn_config = get_unlearn_config(args)
    train_remain_index_map, train_forget_index_map, test_remain_index_map, test_forget_index_map, unlearn_info = \
        get_unlearn_index_map(server.args, unlearn_config, server.train_dataset, server.train_index_map, server.test_dataset, server.test_index_map)

    unlearn_related_info["unlearn_config"] = unlearn_config
    unlearn_related_info["unlearn_info"] = unlearn_info
    unlearn_related_info["train_remain_index_map"] = train_remain_index_map
    unlearn_related_info["train_forget_index_map"] = train_forget_index_map
    unlearn_related_info["test_remain_index_map"] = test_remain_index_map
    unlearn_related_info["test_forget_index_map"] = test_forget_index_map

    # edit origin model
    edit_origin_model(all_model_dict["Origin"]["model"], unlearn_config["after_task_id"])

    # unlearn
    s_unlearn = time.time()
    print(f"start unlearn")
    unlearned_model_dict = perform_all_unlearn_method(all_model_dict["Origin"]["model"], unlearn_config, unlearn_info, unlearn_related_info)
    print(f"finish unlearn Time[{time.time()-s_unlearn:.2f}s]")
    all_model_dict.update(unlearned_model_dict)

    # unlearn metric
    s_unlearn_test = time.time()
    unlearn_test_res = all_unlearn_metirc(
        all_model_dict, unlearn_config, unlearn_info, unlearn_related_info,
        server.train_dataset, server.train_index_map, server.test_dataset, server.test_index_map,
        train_remain_index_map, train_forget_index_map, test_remain_index_map, test_forget_index_map,
        server.name_to_index, all_args.device, all_args.batch_size, all_args.num_workers,
    )
    print(f"finish unlearn test Time[{time.time()-s_unlearn_test:.2f}s]")

    # save result
    res_log_dict = {
        "log_name": "TestUnlearn",
        "last_task_id": unlearn_config["after_task_id"],
        "detail": unlearn_test_res,
    }
    logger.log_test_result(res_log_dict)

    # summarize and print unlearning result
    print_unlearn_test_result(unlearn_test_res)

    pass