import argparse
import torch
import importlib

from lib.parser import args_parser
from lib.dataset import get_dataset_and_info
from lib.network import get_network
from lib.utils import seed_everything
from lib.logger import ExpLogger


if __name__ == '__main__':

    # get args
    args, extra_args = args_parser()
    # get approach args
    approach_parser = getattr(importlib.import_module(name='approach.' + args.approach), 'approach_parser')
    approach_args, extra_args = approach_parser(extra_args)

    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    args.device = device

    # get dataset for training
    seed_everything(seed=args.seed)
    train_dataset, train_index_map, test_dataset, test_index_map, name_to_index, index_to_name = get_dataset_and_info(args)
    
    # get backbone network
    seed_everything(seed=args.seed)
    network = get_network(args.network, args.pretrained, approach_args.network_type)
    
    # init logger
    full_exp_name = args.approach + '_' + args.dataset
    if args.exp_name:
        full_exp_name += ('_' + args.exp_name)
    if args.unlearn:
        full_exp_name += '_unlearn'
    logger = ExpLogger(args.results_dir, full_exp_name)

    # log args
    all_args = {**(args.__dict__),**(approach_args.__dict__)}
    logger.log_args(all_args)
    all_args = argparse.Namespace(**all_args)

    # init FCL server
    ServerClass = getattr(importlib.import_module(name='approach.' + args.approach), 'Server')
    fcl_server = ServerClass(
        all_args, device, network, logger, 
        train_dataset, train_index_map, test_dataset, test_index_map, name_to_index, index_to_name,
    )
    # training
    fcl_server.train()
