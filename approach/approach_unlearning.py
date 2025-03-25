from argparse import ArgumentParser
import time
import random
from copy import deepcopy
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import numpy as np

from approach.base import Server
from lib.dataset import DatasetSplit, get_TinyImageNet_as_public_dataset, get_SVHN_as_public_dataset
from lib.utils import weights_average, weights_sub, weights_multi, weights_div, weights_add, loss_EWC,\
    compute_fisher_matrix_diag, weight_distance, cross_entropy_with_temp, cross_entropy_with_temp_mask, weights_device_convert
from lib.unlearn_utils import FedEraser_unlearn_step, Ours_loss_fn, calculate_client_prototype, calculate_client_proto_dist,\
    CDP_acculumate_feature, CDP_calculate_cp, CDP_get_threshold_by_sparsity, CDP_TFIDFPruner
from nni.compression.speedup import ModelSpeedup

def unlearn_extra_parser(args):
    """Returns a parser containing the approach specific parameters"""
    parser = ArgumentParser()

    parser.add_argument('--only-ours', default=1, type=int, required=False)

    # Ours
    parser.add_argument('--unlearn-ours-num-epoch', default=1, type=int, required=False)
    parser.add_argument('--unlearn-ours-lr', default=0.001, type=float, required=False)
    parser.add_argument('--unlearn-ours-ft-data-num', default=1000, type=int, required=False)
    parser.add_argument('--unlearn-ours-use-kd', default=1, type=int, required=False)
    parser.add_argument('--unlearn-ours-selective-kd', default=1, type=int, required=False)
    parser.add_argument('--unlearn-ours-lambda-KI', default=1, type=float, required=False)
    parser.add_argument('--unlearn-ours-KD-temp', default=2, type=float, required=False)
    
    # FedEraser
    parser.add_argument('--FedEraser-epoch-cali', default=3, type=int, required=False)

    # KnowledgeDistillation
    parser.add_argument('--KD-num-epoch', default=5, type=int, required=False)
    parser.add_argument('--KD-temp', default=2, type=float, required=False)
    parser.add_argument('--KD-lr', default=1e-2, type=float, required=False)
    parser.add_argument('--KD-num-data', default=5000, type=int, required=False)

    # unlearn_Projected_GA
    parser.add_argument('--pGA-num-epoch', default=10, type=int, required=False)
    parser.add_argument('--pGA-lr', default=0.00001, type=float, required=False)
    parser.add_argument('--pGA-distance-threshold', default=1e-5, type=float, required=False)

    # unlearn_EWC_SGA
    parser.add_argument('--ewcsga-num-epoch', default=5, type=int, required=False)
    parser.add_argument('--ewcsga-lr', default=1e-4, type=float, required=False)
    parser.add_argument('--ewcsga-lamb', default=5000, type=float, required=False,
                        help='Forgetting-intransigence trade-off (default=%(default)s)')
    parser.add_argument('--ewcsga-alpha', default=0.5, type=float, required=False,
                        help='EWC alpha (default=%(default)s)')
    parser.add_argument('--ewcsga-sampling-type', default='max_pred', type=str, required=False,
                        choices=['true', 'max_pred', 'multinomial'],
                        help='Sampling type for Fisher information (default=%(default)s)')
    parser.add_argument('--ewcsga-fisher-num-samples', default=-1, type=int, required=False,
                        help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')
    
    # RapidRetraining
    parser.add_argument('--RR-use-AdaHessian', default=0, type=int, required=False)
    parser.add_argument('--RR-num-epoch', default=5, type=int, required=False)
    parser.add_argument('--RR-lr', default=1e-2, type=float, required=False)
    parser.add_argument('--RR-fed-iters', default=2, type=int, required=False)

    # unlearn_ClassDiscriminativePruning
    parser.add_argument('--CDP-num-epoch', default=5, type=int, required=False)
    parser.add_argument('--CDP-lr', default=1e-2, type=float, required=False)
    parser.add_argument('--CDP-fed-iters', default=6, type=int, required=False)
    parser.add_argument('--CDP-stop-batch', default=100, type=int, required=False)
    parser.add_argument('--CDP-coe', default=1.0, type=float, required=False)
    parser.add_argument('--CDP-sparsity', default=1e-4, type=float, required=False)

    return parser.parse_known_args(args)

def unlearn_Natural(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:
    # natural
    return deepcopy(model)


def unlearn_Retrain(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"

    train_remain_index_map = other_info["train_remain_index_map"]
    test_remain_index_map = other_info["test_remain_index_map"]

    retrain_global_model, _ = server._train(train_remain_index_map, test_remain_index_map,\
                                             unlearn_config["after_task_id"], False, "Retrain")

    return retrain_global_model


def unlearn_Ours(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    args = other_info["args"]
    client_dict = other_info["client_dict"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"
    index_to_name = server.index_to_name

    # unlearn_class_set
    unlearn_class_list = []
    for c_id in unlearn_info.keys():
        unlearn_class_list += unlearn_info[c_id]['classes']
    unlearn_class_set = set(unlearn_class_list)

    # dataset
    auxiliary_dataset = get_TinyImageNet_as_public_dataset(num=args.unlearn_ours_ft_data_num)
    train_loader = DataLoader(auxiliary_dataset, batch_size=32, num_workers=0)

    # model
    unlearned_model = deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)
    unlearned_model.train()

    if args.unlearn_ours_use_kd:
        original_model = deepcopy(model)
        original_model = original_model.to(args.device)
        original_model.eval()
        print('use_kd')

        if args.unlearn_ours_selective_kd:
            print('use_selective_kd')

    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=args.unlearn_ours_lr)
    
    # get client proto dict
    client_proto_dict = calculate_client_prototype(client_dict, deepcopy(model), server.train_dataset, args)

    # get proto dist
    client_proto_dist_dict,  client_neighbor_dist_dict = calculate_client_proto_dist(client_proto_dict)

    # training
    for epoch in range(args.unlearn_ours_num_epoch):
        for batch, (images, _) in enumerate(train_loader):
            images = images.to(args.device)
            outputs, features = unlearned_model(images, return_features=True)
            outputs = torch.cat(outputs, dim=1)
            preds = outputs.argmax(1)
            loss = Ours_loss_fn(features, preds, unlearn_info, client_proto_dict, client_proto_dist_dict, client_neighbor_dist_dict, index_to_name)

            if args.unlearn_ours_use_kd:

                soft_labels = original_model(images, return_features=False)
                soft_labels = torch.cat(soft_labels, dim=1)
                soft_preds = soft_labels.argmax(1)
                mask = [1 for _ in range(soft_preds.shape[0])]

                if args.unlearn_ours_selective_kd:
                    for i in range(soft_preds.shape[0]):
                        if (index_to_name[soft_preds[i].item()] in unlearn_class_set):
                            mask[i] = -1
                
                loss = args.unlearn_ours_lambda_KI * loss + \
                    (1-args.unlearn_ours_lambda_KI) * cross_entropy_with_temp_mask(outputs, soft_labels, mask, exp=1.0/args.unlearn_ours_KD_temp)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\rOurs E[{epoch}/{args.unlearn_ours_num_epoch}] B[{batch}] L[{f'{loss.item():.2f}'}]", end=" ")

    return unlearned_model


def unlearn_Accum(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:
    """
    new_GM_{t+1} = new_GM_{t} + (old_CMs_{t} - old_GM_{t})
    """
    assert unlearn_config["unlearn_content"]["unlearn_all"], "only for client level unlearning"

    client_weight_dict = other_info["client_weights"]
    global_weight_dict = other_info["global_weights"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"
    stop_task_id = unlearn_config["after_task_id"]

    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])
    unlearned_model = deepcopy(model)
    new_global_weight = deepcopy(server.init_model.state_dict())

    for task_id in range(stop_task_id+1):
        # update new classifer
        new_weigth_keys = list(set(global_weight_dict[task_id][0])-set(new_global_weight))
        for k in new_weigth_keys:
            new_global_weight[k] = global_weight_dict[task_id][0][k]

        round_id_list = sorted(client_weight_dict[task_id].keys())
        for round_id in round_id_list:
            # remove unlearned client
            old_client_weights = []
            for client_id, weight in client_weight_dict[task_id][round_id].items():
                if client_id not in remove_client_id_set:
                    old_client_weights.append(weight)
            old_client_model_avg = weights_average(old_client_weights)

            # new_GM_{t+1} = new_GM_{t} + (old_CMs_{t} - old_GM_{t})
            previous_global_model = None
            if round_id > 0:
                previous_global_model = global_weight_dict[task_id][round_id-1]
            elif task_id > 0:
                last_round_id = sorted(global_weight_dict[task_id-1].keys())[-1]
                previous_global_model = global_weight_dict[task_id-1][last_round_id]
            else:
                previous_global_model = new_global_weight
            
            delta_weight = weights_sub(old_client_model_avg, previous_global_model)
            new_global_weight = weights_add(delta_weight, new_global_weight)

    removed_keys = [k for k in new_global_weight if 'class' in k]
    unlearned_model_weight = unlearned_model.state_dict()
    for key in removed_keys:
        new_global_weight[key] = deepcopy(unlearned_model_weight[key])

    unlearned_model.load_state_dict(new_global_weight)

    return unlearned_model


def unlearn_FedEraser(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    assert unlearn_config["unlearn_content"]["unlearn_all"], "only for client level unlearning"

    args = other_info["args"]
    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])
    client_weight_dict = other_info["client_weights"]
    global_model_dict = other_info["global_weights"]
    server = other_info["server"]
    stop_task_id = unlearn_config["after_task_id"]

    # prepare client
    calibrated_client_dict = {}
    for c_id, c in other_info["client_dict"].items():
        if c_id not in remove_client_id_set:
            client = deepcopy(c)
            # clear classifers
            client.model = deepcopy(model)
            client.model.classifiers = nn.ModuleList()
            client.args.fed_iters = args.FedEraser_epoch_cali
            client.learned_global_task_id = []
            client.examplar_index_map = {}
            # add info
            client.train_set = server.train_dataset
            client.test_set = server.test_dataset
            calibrated_client_dict[c_id] = client

    unlearned_global_model = deepcopy(model)
    new_global_weight = weights_device_convert(deepcopy(server.init_model.state_dict()), args.device)

    for task_id in range(stop_task_id+1):

        # update new classifer for global model
        new_weight_keys = list(set(global_model_dict[task_id][0])-set(new_global_weight))
        for k in new_weight_keys:
            new_global_weight[k] = global_model_dict[task_id][0][k]

        # add new classifier for client
        for c_id, client in calibrated_client_dict.items():
            num_classes = len(server.train_index_map[task_id]["classes"])
            client.model.add_head(num_classes)

        round_id_list = sorted(client_weight_dict[task_id].keys(), key=lambda x: int(x))
        end_round_id = round_id_list[-1]
        for round_id in round_id_list:
            # 1.Clients Cali Training
            for c_id, client in calibrated_client_dict.items():
                client.set_local_info(new_global_weight)
                client.model = client.model.to(client.device)
                client.local_update(task_id, round_id, end_round_id)
            new_client_updates = [client.get_local_info() for client in calibrated_client_dict.values()]

            old_client_updates = []
            for client_id, weight in client_weight_dict[task_id][round_id].items():
                if client_id not in remove_client_id_set:
                    old_client_updates.append(weight)
            # 2.Update Calibrating
            # 3.Update Aggregating
            # 4.Model Updating
            new_global_weight = FedEraser_unlearn_step(global_model_dict[task_id][round_id], old_client_updates, new_global_weight, new_client_updates, args.device)

    unlearned_global_model.load_state_dict(new_global_weight)

    return unlearned_global_model


def unlearn_ClassDiscriminativePruning(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:
    args = other_info["args"]
    is_client_unlearn = unlearn_config["unlearn_content"]["unlearn_all"]
    stop_task_id = unlearn_config["after_task_id"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"

    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])
    all_unlearn_classes = []
    for v in unlearn_info.values():
        all_unlearn_classes += v["classes"]
    unlearn_classes = sorted(list(set(all_unlearn_classes)))

    train_dataset = server.train_dataset
    test_dataset = server.test_dataset
    name_to_index = server.name_to_index

    train_remain_index_map = other_info["train_remain_index_map"]
    test_remain_index_map = other_info["test_remain_index_map"]
    train_index_map = server.train_index_map

    unlearned_model = deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)

    # Step 1. Local Processing in FL Clients
    # Step 2. Processing in the Federated Server
    idxs = []
    num_classes = 0
    for task_id, task_info in train_index_map.items():
        if task_id > stop_task_id:
            break
        num_classes += len(task_info['classes'])
        for c_id, c_task in task_info['shards'].items():
            idxs += c_task["idxs"]
    train_set = DatasetSplit(train_dataset, idxs)
    train_all_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    
    for u_class_name in unlearn_classes:
        u_class_time = time.time()
        u_class_index = name_to_index[u_class_name]
        # removing one class each time
        feature_iit, targets = CDP_acculumate_feature(unlearned_model, train_all_loader, args.CDP_stop_batch, args.device)
        tf_idf_map = CDP_calculate_cp(feature_iit, targets, num_classes, name_to_index, args.CDP_coe, unlearn_class=u_class_index)
        threshold = CDP_get_threshold_by_sparsity(tf_idf_map, args.CDP_sparsity)

        cp_config={
            "threshold": threshold,
            "map": tf_idf_map
        }
        config_list = [{
            'op_types': ['Conv2d'],
            'sparse_ratio': args.CDP_sparsity,
        }]
        pruner = CDP_TFIDFPruner(unlearned_model.feature_extractor, config_list, cp_config)
        tf_idf_masks = pruner.get_tf_idf_masks()
        pruner.unwrap_model()
        unlearned_model.feature_extractor = ModelSpeedup(unlearned_model.feature_extractor, torch.rand(32, 3, 32, 32).to(args.device), tf_idf_masks).speedup_model()
        print(f"unlearn class [{u_class_name}] Time[{time.time()-u_class_time:.2f}]")

    # Step 3. Fine-tuning Processing
    # prepare client
    all_client_dict = {}
    for c_id, c in other_info["client_dict"].items():
        if is_client_unlearn and c_id in remove_client_id_set:
            continue
        else:
            client = deepcopy(c)
            # clear classifers
            client.model = deepcopy(unlearned_model)
            client.model.classifiers = nn.ModuleList()
            # reset args
            client.args.fed_iters = args.CDP_fed_iters
            client.args.lr = args.CDP_lr
            # clear dataset
            client.learned_global_task_id = []
            client.local_train_index_map = {}
            client.local_test_index_map = {}
            client.examplar_index_map = {}
            all_client_dict[c_id] = client


    for task_id in range(0, stop_task_id+1):
        # client configuration
        for client in all_client_dict.values():
            # config dataset
            client.configure(
                task_id, train_remain_index_map[task_id]['classes'],
                train_dataset, train_remain_index_map[task_id]['shards'][client.id],
                test_dataset, test_remain_index_map[task_id]['shards'][client.id]
            )

        for round_id in range(args.CDP_num_epoch):
            # local update
            for c_id, client in all_client_dict.items():
                client.local_update(task_id, round_id, args.RR_num_epoch)
            # aggregate
            new_client_updates = [client.get_local_info() for client in all_client_dict.values()]
            new_global_weight = weights_average(new_client_updates)
            # dispatch
            for c_id, client in all_client_dict.items():
                client.set_local_info(new_global_weight)

    unlearned_model.load_state_dict(new_global_weight)

    return unlearned_model


def unlearn_KnowledgeDistillation(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    assert unlearn_config["unlearn_content"]["unlearn_all"], "only for client level unlearning"

    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])
    assert len(remove_client_id_set)==1, "only one unlearned client"

    args = other_info["args"]
    client_weight_dict = other_info["client_weights"]
    global_weight_dict = other_info["global_weights"]
    server = other_info["server"]
    stop_task_id = unlearn_config["after_task_id"]
    assert isinstance(server, Server), f"server class is {type(server)}!"

    init_weight = deepcopy(server.init_model.state_dict())

    original_model = deepcopy(model)
    original_model = original_model.to(args.device)
    unlearned_model = deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)
    unlearned_model_weight = deepcopy(unlearned_model.state_dict())
    
    # Step1. Subtract the accumulated historical updates from the unlearned client
    for task_id in range(stop_task_id+1):
        unlearned_client_weight = None

        round_id_list = sorted(client_weight_dict[task_id].keys(), key=lambda x: int(x))
        for round_id in round_id_list:

            for client_id, weight in client_weight_dict[task_id][round_id].items():
                if client_id in remove_client_id_set:
                    
                    if round_id > 0:
                        accum_weight = weights_sub(weight, global_weight_dict[task_id][round_id-1])
                    elif task_id > 0:
                        last_round = sorted(global_weight_dict[task_id-1].keys())[-1]
                        accum_weight = weights_sub(weight, global_weight_dict[task_id-1][last_round])
                    else:
                        accum_weight = weights_sub(weight, init_weight)
                    
                    if unlearned_client_weight == None:
                        unlearned_client_weight = accum_weight
                    else:
                        unlearned_client_weight = weights_add(unlearned_client_weight, accum_weight)

        client_num = len(client_weight_dict[task_id][0])
        delta_weight = weights_div(unlearned_client_weight, client_num)
        
        unlearned_model_weight = weights_sub(unlearned_model_weight, weights_device_convert(delta_weight, args.device))
    
    unlearned_model.load_state_dict(unlearned_model_weight)
    
    # Step2. Leverage the knowledge distillation method to restore the model’s performance without using any data from the clients
    auxiliary_dataset = get_SVHN_as_public_dataset(num=args.KD_num_data)
    data_loader = DataLoader(auxiliary_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    original_model.eval()
    unlearned_model.train()

    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=args.KD_lr)

    # training
    for epoch in range(args.KD_num_epoch):
        for batch, (images, labels) in enumerate(data_loader):
            # copy to device
            images = images.to(args.device)
            # Forward current model
            soft_labels = original_model(images, return_features=False)
            outputs = unlearned_model(images, return_features=False)
            loss = cross_entropy_with_temp(torch.cat(outputs, dim=1), torch.cat(soft_labels, dim=1), exp=1.0/args.KD_temp)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"\rKD E[{epoch}/{args.KD_num_epoch}] B[{batch}] L[{f'{loss.item():.2f}'}]", end="")
    
    return unlearned_model


def unlearn_Projected_GA(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    args = other_info["args"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"

    name_to_index = server.name_to_index
    train_forget_index_map = other_info["train_forget_index_map"]
    train_dataset = server.train_dataset
    client_update_dict = other_info["client_weights"]
    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])
    
    # data
    idxs = []
    for task_id, task_info in train_forget_index_map.items():
        for client_id, client_task in task_info['shards'].items():
            idxs += client_task['idxs']

    data_set = DatasetSplit(train_dataset, idxs)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    # model
    unlearned_model = deepcopy(model)
    unlearned_model = unlearned_model.to(args.device)
    unlearned_model.train()

    # compute reference model
    # w_ref = N/(N-1)w^T - 1/(N-1)w^{T-1}_i = \sum{i \ne j}w_j^{T-1}

    lastest_task_id = sorted(client_update_dict.keys())[-1]
    lastest_round_id = sorted(client_update_dict[lastest_task_id].keys())[-1]
    lateset_client_updates = client_update_dict[lastest_task_id][lastest_round_id]
    remain_client_weight_list = [weight for c_id, weight in lateset_client_updates.items() if c_id not in remove_client_id_set]
    reference_model_weight = weights_average(remain_client_weight_list)
    reference_model = deepcopy(model)
    reference_model.load_state_dict(reference_model_weight, strict=False)
    reference_model = reference_model.to(args.device)

    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=args.pGA_lr)

    # training
    for epoch in range(args.pGA_num_epoch):
        for batch, (images, labels) in enumerate(data_loader):
            # label name to local index
            for i in range(labels.shape[0]):
                labels[i] = name_to_index[labels[i].item()]
            # copy to device
            images, labels = images.to(args.device), labels.to(args.device)
            # Forward current model
            outputs = unlearned_model(images, return_features=False)
            ce_loss = F.cross_entropy(torch.cat(outputs, dim=1), labels)
            loss = -ce_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"\rpGA E[{epoch}/{args.pGA_num_epoch}] B[{batch}] L[{f'{loss.item():.2f}'}]", end=" ")

            with torch.no_grad():
                distance = weight_distance(reference_model.state_dict(), unlearned_model.state_dict())
                if distance > args.pGA_distance_threshold:
                    dist_vec = nn.utils.parameters_to_vector(model.parameters()).to(args.device) - nn.utils.parameters_to_vector(reference_model.parameters()).to(args.device)
                    dist_vec = dist_vec/torch.norm(dist_vec)*np.sqrt(args.pGA_distance_threshold)
                    proj_vec = nn.utils.parameters_to_vector(reference_model.parameters()).to(args.device) + dist_vec
                    nn.utils.vector_to_parameters(proj_vec, model.parameters())
                    # p_distance = weight_distance(reference_model.state_dict(), unlearned_model.state_dict())
                    # print(f"d:{distance} pd:{p_distance} t:{args.pGA_distance_threshold}", end="\n")

    return unlearned_model


def unlearn_EWC_SGA(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:
    
    args = other_info["args"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"
    name_to_index = server.name_to_index
    train_dataset = server.train_dataset
    train_forget_index_map = other_info["train_forget_index_map"]
    
    # data
    idxs = []
    for task_id, task_info in train_forget_index_map.items():
        for client_id, client_task in task_info['shards'].items():
            idxs += client_task['idxs']

    data_set = DatasetSplit(train_dataset, idxs)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    # model
    origin_model = deepcopy(model)
    unlearned_model = deepcopy(model)
    origin_model = origin_model.to(args.device)
    unlearned_model = unlearned_model.to(args.device)
    origin_model.eval()
    unlearned_model.train()

    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=args.ewcsga_lr)

    # get fisher matrix
    fisher_mat = compute_fisher_matrix_diag(unlearned_model, data_loader, name_to_index, optimizer, args)

    # training
    for epoch in range(args.ewcsga_num_epoch):
        for batch, (images, labels) in enumerate(data_loader):
            # label name to local index
            for i in range(labels.shape[0]):
                labels[i] = name_to_index[labels[i].item()]
            # copy to device
            images, labels = images.to(args.device), labels.to(args.device)
            # Forward current model
            outputs = unlearned_model(images, return_features=False)
            ce_loss = F.cross_entropy(torch.cat(outputs, dim=1), labels)
            ewc_loss = loss_EWC(origin_model, unlearned_model, fisher_mat)

            loss = -ce_loss + ewc_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"\rEWC-SGA E[{epoch}/{args.pGA_num_epoch}] B[{batch}] L[{f'{loss.item():.2f}'}]", end=" ")

    return unlearned_model


def unlearn_RapidRetraining(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> nn.Module:

    args = other_info["args"]
    server = other_info["server"]
    assert isinstance(server, Server), f"server class is {type(server)}!"

    is_client_unlearn = unlearn_config["unlearn_content"]["unlearn_all"]
    stop_task_id = unlearn_config["after_task_id"]
    remove_client_id_set = set([c_id for c_id in unlearn_info.keys()])

    train_dataset = server.train_dataset
    test_dataset = server.test_dataset
    train_remain_index_map = other_info["train_remain_index_map"]
    test_remain_index_map = other_info["test_remain_index_map"]

    # prepare client
    all_client_dict = {}
    for c_id, c in other_info["client_dict"].items():
        if is_client_unlearn and c_id in remove_client_id_set:
            continue
        else:
            client = deepcopy(c)
            # clear classifers
            client.model = deepcopy(model)
            client.model.classifiers = nn.ModuleList()
            # reset args
            client.args.RR_use_AdaHessian = False
            client.args.fed_iters = args.RR_fed_iters
            client.args.lr = args.RR_lr
            # clear dataset
            client.learned_global_task_id = []
            client.local_train_index_map = {}
            client.local_test_index_map = {}
            client.examplar_index_map = {}
            all_client_dict[c_id] = client

    unlearned_global_model = deepcopy(model)

    for task_id in range(0, stop_task_id+1):
        # client configuration
        for client in all_client_dict.values():
            # config dataset
            client.configure(
                task_id, train_remain_index_map[task_id]['classes'],
                train_dataset, train_remain_index_map[task_id]['shards'][client.id],
                test_dataset, test_remain_index_map[task_id]['shards'][client.id]
            )

        for round_id in range(args.RR_num_epoch):
            # local update
            for c_id, client in all_client_dict.items():
                client.local_update(task_id, round_id, args.RR_num_epoch)
            # aggregate
            new_client_updates = [client.get_local_info() for client in all_client_dict.values()]
            new_global_weight = weights_average(new_client_updates)
            # dispatch
            for c_id, client in all_client_dict.items():
                client.set_local_info(new_global_weight)

    unlearned_global_model.load_state_dict(new_global_weight)

    return unlearned_global_model


CLIENT_LEVEL_UNLEARN_METHOD = {
    "Accum": unlearn_Accum,
    "FedEraser": unlearn_FedEraser,
    "KD": unlearn_KnowledgeDistillation,
    "pGA": unlearn_Projected_GA,
}

CLASS_LEVEL_UNLEARN_METHOD = {
    "Natural": unlearn_Natural,
    "Retrain": unlearn_Retrain,
    "Ours": unlearn_Ours,
    "CDP": unlearn_ClassDiscriminativePruning,
    "EWC-SGA": unlearn_EWC_SGA,
    "RR": unlearn_RapidRetraining,
}

UNLEARN_METHOD_Dict = {**CLIENT_LEVEL_UNLEARN_METHOD, **CLASS_LEVEL_UNLEARN_METHOD}


def perform_all_unlearn_method(model: nn.Module, unlearn_config: Dict, unlearn_info: Dict, other_info: Dict) -> Dict[str, nn.Module]:
    """
    unlearned_model_dict: {
        name: {
            "model": nn.Module,
            "time": time,
        }
    }
    """
    unlearned_model_dict = {}

    running_method_dict = None
    if unlearn_config["unlearn_content"]["unlearn_all"]:
        running_method_dict = UNLEARN_METHOD_Dict
    else:
        running_method_dict = CLASS_LEVEL_UNLEARN_METHOD

    args = other_info["args"]
    if args.only_ours:
        running_method_dict = {"Ours": unlearn_Ours}

    for name, unlearn_method in running_method_dict.items():
        print(f"start unlearn {name}")
        s_start = time.time()
        t_model = unlearn_method(model, unlearn_config, unlearn_info, other_info)
        time_cost = time.time() - s_start
        print(f"finish unlearn {name} Time[{time_cost:.2f}s]")
        unlearned_model_dict[name] = {
            "model": t_model,
            "time": time_cost,
        }

    return unlearned_model_dict
