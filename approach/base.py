import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import pickle
import os

from copy import deepcopy
from functools import reduce
from operator import add
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from lib.utils import weights_average, weights_device_convert
from lib.dataset import DatasetSplit
from lib.logger import ExpLogger, summarize_test_results, print_test_result, recursive_dump_dict_to_pkl
from lib.network import CLNN
from lib.unlearn_utils import AdaHessian

def approach_parser(arg_str_list: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-type', type=str, default='CLNN')
    # examplar dataset
    parser.add_argument('--use-exemplar', type=bool, default=True)
    parser.add_argument('--num-per-class', type=int, default=20)
    return parser.parse_known_args(arg_str_list)


class Server():
    """
    Federated Continual Learning (FCL) server base class
    """
    def __init__(self, args: argparse.Namespace, device: str, network: nn.Module, logger: ExpLogger, 
        train_dataset: Dataset, train_index_map: Dict, test_dataset: Dataset, test_index_map: Dict, name_to_index: Dict, index_to_name: Dict,
        ) -> None:
        """
        Generally no modifications are required in the inherited class
        """
        self.args = args
        self.device = device
        self.init_model = network
        # self.global_model = network
        # self.client_list = []
        self.name_to_index = name_to_index
        self.index_to_name = index_to_name
        self.logger = logger
        # dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_index_map = train_index_map
        self.test_index_map = test_index_map
        # need to be modified
        self.client_class_path = "approach.base"


    def _train(self, train_index_map: Dict, test_index_map: Dict, stop_task_id: int, save_unlearn_related_info: bool = False, log_name_postfix: str = ""):
        # clear state
        unlearn_related_info = {
            "client_weights": {}, # {task_id: {round: {client_id: OrderedDict}}}
            "global_weights": {}, # {task_id: {round: OrderedDict}}
        }
        # create clients and global model
        client_list = self._create_client()
        global_model = deepcopy(self.init_model)
        print(f"start FCL training!!!, create {len(client_list)} clients!")

        # continual learning for each task
        assert stop_task_id < self.args.num_tasks, "stop_task_id >= self.args.num_tasks"
        for task_id in range(0, stop_task_id+1):
            s_task = time.time()
            print("-"*100)
            print(f"start task [{task_id+1}/{len(train_index_map)}]")

            task_train_idx_map, task_test_idx_map = train_index_map[task_id], test_index_map[task_id]
            
            # client selection
            task_client_list = [client_list[c_id] for c_id in sorted(task_train_idx_map["shards"].keys())]
            print(f"select {len(task_client_list)} clients, they are {[c.id for c in task_client_list]}")

            # global model config
            self._global_model_configure(task_id, task_train_idx_map['classes'], global_model)

            # client configuration
            for client in task_client_list:
                # config dataset
                client.configure(
                    task_id, task_train_idx_map['classes'],
                    self.train_dataset, task_train_idx_map['shards'][client.id],
                    self.test_dataset, task_test_idx_map['shards'][client.id]
                )
                print(f"Client: {client.id} new_classes: {task_train_idx_map['shards'][client.id]['classes']}")
            print(f"Clients configure OK")

            # federated learning
            for round in range(self.args.nepochs):
                s_round = time.time()
                # clients local update
                for client in task_client_list:
                    s_local_update = time.time()
                    client.local_update(task_id, round, self.args.nepochs)
                    print(f"Client[{client.id}] Round[{round+1}/{self.args.nepochs}] Time[{(time.time()-s_local_update):.2f}s]")
                
                if save_unlearn_related_info:
                    # historical parameter updates
                    print("update unlearn_related_info client_updates")
                    # client_updates
                    if task_id not in unlearn_related_info["client_weights"]:
                        unlearn_related_info["client_weights"][task_id] = {}
                    unlearn_related_info["client_weights"][task_id][round] = {a.id: weights_device_convert(a.get_local_info(), "cpu") for a in task_client_list}
                
                # server aggregation
                s_aggregate = time.time()
                self._server_aggregation(task_client_list, global_model)
                if self.args.eval_on_train:
                    s_test_round = time.time()
                    self._test_per_round(task_client_list, task_id, round, log_name_postfix)
                    print(f"TEST PER ROUND Time:[{(time.time()-s_test_round):.2f}s]")
                print(f"Task[{task_id+1}/{len(train_index_map)}] Round[{round+1}/{self.args.nepochs}] Time[{(time.time()-s_round):.2f}s] Aggregate[{(time.time()-s_aggregate):.2f}s]")

                if save_unlearn_related_info:
                    # global_state_dict
                    print("update unlearn_related_info global_state_dict")
                    if task_id not in unlearn_related_info["global_weights"]:
                        unlearn_related_info["global_weights"][task_id] = {}
                    unlearn_related_info["global_weights"][task_id][round] = weights_device_convert(deepcopy(global_model.state_dict()), "cpu")

            # test
            s_test = time.time()
            self._test(task_client_list, task_id, log_name_postfix)
            print(f"Task[{task_id+1}/{len(self.train_index_map)}] test OK Time[{(time.time()-s_test):.2f}s]")
            
            # summarize test result
            test_log_dict = summarize_test_results(self.logger.test_result_file_path, task_id, log_name_postfix)
            self.logger.log_test_result(test_log_dict)
            print_test_result(test_log_dict)

            print(f"Task[{task_id+1}/{len(self.train_index_map)}] finish! Total Time[{(time.time()-s_task):.2f}s]")

        # save client list
        if save_unlearn_related_info:
            print("save client list")
            unlearn_related_info["client_dict"] = {}
            for client in client_list:
                client.train_set = None
                client.test_set = None
                client.model = None
                client.old_network = None
                unlearn_related_info["client_dict"][client.id] = client

        return global_model, unlearn_related_info

    def train(self):
        """
        start a FCL exp
        Generally no modifications are required in the inherited class
        """
        s_fcl = time.time()
        # training
        stop_task_id = self.args.num_tasks-1
        if self.args.stop_at_task != -1:
            stop_task_id = self.args.stop_at_task
        print(f'stop_task_id: {stop_task_id}')
        origin_global_model, unlearn_related_info = self._train(self.train_index_map, self.test_index_map, stop_task_id, self.args.unlearn, "")
        origin_model_dict = {
            "model": origin_global_model,
            "time": time.time() - s_fcl,
        }

        if self.args.unlearn:

            all_model_dict = {}
            all_model_dict["Origin"] = origin_model_dict

            self._save_unlearn_related_info(all_model_dict, unlearn_related_info)

            print("unlearn after global model is obtained, then finish")

        print(f"FCL training finish!!! Time[{(time.time()-s_fcl)/3600:.2f}h]")

    def _save_unlearn_related_info(self, all_model_dict, unlearn_related_info):
        # create save dirs
        log_dir = self.logger.exp_log_dir
        save_dir = os.path.join(log_dir, 'unlearn_save')
        os.makedirs(save_dir)

        # remove reduplicate data
        self.logger = None
        self.train_dataset = None
        self.test_dataset = None

        saved_info = {
            'server': self,
            'all_model_dict': all_model_dict,
            'unlearn_related_info': unlearn_related_info,
        }

        recursive_dump_dict_to_pkl(saved_info, save_dir)

    def _create_client(self):
        """
        Generally no modifications are required in the inherited class
        """
        ClientClass = getattr(importlib.import_module(name=self.client_class_path), 'Client')
        return [ClientClass(i, self.device, self.init_model, self.args, self.name_to_index, self.index_to_name) for i in range(self.args.num_clients)]

    def _client_selection(self, client_list: List, target_num: int):
        """
        Generally no modifications are required in the inherited class
        """
        # random select
        return random.sample(client_list, target_num)

    def _global_model_configure(self, global_task_id: int, task_all_classes: List, global_model: nn.Module):
        self._add_new_classes(task_all_classes, global_model)
        global_model = global_model.to(self.device)

    def _add_new_classes(self, classes: List[int], global_model: nn.Module):
        """
        Generally no modifications are required in the inherited class
        """
        global_model.add_head(len(classes))

    def _server_aggregation(self, client_list: List, global_model: nn.Module):
        # server aggregate
        weight_list = [client.get_local_info() for client in client_list]
        weight_avg = weights_average(weight_list)
        self._update_all_model(client_list, global_model, weight_avg)
        return weight_avg

    def _update_all_model(self, client_list: List, global_model: nn.Module, update_weight_dict: Dict):
        for client in client_list:
            client.set_local_info(update_weight_dict)
        global_model.load_state_dict(update_weight_dict)

    def _test(self, client_list: List, task_id: int, log_name_postfix: str = ""):
        """
        Generally no modifications are required in the inherited class
        """
        all_test_log_dict_list = []
        for client in client_list:
            all_test_log_dict_list += client.local_test()
        # save test log to file
        for test_res_log in all_test_log_dict_list:
            test_res_log["global_task_id"] = task_id
            test_res_log["log_name"] = "Test"+log_name_postfix
            self.logger.log_test_result(test_res_log)

    def _test_per_round(self, client_list: List, task_id: int, round: int, log_name_postfix: str = ""):
        """
        Generally no modifications are required in the inherited class
        """
        all_test_log_dict_list = []
        for client in client_list:
            all_test_log_dict_list += client.local_test()
        # save test log to file
        for test_res_log in all_test_log_dict_list:
            test_res_log["global_task_id"] = task_id
            test_res_log["round"] = round
            test_res_log["log_name"] = "TestPerRound"+log_name_postfix
            self.logger.log_test_result(test_res_log)


class Client():
    """
    Federated Continual Learning (FCL) client base class
    """

    def __init__(self, id: int, device: str, model: CLNN, args: argparse.Namespace, name_to_index: Dict, index_to_name: Dict) -> None:
        """
        Generally no modifications are required in the inherited class
        """
        self.id = id
        self.device = device
        self.args = args
        self.args.RR_use_AdaHessian = False

        self.model = deepcopy(model)
        self.old_network = None

        self.name_to_index = name_to_index
        self.index_to_name = index_to_name

        self.train_set = None
        self.test_set = None
        self.learned_global_task_id = []
        self.local_train_index_map = {} # {global_task_id: {'classes':List[int], 'idxs':List[int]}}
        self.local_test_index_map = {}  # {global_task_id: {'classes':List[int], 'idxs':List[int]}}
        self.examplar_index_map = {}    # {class_name: List[int]}

    def get_local_info(self):
        return deepcopy(self.model.state_dict())

    def set_local_info(self, weight_avg):
        self.model.load_state_dict(deepcopy(weight_avg))

    def configure(self, global_task_id, task_all_classes, train_set, train_shard, test_set, test_shard):
        """
        Generally no modifications are required in the inherited class
        """
        # config learning task info
        self.train_set = train_set
        self.test_set = test_set
        self.learned_global_task_id.append(global_task_id)
        self.local_train_index_map[global_task_id] = train_shard
        self.local_test_index_map[global_task_id] = test_shard
        # NOTE add classifier for all task classes, ensure same model architecture
        new_class_num = self._add_new_classes(task_all_classes)
        # copy network to device
        self.model = self.model.to(self.device)
        return new_class_num

    def local_update(self, global_task_id: int, cur_round: int, total_round: int):
        # train dataset add examplar dataset
        task_idxs = self.local_train_index_map[global_task_id]['idxs']
        if self.examplar_index_map and self.args.use_exemplar:
            examplar_idxs = reduce(add, [i_list for i_list in self.examplar_index_map.values()])
            # print(f"examplar_idxs:{len(examplar_idxs)}")
        else:
            examplar_idxs = []
        idxs = task_idxs + examplar_idxs
        if len(idxs) == 0:
            return
        # get train dataloader
        train_loader = self._get_data_loader(self.train_set, idxs)
        # prepare model
        self.model.train()
        # get optimizer
        optimizer = self._get_optimizer()

        # training
        for epoch in range(self.args.fed_iters):
            for batch, (images, labels) in enumerate(train_loader):
                # label name to local index
                for i in range(labels.shape[0]):
                    labels[i] = self.name_to_index[labels[i].item()]
                # copy to device
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward current model
                outputs = self.model(images, return_features=False)
                loss = self._loss_new_task_learning(outputs, labels)
                # backward
                optimizer.zero_grad()
                self._loss_backword(loss)
                optimizer.step()
        # last global round
        if (cur_round == (total_round-1)):
            if self.args.use_exemplar:
                self._exemplar_memory_update(self.train_set, self.local_train_index_map[global_task_id])

    def local_test(self):
        """
        Generally no modifications are required in the inherited class
        """
        test_log_dict_list = []
        for task_id, shard in self.local_test_index_map.items():
            if len(shard['idxs'])==0:
                continue
            test_loader = self._get_data_loader(self.test_set, shard['idxs'])
            test_log_dict = self._test_signle_task(test_loader)
            test_log_dict["client_id"] = self.id
            test_log_dict["test_task_id"] = task_id
            test_log_dict_list.append(test_log_dict)
        return test_log_dict_list
 
    def _test_signle_task(self, test_loader: DataLoader) -> Dict:
        # prepare model
        self.model.eval()
        # test
        with torch.no_grad():
            total_hits_taw, total_hits_tag, total_num = 0, 0, 0
            total_loss_list = [0]
            for batch, (images, labels) in enumerate(test_loader):
                # label name to local index
                for i in range(labels.shape[0]):
                    labels[i] = self.name_to_index[labels[i].item()]
                # copy to device
                images, labels = images.to(self.device), labels.to(self.device)
                # Forward current model
                outputs = self.model(images, return_features=False)
                # loss
                loss = self._loss_new_task_learning(outputs, labels)
                # metrics
                hits_taw, hits_tag = self._metrics(outputs, labels)
                # save result
                total_loss_list[0] += loss.item() * len(labels)
                total_hits_taw += hits_taw.sum().item()
                total_hits_tag += hits_tag.sum().item()
                total_num += len(labels)
        # create log dict
        test_log_dict = {
            "log_name": "Test",
            "global_task_id": -1,
            "client_id": -1,
            "test_task_id": -1,
            "total_hits_taw": total_hits_taw,
            "total_hits_tag": total_hits_tag,
            "total_num": total_num,
            "total_loss_list": [loss.item()],
        }
        return test_log_dict

    def _loss_new_task_learning(self, outputs: List[Tensor], labels: Tensor):
        # classification loss
        loss = F.cross_entropy(torch.cat(outputs, dim=1), labels)
        return loss

    def _get_optimizer(self):
        """
        Generally no modifications are required in the inherited class
        """
        if self.args.RR_use_AdaHessian == True:
            optimizer = AdaHessian(self.model.parameters(), lr=self.args.lr)
            print(f"Client[{self.id}] using AdaHessian for Rapid Retraining!")
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        return optimizer

    def _loss_backword(self, loss):
        if self.args.RR_use_AdaHessian == True:
            loss.backward(create_graph=True)
        else:
            loss.backward()

    def _get_data_loader(self, dataset, idxs):
        """
        Generally no modifications are required in the inherited class
        """
        data_set = DatasetSplit(dataset, idxs)
        data_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        return data_loader

    def _add_new_classes(self, classes: List[int]):
        """
        Generally no modifications are required in the inherited class
        """
        new_class_num = len(classes)
        self.model.add_head(new_class_num)
        return new_class_num

    def _exemplar_memory_update(self, train_set: Dataset, train_shard: Dict):
        """
        Generally no modifications are required in the inherited class
        """
        idxs = np.array(train_shard['idxs'])
        labels = np.array(train_set.targets)[idxs]
        # prepare model
        self.model.eval()
        for class_name in train_shard['classes']:
            t_idxs = idxs[np.where(labels==class_name)[0]]
            # get trained images of this class, images [N,C,H,W]
            images = torch.stack([train_set[i][0] for i in t_idxs])
            # compute mean and distance
            with torch.no_grad():
                images = images.to(self.device)
                # features [N,C]
                _, features = self.model(images, return_features=True)
                # compute the mean of features, [C]
                feature_mean = torch.mean(features, dim=0).detach()
                # compute the distances
                dists = (features - feature_mean).pow(2).sum(1).squeeze()
                dists = dists.cpu().numpy()
                idxs_dists = np.vstack((t_idxs, dists))
                # sorted by distance
                idxs_sorted = idxs_dists[:, idxs_dists[1, :].argsort()]
                # store index
                self.examplar_index_map[class_name] = idxs_sorted[:,:self.args.num_per_class][0].astype(int).tolist()
        print(f"Client[{self.id}] update exemplar! classes_num[{len(self.examplar_index_map)}] exemplars_num[{sum([len(b) for b in self.examplar_index_map.values()])}]")

    def _metrics(self, outputs, labels):
        """
        Generally no modifications are required in the inherited class
        """
        pred = torch.zeros_like(labels)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= labels[m].to('cpu')).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == labels).float()
        # Task-Agnostic Multi-Head
        if self.args.multi_softmax:
            outputs = [F.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == labels).float()
        return hits_taw, hits_tag
