import argparse
import json
import random
from typing import Dict, Tuple, List
from torch.utils.data import Dataset
from copy import deepcopy

def get_unlearn_config(args: argparse.Namespace) -> Dict:
    assert args.unlearn_config_path != '', "unlearn_config_path is empty"
    unlearn_config = {}
    with open(args.unlearn_config_path) as f:
        unlearn_config = json.load(f)
    return unlearn_config


def _split_index_map_to_remain_and_forget(unlearn_info: Dict, dataset: Dataset, index_map: Dict):
    """
    # index_map : Dict
    {
        task_id: {
            "classes" : List[int],
            "shards" : {
                client_id: {
                    "classes" : List[int],
                    "idxs" : List[int]
                },
            }
        }
    }
    """
    remain_index_map, forget_index_map = deepcopy(index_map), deepcopy(index_map)

    for task_id, task_info in index_map.items():
        for client_id, client_shard in task_info["shards"].items():
            if client_id not in unlearn_info:
                forget_classes_set = set()
            else:
                forget_classes_set = set(client_shard["classes"]) & set(unlearn_info[client_id]["classes"])
            forget_classes = list(forget_classes_set)
            if forget_classes:
                forget_idxs, remain_idxs = [], []
                for i in client_shard["idxs"]:
                    if dataset.targets[i] in forget_classes_set:
                        forget_idxs.append(i)
                    else:
                        remain_idxs.append(i)
                # to keep the index_name_map same in learning and unlearning
                remain_index_map[task_id]["shards"][client_id]["origin_classes"] = client_shard["classes"]
                remain_index_map[task_id]["shards"][client_id]["classes"] = list(set(client_shard["classes"]) - set(forget_classes))
                remain_index_map[task_id]["shards"][client_id]["idxs"] = remain_idxs

                forget_index_map[task_id]["shards"][client_id]["origin_classes"] = client_shard["classes"]
                forget_index_map[task_id]["shards"][client_id]["classes"] = forget_classes
                forget_index_map[task_id]["shards"][client_id]["idxs"] = forget_idxs
            else:
                forget_index_map[task_id]["shards"][client_id]["classes"] = []
                forget_index_map[task_id]["shards"][client_id]["idxs"] = []

        remain_index_map[task_id]["classes"] = index_map[task_id]["classes"]
        forget_index_map[task_id]["classes"] = index_map[task_id]["classes"]

        remain_index_map[task_id]["remain_classes"] = list(set(sum([a["classes"] for a in remain_index_map[task_id]["shards"].values()], [])))
        forget_index_map[task_id]["forget_classes"] = list(set(sum([a["classes"] for a in forget_index_map[task_id]["shards"].values()], [])))
    
    return remain_index_map, forget_index_map


def get_unlearn_index_map(
        args: argparse.Namespace, unlearn_config: Dict, 
        train_dataset: Dataset, train_index_map: Dict, test_dataset: Dataset, test_index_map: Dict, 
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    # unlearn_info
    {
        client_id: {
            "classes": List[int],
        }
    }

    # index_map
    {
        task_id: {
            "classes" : List[int],
            "shards" : {
                client_id: {
                    "classes" : List[int],
                    "idxs" : List[int]
                },
            }
        }
    }
    """
    unlearn_info = {}
    client_num = unlearn_config["unlearn_content"]["client_num"]
    client_id_list = random.sample(range(args.num_clients), client_num)
    for client_id in client_id_list:
        all_classes = []
        for task_id, info in train_index_map.items():
            # select learned classes
            if task_id > unlearn_config["after_task_id"]:
                break
            if client_id in info["shards"]:
                all_classes += info["shards"][client_id]["classes"]

        if unlearn_config["unlearn_content"]["unlearn_all"]:
            unlearn_info[client_id] = {
                "classes": all_classes,
            }
        else:
            selected_classes = random.sample(all_classes, unlearn_config["unlearn_content"]["unlearn_classes_num"])
            unlearn_info[client_id] = {
                "classes": selected_classes,
            }

    train_remain_index_map, train_forget_index_map = _split_index_map_to_remain_and_forget(unlearn_info, train_dataset, train_index_map)
    test_remain_index_map, test_forget_index_map = _split_index_map_to_remain_and_forget(unlearn_info, test_dataset, test_index_map)

    return train_remain_index_map, train_forget_index_map, test_remain_index_map, test_forget_index_map, unlearn_info