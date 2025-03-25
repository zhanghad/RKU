import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from unlearning_test.utils import DatasetSplit

def all_unlearn_metirc(all_model_dict: Dict, unlearn_config: Dict, unlearn_info: Dict, unlearn_related_info: Dict,
        train_dataset: Dataset, train_index_map: Dict, test_dataset: Dataset, test_index_map: Dict,        
        train_remain_index_map: Dict, train_forget_index_map: Dict, test_remain_index_map: Dict, test_forget_index_map: Dict,
        name_to_index: Dict, device: str, batch_size: int = 32, num_workers: int = 0,
    ) -> Dict:

    args = unlearn_related_info['args']
    server = unlearn_related_info['server']

    test_res = {
        "acc_train_remain": {},
        "acc_train_forget": {},
        "acc_test_remain": {},
        "acc_test_forget": {},
        "time": {},
        "commu_cost": {},
        "storage_cost": {},
    }

    # test data
    def get_all_idxs(index_map: Dict):
        idxs = []
        for task_id, task_info in index_map.items():
            # only test task learned
            if task_id > unlearn_config["after_task_id"]:
                break
            for client_id, client_task in task_info["shards"].items():
                idxs += client_task["idxs"]
        return idxs

    train_remain_dataset = DatasetSplit(train_dataset, get_all_idxs(train_remain_index_map))
    train_forget_dataset = DatasetSplit(train_dataset, get_all_idxs(train_forget_index_map))
    test_remain_dataset = DatasetSplit(test_dataset, get_all_idxs(test_remain_index_map))
    test_forget_dataset = DatasetSplit(test_dataset, get_all_idxs(test_forget_index_map))

    train_remain_dataloader = DataLoader(train_remain_dataset, batch_size=batch_size, num_workers=num_workers)
    train_forget_dataloader = DataLoader(train_forget_dataset, batch_size=batch_size, num_workers=num_workers)
    test_remain_dataloader = DataLoader(test_remain_dataset, batch_size=batch_size, num_workers=num_workers)
    test_forget_dataloader = DataLoader(test_forget_dataset, batch_size=batch_size, num_workers=num_workers)

    commu_cost_dict = get_commu_cost(args, server.args, unlearn_config, train_dataset, train_remain_index_map, train_forget_index_map)
    storage_cost_dict = get_storage_cost(args, server.args, unlearn_config, train_dataset, train_remain_index_map, train_forget_index_map)

    for name, v in all_model_dict.items():
        s_test = time.time()
        print(f"start test {name}")
        
        model = v["model"].to(device)

        # Time
        test_res["time"][name] = v["time"]

        # Accuracy on remain train set
        test_res["acc_train_remain"][name] = test_acc(train_remain_dataloader, model, device, name_to_index)

        # Accuracy on forget train set
        test_res["acc_train_forget"][name] = test_acc(train_forget_dataloader, model, device, name_to_index)

        # Accuracy on remain test set
        test_res["acc_test_remain"][name] = test_acc(test_remain_dataloader, model, device, name_to_index)

        # Accuracy on forget test set
        test_res["acc_test_forget"][name] = test_acc(test_forget_dataloader, model, device, name_to_index)

        # Accuracy on forget test set
        if name in commu_cost_dict:
            test_res["commu_cost"][name] = commu_cost_dict[name]

        # Accuracy on forget test set
        if name in storage_cost_dict:
            test_res["storage_cost"][name] = storage_cost_dict[name]

        print(f"finish test {name} Time[{time.time()-s_test:.2f}]s")

    return test_res

def test_acc(test_loader: DataLoader, model: nn.Module, device: str, name_to_index: Dict) -> Dict:
    # prepare model
    model.eval()
    # test
    with torch.no_grad():
        total_hits_taw, total_hits_tag, total_num = 0, 0, 0
        for batch, (images, labels) in enumerate(test_loader):
            # label name to local index
            for i in range(labels.shape[0]):
                labels[i] = name_to_index[labels[i].item()]
            # copy to device
            images, labels = images.to(device), labels.to(device)
            # Forward current model
            outputs = model(images, return_features=False)
            # metrics
            hits_taw, hits_tag = hits_taw_tag_metric(outputs, labels, model)
            # save result
            total_hits_taw += hits_taw.sum().item()
            total_hits_tag += hits_tag.sum().item()
            total_num += len(labels)
    # create log dict
    test_log_dict = {
        "hits_taw": total_hits_taw,
        "hits_tag": total_hits_tag,
        "count": total_num,
    }
    return test_log_dict

def hits_taw_tag_metric(outputs, labels, model, multi_softmax: bool=False):
    """
    Generally no modifications are required in the inherited class
    """
    pred = torch.zeros_like(labels)
    # Task-Aware Multi-Head
    for m in range(len(pred)):
        this_task = (model.task_cls.cumsum(0) <= labels[m].to('cpu')).sum()
        pred[m] = outputs[this_task][m].argmax() + model.task_offset[this_task]
    hits_taw = (pred == labels).float()
    # Task-Agnostic Multi-Head
    if multi_softmax:
        outputs = [F.log_softmax(output, dim=1) for output in outputs]
        pred = torch.cat(outputs, dim=1).argmax(1)
    else:
        pred = torch.cat(outputs, dim=1).argmax(1)
    hits_tag = (pred == labels).float()
    return hits_taw, hits_tag