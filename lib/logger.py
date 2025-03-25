import os
import sys
import json
import pickle
import torch
import numpy as np
from datetime import datetime

from typing import Dict, List
from collections import OrderedDict

class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()

class ExpLogger:
    def __init__(self, results_dir: str, exp_name: str):
        self.begin_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_log_dir = os.path.join(results_dir, exp_name, self.begin_time_str)
        if not os.path.exists(self.exp_log_dir):
            os.makedirs(self.exp_log_dir)
        # log files path
        self.stdout_file_path = os.path.join(self.exp_log_dir, f'stdout-{self.begin_time_str}.txt')
        self.stderr_file_path = os.path.join(self.exp_log_dir, f'stderr-{self.begin_time_str}.txt')
        self.test_result_file_path = os.path.join(self.exp_log_dir, f"test_result-{self.begin_time_str}.txt")
        # create log file
        sys.stdout = FileOutputDuplicator(sys.stdout, self.stdout_file_path, 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr, self.stderr_file_path, 'w')
        self.test_result_file = open(self.test_result_file_path, 'a')

    def log_test_result(self, test_result: Dict):
        test_result["time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.test_result_file.write(json.dumps(test_result, sort_keys=True) + "\n")
        self.test_result_file.flush()

    def log_args(self, args: Dict):
        with open(os.path.join(self.exp_log_dir, 'args-{}.txt'.format(self.begin_time_str)), 'w') as f:
            json.dump(args, f, separators=(',\n', ' : '), sort_keys=True)

def summarize_test_results(test_log_path: str, last_task_id: int, log_name_postfix: str="") -> Dict:
    # read test log
    test_results_log_list = []
    with open(test_log_path) as log_file:
        for line in log_file.readlines():
            test_results_log_list.append(json.loads(line))
    # summarize
    sum_dict_list = []
    test_log_name_set = set(["Test", "TestRetrain"])
    task_id_list = sorted(list(set([a["global_task_id"] for a in test_results_log_list if a["log_name"] in test_log_name_set])))
    for task_id in task_id_list:
        task_log_list = [ a for a in test_results_log_list if a["log_name"] in test_log_name_set and a["global_task_id"]==task_id]
        sum_hits_taw = sum([a["total_hits_taw"] for a in task_log_list])
        sum_hits_tag = sum([a["total_hits_tag"] for a in task_log_list])
        sum_num = sum([a["total_num"] for a in task_log_list])
        if sum_num==0:
            sum_dict_list.append({
                "global_task_id": task_id,
                "acc_taw": 0,
                "acc_tag": 0,
            })
        else:
            sum_dict_list.append({
                "global_task_id": task_id,
                "acc_taw": sum_hits_taw/sum_num,
                "acc_tag": sum_hits_tag/sum_num
            })
    log_dict = {
        "log_name": f"TestSummary{log_name_postfix}",
        "last_task_id": last_task_id,
        "detail": sum_dict_list
    }
    return log_dict


def compute_forgetting(test_log_path: str):
    # read test log
    test_results_log_list = []
    with open(test_log_path) as log_file:
        for line in log_file.readlines():
            test_results_log_list.append(json.loads(line))
    task_id_list = sorted(list(set([a["global_task_id"] for a in test_results_log_list if a["log_name"]=="Test"])))

    # store every task forgetting
    forgetting_list = []
    for test_task_id in task_id_list[:-1]:
        # for each global task id
        test_task_acc_list = []
        for task_id in task_id_list[:-1]:
            task_log_list = [ a for a in test_results_log_list if a["log_name"]=="Test" and a["global_task_id"]==task_id and a["test_task_id"]==test_task_id]
            sum_hits_tag = sum([a["total_hits_tag"] for a in task_log_list])
            sum_num = sum([a["total_num"] for a in task_log_list])
            if sum_num != 0:
                test_task_acc_list.append(sum_hits_tag/sum_num)
        # last global task test task acc
        last_task_log_list = [ a for a in test_results_log_list if a["log_name"]=="Test" and a["global_task_id"]==task_id_list[-1] and a["test_task_id"]==test_task_id]
        last_sum_hits_tag = sum([a["total_hits_tag"] for a in last_task_log_list])
        last_sum_num = sum([a["total_num"] for a in last_task_log_list])
        last_acc = last_sum_hits_tag/last_sum_num

        t_forgetting_list = [a-last_acc for a in test_task_acc_list]

        forgetting_list.append(max(t_forgetting_list))
    
    res_dict = {
        "forgetting_list": forgetting_list,
        "forgetting_avg": np.mean(forgetting_list),
    }
    return res_dict

def round_level_acc_for_each_task(test_log_path: str) -> Dict:
    # read test log
    test_results_log_list = []
    with open(test_log_path) as log_file:
        for line in log_file.readlines():
            test_results_log_list.append(json.loads(line))
    task_id_list = sorted(list(set([a["global_task_id"] for a in test_results_log_list if a["log_name"]=="TestPerRound"])))

    res_dict = {}
    for test_task_id in task_id_list:
        
        # for each global task id
        acc_tag_list = []
        task_round_list = []
        for task_id in task_id_list:

            round_list = sorted(list(set([a["round"] for a in test_results_log_list if a["log_name"]=="TestPerRound" and a["global_task_id"]==task_id])))

            for round in round_list:

                task_round_log_list = [ a for a in test_results_log_list if a["log_name"]=="TestPerRound" and a["global_task_id"]==task_id and a["round"]==round and a["test_task_id"]==test_task_id]
                # sum_hits_taw = sum([a["total_hits_taw"] for a in task_round_log_list])
                sum_hits_tag = sum([a["total_hits_tag"] for a in task_round_log_list])
                sum_num = sum([a["total_num"] for a in task_round_log_list])

                if sum_num != 0:
                    acc_tag_list.append(sum_hits_tag/sum_num)
                    task_round_list.append((task_id, round))

        res_dict[test_task_id] = {
            "test_task_id": test_task_id,
            "task_round_list": task_round_list,
            "acc_tag_list": acc_tag_list,
        }

    return res_dict


def summarize_eval_on_train_results(test_log_path: str, last_task_id: int) -> Dict:
    # read test log
    test_results_log_list = []
    with open(test_log_path) as log_file:
        for line in log_file.readlines():
            test_results_log_list.append(json.loads(line))
    # summarize
    sum_dict_list = []
    task_id_list = sorted(list(set([a["global_task_id"] for a in test_results_log_list if a["log_name"]=="TestPerRound"])))
    for task_id in task_id_list:
        round_list = sorted(list(set([a["round"] for a in test_results_log_list if a["log_name"]=="TestPerRound" and a["global_task_id"]==task_id])))

        for round in round_list:
            task_log_list = [ a for a in test_results_log_list if a["log_name"]=="TestPerRound" and a["global_task_id"]==task_id and a["round"]==round]

            sum_hits_taw = sum([a["total_hits_taw"] for a in task_log_list])
            sum_hits_tag = sum([a["total_hits_tag"] for a in task_log_list])
            sum_num = sum([a["total_num"] for a in task_log_list])
            sum_dict_list.append({
                "task_round": [task_id, round],
                "acc_taw": sum_hits_taw/sum_num,
                "acc_tag": sum_hits_tag/sum_num
            })
    log_dict = {
        "log_name": "TestSummary",
        "last_task_id": last_task_id,
        "detail": sum_dict_list
    }
    return log_dict

def print_test_result(test_log_dict: Dict):

    def list_to_str(data: List, t: str) -> str:
        res_str = ""
        for d in data:
            if t == "float":
                res_str += f"{(d):>6.1f}"
            elif t == "int":
                res_str += f"{(d):>6}"
        return res_str

    acc_taw_list = [0]*len(test_log_dict["detail"])
    acc_tag_list = [0]*len(test_log_dict["detail"])
    for log in test_log_dict["detail"]:
        acc_taw_list[log["global_task_id"]] = log["acc_taw"] * 100
        acc_tag_list[log["global_task_id"]] = log["acc_tag"] * 100
    print("Test Result:")
    print(f"task   : {list_to_str(range(10), 'int')}")
    print(f"acc_taw: {list_to_str(acc_taw_list, 'float')}")
    print(f"acc_tag: {list_to_str(acc_tag_list, 'float')}")

def get_unlearn_res_dict(res_path: str) -> Dict:
    # read test log
    test_results_log_list = []
    with open(res_path) as log_file:
        for line in log_file.readlines():
            test_results_log_list.append(json.loads(line))

    unlearn_res_dict = None
    for log_dict in test_results_log_list:
        if log_dict["log_name"] == "TestUnlearn":
            unlearn_res_dict = log_dict

    return unlearn_res_dict['detail']

def recursive_dump_dict_to_pkl(data_dict: Dict, save_dir: str):

    for name, data in data_dict.items():
        if isinstance(data, OrderedDict):
            tmp_save_path = os.path.join(save_dir, f"{name}.pkl")
            with open(tmp_save_path, 'wb') as f:
                pickle.dump(data, f)
        elif isinstance(data, Dict):
            sub_dir = os.path.join(save_dir, f"{name}")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            recursive_dump_dict_to_pkl(data, sub_dir)
        else:
            tmp_save_path = os.path.join(save_dir, f"{name}.pkl")
            with open(tmp_save_path, 'wb') as f:
                pickle.dump(data, f)

def recursive_load_pkl_to_dict(save_dir: str, result_dict: Dict):

    file_list = os.listdir(save_dir)
    for file_name in file_list:
        tmp_path = os.path.join(save_dir, file_name)
        if file_name.endswith('.pkl'):
            with open(tmp_path, 'rb') as f:
                result_dict[file_name[:-4]] = pickle.load(f)
        elif os.path.isdir(tmp_path):
            result_dict[file_name] = {}
            recursive_load_pkl_to_dict(tmp_path, result_dict[file_name])
        else:
            print(f'Unknow path: {tmp_path}')

