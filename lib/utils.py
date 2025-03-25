import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import itertools
from torch import nn
from typing import List, Dict
from torch import Tensor
from torch.utils.data import DataLoader
from copy import deepcopy
cudnn_deterministic = True
cudnn_benchmark = False

TORCH_INT_TYPE_SET = set([torch.int, torch.int16, torch.int32, torch.int64])

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark

def cal_fedprox_proximal_term(model: nn.Module, global_model: nn.Module):
    proximal_term = 0.0
    for w, w_t in zip(model.parameters(), global_model.parameters()):
        proximal_term += (w - w_t).norm(2)
    return proximal_term

def weights_device_convert(weight: Dict[str,Tensor], device: str) -> Dict[str,Tensor]:
    for k, v in weight.items():
        if isinstance(v, torch.Tensor):
            weight[k] = v.to(device)
    return weight

def weights_average(weight_list: List[Dict[str,Tensor]]) -> Dict[str,Tensor]:
    weight_avg = deepcopy(weight_list[0])
    for k in weight_avg.keys():
        for i in range(1, len(weight_list)):
            weight_avg[k] += weight_list[i][k]
        if weight_avg[k].dtype in TORCH_INT_TYPE_SET:
            weight_avg[k] = torch.div(weight_avg[k], len(weight_list), rounding_mode="floor")
        else:
            weight_avg[k] = torch.div(weight_avg[k], len(weight_list))
    return weight_avg

def weight_distance(weight_a: List[Dict[str,Tensor]], weight_b: List[Dict[str,Tensor]]):
    count = 0
    distance = 0.0
    for k in weight_a.keys():
        distance += torch.square(torch.norm(weight_a[k].float()-weight_b[k].float()))
        count += weight_a[k].numel()
    return distance / count

def weights_add(weight_a: List[Dict[str,Tensor]], weight_b: List[Dict[str,Tensor]]) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        weight_res[k] += weight_b[k]
    return weight_res

def weights_sub(weight_a: List[Dict[str,Tensor]], weight_b: List[Dict[str,Tensor]]) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        if k in weight_b.keys():
            weight_res[k] -= weight_b[k]
    return weight_res

def weights_multi(weight_a: List[Dict[str,Tensor]], weight_b: List[Dict[str,Tensor]]) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        weight_res[k] *= weight_b[k]
    return weight_res

def weights_div(weight_a: List[Dict[str,Tensor]], num: float) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        if weight_res[k].dtype in TORCH_INT_TYPE_SET:
            weight_res[k] = torch.div(weight_res[k], num, rounding_mode="floor")
        else:
            weight_res[k] = torch.div(weight_res[k], num)
    return weight_res

def weights_norm(weight_a: Dict[str,Tensor]) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        weight_res[k] = torch.norm(weight_res[k])
    return weight_res

def weights_div_norm(weight_a: Dict[str,Tensor]) -> Dict[str,Tensor]:
    weight_res = deepcopy(weight_a)
    for k in weight_res.keys():
        if weight_res[k].dtype in TORCH_INT_TYPE_SET:
            weight_res[k] = torch.div(weight_res[k], torch.norm(weight_res[k]), rounding_mode="floor")
        else:
            weight_res[k] = torch.div(weight_res[k], torch.norm(weight_res[k]))
    return weight_res

def tensor_average(tensor_list: List[Tensor]):
    tensor_avg = deepcopy(tensor_list[0])
    for i in range(1, len(tensor_list)):
        tensor_avg += tensor_list[i]
    if tensor_avg.dtype in TORCH_INT_TYPE_SET:
        tensor_avg = torch.div(tensor_avg, len(tensor_list), rounding_mode="floor")
    else:
        tensor_avg = torch.div(tensor_avg, len(tensor_list))
    return tensor_avg

def prototypes_average(protos: List[Dict[int,Tensor]]) -> Dict[int,Tensor]:
    result_protos = {}
    # calculate classes
    all_classes = []
    for d in protos:
        all_classes += list(d.keys())
    classes = list(set(all_classes))
    # averaging
    for c in classes:
        t_protos = []
        for d in protos:
            if c in d.keys():
                t_protos.append(d[c])
        result_protos[c] = torch.mean(torch.stack(t_protos, dim=0), dim=0)
    return result_protos

def cross_entropy_with_temp_mask(outputs, targets, mask=None, exp=1.0, size_average=True, eps=1e-5):
    """
    Calculates cross-entropy with temperature scaling
    """
    if mask:
        for i in range(len(mask)):
            if mask[i] == -1:
                targets[i] = torch.rand_like(targets[i])

    out = F.softmax(outputs, dim=1)
    tar = F.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)

    if size_average:
        ce = ce.mean()
    return ce

def cross_entropy_with_temp(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    """
    Calculates cross-entropy with temperature scaling
    """
    out = F.softmax(outputs, dim=1)
    tar = F.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


def loss_EWC(model_1: nn.Module, model_2: nn.Module, fisher_mat: Dict):
    # fisher = {n: torch.zeros(p.shape).to(device) for n, p in model_1.named_parameters() if p.requires_grad}
    loss_ewc = 0
    for (n, p), (_, p_old) in zip(model_1.named_parameters(), model_2.named_parameters()):
        if n in fisher_mat.keys():
            loss_ewc += torch.sum(fisher_mat[n] * (p - p_old).pow(2)) / 2
    return loss_ewc


def compute_fisher_matrix_diag(model: nn.Module, train_loader: DataLoader, name_to_index: Dict, optimizer, args):
    # Store Fisher Information
    fisher = {n: torch.zeros(p.shape).to(args.device) for n, p in model.named_parameters() if p.requires_grad}
    # Compute fisher information for specified number of samples -- rounded to the batch size
    n_samples_batches = (args.ewcsga_fisher_num_samples // train_loader.batch_size + 1) if args.ewcsga_fisher_num_samples > 0 \
        else (len(train_loader.dataset) // train_loader.batch_size)
    # Do forward and backward pass to compute the fisher information
    model.train()
    for images, labels in itertools.islice(train_loader, n_samples_batches):
        # label name to local index
        for i in range(labels.shape[0]):
            labels[i] = name_to_index[labels[i].item()]

        outputs = model.forward(images.to(args.device), return_features=False)

        if args.ewcsga_sampling_type == 'true':
            # Use the labels to compute the gradients based on the CE-loss with the ground truth
            preds = labels.to(args.device)
        elif args.ewcsga_sampling_type == 'max_pred':
            # Not use labels and compute the gradients related to the prediction the model has learned
            preds = torch.cat(outputs, dim=1).argmax(1).flatten()
        elif args.ewcsga_sampling_type == 'multinomial':
            # Use a multinomial sampling to compute the gradients
            probs = F.softmax(torch.cat(outputs, dim=1), dim=1)
            preds = torch.multinomial(probs, len(labels)).flatten()

        loss = F.cross_entropy(torch.cat(outputs, dim=1), preds)
        optimizer.zero_grad()
        loss.backward()
        # Accumulate all gradients from loss with regularization
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(labels)
    # Apply mean across all samples
    n_samples = n_samples_batches * train_loader.batch_size
    fisher = {n: (p / n_samples) for n, p in fisher.items()}
    return fisher