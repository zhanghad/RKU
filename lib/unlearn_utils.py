import time
import random
from copy import deepcopy
from typing import Dict, List, Tuple
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from nni.compression.pruning import L1NormPruner

from lib.utils import weights_average

def FedEraser_unlearn_step(old_global_weight: Dict, old_client_updates: List[Dict], 
                            new_global_weight: Dict, new_client_updates: List[Dict], device: str):

    old_CM = weights_average(old_client_updates)
    new_CM = weights_average(new_client_updates)

    for layer_name in new_global_weight.keys():

        old_CM_sub_old_GM = old_CM[layer_name].to(device) - old_global_weight[layer_name].to(device)
        new_CM_sub_new_GM = new_CM[layer_name].to(device) - new_global_weight[layer_name].to(device)

        step_length = torch.norm(old_CM_sub_old_GM.float())
        # print(f"step_length {step_length.item()} torch.norm(new_CM_sub_new_GM) {torch.norm(new_CM_sub_new_GM).item()}")
        step_direction = new_CM_sub_new_GM / torch.norm(new_CM_sub_new_GM.float())

        new_global_weight[layer_name] = new_global_weight[layer_name].to(device) + step_length*step_direction

    return new_global_weight


def Ours_loss_fn(
        features: Tensor, preds: Tensor,
        unlearn_info: Dict[int, List],
        client_proto_dict: Dict, client_proto_dist_dict: Dict,  client_neighbor_dist_dict: Dict,
        index_to_name: Dict,
    ):
    # features [B,C] labels [B]
    # prototypes [N,C] 
    # features-prototypes [B,N] 欧式距离
    # weights [B,N]
    # dists*weights [B,N] -> loss

    A = features.shape[0]

    # [{c_id:{class_name: weight}}]
    weights = [{} for i in range(A)]

    for i in range(A):
        for c_id in unlearn_info.keys():
            for class_name in unlearn_info[c_id]['classes']:
                pred = index_to_name[preds[i].item()]
                if pred==class_name:
                    if c_id not in weights[i]:
                        weights[i][c_id] = {}
                    
                    # unlearned knowledge
                    weights[i][c_id][class_name] = -1

                    # relevent konwledge
                    for c_id_cls_name in client_neighbor_dist_dict[(c_id, class_name)]:
                        if c_id_cls_name[0] not in weights[i]:
                             weights[i][c_id_cls_name[0]] = {}
                        weights[i][c_id_cls_name[0]][c_id_cls_name[1]] = 1

    loss = torch.tensor(0.0, requires_grad=True)

    for i in range(A):
        for c_id in weights[i].keys():
            for class_name, w in weights[i][c_id].items():
                tmp_proto = client_proto_dict[c_id][class_name]['mean']
                loss = loss + w*F.mse_loss(features[i], tmp_proto)

    loss = loss / A
    return loss

def calculate_client_prototype(client_dict: Dict, model: nn.Module, data_set: Dataset, args) -> Dict[str, Dict[str, Tensor]]:
    client_proto_dict = {int(i):{} for i in client_dict.keys()}
    """
    client_id: {
        class_id: {
            "mean": tensor[N]
            "std": float
        }
    }
    """
    model.eval()
    model = model.to(args.device)

    with torch.no_grad():
        for c_id, client in client_dict.items():
            c_id = int(c_id)
            
            for class_name, idxs in client.examplar_index_map.items():
                client_proto_dict[c_id][class_name] = {}

                images = torch.stack([data_set[i][0] for i in idxs])
                images = images.to(client.device)
                _, features = model(images, return_features=True)
                feature_mean = torch.mean(features, dim=0).detach()
                client_proto_dict[c_id][class_name]["mean"]=feature_mean

                dist_list = torch.zeros((features.shape[0]))
                for i in range(features.shape[0]):
                    dist_list[i] = F.pairwise_distance(feature_mean, features[i], p=2)
                client_proto_dict[c_id][class_name]["std"]=torch.std(dist_list).item()

    return client_proto_dict

def calculate_client_proto_dist(client_proto_dict: Dict[str, Dict[str, Tensor]]) -> \
    Tuple[Dict, Dict]:
    """
    dist_dict: Dict[(c_id,class_name,c_id,class_name): dist]
    neighbor_dict: Dict[(c_id,class_name): List: [(c_id,class_name)]]
    """
    dist_dict = {}
    neighbor_dict = {}

    c_id_cls_name_list = []
    for c_id in client_proto_dict.keys():
        for c_name in client_proto_dict[c_id].keys():
            c_id_cls_name_list.append((c_id,c_name))
    c_id_cls_name_list.sort()

    num = len(c_id_cls_name_list)

    for i in range(num):
        for j in range(i, num):
            c_id_1, cls_name_1 = c_id_cls_name_list[i][0], c_id_cls_name_list[i][1]
            c_id_2, cls_name_2 = c_id_cls_name_list[j][0], c_id_cls_name_list[j][1]
            dist = F.pairwise_distance(client_proto_dict[c_id_1][cls_name_1]["mean"], client_proto_dict[c_id_2][cls_name_2]["mean"], p=2)
            dist_dict[(c_id_1, cls_name_1, c_id_2, cls_name_2)] = dist.item()
            dist_dict[(c_id_2, cls_name_2, c_id_1, cls_name_1)] = dist.item()

    for i in range(num):
        c_id_1, cls_name_1 = c_id_cls_name_list[i][0], c_id_cls_name_list[i][1]
        tmp_list = []
        std = client_proto_dict[c_id_1][cls_name_1]["std"]
        for j in range(num):
            if i==j:
                continue
            c_id_2, cls_name_2 = c_id_cls_name_list[j][0], c_id_cls_name_list[j][1]
            if dist_dict[(c_id_1, cls_name_1, c_id_2, cls_name_2)] <= std:
                tmp_list.append([c_id_2, cls_name_2])

        neighbor_dict[(c_id_1, cls_name_1)] = tmp_list

    return dist_dict, neighbor_dict

def CDP_acculumate_feature(model: nn.Module, loader: DataLoader, stop: int, device: str):
    features = {}
    targets = []
    
    def hook_func(m, x, y, name, feature_iit):
        '''ReLU'''
        f = F.relu(y)    
        '''Average Pool'''
        feature = F.avg_pool2d(f, f.size()[3])
        feature = feature.view(f.size()[0], -1)
        feature = feature.transpose(0, 1)
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)

    hook = functools.partial(hook_func, feature_iit=features)

    handler_list=[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)
    
    all_targets = []

    # prepare model
    model = model.to(device)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        all_targets += targets.tolist()
        with torch.no_grad():
            inputs = inputs.to(device)
            model(inputs)

    [k.remove() for k in handler_list]
    '''Image-wise Activation'''
    return features, all_targets

def CDP_calculate_cp(features: Dict, targets: List, class_num: int, name_to_index: Dict, coe: int, unlearn_class: int):
    tf_idf_map = {}
    class_idxs_dict = {}
    features_class_wise = {}

    for z in range(class_num):
        same_class_idxs = [i for i in range(len(targets)) if name_to_index[targets[i]] == z]
        class_idxs_dict[z] = same_class_idxs

    for fea_name in features:
        '''Class-wise Activation'''
        class_wise_features = torch.zeros(class_num, features[fea_name].shape[0])
        image_wise_features = features[fea_name].transpose(0, 1)
        for i, v in class_idxs_dict.items():
            for j in v:
                class_wise_features[i] += image_wise_features[j]
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)
        features_class_wise[fea_name] = class_wise_features.transpose(0, 1)
        '''TF-IDF'''
        CDP_calc_tf_idf(features_class_wise[fea_name], fea_name, coe, unlearn_class, tf_idf_map)
    return tf_idf_map

def CDP_calc_tf_idf(feature: Tensor, name: str, coe: float, unlearn_class: int, tf_idf_map: Dict):
    # c - filters; n - classes
    # feature = [c, n] ([64, 10])
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0)
    balance_coe = np.log((feature.shape[0]/coe)*np.e) if coe else 1.0
    tf = (feature / sum_on_filters) * balance_coe
    tf_unlearn_class = tf.transpose(0,1)[unlearn_class]
    
    # calc idf for filters
    classes_quant = float(feature.shape[1])
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1)
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor)
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0))
    
    importance = tf_unlearn_class * idf
    tf_idf_map[name] = importance

def CDP_get_threshold_by_sparsity(mapper:dict, sparsity:float):
    assert 0 < sparsity < 1
    tf_idf_array=torch.cat([v for v in mapper.values()], 0)
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-sparsity)))[0].min()
    return threshold

class CDP_TFIDFPruner(L1NormPruner):

    def __init__(self, model, config_list, cp_config: Dict):
        super().__init__(model, config_list)
        self.tf_idf_map = cp_config['map']
        self.threshold = cp_config['threshold']

    def get_tf_idf_masks(self, prefix='feature_extractor.'):

        curr_mask = self.get_masks()    # Dict[str, Dict[str, torch.Tensor]]
        tf_idf_masks = deepcopy(curr_mask)

        exclude_names = sorted(list(tf_idf_masks.keys()))[-1:]
        for module_name in exclude_names:
            tf_idf_masks.pop(module_name)

        for module_name, mask_dict in tf_idf_masks.items():
            
            mask_weight = torch.gt(self.tf_idf_map[prefix+module_name], self.threshold)[:, None, None, None]
            mask_weight = mask_weight.expand_as(mask_dict['weight']).type_as(mask_dict['weight'])
            tf_idf_masks[module_name]['weight'] = mask_weight

            if mask_dict['bias'] is None:
                mask_bias = None
            else:
                mask_bias = torch.gt(self.tf_idf_map[prefix+module_name], self.threshold)
                mask_bias = mask_bias.expand_as(mask_dict['bias']).type_as(mask_dict['bias'])

                tf_idf_masks[module_name]['bias'] = mask_bias

        return tf_idf_masks


class AdaHessian(torch.optim.Optimizer):
    """
    For Rapid Retraining
   Adaptive Hessian-free Method for Federated Learning - Code

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.01)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        grads = [p.grad for p in params]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss