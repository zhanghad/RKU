from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import random
import os
from typing import Dict, List
from PIL import Image

def get_TinyImageNet_as_public_dataset(train_data_dir='~/data/TinyImageNet/train', num=5000):
    # 3*64*64 100000
    trans_TinyImageNet_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.449, 0.398], std=[0.254, 0.245, 0.260]),
    ])
    dataset_TinyImageNet = datasets.ImageFolder(train_data_dir, trans_TinyImageNet_train)
    idxs = list(range(len(dataset_TinyImageNet.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_TinyImageNet, sampled_idxs)
    return sampled_train_dataset

def get_FashionMNIST_as_public_dataset(data_dir='~/data/', num=5000):
    # 1*28*28 60000
    trans_FashionMNIST_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([32 ,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286, 0.286, 0.286], std=[0.338, 0.338, 0.338]),
    ])
    all_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=trans_FashionMNIST_train,
        download=True
    )
    idxs = list(range(len(all_train_dataset.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(all_train_dataset, sampled_idxs)
    return sampled_train_dataset

def get_SVHN_as_public_dataset(data_dir='~/data/SVHN', num=5000):
    # 3*32*32 73257
    trans_SVHN_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.438, 0.444, 0.473], std=[0.198, 0.201, 0.197]),
    ])
    dataset_SVHN = datasets.SVHN(
        root=data_dir,
        split='train',
        transform=trans_SVHN_train,
        download=True
    )
    idxs = list(range(len(dataset_SVHN.labels)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_SVHN, sampled_idxs)
    return sampled_train_dataset

def get_sampled_CIFAR100(data_dir='~/data/CIFAR100', num=500):
    trans_cifar100_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
    dataset_CIFAR100 = datasets.CIFAR100(
        data_dir,
        train=True,
        download=True,
        transform=trans_cifar100_train
    )
    idxs = list(range(len(dataset_CIFAR100.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_CIFAR100, sampled_idxs)
    return sampled_train_dataset

def get_sampled_TT100K(data_dir='~/data/TT100K', num=500):
    trans_tt100k_train = transforms.Compose([
        transforms.Resize([32 ,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.437, 0.392, 0.437], std=[0.245, 0.222, 0.233]),
    ])
    train_dir = os.path.join(data_dir, 'train')
    dataset_TT100K = datasets.ImageFolder(train_dir, trans_tt100k_train)
    idxs = list(range(len(dataset_TT100K.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_TT100K, sampled_idxs)
    return sampled_train_dataset

def get_sampled_CTSDB(data_dir='~/data/CTSDB-TSRD', num=500):
    trans_CTSDB_TSRD_train = transforms.Compose([
        transforms.Resize([32 ,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.413, 0.376, 0.405], std=[0.243, 0.213, 0.234]),
    ])
    train_dir = os.path.join(data_dir, 'train')
    dataset_CTSDB = datasets.ImageFolder(train_dir, trans_CTSDB_TSRD_train)
    idxs = list(range(len(dataset_CTSDB.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_CTSDB, sampled_idxs)
    return sampled_train_dataset

def get_sampled_GTSRB(data_dir='~/data/GTSRB', num=500):
    trans_GTSRB_train = transforms.Compose([
        transforms.Resize([32 ,32]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.357, 0.317, 0.335], std=[0.267, 0.252, 0.257]),
    ])
    train_dir = os.path.join(data_dir, 'train')
    dataset_GTSRB = datasets.ImageFolder(train_dir, trans_GTSRB_train)
    idxs = list(range(len(dataset_GTSRB.targets)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_GTSRB, sampled_idxs)
    return sampled_train_dataset

class ExpandOneChannelToThree(object):
    def __init__(self) -> None:
        pass
    def __call__(self, img: Image.Image):
        if img.layers == 1:
            return transforms.Grayscale(num_output_channels=3)(img)
        else:
            return img

def get_Caltech101_as_public_dataset(data_dir='~/data/', num=5000):
    # 3*x*y 8677
    trans_Caltech101_train = transforms.Compose([
        ExpandOneChannelToThree(),
        transforms.CenterCrop([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.465, 0.441, 0.417], std=[0.347, 0.337, 0.341]),
    ])
    dataset_Caltech101 = datasets.Caltech101(
        root=data_dir,
        download=True,
        transform=trans_Caltech101_train,
    )
    idxs = list(range(len(dataset_Caltech101.y)))
    sampled_idxs = random.sample(idxs, num)
    sampled_train_dataset = DatasetSplit(dataset_Caltech101, sampled_idxs)
    return sampled_train_dataset

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset:Dataset, idxs:List[int]):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label


def get_dataset(args):

    # get original dataset and split it to train and test
    data_dir = os.path.join(args.data_dir, args.dataset)

    if args.dataset == 'CIFAR100':
        # 3*224*224
        trans_cifar100_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        trans_cifar100_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        train_dataset = datasets.CIFAR100(
            data_dir,
            train=True,
            download=True,
            transform=trans_cifar100_train
        )
        test_dataset = datasets.CIFAR100(
            data_dir,
            train=False,
            download=True,
            transform=trans_cifar100_test
        )

    elif args.dataset == 'TinyImageNet':
        # 3*64*64
        trans_TinyImageNet_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.449, 0.398], std=[0.254, 0.245, 0.260]),
        ])
        trans_TinyImageNet_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.449, 0.398], std=[0.254, 0.245, 0.260]),
        ])
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        train_dataset = datasets.ImageFolder(train_dir, trans_TinyImageNet_train)
        test_dataset = datasets.ImageFolder(test_dir, trans_TinyImageNet_test)

    elif args.dataset == 'TT100K':
        """
        https://cg.cs.tsinghua.edu.cn/traffic-sign/
        """
        # print("TT100K use debug size [28, 28]")
        trans_tt100k_train = transforms.Compose([
            # test
            # transforms.Resize([28 ,28]),
            transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.437, 0.392, 0.437], std=[0.245, 0.222, 0.233]),
        ])
        trans_tt100k_test = transforms.Compose([
            # test
            # transforms.Resize([28 ,28]),
            transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.437, 0.392, 0.437], std=[0.245, 0.222, 0.233]),
        ])
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        train_dataset = datasets.ImageFolder(train_dir, trans_tt100k_train)
        test_dataset = datasets.ImageFolder(test_dir, trans_tt100k_test)

    elif args.dataset == 'CTSDB-TSRD':
        """
        http://www.nlpr.ia.ac.cn/pal/trafficdata/index.html
        """
        # print("CTSDB-TSRD use debug size [28, 28]")
        trans_CTSDB_TSRD_train = transforms.Compose([
            # test
            # transforms.Resize([28 ,28]),
            transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.413, 0.376, 0.405], std=[0.243, 0.213, 0.234]),
        ])
        trans_CTSDB_TSRD_test = transforms.Compose([
            # test
            # transforms.Resize([28 ,28]),
            transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.413, 0.376, 0.405], std=[0.243, 0.213, 0.234]),
        ])
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        train_dataset = datasets.ImageFolder(train_dir, trans_CTSDB_TSRD_train)
        test_dataset = datasets.ImageFolder(test_dir, trans_CTSDB_TSRD_test)
        
    elif args.dataset == 'GTSRB':
        """
        https://benchmark.ini.rub.de/index.html
        """
        # print("CTSDB-TSRD use debug size [28, 28]")
        trans_GTSRB_train = transforms.Compose([
            # test
            # transforms.Resize([28 ,28]),
            transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.357, 0.317, 0.335], std=[0.267, 0.252, 0.257]),
        ])
        trans_GTSRB_test = transforms.Compose([
            # test
            transforms.Resize([28 ,28]),
            # transforms.Resize([32 ,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.357, 0.317, 0.335], std=[0.267, 0.252, 0.257]),
        ])
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val')
        train_dataset = datasets.ImageFolder(train_dir, trans_GTSRB_train)
        test_dataset = datasets.ImageFolder(test_dir, trans_GTSRB_test)

    return train_dataset, test_dataset

def get_dataset_and_info(args):

    train_dataset, test_dataset = get_dataset(args)
    # classes in each task
    # List[List[int]]
    num_tasks = args.num_tasks
    num_class = len(train_dataset.classes)
    classes_per_task = get_classes_per_task(num_class, num_tasks)

    # client selected in each task
    selected_client_per_task = []
    # classes in each client in each task
    classes_per_client_task = []
    for t in range(num_tasks):
        # random client num with range [2, 100%]
        min_client_num = max(int(args.num_clients*args.client_percent), 2)
        t_num_client = random.randint(min_client_num, args.num_clients)
        selected_client_per_task.append(sorted(random.sample(range(args.num_clients), t_num_client)))
        t_class_per_client = []
        if args.client_no_intersection:
            assert t_num_client < len(classes_per_task[t]), "classes num is low"
            t_class_per_client = [[] for _ in range(t_num_client)]
            i = 0
            for class_name in classes_per_task[t]:
                t_class_per_client[i].append(class_name)
                i = (i+1)%t_num_client
        else:
            for c in range(t_num_client):
                # random classes num with range [args.min_class, 100%] for each client
                min_classes_num = int(len(classes_per_task[t]) * args.min_class)
                max_classes_num = int(len(classes_per_task[t]) * args.max_class)
                t_num_class = random.randint(min_classes_num, max_classes_num)
                t_classes = random.sample(classes_per_task[t], t_num_class)
                t_class_per_client.append(sorted(t_classes))
        classes_per_client_task.append(t_class_per_client)

    # get train data index map
    train_index_map = get_train_index_map(train_dataset, num_tasks,
        classes_per_task, classes_per_client_task, selected_client_per_task)
    # get test data index map
    test_index_map = get_test_index_map(test_dataset, num_tasks,
        classes_per_task, classes_per_client_task, selected_client_per_task)

    # get name_to_index and index_to_name
    name_to_index, index_to_name = get_name_index_map(train_index_map)

    return train_dataset, train_index_map, test_dataset, test_index_map, name_to_index, index_to_name

def get_name_index_map(train_index_map: Dict):
    # get name_to_index and index_to_name
    name_to_index, index_to_name = {}, {}
    
    for task_id, task_info in train_index_map.items():
        for name in task_info["classes"]:
            if name not in name_to_index.keys():
                index = len(name_to_index)
                name_to_index[name] = index
                index_to_name[index] = name

    return name_to_index, index_to_name

def get_train_index_map(
        dataset: Dataset, 
        num_task: int,
        classes_per_task: List[List[int]],
        classes_per_client_task: List[List[List[int]]],
        selected_client_per_task: List[List[int]],
    ) -> Dict:
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
    index_map = {}

    idxs = np.arange(len(dataset.targets))
    labels = np.array(dataset.targets)
    # (index,label)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # (label,begin_index)
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt+=1
    # rear (max_class+1,begin)
    label_begin[max(idxs_labels[1,:])+1] = cnt

    for task_id in range(num_task):
        t_dict = {
            'classes': classes_per_task[task_id],
            'shards': {}
        }
        for c_i, classes in enumerate(classes_per_client_task[task_id]):
            t_shard = {
                'classes' : classes,
                'idxs' : []
            }
            for each_class in classes:
                len_block = (label_begin[each_class+1] - label_begin[each_class])//len(classes_per_client_task[task_id])
                s = label_begin[each_class] + (c_i*len_block)
                e = label_begin[each_class] + ((c_i+1)*len_block)
                t_shard['idxs'] += list(idxs_labels[0,:])[s:e]
            client_id = selected_client_per_task[task_id][c_i]
            t_dict['shards'][client_id] = t_shard
        index_map[task_id] = t_dict
    return index_map


def get_test_index_map(
        dataset: Dataset, 
        num_task: int,
        classes_per_task: List[List[int]],
        classes_per_client_task: List[List[List[int]]],
        selected_client_per_task: List[List[int]],
    ) -> Dict:
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
    index_map = {}

    idxs = np.arange(len(dataset.targets))
    labels = np.array(dataset.targets)
    # (index, label)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # (label, begin_index)
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt+=1
    # rear (max_class+1, cnt)
    label_begin[max(idxs_labels[1,:])+1] = cnt

    for task_id in range(num_task):
        t_dict = {
            'classes': classes_per_task[task_id],
            'shards': {}
        }
        for c_i, classes in enumerate(classes_per_client_task[task_id]):
            t_shard = {
                'classes' : classes,
                'idxs' : []
            }
            for each_class in classes:
                # use all test data for a class
                s = label_begin[each_class]
                e = label_begin[each_class+1]
                t_shard['idxs'] += list(idxs_labels[0,:])[s:e]
            client_id = selected_client_per_task[task_id][c_i]
            t_dict['shards'][client_id] = t_shard
        index_map[task_id] = t_dict
    return index_map


def get_classes_per_task(num_class: int, num_tasks: int):
    # make the number of classes for each task the same
    num_class_per_task = [num_class//num_tasks]*num_tasks
    for t in range(num_class - sum(num_class_per_task)):
        num_class_per_task[t] += 1
    # random select classes for each task
    classes_per_task = []
    t_class_list = list(range(num_class))
    for t in range(num_tasks):
        t_sample = random.sample(t_class_list, num_class_per_task[t])
        classes_per_task.append(sorted(t_sample))
        t_class_list = list(set(t_class_list)-set(t_sample))
    return classes_per_task


def get_stat_of_dataset(train_dataset: Dataset):
    # calculate mean first
    mean_list = [0, 0, 0]
    for X, _ in train_dataset:
        for d in range(3):
            mean_list[d] += X[d, :, :].mean().item()
    mean_list = [a / len(train_dataset) for a in mean_list]
    # then calculate std
    var_list = [0, 0, 0]
    for X, _ in train_dataset:
        for d in range(3):
            var_list[d] += ((X[d, :, :] - mean_list[d])**2).mean().item()
    var_list = [a / len(train_dataset) for a in var_list]
    std_list = [ a**0.5 for a in var_list]
    return mean_list, std_list


def calculate_means_stds_of_traindataset():
    train_dataset_cifar100 = datasets.CIFAR100(
        "~/data/CIFAR100",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    train_dataset_TT100K = datasets.ImageFolder(
        "~/data/TT100K/train", 
        transform=transforms.Compose([
            transforms.Resize([32 ,32]),
            transforms.ToTensor()
        ])
    )
    train_dataset_CTSDB = datasets.ImageFolder(
        "~/data/CTSDB-TSRD/train", 
        transform=transforms.Compose([
            transforms.Resize([32 ,32]),
            transforms.ToTensor()
        ])
    )
    train_dataset_GTSRB = datasets.ImageFolder(
        "~/data/GTSRB/train", 
        transform=transforms.Compose([
            transforms.Resize([32 ,32]),
            transforms.ToTensor()
        ])
    )

    dataset_dict = {
        "CIFAR100": train_dataset_cifar100,
        "TT100K": train_dataset_TT100K,
        "CTSDB-TSRD": train_dataset_CTSDB,
        "GTSRB": train_dataset_GTSRB
    }

    for name, dataset in dataset_dict.items():
        means, stds, = get_stat_of_dataset(dataset)
        print(f"{name} means: {means} stds: {stds}")


if __name__ == '__main__':
    # test
    # args = Namespace(
    #     data_dir = 'data',
    #     dataset = 'cifar100',
    #     num_tasks = 10,
    # )
    # seed_everything(seed=0)
    # train_dataset, train_index_map, test_dataset, test_index_map = get_dataset(args)

    dataset = get_TinyImageNet_as_public_dataset()

    dataset = get_FashionMNIST_as_public_dataset()

    dataset = get_SVHN_as_public_dataset()

    dataset = get_Caltech101_as_public_dataset()

    # img = transforms.ToPILImage()(dataset[0][0]) # Alternatively
    # img.save("test.jpg")

    # dataset = datasets.Omniglot(
    #     root="~/data", download=True, transform=transforms.ToTensor()
    # )

    # dataset_SVHN = datasets.SVHN(
    #     root="~/data/SVHN",
    #     split='train',
    #     transform=transforms.ToTensor(),
    #     download=True
    # )

    # dataset_Caltech101 = datasets.Caltech101(
    #     root='~/data',
    #     download=True,
    #     transform=transforms.ToTensor()
    # )

    # dataset_ImageNet = datasets.ImageFolder(
    #     root='~/data/TinyImageNet/train', 
    #     transform=transforms.ToTensor()
    # )

    # dataset_FashionMNIST = datasets.FashionMNIST(
    #     root="~/data/",
    #     train=True,
    #     transform=transforms.ToTensor(),
    #     download=True
    # )

    pass