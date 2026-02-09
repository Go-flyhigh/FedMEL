import random
from collections import defaultdict
import os
import pickle

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST


class TinyImageNet64(torch.utils.data.Dataset):
    """
    Tiny-ImageNet dataset (CIFAR-100 format, 64x64 images)
    Load data from converted pickle files
    """
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        
        self.data = None
        self.targets = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from pickle file"""
        if self.train:
            file_path = os.path.join(self.root, 'train')
        else:
            file_path = os.path.join(self.root, 'test')
        
        if not os.path.exists(file_path):
            raise RuntimeError(f"Data file not found: {file_path}\n"
                             f"Please run conversion script first")
        
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        self.data = data_dict[b'data']
        self.targets = data_dict[b'fine_labels']
        
        # Convert from (N, 12288) to (N, 64, 64, 3)
        self.data = self.data.reshape(-1, 3, 64, 64).transpose(0, 2, 3, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100, mnist, tiny-imagenet]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='cifar10':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        data_obj = CIFAR10
        img_size = 32
    elif data_name == 'cifar100':
        norm_mean = [0.5071, 0.4865, 0.4409]
        norm_std = [0.2673, 0.2564, 0.2762]
        data_obj = CIFAR100
        img_size = 32
    elif data_name == 'mnist':
        norm_mean = [0.5071]
        norm_std = [0.2673]
        data_obj = MNIST
        img_size = 28
    elif data_name == 'fashion-mnist':
        norm_mean = [0.5071]
        norm_std = [0.2673]
        data_obj = FashionMNIST
        img_size = 28
    elif data_name == 'tiny-imagenet':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        data_obj = TinyImageNet64
        img_size = 64
    else:
        raise ValueError("choose data_name from ['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny-imagenet']")

    # Set different data augmentation based on dataset
    if data_name == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    elif data_name in ['mnist', 'fashion-mnist']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomCrop(28, padding=4)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
            transforms.RandomCrop(32, padding=4)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    # Load dataset
    if data_name == 'tiny-imagenet':
        dataset = data_obj(
            dataroot,
            train=True,
            transform=transform_train
        )
        
        test_set = data_obj(
            dataroot,
            train=False,
            transform=transform_test
        )
    else:
        dataset = data_obj(
            dataroot,
            train=True,
            download=True,
            transform=transform_train
        )

        test_set = data_obj(
            dataroot,
            train=False,
            download=True,
            transform=transform_test
        )

    # Adjust val_size for tiny-imagenet
    if data_name == 'tiny-imagenet':
        val_size = min(val_size, len(dataset) // 10)
    
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    if isinstance(dataset, torch.utils.data.Subset):
        if hasattr(dataset.dataset, 'targets'):
            if isinstance(dataset.dataset.targets, list):
                data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
            else:
                data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = np.array([dataset.dataset[i][1] for i in dataset.indices])
    else:
        if hasattr(dataset, 'targets'):
            if isinstance(dataset.targets, list):
                data_labels_list = np.array(dataset.targets)
            else:
                data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = np.array([dataset[i][1] for i in range(len(dataset))])
    
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user, LowProb):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100, tiny-imagenet]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param bz: batch size
    :param classes_per_user: number of classes assigned to each client
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=True)
    
    for i, d in enumerate(datasets):
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user, 0.6, LowProb)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)

        if i == 0:
            train_sample_count = [len(idx) for idx in usr_subset_idx]
        elif i == 1:
            eval_sample_count = [len(idx) for idx in usr_subset_idx]
        else:
            test_sample_count = [len(idx) for idx in usr_subset_idx]
        
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders[0], dataloaders[1], dataloaders[2], train_sample_count, eval_sample_count, test_sample_count