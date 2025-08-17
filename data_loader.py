import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os

# ============================================================================
# Base Dataset Split Functions
# ============================================================================

class DatasetSplit(Dataset):
    """Dataset wrapper for federated learning splits"""
    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else list(range(len(dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def create_class_dict(dataset):
    """Create dictionary mapping class labels to sample indices"""
    class_dict = {}
    for i, label in enumerate(dataset.targets):
        label = label.item() if hasattr(label, 'item') else label
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(i)
    return class_dict


def iid_split(dataset, num_users):
    """Create IID data split across users"""
    np.random.seed(2021)
    num_items = len(dataset) // num_users
    all_indices = list(range(len(dataset)))
    
    user_groups = {}
    for i in range(num_users):
        selected = set(np.random.choice(all_indices, num_items, replace=False))
        all_indices = list(set(all_indices) - selected)
        user_groups[i] = list(selected)
    
    return user_groups


def noniid_split(dataset, num_users, shards_per_user=3):
    """Create non-IID data split with specified shards per user"""
    np.random.seed(2022)
    class_dict = create_class_dict(dataset)
    num_classes = len(class_dict)
    imgs_per_shard = len(dataset) // (num_users * shards_per_user)
    
    # Assign random classes to each user
    user_groups = {i: np.array([], dtype='int64') for i in range(num_users)}
    class_assignments = [np.random.choice(num_classes, shards_per_user, replace=False) 
                        for _ in range(num_users)]
    
    for i in range(num_users):
        user_shards = []
        for class_label in class_assignments[i]:
            selected = np.random.choice(class_dict[class_label], imgs_per_shard, replace=False)
            user_shards.append(selected)
        user_groups[i] = np.concatenate(user_shards)
    
    return user_groups


def noniid_s_split(dataset, num_users, noniid_s=20, local_size=600, predefined_labels=None):
    """Create mixed IID/non-IID split with specified non-IID percentage"""
    np.random.seed(2022)
    s = noniid_s / 100
    num_classes = len(np.unique(dataset.targets))
    
    # Default label groupings for different datasets
    if predefined_labels is None:
        if num_classes == 10:
            predefined_labels = [[0,1,2], [2,3,4], [4,5,6], [6,7,8], [8,9,0]]
        elif num_classes == 100:
            predefined_labels = [list(range(i, min(i+10, num_classes))) for i in range(0, num_classes, 10)]
        else:
            predefined_labels = [list(range(num_classes))] * num_users
    
    # Sort indices by labels
    labels = np.array(dataset.targets)
    sorted_indices = np.argsort(labels)
    
    user_groups = {}
    samples_per_class = len(dataset) // num_classes
    class_used = [0] * num_classes
    
    # Calculate splits
    iid_samples = int(local_size * s)
    noniid_samples = local_size - iid_samples
    iid_per_class = iid_samples // num_classes
    
    for i in range(num_users):
        user_indices = []
        
        # Add IID samples from each class
        for class_id in range(num_classes):
            start_idx = class_id * samples_per_class + class_used[class_id]
            if class_used[class_id] + iid_per_class > samples_per_class:
                start_idx = class_id * samples_per_class
                class_used[class_id] = 0
            
            user_indices.extend(sorted_indices[start_idx:start_idx + iid_per_class])
            class_used[class_id] += iid_per_class
        
        # Add non-IID samples from specific classes
        noniid_classes = predefined_labels[i % len(predefined_labels)]
        noniid_per_class = noniid_samples // len(noniid_classes)
        
        for class_id in noniid_classes:
            start_idx = class_id * samples_per_class + class_used[class_id]
            if class_used[class_id] + noniid_per_class > samples_per_class:
                start_idx = class_id * samples_per_class
                class_used[class_id] = 0
            
            user_indices.extend(sorted_indices[start_idx:start_idx + noniid_per_class])
            class_used[class_id] += noniid_per_class
        
        user_groups[i] = np.array(user_indices[:local_size], dtype=int)
    
    return user_groups


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_cifar10():
    """Load CIFAR-10 with standard augmentations"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    
    return trainset, testset


def load_cifar100():
    """Load CIFAR-100 with standard augmentations"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    trainset = datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
    
    return trainset, testset


def load_mnist():
    """Load MNIST with standard normalization"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, transform=transform)
    
    return trainset, testset


def load_fashion_mnist():
    """Load Fashion-MNIST with standard normalization"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST('data', train=False, transform=transform)
    
    return trainset, testset


def load_emnist():
    """Load EMNIST with standard normalization"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.EMNIST('data', 'byclass', train=True, download=True, transform=transform)
    testset = datasets.EMNIST('data', 'byclass', train=False, transform=transform)
    
    return trainset, testset


def load_animal10():
    """Load Animal-10 dataset from local directory"""
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    trainset = datasets.ImageFolder(root='data/animal10/train', transform=transform_train)
    testset = datasets.ImageFolder(root='data/animal10/test', transform=transform_test)
    
    return trainset, testset


# ============================================================================
# Advanced Split Functions
# ============================================================================

def add_duplicates(user_groups, num_repeat_clients, repeat_ratio):
    """Add duplicate samples to specified clients"""
    for i in range(min(num_repeat_clients, len(user_groups))):
        user_data = user_groups[i]
        num_repeat = int(len(user_data) * repeat_ratio)
        
        if num_repeat > 0:
            # Select random sample to duplicate
            repeat_sample = np.random.choice(user_data, 1)[0]
            # Select random positions to replace
            repeat_positions = np.random.choice(len(user_data), num_repeat, replace=False)
            # Replace with duplicate sample
            user_data[repeat_positions] = repeat_sample
    
    return user_groups


def calculate_class_balance(user_groups, dataset, user_id):
    """Calculate class balance metric for a user"""
    user_indices = user_groups[user_id]
    class_counts = {}
    
    for idx in user_indices:
        label = dataset.targets[idx]
        if hasattr(label, 'item'):
            label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1
    
    if not class_counts:
        return 0
    
    min_samples = min(class_counts.values())
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    
    return (min_samples * num_classes) / total_samples


# ============================================================================
# Main Dataset Creation Function
# ============================================================================

def get_dataset(args):
    """Main function to load datasets and create federated splits"""
    
    # Dataset loading mapping
    dataset_loaders = {
        'cifar10': load_cifar10,
        'cifar': load_cifar10,
        'cifar100': load_cifar100,
        'mnist': load_mnist,
        'fmnist': load_fashion_mnist,
        'emnist': load_emnist,
        'animal10': load_animal10,
        'animal': load_animal10,
    }
    
    if args.dataset not in dataset_loaders:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Load datasets
    train_dataset, test_dataset = dataset_loaders[args.dataset]()
    print(f"{args.dataset.upper()} Data Loading...")
    
    # Create federated splits
    if args.iid:
        user_groups_train = iid_split(train_dataset, args.num_users)
        user_groups_test = iid_split(test_dataset, args.num_users)
        print('IID Data Loading...')
    else:
        # Choose split strategy based on heterogeneity type
        if getattr(args, 'heter', 'weakpath') == 'weakpath':
            user_groups_train = noniid_s_split(
                train_dataset, args.num_users, 
                noniid_s=getattr(args, 'noniid_s', 20),
                local_size=getattr(args, 'local_size', 600)
            )
            
            if getattr(args, 'testiid', 0):
                user_groups_test = iid_split(test_dataset, args.num_users)
            else:
                user_groups_test = noniid_s_split(
                    test_dataset, args.num_users,
                    noniid_s=getattr(args, 'noniid_s', 20),
                    local_size=300
                )
        else:  # 'path' heterogeneity
            shards = 10 if args.dataset == 'cifar100' else 3
            user_groups_train = noniid_split(train_dataset, args.num_users, shards)
            
            if getattr(args, 'testiid', 0):
                user_groups_test = iid_split(test_dataset, args.num_users)
            else:
                user_groups_test = noniid_split(test_dataset, args.num_users, shards)
        
        # Add duplicates if specified
        if hasattr(args, 'num_repeat_users') and hasattr(args, 'repeat_ratio'):
            user_groups_train = add_duplicates(
                user_groups_train, args.num_repeat_users, args.repeat_ratio
            )
        
        print('Non-IID Data Loading...')
    
    # Create data loaders
    train_loaders = []
    test_loaders = []
    
    for i in range(args.num_users):
        # Calculate class balance metric
        h_l = calculate_class_balance(user_groups_train, train_dataset, i)
        
        # Create loaders
        train_loader = DataLoader(
            DatasetSplit(train_dataset, user_groups_train[i]),
            batch_size=args.local_bs, shuffle=True
        )
        test_loader = DataLoader(
            DatasetSplit(test_dataset, user_groups_test[i]),
            batch_size=args.local_bs, shuffle=False
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Global test loader
    global_test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
    
    return train_loaders, test_loaders, global_test_loader, h_l