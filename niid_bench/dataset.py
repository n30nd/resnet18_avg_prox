"""Partition the data and create the dataloaders."""

from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# def get_custom_dataset(data_path: str = "/media/namvq/Data/chest_xray"):
#     """Load custom dataset and apply transformations."""
#     transform = Compose([
#         Resize((100, 100)),
#         Grayscale(num_output_channels=1),
#         ToTensor()
#     ])
#     trainset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
#     testset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
#     return trainset, testset

# def get_custom_dataset(data_path: str = "/kaggle/input/chest-xray-pneumonia/chest_xray"):
#     """Load custom dataset and apply transformations."""
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Kích thước ảnh cho EfficientNet
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
#                              [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
#     ])
#     trainset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
#     testset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
#     return trainset, testset

# def get_custom_dataset(data_path: str = "/media/namvq/Data/chest_xray"):
#     """Load custom dataset and apply transformations."""
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Kích thước ảnh cho EfficientNet
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
#                              [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Kích thước ảnh cho EfficientNet
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
#                              [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
#     ])
#     trainset = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
#     testset = ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
#     return trainset, testset
# def get_custom_dataset(data_path: str = "/media/namvq/Data/chest_xray"):
#     """Load custom dataset and apply transformations."""
#     train_transform = transforms.Compose([
#         transforms.Resize((150, 150)),  # Kích thước ảnh cho VGG
#         transforms.RandomAffine(degrees=0, shear=10),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
#         transforms.RandomAffine(degrees=0, translate=(0.2, 0)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
#                              [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((150, 150)),  # Kích thước ảnh cho VGG
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
#                              [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
#     ])
#     trainset = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
#     testset = ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
#     return trainset, testset
# /kaggle/input/chest-xray-reduce-4-test
#/media/namvq/Data/chest_xray
# /kaggle/input/chest-xray-pneumonia/chest_xray
def get_custom_dataset(data_path: str = "/kaggle/input/chest-xray-pneumonia/chest_xray"):
    """Load custom dataset and apply transformations."""
    train_transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
                             [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Kích thước ảnh cho VGG
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean chuẩn của ImageNet
                             [0.229, 0.224, 0.225])  # Std chuẩn của ImageNet
    ])
    trainset = ImageFolder(os.path.join(data_path, 'train'), transform=train_transform)
    testset = ImageFolder(os.path.join(data_path, 'test'), transform=test_transform)
    return trainset, testset


def prepare_dataset_for_centralized_train(batch_size: int, val_ratio: float = 0.1, seed: int = 42):
    trainset, testset = get_custom_dataset()
    # Split trainset into trainset and valset
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(seed))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloader, valloader, testloader


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, alpha: float = 100, seed: int = 42):
    """Load custom dataset and generate non-IID partitions using Dirichlet distribution."""
    trainset, testset = get_custom_dataset()
    
    # Split trainset into trainset and valset
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], torch.Generator().manual_seed(seed))
    
    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])
    
    # Generate Dirichlet distribution for each class
    class_indices = [np.where(train_labels == i)[0] for i in range(len(np.unique(train_labels)))]
    partition_indices = [[] for _ in range(num_partitions)]
    
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        class_partitions = np.split(class_idx, proportions)
        for i in range(num_partitions):
            partition_indices[i].extend(class_partitions[i])
    
    # Create Subsets for each partition
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]
    
    # Split valset into partitions
    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1

    valsets = random_split(valset, partition_len_val, torch.Generator().manual_seed(seed))
    
    # Create DataLoaders for each partition
    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=2) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=2) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Calculate class distribution for each partition in trainloaders
    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')

    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloaders, valloaders, testloader

def prepare_partitioned_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, num_labels_each_party: int = 1, seed: int = 42):
    """Load custom dataset and generate partitions where each party has a fixed number of labels."""
    trainset, testset = get_custom_dataset()  # Load datasets

    # Split the trainset into trainset and valset based on the validation ratio
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))

    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])

    # Define partitions: each party has k labels
    num_labels = len(np.unique(train_labels))  # Assuming labels are 0 and 1 for binary classification
    times = [0 for i in range(num_labels)]
    contain = []
    #Phan label cho cac client
    for i in range(num_partitions):
        current = [i%num_labels]
        times[i%num_labels] += 1
        if num_labels_each_party > 1:
            current.append(1-i%num_labels)
            times[1-i%num_labels] += 1
        contain.append(current)
    print(times)
    print(contain)
    # Create Subsets for each partition

    partition_indices = [[] for _ in range(num_partitions)]
    for i in range(num_labels):
        idx_i = np.where(train_labels == i)[0]  # Get indices of label i in train_labels
        idx_i = [trainset.indices[j] for j in idx_i]  # Convert indices to indices in trainset
        # #print label of idx_i
        # print("Label of idx: ", i)
        # for j in range(len(idx_i)):
        #     idx_in_dataset = trainset.indices[idx_i[j]]
        #     print(trainset.dataset.targets[idx_in_dataset])
        np.random.shuffle(idx_i)
        split = np.array_split(idx_i, times[i])
        ids = 0
        for j in range(num_partitions):
            if i in contain[j]:
                partition_indices[j].extend(split[ids])
                ids += 1
    
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    # #print label of client 0
    # print("Client 0")
    # for i in range(len(trainsets[0])):
    #     print(trainsets[0][i][1])

    # Split valset into partitions
    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))

    # Create DataLoaders for each partition
    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=4) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=4) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Calculate class distribution for each partition in trainloaders
    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    # plt.show()

    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()



    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    return trainloaders, valloaders, testloader

def prepare_imbalance_label_dirichlet(num_partitions: int, batch_size: int, val_ratio: float = 0.1, beta: float = 0.5, seed: int = 42):
    """Load custom dataset and generate partitions where each party has a fixed number of labels."""
    trainset, testset = get_custom_dataset()  # Load datasets

    # Split the trainset into trainset and valset based on the validation ratio
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))

    # Get labels for the entire trainset
    train_labels = np.array([trainset.dataset.targets[i] for i in trainset.indices])

    # Define partitions: each party has k labels
    num_labels = len(np.unique(train_labels))  # Assuming labels are 0 and 1 for binary classification
    min_size = 0
    min_require_size = 2

    N = len(trainset)


    while(min_size < min_require_size):
        partition_indices = [[] for _ in range(num_partitions)]
        for label in range(num_labels):
            idx_label = np.where(train_labels == label)[0]
            idx_label = [trainset.indices[j] for j in idx_label]
            np.random.shuffle(idx_label)

            proportions = np.random.dirichlet(np.repeat(beta, num_partitions))
            # proportions = np.array( [p * len(idx_j) < N/num_partitions] for p, idx_j in zip(proportions, partition_indices))
            proportions = np.array([p if p * len(idx_j) < N / num_partitions else 0 for p, idx_j in zip(proportions, partition_indices)])

            proportions = proportions / np.sum(proportions)
            proportions = (np.cumsum(proportions) * len(idx_label)).astype(int)[:-1]

            partition_indices = [idx_j + idx.tolist() for idx_j, idx in zip(partition_indices, np.split(idx_label, proportions))]
            min_size = min([len(idx_j) for idx_j in partition_indices])
        
    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))

    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=4) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=4) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    # Plot class distribution
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')

    return trainloaders, valloaders, testloader



def apply_gaussian_noise(tensor, std_dev):
    noise = torch.randn_like(tensor) * std_dev
    return tensor + noise

# Hàm đảo ngược chuẩn hóa
def unnormalize_image(image_tensor, mean, std):
    # Đảo ngược Normalize: (image * std) + mean
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)  # Thực hiện từng kênh
    return image_tensor

# Hàm hiển thị ảnh từ một tensor
def display_image(image_tensor, mean, std):
    # Đảo ngược chuẩn hóa
    image_tensor = unnormalize_image(image_tensor, mean, std)
    # Chuyển tensor thành NumPy array và điều chỉnh thứ tự kênh màu (CHW -> HWC)
    image_numpy = image_tensor.permute(1, 2, 0).numpy()
    # Cắt giá trị ảnh về phạm vi [0, 1] để hiển thị đúng
    image_numpy = image_numpy.clip(0, 1)
    # Trả về ảnh NumPy
    return image_numpy

# def prepare_noise_based_imbalance(num_partitions: int, batch_size: int, val_ratio: float = 0.1, sigma: float = 0.05, seed: int = 42):
#     """
#     Chia du lieu ngau nhien va deu cho cac ben, sau do them noise vao cac ben
#     moi ben i co noise khac nhau Gauss(0, sigma*i/N)
#     """
#     trainset, testset = get_custom_dataset()
#     num_train = int((1 - val_ratio) * len(trainset))
#     num_val = len(trainset) - num_train
#     trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))

#     indices = trainset.indices

#     np.random.shuffle(indices)

#     partition_indices = np.array_split(indices, num_partitions)

#     train_partitions = []

#     for i, part_indices in enumerate(partition_indices):
#         partition_std_dev = sigma * (i + 1) / num_partitions
#         partition_set = Subset(trainset.dataset, part_indices)
        
#         noisy_samples = [apply_gaussian_noise(sample[0], partition_std_dev) for sample in partition_set]
#         noisy_dataset = [(noisy_samples[j], trainset.dataset[part_indices[j]][1]) for j in range(len(part_indices))]
#         # train_partitions.append((noisy_samples, [sample[1] for sample in partition_set]))
#         train_partitions.append(noisy_dataset)
#     trainloaders = [DataLoader(train_partitions[i], batch_size=batch_size, shuffle=True, num_workers=4) for i in range(num_partitions)]
#     partition_len_val = [len(valset) // num_partitions] * num_partitions
#     for i in range(len(valset) % num_partitions):
#         partition_len_val[i] += 1
    
#     valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))
#     valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=4) for vs in valsets]
#     testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# ####
#     class_distributions = []
#     for i, trainloader in enumerate(trainloaders):
#         class_counts = Counter()
#         for _, labels in trainloader:
#             class_counts.update(labels.numpy())
#         class_distributions.append(class_counts)
#         print(f'Partition {i} class distribution: {dict(class_counts)}')
    
#     partitions = range(num_partitions)
#     class_0_counts = [class_distributions[i][0] for i in partitions]
#     class_1_counts = [class_distributions[i][1] for i in partitions]

#     bar_width = 0.5
#     plt.figure(figsize=(12, 8))
#     plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
#     plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
#     plt.xlabel('Partition')
#     plt.ylabel('Number of Samples')
#     plt.title('Class Distribution in Each Partition')
#     plt.legend()
#     plt.grid(True)
#     # plt.show()
#     #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
#     output_dir = 'running_outputs'
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, 'data_partition.png'))
#     plt.close()

#     #Lưu ảnh nhiễu vào running_outputs
#     # Mean và std từ Normalize
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     # Tạo thư mục lưu ảnh nếu chưa tồn tại
#     output_dir = "running_outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     # Khởi tạo một lưới 10x6 để hiển thị ảnh
#     fig, axes = plt.subplots(10, 6, figsize=(15, 25))

#     # Duyệt qua 60 trainloaders và hiển thị ảnh đầu tiên
#     for i, trainloader in enumerate(trainloaders[:num_partitions]):
#         # Lấy ảnh đầu tiên từ trainloader
#         image_tensor = trainloader.dataset[0][0].clone()  # Clone để tránh thay đổi dữ liệu gốc
        
#         # Tìm vị trí hàng, cột trong lưới
#         row, col = divmod(i, 6)
#         plt.sca(axes[row, col])  # Đặt trục hiện tại là vị trí hàng, cột trong lưới
        
#         # Hiển thị ảnh
#         image_numpy = display_image(image_tensor, mean, std)
#         axes[row, col].imshow(image_numpy)
#         axes[row, col].axis('off')
#     plt.title(f"Noise image with sigma from {sigma * 1 / num_partitions} to {sigma}")
#     # Điều chỉnh layout để không bị chồng lấn
#     plt.tight_layout()

#     # Lưu ảnh thay vì hiển thị
#     output_path = os.path.join(output_dir, "image_noise.png")
#     plt.savefig(output_path, dpi=300)  # Lưu ảnh với chất lượng cao

#     plt.close()  # Đóng figure

#     print(f"Ảnh đã được lưu tại {output_path}")

#     print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')

# ###
#     return trainloaders, valloaders, testloader
def prepare_noise_based_imbalance(num_partitions: int, batch_size: int, val_ratio: float = 0.1, sigma: float = 0.05, seed: int = 42):
    """
    Chia dữ liệu ngẫu nhiên và đều cho các bên, sau đó thêm noise vào các bên.
    Mỗi bên i có noise khác nhau Gauss(0, sigma*i/N). Nếu dữ liệu đã tồn tại, tải từ thư mục dataset_noise_{sigma}.
    """
    # noise_dir = f'chest_xray_noise_{sigma}'
    noise_dir = f'/kaggle/input/chest-xray-noise-60-partitions/chest_xray_noise_{sigma}'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    noisy_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if os.path.exists(noise_dir):
        print(f"Loading noisy dataset from {noise_dir}...")
        # Sử dụng ImageFolder để tải dữ liệu đã được thêm nhiễu với transform phù hợp
        train_partitions = [ImageFolder(os.path.join(noise_dir, f'partition_{i}'), transform=noisy_transform) for i in range(num_partitions)]
        
        # Tải val và test set như bình thường
        trainset, testset = get_custom_dataset()
        num_train = int((1 - val_ratio) * len(trainset))
        num_val = len(trainset) - num_train
        _, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))
        
        # Chia valset thành các partition
        partition_len_val = [len(valset) // num_partitions] * num_partitions
        for i in range(len(valset) % num_partitions):
            partition_len_val[i] += 1
        valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))
        
        # Tạo DataLoaders
        trainloaders = [DataLoader(part, batch_size=batch_size, shuffle=True, num_workers=6) for part in train_partitions]
        valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=6) for vs in valsets]
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)
        
        print("Dữ liệu đã được tải thành công từ thư mục lưu trữ.")
    else:
        print(f"Creating noisy dataset and saving to {noise_dir}...")
        os.makedirs(noise_dir, exist_ok=True)
        
        trainset, testset = get_custom_dataset()
        num_train = int((1 - val_ratio) * len(trainset))
        num_val = len(trainset) - num_train
        trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))
    
        indices = trainset.indices
        np.random.shuffle(indices)
        partition_indices = np.array_split(indices, num_partitions)
    
        # Mean và std từ Normalize đã được định nghĩa trước
        for i, part_indices in enumerate(partition_indices):
            partition_std_dev = sigma * (i + 1) / num_partitions
            partition_set = Subset(trainset.dataset, part_indices)
            
            # Tạo thư mục cho partition và các lớp
            partition_dir = os.path.join(noise_dir, f'partition_{i}')
            os.makedirs(partition_dir, exist_ok=True)
            class_dirs = {}
            for _, label in partition_set:
                if label not in class_dirs:
                    class_dirs[label] = os.path.join(partition_dir, f'class_{label}')
                    os.makedirs(class_dirs[label], exist_ok=True)
            
            for j, (image, label) in enumerate(partition_set):
                noisy_image = apply_gaussian_noise(image, partition_std_dev)
                # Đảo ngược chuẩn hóa để lưu ảnh đúng định dạng
                noisy_image = unnormalize_image(noisy_image, mean, std)
                # Chuyển tensor thành PIL Image
                noisy_image_pil = transforms.ToPILImage()(noisy_image.clamp(0, 1))
                # Lưu ảnh với tên duy nhất
                image_filename = f'image_{j}.png'
                noisy_image_pil.save(os.path.join(class_dirs[label], image_filename))
        
        # Tải dữ liệu từ thư mục đã lưu với transform phù hợp
        train_partitions = [ImageFolder(os.path.join(noise_dir, f'partition_{i}'), transform=noisy_transform) for i in range(num_partitions)]
        
        # Chia valset thành các partition
        partition_len_val = [len(valset) // num_partitions] * num_partitions
        for i in range(len(valset) % num_partitions):
            partition_len_val[i] += 1
        valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))
        
        # Tạo DataLoaders
        trainloaders = [DataLoader(part, batch_size=batch_size, shuffle=True, num_workers=6) for part in train_partitions]
        valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=6) for vs in valsets]
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    # Phân tích phân bố lớp
    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i].get(0, 0) for i in partitions]
    class_1_counts = [class_distributions[i].get(1, 0) for i in partitions]
    
    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    # Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()
    
    # Lưu ảnh nhiễu vào running_outputs
    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    output_dir = "running_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo một lưới 10x6 để hiển thị ảnh
    fig, axes = plt.subplots(10, 6, figsize=(15, 25))
    
    # Duyệt qua trainloaders và hiển thị ảnh đầu tiên từ mỗi partition
    for i, trainloader in enumerate(trainloaders[:min(num_partitions, 60)]):
        if len(trainloader.dataset) == 0:
            continue
        # Lấy ảnh đầu tiên từ trainloader
        image_tensor, label = trainloader.dataset[0]
        
        # Tìm vị trí hàng, cột trong lưới
        row, col = divmod(i, 6)
        if row >= 10:
            break  # Chỉ hiển thị tối đa 60 ảnh
        # Hiển thị ảnh
        image_numpy = unnormalize_image(image_tensor.clone(), mean, std).permute(1, 2, 0).numpy().clip(0, 1)
        axes[row, col].imshow(image_numpy)
        axes[row, col].axis('off')
    # Điều chỉnh layout để không bị chồng lấn
    plt.tight_layout()
    
    # Lưu ảnh thay vì hiển thị
    output_path = os.path.join(output_dir, "image_noise.png")
    plt.savefig(output_path, dpi=300)  # Lưu ảnh với chất lượng cao
    
    plt.close()  # Đóng figure
    
    print(f"Ảnh minh họa đã được lưu tại {output_path}")
    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')
    
    return trainloaders, valloaders, testloader

def prepare_quantity_skew_dirichlet(num_partitions: int, batch_size: int, val_ratio: float = 0.1, beta: float = 10, seed: int = 42):
    trainset, testset = get_custom_dataset()
    num_train = int((1 - val_ratio) * len(trainset))
    num_val = len(trainset) - num_train
    trainset, valset = random_split(trainset, [num_train, num_val], generator=torch.Generator().manual_seed(seed))

    all_indices = trainset.indices

    min_size = 0
    while min_size < 1:
        proportions = np.random.dirichlet(np.repeat(beta, num_partitions))
        proportions = (np.cumsum(proportions) * len(all_indices)).astype(int)[:-1]

        partition_indices = np.split(all_indices, proportions)

        min_size = min([len(partition) for partition in partition_indices])
        print('Partition sizes:', [len(partition) for partition in partition_indices])
        print('Min partition size:', min_size)

    trainsets = [Subset(trainset.dataset, indices) for indices in partition_indices]

    partition_len_val = [len(valset) // num_partitions] * num_partitions
    for i in range(len(valset) % num_partitions):
        partition_len_val[i] += 1
    
    valsets = random_split(valset, partition_len_val, generator=torch.Generator().manual_seed(seed))

    trainloaders = [DataLoader(ts, batch_size=batch_size, shuffle=True, num_workers=4) for ts in trainsets]
    valloaders = [DataLoader(vs, batch_size=batch_size, shuffle=False, num_workers=4) for vs in valsets]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_distributions = []
    for i, trainloader in enumerate(trainloaders):
        class_counts = Counter()
        for _, labels in trainloader:
            class_counts.update(labels.numpy())
        class_distributions.append(class_counts)
        print(f'Partition {i} class distribution: {dict(class_counts)}')
    
    partitions = range(num_partitions)
    class_0_counts = [class_distributions[i][0] for i in partitions]
    class_1_counts = [class_distributions[i][1] for i in partitions]

    bar_width = 0.5
    plt.figure(figsize=(12, 8))
    plt.bar(partitions, class_0_counts, bar_width, label='Class 0', color='blue')
    plt.bar(partitions, class_1_counts, bar_width, bottom=class_0_counts, label='Class 1', color='red')
    plt.xlabel('Partition')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Each Partition')
    plt.legend()
    plt.grid(True)
    # plt.show()
    #  Lưu đồ thị vào thư mục running_outputs với tên data_partition
    output_dir = 'running_outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'data_partition.png'))
    plt.close()

    print(f'Number of train samples: {len(trainset)}, val samples: {len(valset)}, test samples: {len(testloader.dataset)}')

    return trainloaders, valloaders, testloader


def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoaders for training, validation, and testing.
    """
    print(f"Dataset partitioning config: {config}")
    batch_size = -1
    if "batch_size" in config:
        batch_size = config.batch_size
    elif "batch_size_ratio" in config:
        batch_size_ratio = config.batch_size_ratio
    else:
        raise ValueError
    partitioning = ""
    
    if "partitioning" in config:
        partitioning = config.partitioning

    # partition the data
    if partitioning == "imbalance_label":
        return prepare_partitioned_dataset(num_clients, batch_size, val_ratio, config.labels_per_client, config.seed)

    if partitioning == "imbalance_label_dirichlet":
        return prepare_imbalance_label_dirichlet(num_clients, batch_size, val_ratio, config.alpha, config.seed)

    if partitioning == "noise_based_imbalance":
        return prepare_noise_based_imbalance(num_clients, batch_size, val_ratio, config.sigma, config.seed)

    if partitioning == "quantity_skew_dirichlet":
        return prepare_quantity_skew_dirichlet(num_clients, batch_size, val_ratio, config.alpha, config.seed)
    

if __name__ == "__main__":
    # prepare_imbalance_label_dirichlet(5, 32, 0.1, 0.5)
    # prepare_partitioned_dataset(5, 32, 0.1, 1)
    prepare_noise_based_imbalance(5, 10, 0.1, 0.1)
    # prepare_quantity_skew_dirichlet(5, 32, 0.1, 10)









# from niid_bench.dataset_preparation import (
#     partition_data,
#     partition_data_dirichlet,
#     partition_data_label_quantity,
# )


# pylint: disable=too-many-locals, too-many-branches
# def load_datasets(
#     config: DictConfig,
#     num_clients: int,
#     val_ratio: float = 0.1,
#     seed: Optional[int] = 42,
# ) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
#     """Create the dataloaders to be fed into the model.

#     Parameters
#     ----------
#     config: DictConfig
#         Parameterises the dataset partitioning processa
#     num_clients : int
#         The number of clients that hold a part of the data
#     val_ratio : float, optional
#         The ratio of training data that will be used for validation (between 0 and 1),
#         by default 0.1
#     seed : int, optional
#         Used to set a fix seed to replicate experiments, by default 42

#     Returns
#     -------
#     Tuple[DataLoader, DataLoader, DataLoader]
#         The DataLoaders for training, validation, and testing.
#     """
#     print(f"Dataset partitioning config: {config}")
#     partitioning = ""
#     if "partitioning" in config:
#         partitioning = config.partitioning
#     # partition the data
#     if partitioning == "dirichlet":
#         alpha = 0.5
#         if "alpha" in config:
#             alpha = config.alpha
#         datasets, testset = partition_data_dirichlet(
#             num_clients,
#             alpha=alpha,
#             seed=seed,
#             dataset_name=config.name,
#         )
#     elif partitioning == "label_quantity":
#         labels_per_client = 2
#         if "labels_per_client" in config:
#             labels_per_client = config.labels_per_client
#         datasets, testset = partition_data_label_quantity(
#             num_clients,
#             labels_per_client=labels_per_client,
#             seed=seed,
#             dataset_name=config.name,
#         )
#     elif partitioning == "iid":
#         datasets, testset = partition_data(
#             num_clients,
#             similarity=1.0,
#             seed=seed,
#             dataset_name=config.name,
#         )
#     elif partitioning == "iid_noniid":
#         similarity = 0.5
#         if "similarity" in config:
#             similarity = config.similarity
#         datasets, testset = partition_data(
#             num_clients,
#             similarity=similarity,
#             seed=seed,
#             dataset_name=config.name,
#         )

#     batch_size = -1
#     if "batch_size" in config:
#         batch_size = config.batch_size
#     elif "batch_size_ratio" in config:
#         batch_size_ratio = config.batch_size_ratio
#     else:
#         raise ValueError

#     # split each partition into train/val and create DataLoader
#     trainloaders = []
#     valloaders = []
#     for dataset in datasets:
#         len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
#         lengths = [len(dataset) - len_val, len_val]
#         ds_train, ds_val = random_split(
#             dataset, lengths, torch.Generator().manual_seed(seed)
#         )
#         if batch_size == -1:
#             batch_size = int(len(ds_train) * batch_size_ratio)
#         trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
#         valloaders.append(DataLoader(ds_val, batch_size=batch_size))
#     return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset))

