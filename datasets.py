from torchvision import transforms
from torchvision.datasets import (CIFAR100,CIFAR10,MNIST)
import torchvision
import torchvision.transforms as transforms
import torch
import random
import numpy as np
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# CuDNNのバックエンドにも乱数シードを設定します（GPUを使用する場合）
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datasets(dataset_name):
    if dataset_name=='CIFAR100':
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761)),
        ])
        return (CIFAR100('./data',
        train=True,download=True,transform=transforms.ToTensor()),
        CIFAR100('./data',
        train=False,download=True,transform=transforms.ToTensor()))
    
    elif dataset_name=='MNIST':
        return (MNIST('./data',
        train=True,download=True,transform=transforms.ToTensor()),
        MNIST('./data',
        train=False,download=True,transform=transforms.ToTensor()))
 
    elif dataset_name=='CIFAR10':
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])
        return (CIFAR10('./data',
        train=True,download=True,transform=transform_function),
        CIFAR10('./data',
        train=False,download=True,transform=transform_test))
 
    elif dataset_name=='imagenet':
        transform = transforms.Compose(
        [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        trainset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_train', transform=transform)

        testset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_val_for_ImageFolder', transform=transform)


        return trainset,testset

    elif dataset_name=='imdb':
        TEXT = Field(tokenize='spacy', include_lengths=True)
        LABEL = LabelField(dtype=torch.float)

        # IMDbデータセットをダウンロード
        train_data, test_data = IMDB.splits(TEXT, LABEL)

        # ボキャブラリの構築
        TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
        LABEL.build_vocab(train_data)

        return train_data,test_data
