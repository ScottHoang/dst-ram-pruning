import os
import random

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class EvenClassSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        if isinstance(data_source, DatasetSplitter):
            self.data_source = data_source.parent_dataset
            self.split_end = data_source.split_end
            self.split_start = data_source.split_start
        else:
            self.data_source = data_source
            self.split_end = len(data_source)
            self.split_start = 0

        self.indices_by_class = {
            k: [] for k in range(len(np.unique(self.data_source.targets)))
        }
        for idx, target in enumerate(self.data_source.targets):
            if idx >= self.split_start and idx < self.split_end:
                self.indices_by_class[target].append(idx)

    def __iter__(self):
        for k in self.indices_by_class:
            random.shuffle(self.indices_by_class[k])

        num_classes = len(self.indices_by_class)
        iterators = [iter(self.indices_by_class[k]) for k in range(num_classes)]
        num_samples = sum(len(v) for v in self.indices_by_class.values())
        for i in range(num_samples):
            # class_idx = random.randint(0, num_classes - 1)
            class_idx = i % num_classes
            try:
                yield next(iterators[class_idx])
            except StopIteration:
                iterators.pop(class_idx)
                num_classes -= 1

    def __len__(self):
        return self.split_end - self.split_start


##################


class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""

    def __init__(self, parent_dataset, split_start=-1, split_end=-1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert (
            split_start <= len(parent_dataset) - 1
            and split_end <= len(parent_dataset)
            and split_start < split_end
        ), "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start

    def __getitem__(self, index):
        assert index < len(self), "index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]


def get_cifar100_dataloaders(
    args, validation_split=0.0, max_threads=10, shuffle=True, even_sampling=False
):
    """Creates augmented train, validation, and test data loaders."""
    cifar_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    cifar_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ]
    )
    sampler = None
    if even_sampling:
        sampler = EvenClassSampler

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    sampler = sampler(trainset) if sampler is not None else None
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=2,
        sampler=sampler,
    )

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, test_loader


def get_cifar10_dataloaders(
    args, validation_split=0.0, max_threads=10, shuffle=True, even_sampling=False
):

    """Creates augmented train, validation, and test data loaders."""

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_dataset = datasets.CIFAR10("_dataset", True, train_transform, download=True)
    test_dataset = datasets.CIFAR10("_dataset", False, test_transform, download=False)

    # we need at least two threads
    max_threads = 2 if max_threads < 2 else max_threads
    if max_threads >= 6:
        val_threads = 2
        train_threads = max_threads - val_threads
    else:
        val_threads = 1
        train_threads = max_threads - 1

    sampler = None
    if even_sampling:
        sampler = EvenClassSampler

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)
        sampler = sampler(train_dataset) if sampler is not None else None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=train_threads,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset, args.test_batch_size, num_workers=val_threads, pin_memory=True
        )
    else:
        sampler = sampler(full_dataset) if sampler is not None else None
        train_loader = torch.utils.data.DataLoader(
            full_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True
        )

    print("Train loader length", len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    if validation_split == 0.0:
        valid_loader = test_loader

    return train_loader, valid_loader, test_loader


def get_tinyimagenet_dataloaders(args, validation_split=-1.0):
    traindir = os.path.join(args.datadir, "train")
    valdir = os.path.join(args.datadir, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    data_transform_train = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    data_transform_val = transforms.Compose([transforms.ToTensor(), normalize])

    ###
    # data_transform_train = create_transform(
    # input_size=64,
    # is_training=True,
    # color_jitter=0.3,
    # auto_augment=None,
    # )
    # data_transform_val = create_transform(input_size=64, is_training=False)

    train_dataset = datasets.ImageFolder(traindir, data_transform_train)

    # if args.distributed:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, data_transform_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, val_loader


def get_imagenet_dataloaders(args, validation_split=-1.0):
    traindir = os.path.join("imagenet", "train")
    valdir = os.path.join("imagenet", "val")

    data_transform_train = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
    )
    data_transform_val = create_transform(input_size=224, is_training=False)

    train_dataset = datasets.ImageFolder(
        traindir,
        data_transform_train,
    )

    # if args.distributed:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=10,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, data_transform_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, val_loader, val_loader


def get_imagenet100_dataloaders(args, validation_split=-1.0):
    traindir = os.path.join("imagenet100", "train")
    valdir = os.path.join("imagenet100", "val")

    data_transform_train = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.3,
        auto_augment=None,
    )
    data_transform_val = create_transform(input_size=224, is_training=False)

    train_dataset = datasets.ImageFolder(
        traindir,
        data_transform_train,
    )

    # if args.distributed:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, data_transform_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, val_loader


def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transform = transforms.Compose([transforms.ToTensor(), normalize])

    full_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0 - validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset, split_end=split)
        val_dataset = DatasetSplitter(full_dataset, split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset, args.test_batch_size, num_workers=2, pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset, args.batch_size, num_workers=8, pin_memory=True, shuffle=True
        )

    print("Train loader length", len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


def plot_class_feature_histograms(args, model, device, test_loader, optimizer):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    model.eval()
    agg = {}
    num_classes = 10
    feat_id = 0
    sparse = not args.dense
    model_name = "alexnet"
    # model_name = 'vgg'
    # model_name = 'wrn'

    densities = None
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx % 100 == 0:
            print(batch_idx, "/", len(test_loader))
        with torch.no_grad():
            # if batch_idx == 10: break
            data, target = data.to(device), target.to(device)
            for cls in range(num_classes):
                # print('=='*50)
                # print('CLASS {0}'.format(cls))
                model.t = target
                sub_data = data[target == cls]

                output = model(sub_data)

                feats = model.feats
                if densities is None:
                    densities = []
                    densities += model.densities

                if len(agg) == 0:
                    for feat_id, feat in enumerate(feats):
                        agg[feat_id] = []
                        # print(feat.shape)
                        for i in range(feat.shape[1]):
                            agg[feat_id].append(np.zeros((num_classes,)))

                for feat_id, feat in enumerate(feats):
                    map_contributions = torch.abs(feat).sum([0, 2, 3])
                    for map_id in range(map_contributions.shape[0]):
                        # print(feat_id, map_id, cls)
                        # print(len(agg), len(agg[feat_id]), len(agg[feat_id][map_id]), len(feats))
                        agg[feat_id][map_id][cls] += map_contributions[map_id].item()

                del model.feats[:]
                del model.densities[:]
                model.feats = []
                model.densities = []

    if sparse:
        np.save("./results/{0}_sparse_density_data".format(model_name), densities)

    for feat_id, map_data in agg.items():
        data = np.array(map_data)
        # print(feat_id, data)
        full_contribution = data.sum()
        # print(full_contribution, data)
        contribution_per_channel = (1.0 / full_contribution) * data.sum(1)
        # print('pre', data.shape[0])
        channels = data.shape[0]
        # data = data[contribution_per_channel > 0.001]

        channel_density = np.cumsum(np.sort(contribution_per_channel))
        print(channel_density)
        idx = np.argsort(contribution_per_channel)

        threshold_idx = np.searchsorted(channel_density, 0.05)
        print(data.shape, "pre")
        data = data[idx[threshold_idx:]]
        print(data.shape, "post")

        # perc = np.percentile(contribution_per_channel[contribution_per_channel > 0.0], 10)
        # print(contribution_per_channel, perc, feat_id)
        # data = data[contribution_per_channel > perc]
        # print(contribution_per_channel[contribution_per_channel < perc].sum())
        # print('post', data.shape[0])
        normed_data = np.max(data / np.sum(data, 1).reshape(-1, 1), 1)
        # normed_data = (data/np.sum(data,1).reshape(-1, 1) > 0.2).sum(1)
        # counts, bins = np.histogram(normed_data, bins=4, range=(0, 4))
        np.save(
            "./results/{2}_{1}_feat_data_layer_{0}".format(
                feat_id, "sparse" if sparse else "dense", model_name
            ),
            normed_data,
        )
        # plt.ylim(0, channels/2.0)
        ##plt.hist(normed_data, bins=range(0, 5))
        # plt.hist(normed_data, bins=[(i+20)/float(200) for i in range(180)])
        # plt.xlim(0.1, 0.5)
        # if sparse:
        #    plt.title("Sparse: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_sp.png'.format(feat_id))
        # else:
        #    plt.title("Dense: Conv2D layer {0}".format(feat_id))
        #    plt.savefig('./output/feat_histo/layer_{0}_d.png'.format(feat_id))
        # plt.clf()


class TensorboardXTracker:
    def __init__(self, log_dir):
        self.writer = tensorboardX.SummaryWriter(log_dir)

    def log_scalar(self, var_name, value, step):
        self.writer.add_scalar(var_name, value, step)

    def log_loss(self, loss, step):
        self.log_scalar("loss", loss, step)

    def log_validation_acc(self, acc, step):
        self.log_scalar("validation_acc", acc, step)

    def log_test_acc(self, acc, step):
        self.log_scalar("test_acc", acc, step)

    def close(self):
        self.writer.close()


def get_sparse_state_dict(model):
    _state_dict = {}
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.endswith("weight_orig"):
            mask = state_dict[k.replace("orig", "mask")]
            _state_dict[k] = (v * mask).to_sparse()
        elif k.endswith("weight_mask"):
            _state_dict[k] = v.to_sparse()
        else:
            _state_dict[k] = v
    return _state_dict


def get_dense_state_dict(state_dict):
    for k, v in state_dict.items():
        state_dict[k] = v.to_dense()
    return state_dict
