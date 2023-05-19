from __future__ import print_function

import argparse
import copy
import hashlib
import logging
import math
import os
import time
import typing as typ
import warnings
from os import path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision  # type: ignore[import]
from sparselearning.utils import get_tinyimagenet_dataloaders
import torchvision.transforms as transforms  # type: ignore[import]

import sparselearning
from sparselearning.core import CosineDecay
from sparselearning.core import LinearDecay
from sparselearning.extended_core import ExtendedMasking as Masking
from sparselearning.models import AlexNet
from sparselearning.models import LeNet_300_100
from sparselearning.models import LeNet_5_Caffe
from sparselearning.models import MLP_CIFAR10
from sparselearning.models import ResNet18
from sparselearning.models import ResNet34
from sparselearning.models import VGG16
from sparselearning.models import WideResNet
from sparselearning.utils import get_cifar100_dataloaders
from sparselearning.utils import get_cifar10_dataloaders
from sparselearning.utils import get_mnist_dataloaders
from sparselearning.utils import TensorboardXTracker

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

logger = None

models: typ.Dict[str, typ.Tuple[typ.Any, typ.Any]] = {}

models["MLPCIFAR10"] = (MLP_CIFAR10, [])
models["lenet5"] = (LeNet_5_Caffe, [])
models["lenet300-100"] = (LeNet_300_100, [])
models["ResNet34"] = ()  # type: ignore [assignment]
models["ResNet18"] = ()  # type: ignore[assignment]
models["alexnet-s"] = (AlexNet, ["s", 10])
models["alexnet-b"] = (AlexNet, ["b", 10])
models["vgg-c"] = (VGG16, ["C", 10])
models["vgg-d"] = (VGG16, ["D", 10])
models["vgg-like"] = (VGG16, ["like", 10])
models["wrn-28-2"] = (WideResNet, [28, 2, 10, 0.3])
models["wrn-22-8"] = (WideResNet, [22, 8, 10, 0.3])
models["wrn-16-8"] = (WideResNet, [16, 8, 10, 0.3])
models["wrn-16-10"] = (WideResNet, [16, 10, 10, 0.3])


def init_random(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
    return model


def setup_logger():
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = os.path.join(savedir, "log.log")

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def train(model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    plateau = False
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16:
            data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            plateau = mask.step(epoch)
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% ".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader) * args.batch_size,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    correct,
                    n,
                    100.0 * correct / float(n),
                )
            )
        if plateau:
            break

    # training summary
    print_and_log(
        "\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n".format(
            "Training summary",
            train_loss / batch_idx,
            correct,
            n,
            100.0 * correct / float(n),
        )
    )
    writer.log_loss(train_loss / batch_idx, epoch)
    return plateau


def evaluate(model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16:
                data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log(
        "\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n".format(
            "Test evaluation" if is_test_set else "Evaluation",
            test_loss,
            correct,
            n,
            100.0 * correct / float(n),
        )
    )
    return correct / float(n)


def parser():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument("--output-dir", type=str, default="results")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        metavar="N",
        help="extend training time by multiplier times",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=17, metavar="S", help="random seed (default: 17)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="The optimizer to use. Default: sgd. Options: sgd, adam.",
    )
    randomhash = "".join(str(time.time()).split("."))
    parser.add_argument(
        "--save",
        type=str,
        default=randomhash + ".pt",
        help="path to save the final model",
    )
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--decay_frequency", type=int, default=25000)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true", help="Run in fp16 mode.")
    parser.add_argument("--valid_split", type=float, default=0.1)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--l2", type=float, default=5.0e-4)
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="How many times the model should be run after each other. Default=1",
    )
    parser.add_argument(
        "--save-features",
        action="store_true",
        help="Resumes a saved model and saves its feature data to disk for plotting.",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Enables the benchmarking of layers and estimates sparse speedups",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=10,
        help="How many threads to use for data loading.",
    )
    # ramanujan settings
    parser.add_argument("--ramanujan", action="store_true")
    parser.add_argument("--ramanujan-max-try", default=3, type=int)
    parser.add_argument("--ramanujan-full-graph", action="store_true")
    parser.add_argument("--ramanujan-soft", type=float, default=0.0)
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-threshold", type=float, default=0.05)
    parser.add_argument(
        "--feed-forward",
        action="store_true",
        help="whether or not we will forward initial weights to new mask",
    )
    parser.add_argument("--disable-scheduler", action="store_true")

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    return parser.parse_args()


def main():
    # Training settings
    setup_logger()
    print_and_log(args)

    if args.fp16:
        print("haha nice try")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log("\n\n")
    print_and_log("=" * 80)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        args.seed = i
        torch.manual_seed(args.seed)

        if args.data == "mnist":
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(
                args, validation_split=args.valid_split
            )
        elif args.data == "cifar10":
            num_classes = 100
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(
                args, args.valid_split, max_threads=args.max_threads
            )
        elif args.data == "cifar100":
            num_classes = 100
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(
                args, args.valid_split, max_threads=args.max_threads
            )
        elif args.data == "tinyimnet":
            args.datadir = "./tiny-imagenet-200"
            # scaler = th.cuda.amp.GradScaler()
            num_classes = 200
            train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(args)
            train_loader_full = train_loader
        if args.model not in models:
            print(
                "You need to select an existing model via the --model argument. Available models include: "
            )
            for key in models:
                print("\t{0}".format(key))
            raise Exception("You need to select a model")
        elif args.model == "ResNet18":
            model = ResNet18(c=num_classes).to(device)
        elif args.model == "ResNet34":
            model = ResNet34(c=num_classes).to(device)
        elif args.model == "vgg-d":
            num_classes = 100  # if args.data == "imnet100" else 10
            model = VGG16("D", num_classes).to(device)
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

        model = init_random(model)

        print_and_log(model)
        print_and_log("=" * 60)
        print_and_log(args.model)
        print_and_log("=" * 60)

        print_and_log("=" * 60)
        print_and_log("Prune mode: {0}".format(args.death))
        print_and_log("Growth mode: {0}".format(args.growth))
        print_and_log("Redistribution mode: {0}".format(args.redistribution))
        print_and_log("=" * 60)

        optimizer = None
        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.l2,
                nesterov=True,
            )
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print("Unknown optimizer: {0}".format(args.optimizer))
            raise Exception("Unknown optimizer.")

        if not args.disable_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(args.epochs / 2) * args.multiplier,
                    int(args.epochs * 3 / 4) * args.multiplier,
                ],
                last_epoch=-1,
            )
        else:
            print("we are disabling lr_scheduler for this run")
            lr_scheduler = None

        mask = None
        if args.sparse:
            decay = CosineDecay(
                args.death_rate, len(train_loader) * (args.epochs * args.multiplier)
            )
            mask = Masking(
                optimizer,
                death_rate=args.death_rate,
                death_mode=args.death,
                death_rate_decay=decay,
                growth_mode=args.growth,
                redistribution_mode=args.redistribution,
                args=args,
                writer=writer,
                dataloader=train_loader,
                criterion=F.nll_loss,
                device=device,
                feed_forward=args.feed_forward,
            )
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0

        for epoch in range(1, args.epochs * args.multiplier + 1):
            t0 = time.time()
            is_plateau = train(model, device, train_loader, optimizer, epoch, mask)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc = evaluate(model, device, valid_loader)
                writer.log_validation_acc(val_acc, epoch)

            # if val_acc > best_acc:
            # print("Saving model")
            # best_acc = val_acc
            # torch.save(model.state_dict(), os.path.join(savedir, "ckpt.pth"))

            print_and_log(
                "Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n".format(
                    optimizer.param_groups[0]["lr"], time.time() - t0
                )
            )
            writer.log_scalar("best-acc", best_acc, epoch)
            if is_plateau:
                break

        print("Testing model")
        test_acc = evaluate(model, device, test_loader, is_test_set=True)
        layer_fired_weights, total_fired_weights = mask.fired_masks_update()

        torch.save(
            {
                "test_acc": test_acc,
                "total_fired_weights": total_fired_weights,
                "layer_fired_weights": layer_fired_weights,
            },
            osp.join(savedir, f"seed_{args.seed}_dst_final_stats.pth"),
        )

        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))

        for name in layer_fired_weights:
            print(
                "The final percentage of fired weights in the layer",
                name,
                "is:",
                layer_fired_weights[name],
            )
        print(
            "The final percentage of the total fired weights is:", total_fired_weights
        )


if __name__ == "__main__":
    args = parser()
    timestr = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    ram = "ramanujan" if args.ramanujan else "vanilla"
    savedir = os.path.join(
        args.output_dir,
        f"{args.growth}+{args.death}+{ram}",
        args.sparse_init,
        args.model,
        str(args.density),
    )
    args.savedir = savedir
    os.makedirs(ram, exist_ok=True)
    writer = TensorboardXTracker(savedir)
    main()
