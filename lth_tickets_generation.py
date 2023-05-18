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
import torch as th
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision  # type: ignore[import]
import torchvision.transforms as transforms  # type: ignore[import]

import sparselearning
from sparselearning.core import CosineDecay
from sparselearning.core import LinearDecay
from sparselearning.extended_core import detect_plateau
from sparselearning.extended_core import generate_sparse_masks
from sparselearning.models import AlexNet
from sparselearning.models import LeNet_300_100
from sparselearning.models import LeNet_5_Caffe
from sparselearning.models import MLP_CIFAR10
from sparselearning.models import ResNet18
from sparselearning.models import ResNet34
from sparselearning.models import VGG16
from sparselearning.models import WideResNet
from sparselearning.PAI import check_sparsity
from sparselearning.PAI import generate_mask_parameters
from sparselearning.PAI import SNIP
from sparselearning.ramanujan import Ramanujan
from sparselearning.utils import get_cifar100_dataloaders
from sparselearning.utils import get_cifar10_dataloaders
from sparselearning.utils import get_imagenet100_dataloaders
from sparselearning.utils import get_mnist_dataloaders
from sparselearning.utils import TensorboardXTracker

# from sparselearning.extended_core import ExtendedMasking as Masking

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
            n = m.kernel_size[1] * m.kernel_size[1] * m.out_channels
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
        with th.cuda.amp.autocast():
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


def get_sparse_state_dict(weights, masks):
    state_dict = {}
    for k, v in weights.items():
        if k in masks:
            state_dict[k + "_orig"] = (v * masks[k]).to_sparse()
            state_dict[k + "_mask"] = masks[k].to_sparse()
        else:
            state_dict[k] = v.to_sparse()
    return state_dict


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
    parser.add_argument("--valid_split", type=float, default=0.0)
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
    parser.add_argument("--mask-population", type=int, default=2000)
    parser.add_argument("--mask-sampling", type=int, default=10)
    parser.add_argument("--num-iteration", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained-iter", type=int, default=500)
    parser.add_argument("--alt", action="store_true")

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    return parser.parse_args()


import typing as typ
from tqdm import tqdm


def obtain_supernet(model, criteria, train_loader, device, generator):
    global args
    super_mask: dict = {}
    pbar = tqdm(range(args.max_supernet_iteration))
    for it in pbar:
        masks = generator(
            args.sparse_init,
            model,
            criteria,
            train_loader,
            args.density,
            device,
            num_iteration=args.num_iteration,
            skip_check_sparsity=True,
        )
        if len(super_mask) == 0:
            super_mask = masks
        else:
            for k, v in masks.items():
                super_mask[k] = v.data.byte() | super_mask[k].data.byte()
        total_explored = sum(v.sum() for k, v in super_mask.items())
        total_params = sum(v.numel() for k, v in super_mask.items())
        density = total_explored / total_params * 100
        pbar.set_description(f"Explored: {density:.4f}")

    return super_mask


def generate_ticket_criteria(
    masks: typ.Dict[str, th.Tensor],
    model: th.nn.Module,
    explored_masks: typ.Optional[typ.Dict],
):
    cnt = 0
    # layer_imdb = {}
    target = len(masks)
    layer_db = th.zeros(target).cuda()
    layer_imdb = th.zeros(target).cuda()
    layer_wimdb = th.zeros(target).cuda()
    layer_full_spectrum = th.zeros(target).cuda()
    mask_density = th.zeros(target).cuda()
    ramanujan_criteria = Ramanujan()
    state_dict = model.state_dict()

    if explored_masks is None:
        explored_masks = {}
        for name, mask in masks.items():
            explored_masks[name] = th.zeros_like(mask)

    for cnt, (name, mask) in enumerate(masks.items()):
        weight = state_dict[name]
        full_spectrum = ramanujan_criteria.total_spectrum_measurement(
            mask, weight, return_imdb=True
        )
        # imdb = ramanujan_criteria(mask, weight)
        layer_imdb[cnt] = full_spectrum[1]  # updating layer_imdb
        layer_wimdb[cnt] = full_spectrum[2]  # updating layer_imdb
        layer_db[cnt] = full_spectrum[3]  # updating layer_imdb
        layer_full_spectrum[cnt] = full_spectrum[0]  # updating layer_imdb
        mask_density[cnt] = masks[name].sum() / masks[name].numel()

        explored_masks[name] = (
            masks[name].data.byte() | explored_masks[name].data.byte()
        )

    # print(f"layerwise score: {layer_imdb}")

    # print(f"density per layer: {mask_density}")
    # print(mask_density)

    layer_imdb = inf_to_zero(layer_imdb)
    layer_wimdb = inf_to_zero(layer_wimdb)
    avg_imdb = layer_imdb.mean(dim=-1)
    avg_wimdb = layer_wimdb.mean(dim=-1)

    return {
        "layer_imdb": layer_imdb,
        "layer_wimdb": layer_wimdb,
        "layer_db": layer_db,
        "layer_full_spectrum": layer_full_spectrum,
        "avg_wimdb": avg_wimdb,
        "avg_imdb": avg_imdb,
        "mask_density": mask_density,
        "masks": masks,
    }, explored_masks


def inf_to_zero(batch):
    batch[batch == float("inf")] = 0
    return batch


def main():
    # Training settings
    setup_logger()
    print_and_log(args)

    if args.fp16:
        print("haha nice try")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    noise_scaler = 1e-3
    if args.model == "ResNet34":
        noise_scaler = 1e-6

    print_and_log("\n\n")
    print_and_log("=" * 80)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        args.seed = i
        torch.manual_seed(args.seed)

        exception = None
        scaler = None

        if args.data == "mnist":
            train_loader_full, _, _ = get_mnist_dataloaders(
                args,
                validation_split=args.valid_split,
            )
            train_sample = 60000 * (1 - args.valid_split)
            num_iteration = args.num_iteration  # int(train_sample // args.batch_size)
            args.batch_size = args.batch_k * 10

            train_loader, valid_loader, test_loader = get_mnist_dataloaders(
                args,
                validation_split=args.valid_split,
            )
        elif args.data == "cifar10":
            num_classes = 100
            train_loader_full, _, _ = get_cifar10_dataloaders(
                args,
                args.valid_split,
                max_threads=args.max_threads,
                shuffle=True
                # even_sampling=True,
            )

            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(
                args,
                args.valid_split,
                max_threads=args.max_threads,
                # shuffle=True,
                shuffle=True,
                # even_sampling=True,
            )
        elif args.data == "cifar100":
            num_classes = 100
            train_loader_full, _, _ = get_cifar100_dataloaders(
                args,
                args.valid_split,
                max_threads=args.max_threads,
            )

            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(
                args,
                args.valid_split,
                max_threads=args.max_threads,
                shuffle=True,
                # even_sampling=True,
            )
        elif args.data == "imnet100":
            # scaler = th.cuda.amp.GradScaler()
            exception = "classifier"
            num_classes = 100
            train_loader, valid_loader, test_loader = get_imagenet100_dataloaders(args)
            train_loader_full = train_loader

        if args.model not in models:
            print(
                "You need to select an existing model via the --model argument. Available models include: "
            )
            for key in models:
                print("\t{0}".format(key))
            raise Exception("You need to select a model")
        elif args.model == "ResNet18":
            num_classes = 100
            model = ResNet18(c=num_classes).to(device)
        elif args.model == "ResNet34":
            num_classes = 100
            model = ResNet34(c=num_classes).to(device)
        elif args.model == "vgg-d":
            num_classes = 100 if args.data == "imnet100" else 10
            model = VGG16("D", num_classes).to(device)
        else:
            raise NotImplementedError

        # else:
        # cls, cls_args = models[args.model]
        # model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

        if args.pretrained:
            folder = "pretraining_imnet" if args.data == "imnet100" else "pretraining"
            path = osp.join(
                folder, args.model, f"seed_0_iter_{args.pretrained_iter}.pth"
            )
            print("loading pretrained weights from ", path)
            model.load_state_dict(torch.load(path)["state_dict"])
        else:
            model = init_random(model)

        explored_masks = None
        best_ticket = {}
        masks = generate_sparse_masks(
            args.sparse_init,
            model,
            F.nll_loss,
            train_loader_full,
            args.density,
            device,
            exception=exception,
            scaler=scaler,
            skip_check_sparsity=True,
        )
        ticket, explored_masks = generate_ticket_criteria(masks, model, None)
        state_dict = get_sparse_state_dict(model.state_dict(), masks)
        th.save(
            {"state_dict": state_dict},
            osp.join(savedir, f"seed_{args.seed}_mask_0_step_start.pth"),
        )
        print(
            f"{args.sparse_init} full: imdb {ticket['layer_imdb'].mean()} \
        spectrum {ticket['layer_full_spectrum'].mean()}"
        )

        best_ticket = ticket  # we keep full data ticket
        save_flag = False
        last_saved = 0
        persisted_mask = None
        for mask_no in range(1, args.mask_population + 1):
            masks = generate_sparse_masks(
                args.sparse_init,
                model,
                F.nll_loss,
                train_loader,
                args.density,
                device,
                num_iteration=args.num_iteration,
                supernet_mask=None,
                skip_check_sparsity=True,
                add_noise=True,  # SynFlow specific
                noise_scaler=noise_scaler,  # SynFlow specific,
                exception=exception,
                scaler=scaler,
            )
            targets = masks.keys()
            ticket, explored_masks = generate_ticket_criteria(
                masks, model, explored_masks
            )
            if len(best_ticket.keys()) == 0:
                best_ticket = ticket
                save_flag = True
            else:
                for j, target in enumerate(targets):
                    loose_condition = (
                        ticket["layer_imdb"][j] >= best_ticket["layer_imdb"][j]
                        and ticket["layer_full_spectrum"][j]
                        >= best_ticket["layer_full_spectrum"][j]
                    )
                    alt_condition = (
                        ticket["layer_db"][j] < best_ticket["layer_db"][j]
                        and ticket["layer_full_spectrum"][j]
                        >= best_ticket["layer_full_spectrum"][j]
                    )
                    if (args.alt and alt_condition) or (
                        not args.alt and loose_condition
                    ):
                        best_ticket["layer_imdb"][j] = ticket["layer_imdb"][j]
                        best_ticket["layer_db"][j] = ticket["layer_db"][j]
                        best_ticket["layer_wimdb"][j] = ticket["layer_wimdb"][j]
                        best_ticket["layer_full_spectrum"][j] = ticket[
                            "layer_full_spectrum"
                        ][j]
                        best_ticket["masks"][target] = ticket["masks"][target]
                        save_flag = True

                        # best_ticket["mask_density"][j] = ticket["masks_density"][j]
            state_dict = get_sparse_state_dict(model.state_dict(), best_ticket["masks"])

            total_explored = sum(v.sum() for k, v in explored_masks.items())
            total_params = sum(v.numel() for k, v in explored_masks.items())
            total_explored = sum(v.sum() for k, v in explored_masks.items())
            total_params = sum(v.numel() for k, v in explored_masks.items())

            density = 0
            total = 0
            for k, v in best_ticket["masks"].items():
                density += v.sum()
                total += v.numel()

            avg_db = best_ticket["layer_db"].mean(dim=-1)
            avg_imdb = best_ticket["layer_imdb"].mean(dim=-1)
            avg_wimdb = best_ticket["layer_wimdb"].mean(dim=-1)
            avg_spec = best_ticket["layer_full_spectrum"].mean(dim=-1)
            explored = total_explored / total_params * 100
            current_density = density / total * 100

            print(
                f"mask_no {mask_no}|"
                f" imdb {avg_imdb:.4f}| wimdb {avg_wimdb:.4f}|"
                f" db {avg_db:.4f}|"
                f" spect. {avg_spec:.4f}| expl. {explored:.2f}%|"
                f" density = {current_density:.2f}%|"
                f" last-saved: {last_saved}| to-save: {save_flag}"
            )

            if mask_no % args.mask_sampling == 0:
                if save_flag:
                    th.save(
                        {"state_dict": state_dict, "explored": explored},
                        osp.join(
                            savedir, f"seed_{args.seed}_mask_{mask_no}_step_start.pth"
                        ),
                    )
                    save_flag = False
                    last_saved = mask_no


if __name__ == "__main__":
    args = parser()
    timestr = time.strftime("%Hh%Mm%Ss_on_%b_%d_%Y")
    subdirname = f"population-{args.mask_population}_sampling-{args.mask_sampling}_iter-{args.num_iteration}"
    savedir = os.path.join(
        args.output_dir, args.sparse_init, args.model, str(args.density), subdirname
    )
    args.savedir = savedir
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "logs"), exist_ok=True)
    writer = TensorboardXTracker(os.path.join(savedir, "logs"))
    main()
