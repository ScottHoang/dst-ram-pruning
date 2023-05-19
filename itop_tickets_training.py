from __future__ import print_function

import argparse
import copy
import hashlib
import logging
import os
import os.path as osp
import time
import typing as typ
import warnings

import torch
import torch as th
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision  # type: ignore[import]
import torchvision.transforms as transforms  # type: ignore[import]

import sparselearning
from sparselearning import ramanujan
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
from sparselearning.utils import get_tinyimagenet_dataloaders
from sparselearning.utils import get_cifar10_dataloaders
from sparselearning.utils import get_dense_state_dict
from sparselearning.utils import get_mnist_dataloaders
from sparselearning.utils import get_sparse_state_dict
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


def get_ramanujan_scores(model, fn=th.abs, use_grad=False, **kwargs):
    criteria = ramanujan.Ramanujan()
    num_layers = 0
    state_dict = model.state_dict()
    for m in state_dict:
        if m.endswith("orig"):
            num_layers += 1

    score_imdb = th.zeros(1, num_layers)  # just the mask
    score_full_imdb = th.zeros(1, num_layers)  # just the mask
    score_weighted_imdb = th.zeros(1, num_layers)
    score_full_weighted_imdb = th.zeros(1, num_layers)
    score_narc = th.zeros(1, num_layers)  # just the mask
    score_full_narc = th.zeros(1, num_layers)  # just the mask
    score_grad_imdb = th.zeros(1, num_layers)
    score_full_grad_imdb = th.zeros(1, num_layers)

    i = 0
    for name, m in model.named_modules():
        if hasattr(m, "weight_orig"):
            mask = m.weight_mask.data
            weight = m.weight_orig.data

            weight = fn(weight * mask)

            imdb = criteria.iterative_mean_score(mask, weight)
            full_imdb = criteria.full_graph_score(mask, weight)

            score_weighted_imdb[0, i] = imdb[1]
            score_imdb[0, i] = imdb[0]
            score_narc[0, 1] = imdb[2]

            score_full_weighted_imdb[0, i] = full_imdb[1]
            score_full_imdb[0, i] = full_imdb[0]
            score_full_narc[0, i] = full_imdb[2]

            if use_grad:
                assert m.weight_orig.grad is not None
                grad = m.weight_orig.grad.data
                grad = fn(grad * mask)
                # grad = grad * (weight != 0.0).float()

                imdb = criteria.iterative_mean_score(mask, grad)
                full_imdb = criteria.full_graph_score(mask, grad)
                score_grad_imdb[0, i] = imdb[1]
                score_full_grad_imdb[0, i] = full_imdb[1]

            i += 1
    ret = {
        "imdb": score_imdb,
        "full_imdb": score_full_imdb,
        "weighted_imdb": score_weighted_imdb,
        "full_weighted_imdb": score_full_weighted_imdb,
        "narc": score_narc,
        "full_narc": score_full_narc,
        "grad_imdb": score_grad_imdb,
        "full_grad_imdb": score_full_grad_imdb,
    }
    return ret


def generate_mask_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = th.ones_like(module.weight)
            prune.CustomFromMask.apply(module, "weight", mask)
    return model


def load_state_dict(file):
    data = torch.load(file)
    data["state_dict"] = get_dense_state_dict(data["state_dict"])
    return data


def cumulate_gradients(model, device, loader, optimizer, **kwargs):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        loss.backward()
    # train_loss /= float(n)
    return train_loss, correct / float(n)


def generate_characteristics(model, file, **kwargs):
    data = load_state_dict(file)
    state_dict = data["state_dict"]
    model.load_state_dict(state_dict)
    check_sparsity(model)
    if kwargs.get("use_grad", False):
        print("cumulating gradients for analysis")
        cumulate_gradients(model, **kwargs)
    return get_ramanujan_scores(model, **kwargs), data["epoch"]


def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            sum_list = sum_list + float(m.weight_mask.nelement())
            zero_sum = zero_sum + float(th.sum(m.weight_mask == 0))
    print("* remain weight = ", 100 * (1 - zero_sum / sum_list), "%")

    return 100 * (1 - zero_sum / sum_list)


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
    parser.add_argument("--skip-exist-partial", action="store_true")
    parser.add_argument("--skip-exist-full", action="store_true")

    parser.add_argument("--from-init", action="store_true")
    parser.add_argument("--use-grad", action="store_true")

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
    #
    files = list(filter(lambda x: "seed" in x, os.listdir(args.savedir)))
    seeds = set(x.split("_")[1] for x in files)

    print(f"number of seed {len(seeds)}")

    for seed in seeds:
        torch.manual_seed(args.seed)
        seed_file = list(filter(lambda x: x.split("_")[1] == seed, files))
        num_masks = set()
        for file in seed_file:
            if file.split("_")[3] in ("final", "-1"):
                continue
            num_masks.add(int(file.split("_")[3]))
        num_masks = list(num_masks)
        # num_masks = list(set(int(x.split("_")[3]) for x in seed_file))
        num_masks.sort()

        for mask_no in num_masks:
            if mask_no == -1:
                continue  # this is init file

            print(f"for {seed=} number of masks {len(num_masks)}")
            if args.from_init:
                savepath = osp.join(
                    args.savedir,
                    f"seed_{seed}_mask_{mask_no}_step_initfinetune.pth",
                )
            else:
                savepath = osp.join(
                    args.savedir, f"seed_{seed}_mask_{mask_no}_step_finetune.pth"
                )

            if os.path.isfile(savepath) and args.skip_exist_full:
                print(f"{savepath} exist! skipping this one completely")
                continue

            #
            # print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

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
                train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(
                    args
                )
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

            model = generate_mask_parameters(model)
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
                optimizer = optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.l2
                )
            else:
                print("Unknown optimizer: {0}".format(args.optimizer))
                raise Exception("Unknown optimizer.")

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(args.epochs / 2) * args.multiplier,
                    int(args.epochs * 3 / 4) * args.multiplier,
                ],
                last_epoch=-1,
            )

            mask = None

            if args.from_init:
                savepath = osp.join(
                    args.savedir,
                    f"seed_{seed}_mask_{mask_no}_step_initfinetune.pth",
                )
            else:
                savepath = osp.join(
                    args.savedir, f"seed_{seed}_mask_{mask_no}_step_finetune.pth"
                )

            print(f"now working on {savepath}")
            partial_characteristics, _ = generate_characteristics(
                model,
                osp.join(args.savedir, f"seed_{seed}_mask_{mask_no}_step_final.pth"),
            )

            sparse_init_weight = None
            if osp.exists(osp.join(args.savedir, f"seed_{seed}_mask_-1_step_init.pth")):
                anchor_dense_weight = load_state_dict(
                    osp.join(args.savedir, f"seed_{seed}_mask_-1_step_init.pth")
                )["state_dict"]
                sparse_init_weight = model.state_dict()
                for k, v in sparse_init_weight.items():
                    if k in anchor_dense_weight:
                        sparse_init_weight[k] = anchor_dense_weight[k]
                    elif k.endswith("_orig"):
                        sparse_init_weight[k] = anchor_dense_weight[k[0:-5]]
                model.load_state_dict(sparse_init_weight)
                anchor_characteristics = get_ramanujan_scores(model)
            else:
                assert args.from_init is False
                anchor_characteristics = None

            initial_characteristics, discovered_epoch = generate_characteristics(
                model,
                osp.join(args.savedir, f"seed_{seed}_mask_{mask_no}_step_start.pth"),
                device=device,
                loader=valid_loader,
                optimizer=optimizer,
                fn=th.abs,
                use_grad=args.use_grad,
            )
            if args.from_init:
                assert sparse_init_weight is not None
                print(f"loading weight from init with mask {mask_no}")
                model.load_state_dict(sparse_init_weight)

            if not osp.exists(savepath) or not args.skip_exist_partial:

                best_acc = 0.0
                for epoch in range(1, args.epochs * args.multiplier + 1):
                    t0 = time.time()
                    train(model, device, train_loader, optimizer, epoch)
                    lr_scheduler.step()
                    if args.valid_split > 0.0:
                        val_acc = evaluate(model, device, valid_loader)
                        writer.log_validation_acc(val_acc, epoch)
                    if val_acc > best_acc:
                        print("Saving model")
                        best_acc = val_acc
                        torch.save(
                            {
                                "state_dict": get_sparse_state_dict(model),
                                "initial_characteristics": initial_characteristics,
                                "partial_characteristics": partial_characteristics,
                                "anchor_characteristics": anchor_characteristics,
                                "valdiation_acc": best_acc,
                                "discovered_epoch": discovered_epoch,
                            },
                            savepath,
                        )

                    print_and_log(
                        "Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n".format(
                            optimizer.param_groups[0]["lr"], time.time() - t0
                        )
                    )
                    writer.log_scalar("best-acc", best_acc, epoch)

            print(f"Testing model {savepath}")
            ckpt = load_state_dict(savepath)
            model.load_state_dict(ckpt["state_dict"])
            test_acc = evaluate(model, device, test_loader, is_test_set=True)
            try:
                final_characteristics = get_ramanujan_scores(model)
            except:
                final_characteristics = None
            torch.save(
                {
                    "state_dict": get_sparse_state_dict(model),
                    "initial_characteristics": initial_characteristics,
                    "partial_characteristics": partial_characteristics,
                    "anchor_characteristics": anchor_characteristics,
                    "final_characteristics": final_characteristics,
                    "test_acc": test_acc,
                    "valdiation_acc": ckpt["valdiation_acc"],
                    "discovered_epoch": discovered_epoch,
                },
                savepath,
            )
            # ckpt["final_characteristics"] = final_characteristics
            # ckpt["test_acc"] = test_acc
            # th.save(ckpt, savepath)


if __name__ == "__main__":
    args = parser()
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
