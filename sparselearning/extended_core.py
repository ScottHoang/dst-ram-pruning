import collections
import copy
import os

import numpy as np
import torch as th
from tqdm import tqdm as tqdm

from .core import Masking
from .PAI import pruning_utils as prune
from .ramanujan import Ramanujan


def prune_loop(
    model, loss, pruner, dataloader, device, density, scope, epochs, train_mode=False
):

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        sparse = 1 - density ** ((epoch + 1) / epochs)
        pruner.mask(sparse, scope)


def get_sparse_model(prunemethod, model, loss, dataloader, density, device):
    model = copy.deepcopy(model)
    if prunemethod in ("SynFlow", "iterSNIP"):
        iteration = 100
        if prunemethod == "iterSNIP":
            prunemethod = "SNIP"
    else:
        iteration = 1
    pruner = eval(f"prune.{prunemethod}")(prune.generate_mask_parameters(model))
    prune_loop(
        model,
        loss,
        pruner,
        dataloader,
        device,
        density,
        "global",
        iteration,
    )
    prune.check_sparsity(model)

    masks = {}
    for k, v in model.state_dict().items():
        if k.endswith("_mask"):
            masks[k[:-5]] = v
    return masks


class ExtendedMasking(Masking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.using_ramanujan = self.args.ramanujan
        self.use_full_graph = self.args.ramanujan_full_graph
        self.max_retry = self.args.ramanujan_max_try
        self.ramanujan_criteria = Ramanujan(self.args.ramanujan_soft)
        self.mask_no = 0
        self.init_weights = {}
        self.layer_imdb = th.zeros(100)
        self.avg_imdb = float("-inf")
        self.angular_dist = float("-inf")
        self.fired_weights = []
        self.fired_weights_slopes = []
        self.plateau_window = self.args.plateau_window
        self.plateau_threshold = self.args.plateau_threshold
        self.imdb_tolerance = 0.25

        self.dataloader = kwargs.get("dataloader", None)
        self.device = kwargs.get("device", 0)
        self.criterion = kwargs.get("criterion", None)

        self.density_threshold = 0.4
        # self.name2nonzeros = collections.defaultdict(int)
        # self.name2zeros = collections.defaultdict(int)
        # self.num_remove = collections.defaultdict(int)
        self.feed_forward_anchor_weight = kwargs.get("feed-forward", False)

    def init(self, *args, **kwargs):
        for module in self.modules:
            for name, weight in module.named_parameters():
                self.init_weights[name] = weight.detach().clone()
        # saving the anchor weights
        path = os.path.join(
            self.args.savedir, f"seed_{self.args.seed}_mask_-1_step_init.pth"
        )

        th.save({"state_dict": self.init_weights, "epoch": -1, "step": -1}, path)

        if kwargs["mode"] in ("ERK", "uniform", "resume", "lottery_ticket", "GMP"):
            super().init(*args, **kwargs)
        else:
            assert self.dataloader is not None
            assert self.criterion is not None
            masks = get_sparse_model(
                kwargs["mode"],
                self.modules[0],
                self.criterion,
                self.dataloader,
                kwargs["density"],
                self.device,
            )
            self.density = kwargs["density"]
            for k in self.masks:
                if k in masks:
                    self.masks[k] = masks[k]

            self.apply_mask(new_mask=True)
            self.fired_masks = copy.deepcopy(self.masks)  # used for ITOP
            # self.print_nonzero_counts()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print("Total Model parameters:", total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print(
                "Total parameters under sparsity level of {0}: {1}".format(
                    self.density, sparse_size / total_size
                )
            )

        self.layer_imdb = th.zeros(len(self.masks))
        self.save_mask()

    def _step(self, epoch):
        if self.steps % self.prune_every_k_steps != 0:
            return False

        self.save_mask(step="final", epoch=epoch)

        if not self.using_ramanujan:
            super()._step(epoch)
            plateau = self.is_plateau()
            print(
                f"{epoch=} {self.mask_no=} {plateau=}, slope={self.fired_weights_slopes[-1]}"
            )
            plateau = False  #  for vanilla mode; i want to fully explore

        else:
            self.iterative_truncate_weights(self.max_retry)
            self.apply_mask(True)
            _, total_fired_weights = self.fired_masks_update()
            self.fired_weights.append(total_fired_weights * 100)
            # ^ this is in percentage
            plateau = self.is_plateau()
            print(
                f"{epoch=} {self.mask_no=} {plateau=}, slope={self.fired_weights_slopes[-1]}"
            )
            if self.writer is not None:
                self.writer.log_scalar(
                    "exploration/total-fired-weight", total_fired_weights, epoch
                )
                self.writer.log_scalar(
                    "exploration/total-fired-weight-slope",
                    self.fired_weights_slopes[-1],
                    epoch,
                )
            self.print_nonzero_counts()

        self.mask_no += 1
        if not plateau:
            self.save_mask(step="start", epoch=epoch)
        return plateau

    def _generate_ticket_criteria(self, masks, modules, targets):
        cnt = 0
        layer_imdb = th.zeros(len(targets)).cuda()
        mask_density = th.zeros(len(targets)).cuda()
        # layer_imdb = {}

        for module in modules:
            for name, weight in module.named_parameters():
                if name in targets:
                    imdb = self.ramanujan_criteria._iterative_mean_score(
                        masks[name], masks[name]
                    )
                    layer_imdb[cnt] = imdb[0]  # updating layer_imdb
                    mask_density[cnt] = masks[name].sum() / masks[name].numel()
                if name in self.masks:
                    cnt += 1

        layer_imdb = inf_to_zero(layer_imdb)
        avg_imdb = layer_imdb.mean(dim=-1)
        # exclusion_mask = mask_density <= self.density_threshold
        # avg_imdb = (layer_imdb * exclusion_mask).sum() / exclusion_mask.sum()

        return {
            "layer_imdb": layer_imdb,
            "avg_imdb": avg_imdb,
            "mask_density": mask_density,
        }

    def iterative_truncate_weights(self, max_retry=10):
        targets = list(self.masks.keys())
        name2idx = {k: i for i, k in enumerate(targets)}
        cnt = 0

        current_best_ticket: dict = None
        for i in range(max(1, max_retry)):
            # obtain new future settings
            (
                _new_masks,
                _num_remove,
                _name2zeros,
                _name2nonzeros,
            ) = self._truncate_weights(targets=targets)
            # characterize the current settings
            criteria = self._generate_ticket_criteria(_new_masks, self.modules, targets)
            ticket = {
                "masks": _new_masks,
                "num_remove": _num_remove,
                "name2zeros": _name2zeros,
                "name2nonzeros": _name2nonzeros,
            }
            ticket.update(criteria)

            if current_best_ticket is None:
                current_best_ticket = ticket
                continue

            for j, target in enumerate(targets):
                if ticket["layer_imdb"][j] >= current_best_ticket["layer_imdb"][j]:
                    current_best_ticket["layer_imdb"][j] = ticket["layer_imdb"][j]
                    for key in (
                        "num_remove",
                        "name2zeros",
                        "name2nonzeros",
                        "masks",
                    ):
                        current_best_ticket[key][target] = ticket[key][target]
            print(
                f"ticket is {i}: imdb {current_best_ticket['layer_imdb'].mean(dim=-1)}"
            )

        assert current_best_ticket is not None

        # if self.mask_no > 0:
        # for j, target in enumerate(targets):
        # if current_best_ticket["layer_imdb"][j] < self.layer_imdb[j]:
        # current_best_ticket["layer_imdb"][j] = self.layer_imdb[j]
        # for key in (
        # "num_remove",
        # "name2zeros",
        # "name2nonzeros",
        # "masks",
        # ):
        # attr = getattr(self, key)
        # current_best_ticket[key][target] = attr[target]

        for k, v in current_best_ticket.items():
            setattr(self, k, v)

        # global_ticket = self._generate_ticket_criteria(
        # self.masks, self.modules, targets
        # )
        # for j, target in enumerate(targets):
        # if global_ticket["layer_imdb"][j] < current_best_ticket["layer_imdb"][j]:
        # self.layer_imdb[j] = current_best_ticket["layer_imdb"][j]
        # for key in (
        # "num_remove",
        # "name2zeros",
        # "name2nonzeros",
        # "masks",
        # ):
        # setting = getattr(self, key)
        # setting[target] = current_best_ticket[key][target]

    def get_sparse_state_dict(self):
        self.apply_mask()

        assert len(self.modules) == 1, print(
            "currently we are considering only a single model"
        )
        model = self.modules[0]

        state_dict = {}
        for k, v in model.state_dict().items():
            if k in self.masks:
                state_dict[k + "_orig"] = v.to_sparse()
                state_dict[k + "_mask"] = self.masks[k].to_sparse()
            else:
                state_dict[k] = v

        return state_dict

    def save_mask(self, step: str = "start", epoch: int = 0):
        args = self.args
        state_dict = self.get_sparse_state_dict()
        layer_fired_weights, total_fired_weights = self.fired_masks_update(
            verbose=False
        )
        path = os.path.join(
            args.savedir, f"seed_{args.seed}_mask_{self.mask_no}_step_{step}.pth"
        )
        th.save(
            {
                "state_dict": state_dict,
                "epoch": epoch,
                "step": self.steps,
                "layer_fired_weights": layer_fired_weights,
                "total_fired_weights": total_fired_weights,
            },
            path,
        )

    def is_plateau(self):
        slope, boolean = detect_plateau(
            self.fired_weights,
            self.plateau_window,
            self.plateau_threshold,
        )
        self.fired_weights_slopes.append(slope)
        return boolean

    def pprint(self, *args, verbose=True):
        if verbose:
            print(*args)

    def fired_masks_update(self, verbose=True):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        cnt = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                self.fired_masks[name] = (
                    self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                )
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(
                    self.fired_masks[name].sum().item()
                ) / float(self.fired_masks[name].numel())

                self.pprint(
                    "Layerwise percentage of the fired weights of",
                    name,
                    "is:",
                    f"{layer_fired_weights[name]:.4f} ",
                    f"imdb:{self.layer_imdb[cnt]:.4f}",
                    verbose=verbose,
                )
                cnt += 1
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        avg_imdb = self.layer_imdb.mean(dim=-1)
        self.pprint(
            "The percentage of the total fired weights is:",
            total_fired_weights,
            verbose=verbose,
        )
        self.pprint(f"The new average Imdb: {avg_imdb:.4f}", verbose=verbose)
        self.pprint(f"The new angular dist: {self.angular_dist:.4f}", verbose=verbose)
        return layer_fired_weights, total_fired_weights

    def apply_mask(self, new_mask=False):
        new_mask = new_mask and self.feed_forward_anchor_weight
        super().apply_mask(new_mask)


def detect_plateau(seq, window, threshold):
    """
    Detects a plateau in the vector x within a certain window period.

    Arguments:
    seq -- a list or numpy array of values
    window -- an integer specifying the size of the window
    threshold -- a float specifying the maximum slope for a plateau

    Returns:
    A boolean value indicating whether a plateau was detected.
    """
    if len(seq) < window:
        return 0, False

    # for i in range(len(seq) - window + 1):
    window_slice = seq[-window::]  # i + window]
    slope = (window_slice[-1] - window_slice[0]) / window

    if abs(slope) <= threshold:
        return slope, True

    return slope, False


def inf_to_zero(batch):
    batch[batch == float("inf")] = 0
    return batch


def get_angular_dist(current_character, anchor_character):
    if current_character is None:
        return float("-inf"), float("-inf"), float("-inf")
    cos = (
        current_character
        @ anchor_character
        / (current_character.norm() * anchor_character.norm())
    )
    l2 = (current_character - anchor_character).norm()
    angular_dist = cos * l2
    return angular_dist, cos, l2
