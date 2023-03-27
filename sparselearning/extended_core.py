import os

import numpy as np
import torch as th

from .core import Masking
from .ramanujan import Ramanujan


class ExtendedMasking(Masking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.using_ramanujan = self.args.ramanujan
        self.use_full_graph = self.args.ramanujan_full_graph
        self.max_retry = self.args.ramanujan_max_try
        self.ramanujan_criteria = Ramanujan(self.args.ramanujan_soft)
        self.mask_no = 0

        self.fired_weights = []
        self.fired_weights_slopes = []
        self.plateau_window = self.args.plateau_window
        self.plateau_threshold = self.args.plateau_threshold

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.layer_imdb = {k: float("-inf") for k in self.masks}
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
            self.apply_mask()
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

    def iterative_truncate_weights(self, max_retry=10):
        names = list(self.masks.keys())
        cnt = 0
        while len(names) > 0 and cnt < max_retry:
            new_names = []
            mask_copy = {n: self.masks[n].clone() for n in names}
            self.truncate_weights()
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name in names:
                        new_mask = self.masks[name]
                        imdb = self.ramanujan_criteria(
                            new_mask, weight.detach(), self.use_full_graph
                        )[0]

                        if imdb > 0 or imdb > self.layer_imdb[name]:
                            self.layer_imdb[name] = imdb
                            # names.remove(name)
                        else:
                            new_names.append(name)
                            new_mask = mask_copy[name]

                        self.masks[name] = new_mask.float()

            names = new_names
            cnt += 1
        # self.apply_mask()

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
        path = os.path.join(
            args.savedir, f"seed_{args.seed}_mask_{self.mask_no}_step_{step}.pth"
        )

        th.save({"state_dict": state_dict, "epoch": epoch}, path)

    def is_plateau(self):
        slope, boolean = detect_plateau(
            self.fired_weights,
            self.plateau_window,
            self.plateau_threshold,
        )
        self.fired_weights_slopes.append(slope)
        return boolean

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
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

                print(
                    "Layerwise percentage of the fired weights of",
                    name,
                    "is:",
                    f"{layer_fired_weights[name]:.4f} ",
                    f"imdb:{self.layer_imdb[name]:.4f}",
                )
        total_fired_weights = ntotal_fired_weights / ntotal_weights
        avg_imdb = sum(v for k, v in self.layer_imdb.items()) / len(self.layer_imdb)
        print("The percentage of the total fired weights is:", total_fired_weights)
        print(f"The new average Imdb: {avg_imdb:.4f}")
        return layer_fired_weights, total_fired_weights


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
