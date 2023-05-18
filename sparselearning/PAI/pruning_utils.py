import collections
import copy
import math
import typing as typ
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy as sp  # type: ignore[import]
import torch as th
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_geometric as pyg  # type: ignore[import]
import tqdm  # type: ignore[import]
from scipy.sparse import coo_array  # type: ignore[import]


th.multiprocessing.set_sharing_strategy("file_system")

# from layers import Conv2d
# from layers import Linear

__all__ = [
    "generate_mask_parameters",
    "SynFlow",
    "Mag",
    "Taylor1ScorerAbs",
    "Rand",
    "RandGlobal",
    "SNIP",
    "GraSP",
    "check_sparsity",
    "check_sparsity_dict",
    "prune_model_identity",
    "prune_model_custom",
    "extract_mask",
    "ERK",
    "PHEW",
    "Ramanujan",
    "get_coo_state_dict",
]


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask."""
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def generate_mask_parameters(model, global_mask, exception=None):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for name, module in model.named_modules():
        if exception is not None and exception in name:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = th.ones_like(module.weight)
            prune.CustomFromMask.apply(module, "weight", mask)
            if global_mask is not None:
                module.weight_mask.copy_(global_mask[f"{name}.weight"])
            yield module.weight_mask, module.weight_orig


class Pruner:
    def __init__(self, masked_parameters: typ.Iterator[typ.List[th.Tensor]]):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}  # type: ignore[var-annotated]

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally."""
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = th.cat([th.flatten(v) for v in self.scores.values()])
        k = int((1 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold = global_scores.topk(k)[0][-1]
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = th.tensor([0.0]).to(mask.device)
                one = th.tensor([1.0]).to(mask.device)
                mask.copy_(th.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise."""
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = th.kthvalue(th.flatten(score), k)
                zero = th.tensor([0.0]).to(mask.device)
                one = th.tensor([1.0]).to(mask.device)
                mask.copy_(th.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope."""
        if scope == "global":
            self._global_mask(sparsity)
        if scope == "local":
            self._local_mask(sparsity)

    @th.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters."""
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, _ in self.masked_parameters:
            shape = mask.shape
            perm = th.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters."""
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):
        @th.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = th.sign(param)
                param.abs_()
            return signs

        @th.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = th.ones([1] + input_dim).to(device)  # , dtype=th.float64).to(device)
        output = model(input)
        th.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, *args, **kwargs): 
        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.clone(p.data).detach().abs_()


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.randn_like(p)


class RandGlobal(Pruner):
    def __init__(self, masked_parameters):
        super(RandGlobal, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.rand_like(p) * p.abs()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):

        scaler = kwargs.get("scaler", None)

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _loss = loss(output, target)
            if scaler:
                scaler.scale(_loss).backward()
            else:
                _loss.backward()

            if batch_idx > num_iteration > 0:
                break

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = th.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = th.cat([th.flatten(v) for v in self.scores.values()])
        norm = th.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = th.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=False
            )
            flatten_grads = th.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

            if batch_idx > num_iteration > 0:
                break

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = th.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=True
            )
            flatten_grads = th.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
            if batch_idx > num_iteration > 0:
                break

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = th.cat([th.flatten(v) for v in self.scores.values()])
        norm = th.abs(th.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class Taylor1ScorerAbs(Pruner):
    def __init__(self, masked_parameters):
        super(Taylor1ScorerAbs, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device, num_iteration=-1, **kwargs):

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()
            if batch_idx > num_iteration > 0:
                break

        for _, p in self.masked_parameters:
            self.scores[id(p)] = th.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()


class ERK(Pruner):
    """ERK pruning method

    code is copy and modified from snippet in FreeTicket github
    """

    def __init__(self, masked_parameters):
        """TODO: to be defined.

        :masked_parameters: TODO

        """
        Pruner.__init__(self, masked_parameters)

    def score(self, *args, **kwargs):
        pass

    def mask(self, sparsity, scope=None):
        r"""Updates masks of model with scores by sparsity according to scope."""
        total_params = 0
        for mask, weight in self.masked_parameters:
            total_params += weight.numel()
        is_epsilon_valid = False
        erk_power_scale = 1.0
        dense_layers = set()
        density = 1 - sparsity
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, (mask, params) in enumerate(self.masked_parameters):
                n_param = np.prod(params.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                        np.sum(mask.shape) / np.prod(mask.shape)
                    ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True
        self.density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, (mask, params) in enumerate(self.masked_parameters):
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                self.density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                self.density_dict[name] = probability_one
            mask.data.copy_((th.rand(mask.shape) < self.density_dict[name]).float())
            total_nonzero += self.density_dict[name] * mask.numel()

        # print(f"{epsilon=}")
        # print(f"{raw_probabilities=}")
        # print(f"{self.density_dict=}")


class PHEW(Pruner):
    """Docstring for PHEW."""

    def __init__(self, masked_parameters):
        """TODO: to be defined.

        :masked_parameters: TODO

        """
        Pruner.__init__(self, masked_parameters)

    def score(self, *args, **kwargs):
        pass

    def mask(self, sparsity, scope=None):
        parameters = [mask[1] for mask in self.masked_parameters]
        prob, reverse_prob, kernel_prob = phew_utils.generate_probability(parameters)

        weight_masks, bias_masks = phew_utils.generate_masks(
            [th.zeros_like(p) for p in parameters]
        )

        prune_perc = sparsity * 100
        weight_masks, bias_masks = phew_utils.phew_masks(
            parameters,
            prune_perc,
            prob,
            reverse_prob,
            kernel_prob,
            weight_masks,
            bias_masks,
            verbose=True,
        )
        for i, (m, _) in enumerate(self.masked_parameters):
            m.data.copy_(weight_masks[i].data)


class Ramanujan(ERK):

    """Ramanujan base PaI pruning approach"""

    def __init__(self, masked_parameters):
        """TODO: to be defined."""
        super().__init__(masked_parameters)
        self.top_masks: typ.List[th.Tensor] = None
        self.top_score: float = 0.0

    @staticmethod
    @th.no_grad()
    def linearize(model):
        # model.double()
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = th.sign(param)
            param.abs_()
        return signs

    @staticmethod
    @th.no_grad()
    def nonlinearize(model, signs):
        # model.float()
        for name, param in model.state_dict().items():
            if "_mask" in name:
                continue
            param.mul_(signs[name])

    def forward(self, model, dataloader, device):
        model.zero_grad()
        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        _input = th.ones([1] + input_dim).to(device)  # , dtype=th.float64).to(device)
        output = model(_input)
        th.sum(output).backward()
        # for batch_idx, (data, target) in enumerate(dataloader):
        # data, target = data.to(device), target.to(device)
        # output = model(data)  # / self.temp
        # L = self.loss(output, target)
        # L.backward()

    def score(self, model, loss, dataloader, device):
        self.model = model
        self.loss = loss
        self.dataloader = dataloader
        self.device = device

        # self.model = self.model.to(device)

    def mask(self, sparsity, scope=None):

        old_masks: typ.Optional[typ.List[th.Tensor]] = None
        dense_grads: typ.Optional[typ.List[th.Tensor]] = None
        old_mask_grads: typ.Optional[typ.List[th.Tensor]] = None

        if not hasattr(self, "density_dict"):
            self.forward(self.model, self.dataloader, self.device)
            self.dense_grads = [
                w.grad.data.norm() / w.numel() for _, w in self.masked_parameters
            ]
            self.model.zero_grad()
            super().mask(sparsity, scope)  # get initial random mask from ERK
            # self.forward(self.model, self.dataloader, self.device)
            # old_mask_grads = [
            # w.grad.data.norm() / m.sum() for m, w in self.masked_parameters
            # ]
            # for mask, weight in self.masked_parameters:
            # mask.data.copy_(th.ones_like(mask))
            # self.forward(self.model, self.dataloader, self.device)
            # dense_grads = [
            # w.grad.data.norm() / w.numel() for _, w in self.masked_parameters
            # ]
            # # first time running mask
            # print(
            # f"norm of diff of previous mask {(th.Tensor(dense_grads) - th.Tensor(old_mask_grads)).norm()}"
            # )
        dense_grads = self.dense_grads

        # else:
        # subsequent time running mask
        old_masks = [m.data.clone() for m, _ in self.masked_parameters]
        self.forward(self.model, self.dataloader, self.device)
        old_mask_grads = [
            w.grad.data.norm() / m.sum() for m, w in self.masked_parameters
        ]

        # for mask, weight in self.masked_parameters:
        # mask.copy_(th.ones_like(mask))

        # for old_mask_data, (mask, weight) in zip(old_masks, self.masked_parameters):
        # mask.data.copy_(old_mask_data)

        print(
            f"difference of norm: {(th.Tensor(dense_grads) - th.Tensor(old_mask_grads)).norm()}"
        )

        input_bias: typ.Optional[th.Tensor] = None
        signs = self.linearize(self.model)

        for name, (mask, weight) in enumerate(self.masked_parameters):
            print(f"#########{name=}##########")

            if (erk_density := self.density_dict[name]) < 1.0:

                num_edges = int(erk_density * math.prod(mask.shape))
                shape = mask.shape

                self.forward(self.model, self.dataloader, self.device)

                # bias = th.clone(weight.grad * weight).detach().cpu().abs_()

                weighted_input_bias = self.get_weighted_input_bias(
                    input_bias, weight.grad
                )

                bias = weight.grad.clone().abs()

                if len(shape) == 4:
                    if math.prod(shape[2::]) > 1:
                        new_mask, input_bias = self.generate_convolutional_bias_mask(
                            num_edges, shape, weighted_input_bias, bias
                        )
                    else:
                        weighted_input_bias = self.get_weighted_input_bias(
                            None, weight.grad
                        )
                        new_mask, _ = self.generate_convolutional_bias_mask(
                            num_edges, mask.shape, weighted_input_bias, bias
                        )
                elif len(shape) == 2:
                    # if name < len(self.masked_parameters) - 1:
                    new_mask, input_bias = self.generate_linear_bias_masks(
                        num_edges,
                        mask.shape,
                        weighted_input_bias,
                        bias,
                    )

                old_mask = mask.data.clone()
                mask.copy_(new_mask)
                if old_masks is not None:
                    old_mask_grad = weight.grad.data.detach().norm() / old_mask.sum()
                    self.forward(self.model, self.dataloader, self.device)
                    new_mask_grad = weight.grad.data.detach().norm() / new_mask.sum()
                    if old_mask_grad > new_mask_grad:
                        print(f"\tfail to reduce norm... restore old mask")
                        print(f"\t{old_mask_grad=} {new_mask_grad=}")
                        mask.data.copy_(old_mask)
                        input_bias = (
                            old_masks[name].view(shape[0], -1).sum(dim=-1).squeeze()
                        )
                        input_bias = input_bias.div(input_bias.sum())

                # weight.data.copy_(mask.data)
                out_density = (mask.sum() / mask.numel()).item()
                new_mask_density = (new_mask.sum() / mask.numel()).item()
                print(
                    f"\t{erk_density=} {out_density=} {new_mask_density=} {num_edges=}"
                )
            else:
                input_bias = None  # full-dense layer has no output bias
                num_edges = "full"

            # degree_bound, random_bound, meta = self.ramanujan_bounds(mask)
            # print(
            # f"{name=} {meta=} {density=} {out_density=} {degree_bound=} {random_bound=}"
            # )
        # check_sparsity(self.model)
        self.nonlinearize(self.model, signs)
        # check_sparsity(self.model)

    @staticmethod
    @th.no_grad()
    def get_weighted_input_bias(
        input_bias: typ.Optional[th.Tensor], gradients: th.Tensor
    ) -> th.Tensor:
        if input_bias is None or input_bias.size(0) != gradients.size(1):
            input_bias = th.ones(gradients.size(1)).softmax(dim=0)

        inc = gradients.size(1)
        gradients = th.transpose(gradients.detach().clone(), 0, 1).reshape(inc, -1)
        input_grad_norm = th.norm(gradients, dim=-1)
        input_grad_prob = input_grad_norm / input_grad_norm.sum()

        new_bias = input_bias * input_grad_prob
        return new_bias.div(new_bias.sum())

    def _find_offset(self, shape: typ.List[int], starting_offset: int) -> int:
        """find the largest fix number of filters closed to starting_offset"""
        if len(shape) == 2:
            while shape[1] % starting_offset:
                starting_offset -= 1
            return starting_offset
        else:
            kH, kW = shape[2::]
            return starting_offset // (kH * kW) * (kH * kW)

    def _edge_balance(
        self, edges_per_channels: th.Tensor, numel: int, max_degree: int
    ) -> typ.List[float]:
        """Balance the assigned edges per input channel by max degree.
        Any extra edges will be assigned to the next most populated channels until there is no more
        carry over.
        """
        edges_per_channels = th.round(edges_per_channels)
        max_edges_per_channel = numel * max_degree
        carry_over_edges = edges_per_channels - max_edges_per_channel
        carry_over_edges = carry_over_edges[carry_over_edges > 0].sum()

        if carry_over_edges > 0:

            edges_per_channels[
                edges_per_channels > max_edges_per_channel
            ] = max_edges_per_channel

            val, index = edges_per_channels.topk(k=edges_per_channels.size(0))

            for vol, idx in zip(val, index):
                if carry_over_edges > 0:
                    if vol < max_edges_per_channel:
                        addon = min(max_edges_per_channel - vol, carry_over_edges)
                        edges_per_channels[idx] = vol + addon
                        carry_over_edges -= addon
                else:
                    break
        return edges_per_channels

    @th.no_grad()
    def generate_convolutional_bias_mask(
        self,
        total_edges: int,
        shape: typ.List[int],
        input_bias: th.Tensor,
        weights_bias: th.Tensor,
    ) -> typ.Tuple[th.Tensor, th.Tensor]:
        """genearate a mask contrained by budgets, and ensures all sub-graphs are ramanujan"""
        assert len(shape) == 4
        if input_bias is not None:
            assert len(input_bias) == shape[1]

        out_dim = shape[0]
        in_dim = math.prod(shape[1::])

        submask_offset = math.prod(shape[2::])  # self._find_offset(shape, out_dim // 2)
        # out_dim//2 is  from expander theory |S| \leq |V|//2

        if input_bias is None:
            edges_per_channels = total_edges * th.ones(shape[1]).div(shape[1])
        else:
            edges_per_channels = total_edges * input_bias

        edges_per_channels = self._edge_balance(
            edges_per_channels, submask_offset, out_dim
        )

        mask = th.zeros(in_dim, out_dim)

        output_bias = th.ones(out_dim)
        weights_bias = weights_bias.reshape(out_dim, -1).T

        for i in range(in_dim // submask_offset):
            offset = min(submask_offset, mask.size(0) - i * submask_offset)
            if total_edges > 0 and edges_per_channels[i] > 0:
                max_degree = max(
                    min(math.ceil(edges_per_channels[i] / offset), out_dim),
                    3,
                )
                subset_weights_bias = weights_bias[
                    i * submask_offset : submask_offset * i + offset
                ]
                submask, total_edges, output_bias = self.regular_permutate(
                    max_degree,
                    offset,
                    out_dim,
                    total_edges,
                    output_bias,
                    subset_weights_bias,
                )  # type: ignore[assignment]
                # print(
                # f"{total_edges=} channel{i} edges:{edges_per_channels[i]} {output_bias=}"
                # )

                mask[i * submask_offset : submask_offset * i + offset] = submask
            elif total_edges == 0:
                break

        # assert total_edges <= 0
        output_bias -= 1
        # percentile = np.percentile(output_bias.numpy(), 3)
        output_mask = (output_bias <= math.prod(shape[2::])).bool()
        output_bias[output_mask] = 0.0
        mask[:, output_mask] = 0
        mask = mask.T.reshape(*shape)

        # print(
        # f"{shape=} {edges_per_channels=} {submask_offset=} {output_bias=}\
        # weightnorm={weights_bias.norm()}"
        # )
        return mask, output_bias.div(th.sum(output_bias))

    @th.no_grad()
    def generate_linear_bias_masks(
        self,
        total_edges: int,
        shape: typ.List[int],
        input_bias: typ.Optional[th.Tensor] = None,
        weight_bias: typ.Optional[th.Tensor] = None,
    ) -> typ.Tuple[th.Tensor, None]:
        """genearate a mask contrained by budgets, and ensures all sub-graphs are ramanujan"""
        assert len(shape) == 2

        out_dim = shape[0]
        in_dim = shape[1]

        avg_edges: typ.Optional[float] = None
        edges_per_channels: typ.Optional[typ.List[float]] = None

        if input_bias is None or input_bias.size(0) != in_dim:
            submask_offset = self._find_offset(shape, out_dim // 2)
            avg_edges = total_edges / (in_dim // submask_offset)
        else:
            assert input_bias.size(0) == in_dim, print(
                f"{in_dim=}, {input_bias.size(0)}"
            )
            submask_offset = 1
            edges_per_channels = total_edges * input_bias  # type: ignore[assignment]

            edges_per_channels = self._edge_balance(
                edges_per_channels, submask_offset, out_dim
            )

        # out_dim//2 is  from expander theory |S| \leq |V|//2

        mask = th.zeros(in_dim, out_dim)
        output_bias: typ.Optional[th.Tensor] = None
        if weight_bias is not None:
            weight_bias = weight_bias.T

        for i in range(in_dim // submask_offset):
            offset = min(submask_offset, mask.size(0) - i * submask_offset)
            if total_edges > 0:
                if avg_edges is not None:
                    max_degree = min(math.ceil(avg_edges / offset), out_dim)

                if edges_per_channels is not None:
                    max_degree = min(math.ceil(edges_per_channels[i]), out_dim)

                if weight_bias is not None:
                    subset_weights_bias = weight_bias[
                        i * submask_offset : submask_offset * i + offset
                    ]
                else:
                    subset_weights_bias = None

                if max_degree > 0:
                    submask, total_edges, output_bias = self.regular_permutate(
                        max_degree,
                        offset,
                        out_dim,
                        total_edges,
                        output_bias,
                        subset_weights_bias,
                    )
                    mask[i * submask_offset : submask_offset * i + offset] = submask
            else:
                break

        # assert total_edges <= 0
        output_bias -= 1
        # percentile = np.percentile(output_bias.numpy(), 3)
        # output_mask = (output_bias <= percentile).bool()
        # output_bias[output_mask] = 0.0
        # mask[:, output_mask] = 0
        mask = mask.T.reshape(*shape)

        return mask, output_bias

    def regular_permutate(
        self,
        degree: int,
        leftnodes: int,
        rightnodes: int,
        total_edges: int,
        output_bias: typ.Optional[th.Tensor] = None,
        weight_bias: typ.Optional[th.Tensor] = None,
        iteration: int = 1,
    ) -> typ.Tuple[th.Tensor, int, th.Tensor]:
        """permute the edge distribution until within Ramanujan"""

        if weight_bias is not None:
            # # __import__("pdb").set_trace()
            softweight = weight_bias.mean(dim=0)
            softweight = softweight / softweight.sum()
            softweight = softweight.tile(leftnodes).reshape(leftnodes, rightnodes)

            # softweight = (weight_bias.T / weight_bias.sum(dim=1)).T
            mask = th.isnan(softweight) | th.isinf(softweight)
            softweight[mask] = 0

            false_index = softweight.sum(dim=1) <= 0

            if th.any(false_index):
                softweight[false_index] = th.ones(rightnodes).softmax(dim=0)

        # assert softweight.sum(dim=1).sum().item() == leftnodes, print(
        # f"{softweight=}, {softweight.sum(dim=1).sum()} {leftnodes=}"
        # )

        def randomize():
            if weight_bias is None:
                indices = th.topk(th.rand(leftnodes, rightnodes), degree, dim=-1)[1]
            else:
                indices = th.multinomial(softweight, degree, False)
            return indices

        total_edges -= degree * leftnodes

        if leftnodes > 1:

            ret: th.Tensor
            max_distance = float("-inf")
            for i in range(iteration):
                submask = th.scatter(
                    th.zeros(leftnodes, rightnodes),
                    dim=-1,
                    index=randomize(),
                    value=1.0,
                )
                degree_bound, random_bound, meta = self.ramanujan_bounds(submask.T)
                if degree_bound > max_distance:
                    ret = submask
                    max_distance = degree_bound

        else:
            ret = th.scatter(
                th.zeros(leftnodes, rightnodes), dim=-1, index=randomize(), value=1.0
            )

        if output_bias is not None:
            output_bias += ret.sum(dim=0)
        else:
            output_bias = ret.sum(dim=0)

        return ret, total_edges, output_bias

    @staticmethod
    def get_biparite_graph(
        tensor: th.Tensor,
    ) -> typ.Tuple[th.Tensor, typ.Dict[str, typ.Any]]:
        """generate bipartite graph adj matrix from tensor"""

        org_shape = tensor.shape

        mask = (tensor > 0.0).float()
        mask = mask.reshape(mask.size(0), -1).T

        in_dim = mask.size(0)
        out_dim = mask.size(1)

        mask = mask[mask.sum(-1) > 0, :]
        mask = mask[:, mask.T.sum(-1) > 0]

        left_nodes, right_nodes = mask.shape
        num_nodes = left_nodes + right_nodes

        bigraph = th.zeros(num_nodes, num_nodes).to(mask.device)
        bigraph[0:left_nodes, left_nodes::] = mask
        bigraph[left_nodes::, 0:left_nodes] = mask.T

        meta = {
            "num_nodes": num_nodes,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "left_nodes": left_nodes,
            "right_nodes": right_nodes,
            "orignal_shape": org_shape,
        }
        return bigraph, meta  # num_nodes, out_dim, in_dim

    @staticmethod
    def get_first_eigen(tensor: th.Tensor) -> float:
        """calculate the first eig value

        :tensor: TODO
        :returns: TODO

        """
        bigraph, meta = Ramanujan.get_biparite_graph(tensor)

        edge_index = bigraph.to_sparse().indices()
        degree = pyg.utils.degree(edge_index[0, :], meta["num_nodes"])
        avg_deg_left = degree[0 : meta["left_nodes"]].mean()
        avg_deg_right = degree[meta["left_nodes"] : :].mean()

        return math.sqrt(avg_deg_left * avg_deg_right)

    @staticmethod
    def ramanujan_bounds(
        tensor: th.Tensor,
    ) -> typ.Tuple[float, float, typ.Dict[str, typ.Any]]:
        """Calculate the Ramanujan bound"""
        bigraph, meta = Ramanujan.get_biparite_graph(tensor)

        edge_index = bigraph.to_sparse().coalesce().indices()
        degree = pyg.utils.degree(edge_index[0, :], meta["num_nodes"])
        avg_deg_left = degree[0 : meta["left_nodes"]].mean()
        avg_deg_right = degree[meta["left_nodes"] : :].mean()

        if avg_deg_left >= 3 or avg_deg_right >= 3:

            eigs = Ramanujan.get_eig_values(bigraph)

            degree_bound = (
                math.sqrt(avg_deg_left - 1) + math.sqrt(avg_deg_right - 1) - eigs[-1]
            )

            random_bound = (eigs[-1] ** 2 / 4 + 1) * math.sqrt(
                meta["left_nodes"] * meta["right_nodes"]
            ) - abs(
                edge_index.size(-1)
                - (avg_deg_left / meta["out_dim"])
                * (meta["left_nodes"] * meta["right_nodes"])
            )
            random_bound /= meta["out_dim"] * meta["in_dim"]
        else:
            random_bound = float("inf")
            degree_bound = float("inf")
        meta["avg_deg_left"] = avg_deg_left.item()
        meta["avg_deg_right"] = avg_deg_right.item()

        return degree_bound, random_bound, meta

    @staticmethod
    def get_eig_values(matrix: th.Tensor, k: int = 3) -> typ.List[float]:
        """
        get the real eig of a square matrix
        for bi-graph, the third largest eig denotes connectivity
        """
        matrix = coo_array(matrix.cpu())
        adj_eigh_val, _ = sp.sparse.linalg.eigsh(matrix, k=k, which="LM")
        abs_eig = [abs(i) for i in adj_eigh_val]
        abs_eig.sort(reverse=True)
        return abs_eig


def mp_proportional_sampling(
    bias: np.ndarray, target: int, num_nodes: int
) -> th.Tensor:
    """multi-threaded proportional_sampling"""
    with Pool() as p:
        func = partial(unique_proportional_sampling, target)
        results = list(p.map(func, list(bias)))
    p.join()
    p.close()
    return th.cat(results, dim=0)


def unique_proportional_sampling(target: int, bias: np.ndarray) -> th.Tensor:
    # seen: typ.Set[int] = set()
    # np.random.seed()
    candidates = np.random.choice(
        range(bias.shape[0]), size=target, replace=False, p=bias
    )

    return th.tensor(candidates).unsqueeze(0)


def proportional_sampling(bias: typ.List[float]) -> int:
    """performe individual index proportional sampling"""
    cumsum = [bias[0]]
    for i in range(1, len(bias)):
        cumsum.append(bias[i] + cumsum[i - 1])

    r = np.random.random(1)[0]
    for i, prob in enumerate(cumsum):
        if r <= prob:
            break
    return i


def check_sparsity(model):

    sum_list = 0
    one_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            sum_list = sum_list + float(m.weight_mask.nelement())
            one_sum = one_sum + float(th.sum(m.weight_mask))
    print("* remain weight = ", 100 * one_sum / sum_list, "%")

    return 100 * one_sum / sum_list


def check_sparsity_dict(model_dict):

    sum_list = 0
    zero_sum = 0

    for key in model_dict.keys():
        if "mask" in key:
            sum_list = sum_list + float(model_dict[key].nelement())
            zero_sum = zero_sum + float(th.sum(model_dict[key] == 0))
    print("* remain weight = ", 100 * (1 - zero_sum / sum_list), "%")

    return 100 * (1 - zero_sum / sum_list)


def prune_model_identity(model):

    print("start pruning with identity mask")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("identity pruning layer {}".format(name))
            prune.Identity.apply(m, "weight")


def prune_model_custom(model, mask_dict):

    print("start pruning with custom mask")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("custom pruning layer {}".format(name))
            prune.CustomFromMask.apply(
                m, "weight", mask=mask_dict[name + ".weight_mask"]
            )


def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict


def get_coo_state_dict(state_dict: dict):
    ret = {}
    for k, v in state_dict.items():
        if k.endswith("orig"):
            mask = state_dict[k.replace("orig", "mask")]
            name = k[:-5]  # .replace("_orig", "")
            ret[name] = (v * mask).to_sparse()
    return ret
