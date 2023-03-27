import collections
import math
import statistics as stats
import typing as typ

import numpy as np
import scipy as sp  # type: ignore[import]
import torch as th
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_geometric as pyg  # type: ignore[import]
import tqdm  # type: ignore[import]
from scipy.sparse import coo_array  # type: ignore[import]


class Ramanujan:

    """Ramanujan base PaI pruning approach"""

    def __init__(self, softness: float = 0.0):
        self.ramanujan_softness = softness
        # a hyper-param in which we control the toughness of ram constraint

    # self.model = self.model.to(device)

    def __call__(self, mask: th.Tensor, weight: th.Tensor, full_graph: bool = False):
        if full_graph:
            return self.full_graph_score(mask, weight)
        else:
            return self.iterative_mean_score(mask, weight)

    def full_graph_score(
        self, mask: th.Tensor, weight: th.Tensor
    ) -> typ.Tuple[float, float, float, float]:
        outc = mask.shape[0]

        flatten_mask = mask.reshape(outc, -1).T
        # shape is [ (in_c x h x w) x out_c]
        # shape is [in_c x out_c] -> linear
        flatten_weight = weight.reshape(outc, -1).T

        total_nodes = (flatten_mask.sum(dim=-1) > 0.0).float().sum() + (
            flatten_mask.sum(dim=0) > 0.0
        ).float().sum()

        layer_score, mixing_lem, _ = self.ramanujan_bounds(flatten_mask, total_nodes)

        weighted_layer_score, weighted_mixing_lem, _ = self.ramanujan_bounds(
            flatten_weight, total_nodes, True
        )
        return layer_score, weighted_layer_score, mixing_lem, weighted_mixing_lem

    def iterative_mean_score(
        self, mask: th.Tensor, weight: th.Tensor
    ) -> typ.Tuple[float, float, float, float]:
        # shape = mask.shape
        # outc, inc = shape[0], shape[1]

        outc = mask.shape[0]

        flatten_mask = mask.reshape(outc, -1).T
        flatten_weight = weight.reshape(outc, -1).T

        fan_out_index = flatten_mask.sum(dim=1)
        degree_out_lut = collections.Counter(fan_out_index.tolist())

        layer_score: typ.List[float] = []  # layer score
        weighted_layer_score: typ.List[float] = []  # weighted layer score

        mixing_lem: typ.List[float] = []
        weighted_mixing_lem: typ.List[float] = []

        total_nodes = (flatten_mask.sum(dim=-1) > 0.0).float().sum() + (
            flatten_mask.sum(dim=0) > 0.0
        ).float().sum()

        for degree_out, num_nodes in degree_out_lut.items():
            if degree_out == 0 or num_nodes < 2:
                continue
            index = fan_out_index == degree_out
            submask = flatten_mask[index].T  # out_c x in_c
            submask_weight = flatten_weight[index].T
            degree_bound, randomness, _ = self.ramanujan_bounds(submask, total_nodes)

            w_degree_bound, w_mix_lemma, _ = self.ramanujan_bounds(
                submask_weight, total_nodes, True
            )

            input_vertices = index.sum()
            output_vertices = (submask.sum(dim=1) > 0).sum()
            cheeger = max(input_vertices, output_vertices) / (
                input_vertices + output_vertices - 1
            )

            if degree_bound == float("inf"):
                # invalid type
                continue
            layer_score.append(cheeger * degree_bound)
            weighted_layer_score.append(cheeger * w_degree_bound)
            mixing_lem.append(randomness)
            weighted_mixing_lem.append(w_mix_lemma)

        def mean(seq):
            if len(seq) == 0:
                return 0
            return sum(seq) / len(seq)

        return (
            mean(layer_score),
            mean(weighted_layer_score),
            mean(mixing_lem),
            mean(weighted_mixing_lem),
        )

    @staticmethod
    def get_biparite_graph(
        tensor: th.Tensor,
    ) -> typ.Tuple[th.Tensor, typ.Dict[str, typ.Any]]:
        """generate bipartite graph adj matrix from tensor"""

        org_shape = tensor.shape

        mask = (tensor != 0.0).float()  # inverted sparsity mask
        mask = mask.reshape(mask.size(0), -1).T
        tensor = tensor.reshape(tensor.size(0), -1).T

        in_dim = mask.size(0)
        out_dim = mask.size(1)

        tensor = tensor[mask.sum(-1) > 0, :]
        tensor = tensor[:, mask.T.sum(-1) > 0]
        mask = mask[mask.sum(-1) > 0, :]
        mask = mask[:, mask.T.sum(-1) > 0]

        left_nodes, right_nodes = mask.shape
        num_nodes = left_nodes + right_nodes

        bigraph = th.zeros(num_nodes, num_nodes).to(mask.device)
        bigraph[0:left_nodes, left_nodes::] = tensor
        bigraph[left_nodes::, 0:left_nodes] = tensor.T

        meta = {
            "num_nodes": num_nodes,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "left_nodes": left_nodes,
            "right_nodes": right_nodes,
            "orignal_shape": org_shape,
        }
        return bigraph, meta  # num_nodes, out_dim, in_dim

    def ramanujan_bounds(
        self, tensor: th.Tensor, total_nodes: th.Tensor, is_weight: bool = False
    ) -> typ.Tuple[float, float, typ.Dict[str, typ.Any]]:
        """Calculate the Ramanujan bound"""
        bigraph, meta = Ramanujan.get_biparite_graph(tensor)

        edge_index = bigraph.to_sparse().coalesce().indices()
        degree = pyg.utils.degree(edge_index[0, :], meta["num_nodes"])
        avg_deg_left = degree[0 : meta["left_nodes"]].mean()
        avg_deg_right = degree[meta["left_nodes"] : :].mean()

        if avg_deg_left >= 3 and avg_deg_right > 1:

            eigs = Ramanujan.get_eig_values(bigraph)
            if is_weight:
                degree_bound = eigs[0] - eigs[-1]  # vD
            else:
                degree_bound = (
                    math.sqrt(avg_deg_left - 1)
                    + math.sqrt(avg_deg_right - 1)
                    + self.ramanujan_softness
                    - eigs[-1]
                )
            # avg_degree = degree.mean()

            expander_mixxing = eigs[-1] * math.sqrt(
                meta["left_nodes"] * meta["right_nodes"]
            ) - abs(
                edge_index.size(1)
                - avg_deg_left * meta["left_nodes"] * meta["right_nodes"] / total_nodes
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
            expander_mixxing = float("inf")

        meta["avg_deg_left"] = avg_deg_left.item()
        meta["avg_deg_right"] = avg_deg_right.item()

        return degree_bound, expander_mixxing, meta

    @staticmethod
    def get_eig_values(matrix: th.Tensor, k: int = 3) -> typ.List[float]:
        """
        get the real eig of a square matrix
        for bi-graph, the third largest eig denotes connectivity
        """
        matrix = coo_array(matrix.cpu())
        adj_eigh_val = sp.sparse.linalg.eigs(matrix, k=k, which="LM")[0].tolist()
        abs_eig = [abs(i) for i in adj_eigh_val]
        abs_eig.sort(reverse=True)
        return abs_eig
