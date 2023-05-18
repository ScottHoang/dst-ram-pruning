import collections
import math
import statistics as stats
import typing as typ

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch as th
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_geometric as pyg  # type: ignore[import]
import tqdm  # type: ignore[import]
from scipy.sparse import coo_array  # type: ignore[import]

# import cupy as cp


def inf_to_zero(batch):
    batch[batch == float("inf")] = 0
    return batch


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

    def total_spectrum_measurement(
        self, mask: th.Tensor, weight: th.Tensor, return_imdb=False
    ):
        mask = mask.data
        weight = weight.data

        weight = th.abs(weight * mask)

        full_graph_analysis = self.full_graph_score(mask, weight)
        iterative_graphs_analysis = self.iterative_mean_score(mask, weight)

        magnitude = inf_to_zero(
            th.tensor([*full_graph_analysis[0:2], *iterative_graphs_analysis[0:2]])
        ).norm()
        ret = [magnitude]
        if return_imdb:
            ret.append(iterative_graphs_analysis[0])
            ret.append(iterative_graphs_analysis[1])
            ret.append(full_graph_analysis[0])
        return ret

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

    def _iterative_mean_score(
        self, mask: th.Tensor, weight: th.Tensor
    ) -> typ.Tuple[float, float, float, float]:
        # shape = mask.shape
        # outc, inc = shape[0], shape[1]

        outc = mask.shape[0]

        flatten_mask = mask.reshape(outc, -1).T
        # flatten_weight = weight.reshape(outc, -1).T

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
            # submask_weight = flatten_weight[index].T
            degree_bound, randomness, _ = self.ramanujan_bounds(submask, total_nodes)

            # w_degree_bound, w_mix_lemma, _ = self.ramanujan_bounds(
            # submask_weight, total_nodes, True
            # )

            input_vertices = index.sum()
            output_vertices = (submask.sum(dim=1) > 0).sum()
            cheeger = max(input_vertices, output_vertices) / (
                input_vertices + output_vertices - 1
            )

            if degree_bound == float("inf"):
                # invalid type
                continue
            layer_score.append(cheeger * degree_bound)
            # weighted_layer_score.append(cheeger * w_degree_bound)
            mixing_lem.append(randomness)
            # weighted_mixing_lem.append(w_mix_lemma)

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

        info = {}

        info["total_left_nodes"] = (flatten_mask.sum(dim=-1) > 0.0).sum()
        info["total_right_nodes"] = (flatten_mask.sum(dim=0) > 0.0).sum()
        info["total_nodes"] = info["total_left_nodes"] + info["total_right_nodes"]

        info["total_edges"] = flatten_mask.sum()
        info["in_deg_avg"] = flatten_mask.sum(dim=-1).mean()
        info["out_deg_avg"] = flatten_mask.T.sum(dim=-1).mean()

        for degree_out, num_nodes in degree_out_lut.items():
            if degree_out == 0 or num_nodes < 2:
                continue
            index = fan_out_index == degree_out
            submask = flatten_mask[index].T  # out_c x in_c
            submask_weight = flatten_weight[index].T

            degree_bound, randomness, _ = self.ramanujan_bounds(
                submask, info["total_nodes"]
            )

            w_degree_bound, w_mix_lemma, _ = self.ramanujan_bounds(
                submask_weight, info["total_nodes"], True
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

            # eigs = Ramanujan.get_eig_values(bigraph)
            eigs = Ramanujan.get_eig_values(bigraph)
            # eigs2 = eigenvalues_sparse_symmetric_cuda(bigraph)
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

            # random_bound = (eigs[-1] ** 2 / 4 + 1) * math.sqrt(
            # meta["left_nodes"] * meta["right_nodes"]
            # ) - abs(
            # edge_index.size(-1)
            # - (avg_deg_left / meta["out_dim"])
            # * (meta["left_nodes"] * meta["right_nodes"])
            # )
            # random_bound /= meta["out_dim"] * meta["in_dim"]
            expander_mixxing /= total_nodes
        else:
            # random_bound = float("inf")
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
        adj_eigh_val = sp.linalg.eigsh(matrix, k=k, which="LM")[0].tolist()
        ###################j
        abs_eig = [abs(i) for i in adj_eigh_val]
        abs_eig.sort(reverse=True)
        return abs_eig


def eigenvalues_sparse_symmetric_cuda(matrix: torch.Tensor) -> torch.Tensor:

    # if not matrix.is_cuda:
    # raise ValueError("Input matrix must be on a GPU")

    # if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    # raise ValueError("Input matrix must be a square matrix")
    dense_cp_mx_cuda = cp.asarray(matrix)

    sparse_cp_mx_cuda = cp.sparse.coo_matrix(dense_cp_mx_cuda)

    # Convert the PyTorch sparse tensor to a CuPy sparse matrix
    # coo = matrix.to_sparse().cpu().numpy()
    # __import__("pdb").set_trace()
    # sparse_matrix_gpu = cp.scipy.sparse.coo_matrix(coo)

    # Check if the input matrix is symmetric
    # if not cp.allclose(sparse_matrix_gpu, sparse_matrix_gpu.transpose()):
    # raise ValueError("Input matrix must be symmetric")
    # Calculate all eigenvalues using eigsh from scipy.sparse.linalg
    eigenvalues_gpu = cuspla.eigsh(sparse_cp_mx_cuda, k=3, return_eigenvectors=False)
    __import__("pdb").set_trace()

    # Convert the eigenvalues back to a PyTorch tensor
    return th.tensor(cp.asnumpy(eigenvalues_gpu), dtype=th.float32)
