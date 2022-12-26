import math
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

    # self.model = self.model.to(device)

    def __call__(self, mask, grow=False):
        mask = mask.data.clone()
        return self.mask_score(mask, grow)

    def mask_score(self, mask: th.Tensor, grow: bool) -> float:
        if grow:
            grad_bound, randomness_score, meta = self.ramanujan_bounds(mask)
            return grad_bound * -1

        shape = mask.shape
        outc, inc = shape[0], shape[1]
        flatten_mask = mask.reshape(outc, -1).T
        fan_out = flatten_mask.sum(dim=1)

        layer_score = []
        for _fan_out in fan_out.unique():
            if _fan_out == 0:
                continue
            index = fan_out == _fan_out
            submask = flatten_mask[index].T  # out_c x in_c
            if index.sum() < 3:
                continue
            if submask.sum() < 3:
                continue
            degree_bound, random_bound, meta = self.ramanujan_bounds(submask)
            if degree_bound == float("inf"):
                pass
            else:
                # total_edges = submask.sum()
                # left_vol = total_edges
                # right_vol = flatten_mask.T[submask.sum(dim=1) > 0].sum()
                left_vert = index.sum()
                right_vert = (submask.sum(dim=1) > 0).sum()
                cheeger = max(left_vert, right_vert) / (left_vert + right_vert - 1)
                layer_score.append(cheeger * degree_bound)
        score = sum(layer_score)  # / len(layer_score)
        return score

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

        if avg_deg_left >= 3 and avg_deg_right > 1:

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
    def weighted_ramanujan_bounds(
        tensor: th.Tensor,
    ) -> typ.Tuple[float, float, typ.Dict[str, typ.Any]]:
        """Calculate the Ramanujan bound"""
        bigraph, meta = Ramanujan.get_biparite_graph(tensor)

        eigs = Ramanujan.get_eig_values(bigraph)
        return eigs[0] - eigs[-1]

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
