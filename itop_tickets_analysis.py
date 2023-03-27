import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline


def inf_to_zero(batch):
    batch[batch == float("inf")] = 0
    return batch


def batch_cosine_sim(src, tgt):

    # src : Num_masks x types x Num_layers
    # tgt : Num_masks x types x Num_layers

    src = inf_to_zero(src)
    tgt = inf_to_zero(tgt)

    src = src.transpose(0, 1)  # types x Num_masks x Num_layers
    tgt = tgt.transpose(0, 1)

    # dot_product = src * tgt # Num_masks x Num_layerj,x
    # norm = src.norm(dim=-1) * tgt.norm(dim=-1)

    dot_product = src @ tgt.transpose(-1, -2)  # Types x Num_masks x Num_masks
    norm_src = src.norm(dim=-1)[:, :, None]  # Types x Num_masks x 1
    norm_tgt = tgt.norm(dim=-1)[:, :, None]  # Types x Num_masks x 1
    norm = norm_src @ norm_tgt.transpose(-1, -2)  # Types x Num_masks x Num_masks

    cosine_distance = th.diagonal(
        dot_product / norm, dim1=1, dim2=2
    )  # Types x Num_masks x 1
    cosine_distance = cosine_distance.transpose(0, 1)
    return cosine_distance


def batch_l2_distance(src, tgt):
    # src : Num_masks x Num_layers
    # tgt : Num_masks x Num_layers

    src = inf_to_zero(src)
    tgt = inf_to_zero(tgt)

    distance = (src - tgt).norm(dim=-1)

    return distance


def read(which_seed=0):
    global savedir, num_layers, files

    seed_files = list(filter(lambda x: int(x.split("_")[1]) == which_seed, files))
    number_of_masks = len(seed_files)

    all_init_characteristics = th.zeros((number_of_masks, 2, num_layers))
    all_partial_characteristics = th.zeros((number_of_masks, 2, num_layers))
    all_validation_acc = th.zeros((number_of_masks, 1))
    all_test_acc = th.zeros((number_of_masks, 1))
    all_epochs = th.zeros((number_of_masks, 1))

    for mask_no in range(number_of_masks):
        path = osp.join(savedir, f"seed_{which_seed}_mask_{mask_no}_step_finetune.pth")
        data = th.load(path)
        all_init_characteristics[mask_no, 0] = data["initial_characteristics"]["imdb"]
        all_init_characteristics[mask_no, 1] = data["initial_characteristics"][
            "full_imdb"
        ]
        all_partial_characteristics[mask_no, 0] = data["partial_characteristics"][
            "imdb"
        ]
        all_partial_characteristics[mask_no, 1] = data["partial_characteristics"][
            "full_imdb"
        ]
        all_validation_acc[mask_no] = data["valdiation_acc"]
        all_test_acc[mask_no] = data["test_acc"]
        all_epochs[mask_no] = data["discovered_epoch"]

    cosine_distance = batch_cosine_sim(
        all_partial_characteristics, all_init_characteristics
    )
    l2_distance = batch_l2_distance(
        all_partial_characteristics, all_init_characteristics
    )
    angular_distance = cosine_distance * l2_distance

    return angular_distance, all_test_acc, all_epochs


def create_scatter_plot(x, y, title, xlabel, ylabel, output_dir):
    # Create scatter plot
    plt.clf()
    plt.scatter(x, y)

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Save plot to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{title}.png")
    plt.savefig(filename)


if __name__ == "__main__":
    savedir = "results_v1/random+magnitude+ramanujan/vgg-c/0.01"
    outputdir = "anlysis/results_v1/random+magnitude+ramanujan/vgg-c/0.01"
    os.makedirs(outputdir, exist_ok=True)
    num_layers = 16
    files = list(filter(lambda x: x.endswith("finetune.pth"), os.listdir(savedir)))
    seeds = set(int(x.split("_")[1]) for x in files)
    for seed in seeds:
        ang_dist, test_acc, epochs = read(seed)
        create_scatter_plot(
            ang_dist[10::, 0],
            test_acc[10::],
            "ram-bound vgg-c 0.01 angular v test",
            "angular distance",
            "accuracy",
            outputdir,
        )
        create_scatter_plot(
            epochs,
            test_acc,
            "ram-bound vgg-c 0.01 epoch v test",
            "discovered_epochs",
            "accuracy",
            outputdir,
        )
