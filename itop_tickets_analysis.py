import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


def mask_iou(src_state_dict: dict, tgt_state_dict: dict):
    global num_layers
    iou = th.zeros(num_layers)
    cnt = 0
    for k in src_state_dict:
        src_mask = src_state_dict[k]
        tgt_mask = tgt_state_dict[k]
        interection = src_mask & tgt_mask
        union = src_mask | tgt_mask
        iou[cnt] = interection.sum() / union.sum()
        cnt += 1
    return iou


def read(which_seed=0):
    global savedir, num_layers, files

    seed_files = list(filter(lambda x: int(x.split("_")[1]) == which_seed, files))
    masks = [int(x.split("_")[3]) for x in seed_files]
    masks.sort()
    number_of_masks = len(masks) if -1 not in masks else len(masks) - 1

    mask_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_init_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_partial_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_anchor_characteristics = th.zeros((number_of_masks, 2, num_layers))

    all_validation_acc = th.zeros((number_of_masks, 1))
    all_test_acc = th.zeros((number_of_masks, 1))
    all_epochs = th.zeros((number_of_masks, 1))
    iou_heatmap = th.zeros((number_of_masks, number_of_masks, num_layers))

    for mask_no in masks:
        if mask_no == -1:
            continue
        path = osp.join(savedir, f"seed_{which_seed}_mask_{mask_no}_step_finetune.pth")
        data = th.load(path)
        weighted_init_characteristics[mask_no, 0] = data["initial_characteristics"][
            "weighted_imdb"
        ]
        weighted_init_characteristics[mask_no, 1] = data["initial_characteristics"][
            "full_weighted_imdb"
        ]
        weighted_partial_characteristics[mask_no, 0] = data["partial_characteristics"][
            "weighted_imdb"
        ]
        weighted_partial_characteristics[mask_no, 1] = data["partial_characteristics"][
            "full_weighted_imdb"
        ]
        weighted_anchor_characteristics[mask_no, 0] = data["anchor_characteristics"][
            "weighted_imdb"
        ]
        weighted_anchor_characteristics[mask_no, 1] = data["anchor_characteristics"][
            "full_weighted_imdb"
        ]

        mask_characteristics[mask_no, 0] = data["initial_characteristics"]["imdb"]
        mask_characteristics[mask_no, 1] = data["initial_characteristics"]["full_imdb"]

        all_validation_acc[mask_no] = data["valdiation_acc"]
        all_test_acc[mask_no] = data["test_acc"]
        all_epochs[mask_no] = data["discovered_epoch"]

    for i in range(len(masks)):
        if masks[i] == -1:
            continue
        path_src = osp.join(
            savedir, f"seed_{which_seed}_mask_{masks[i]}_step_finetune.pth"
        )
        src_state_dict = th.load(path_src)["state_dict"]
        src_state_dict = {
            k: v.to_dense().bool()
            for k, v in src_state_dict.items()
            if k.endswith("_mask")
        }
        for j in range(i + 1, len(masks)):
            if masks[j] == -1:
                continue
            path_tgt = osp.join(
                savedir, f"seed_{which_seed}_mask_{masks[j]}_step_finetune.pth"
            )
            tgt_state_dict = th.load(path_tgt)["state_dict"]
            tgt_state_dict = {
                k: v.to_dense().bool()
                for k, v in tgt_state_dict.items()
                if k.endswith("_mask")
            }
            iou_heatmap[i, j] = mask_iou(src_state_dict, tgt_state_dict)

    cosine_distance = batch_cosine_sim(
        weighted_partial_characteristics, weighted_init_characteristics
    )
    l2_distance = batch_l2_distance(
        weighted_partial_characteristics, weighted_init_characteristics
    )
    delta_angular_distance = cosine_distance * l2_distance

    cosine_distance = batch_cosine_sim(
        weighted_init_characteristics, weighted_anchor_characteristics
    )
    l2_distance = batch_l2_distance(
        weighted_init_characteristics, weighted_anchor_characteristics
    )
    anchor_angular_distance = cosine_distance * l2_distance

    avg_masks_characteristics = mask_characteristics.mean(dim=-1)

    return (
        delta_angular_distance,
        anchor_angular_distance,
        avg_masks_characteristics,
        all_test_acc,
        all_epochs,
        iou_heatmap,
    )


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


def create_heatmap(heatmap, title, output_dir):
    plt.clf()
    __import__("pdb").set_trace()
    heatmap = heatmap + th.transpose(heatmap, -1, -2)
    sns.heatmap(heatmap)
    filename = os.path.join(output_dir, f"{title}.png")
    plt.savefig(filename)


if __name__ == "__main__":
    savedir = "results_v2/random+magnitude+vanilla/ERK/ResNet18/0.01"
    outputdir = "anlysis/results_v2/random+magnitude+vanilla/ResNet18/ERK/0.01"
    os.makedirs(outputdir, exist_ok=True)
    # num_layers = 16
    files = list(filter(lambda x: x.endswith("finetune.pth"), os.listdir(savedir)))
    seeds = set(int(x.split("_")[1]) for x in files)
    for seed in seeds:
        delta, anchor, imdb, test_acc, epochs, iou_heatmap = read(seed)
        create_scatter_plot(
            imdb[:, 0], test_acc, "imdb vs test_acc", "imdb", "test acc", outputdir
        )
        create_scatter_plot(
            delta[:, 0],
            test_acc,
            "delta-T vs test_acc",
            "angular distance",
            "test acc",
            outputdir,
        )
        create_scatter_plot(
            anchor[:, 0],
            test_acc,
            "anchor vs test_acc",
            "angular distance",
            "test acc",
            outputdir,
        )

        create_scatter_plot(
            imdb[:, 1], test_acc, "full imdb vs test_acc", "imdb", "test acc", outputdir
        )
        create_scatter_plot(
            delta[:, 1],
            test_acc,
            "full delta-T vs test_acc",
            "angular distance",
            "test acc",
            outputdir,
        )
        create_scatter_plot(
            anchor[:, 1],
            test_acc,
            "anchor vs test_acc",
            "angular distance",
            "test acc",
            outputdir,
        )
        ###########################
        create_scatter_plot(
            epochs,
            imdb[:, 1],
            "full imdb vs epoch",
            "discovered epoch",
            "imdb",
            outputdir,
        )
        create_scatter_plot(
            epochs,
            imdb[:, 0],
            "imdb vs epoch",
            "discovered epoch",
            "imdb",
            outputdir,
        )
        create_scatter_plot(
            epochs,
            test_acc,
            "test acc vs epoch",
            "discovered epoch",
            "test acc",
            outputdir,
        )
        create_scatter_plot(
            epochs,
            delta[:, 0],
            "delta-T vs epochs",
            "epochs",
            "angular distance",
            outputdir,
        )
        create_scatter_plot(
            epochs,
            anchor[:, 0],
            "anchor vs epochs",
            "epochs",
            "angular distance",
            outputdir,
        )
        create_heatmap(iou_heatmap.mean(dim=-1), "IoU mask heatmap", outputdir)
        # __import__("pdb").set_trace()
        # create_scatter_plot(
        # ang_dist[10::, 0],
        # test_acc[10::],
        # "ram-bound vgg-c 0.01 angular v test",
        # "angular distance",
        # "accuracy",
        # outputdir,
        # )
        # create_scatter_plot(
        # epochs,
        # test_acc,
        # "ram-bound vgg-c 0.01 epoch v test",
        # "discovered_epochs",
        # "accuracy",
        # outputdir,
        # )
