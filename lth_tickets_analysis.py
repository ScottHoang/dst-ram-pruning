import os
import os.path as osp

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator


def create_3d_blank_chart(xs, ys, zs, xtitle, ytitle, ztitle, ax_padding, plot_size):
    # my favorite styling kit
    plt.style.use("dark_background")
    # determining the size of the graph
    fig = plt.figure(figsize=plot_size)
    # 3D mode
    ax = fig.gca(projection="3d")
    # transparent axis pane background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # setting chart axis names
    # ax.set_xlabel("Cosine-similarity")
    # ax.set_ylabel("L2-distance")
    # ax.set_zlabel("Accuracy")
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_zlabel(ztitle)

    ax.set_xlim3d(xs.min() - ax_padding, xs.max() + ax_padding)
    ax.set_ylim3d(ys.min() - ax_padding, ys.max() + ax_padding)
    ax.set_zlim3d(zs.min() - ax_padding, zs.max() + ax_padding)
    return (fig, ax)


def create_2d_blank_chart(xs, ys, xtitle, ytitle, ax_padding, plot_size):
    # my favorite styling kit
    plt.style.use("dark_background")
    # determining the size of the graph
    fig = plt.figure(figsize=plot_size)
    # 3D mode
    ax = fig.gca()
    # transparent axis pane background
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # setting chart axis names
    # ax.set_xlabel("Cosine-similarity")
    # ax.set_ylabel("L2-distance")
    # ax.set_zlabel("Accuracy")
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    # ax.set_zlabel(ztitle)

    ax.set_xlim(xs.min() - ax_padding, xs.max() + ax_padding)
    ax.set_ylim(ys.min() - ax_padding, ys.max() + ax_padding)
    # ax.set_zlim3d(zs.min() - ax_padding, zs.max() + ax_padding)
    return (fig, ax)


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


def sign_iou(src_state_dict: dict, tgt_state_dict: dict):
    global num_layers
    iou = th.zeros(num_layers)
    cnt = 0
    for k in src_state_dict:
        if k.endswith("_mask"):
            src_mask = src_state_dict[k].bool()
            tgt_mask = tgt_state_dict[k].bool()

            src_w = src_state_dict[k[0:-4] + "orig"]
            tgt_w = tgt_state_dict[k[0:-4] + "orig"]

            intersection = src_mask & tgt_mask

            sign_src = src_w * intersection
            sign_tgt = tgt_w * intersection

            sign_src[sign_src > 0] = 1.0
            sign_src[sign_src < 0] = -1.0
            sign_tgt[sign_tgt > 0] = 1.0
            sign_tgt[sign_tgt < 0] = -1.0

            pos_mask = (sign_src == 1.0).bool() & (sign_tgt == 1.0).bool()
            neg_mask = (sign_src == -1.0).bool() & (sign_tgt == -1.0).bool()

            iou[cnt] = (pos_mask | neg_mask).float().sum() / intersection.numel()

            cnt += 1
    return iou


def read_iou(masks, iou_heatmap, which_seed):
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
    return iou_heatmap


def read_sign(masks, iou_heatmap, which_seed):
    for i in range(len(masks)):
        if masks[i] == -1:
            continue
        path_src = osp.join(
            savedir, f"seed_{which_seed}_mask_{masks[i]}_step_finetune.pth"
        )
        src_state_dict = th.load(path_src)["state_dict"]
        src_state_dict = {
            k: v.to_dense()
            for k, v in src_state_dict.items()
            if k.endswith("_mask") or k.endswith("_orig")
        }
        for j in range(i + 1, len(masks)):
            if masks[j] == -1:
                continue
            path_tgt = osp.join(
                savedir, f"seed_{which_seed}_mask_{masks[j]}_step_finetune.pth"
            )
            tgt_state_dict = th.load(path_tgt)["state_dict"]
            tgt_state_dict = {
                k: v.to_dense()
                for k, v in tgt_state_dict.items()
                if k.endswith("_mask") or k.endswith("_orig")
            }
            iou_heatmap[i, j] = sign_iou(src_state_dict, tgt_state_dict)
    return iou_heatmap


def read(which_seed=0):
    global savedir, num_layers, files

    seed_files = list(filter(lambda x: int(x.split("_")[1]) == which_seed, files))
    masks = [int(x.split("_")[3]) for x in seed_files]
    masks.sort()
    number_of_masks = len(masks) if -1 not in masks else len(masks) - 1

    mask_characteristics = th.zeros((number_of_masks, 2, num_layers))
    narc_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_init_characteristics = th.zeros((number_of_masks, 2, num_layers))
    grad_init_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_partial_characteristics = th.zeros((number_of_masks, 2, num_layers))
    weighted_anchor_characteristics = th.zeros((number_of_masks, 2, num_layers))

    all_validation_acc = th.zeros((number_of_masks, 1))
    all_test_acc = th.zeros((number_of_masks, 1))
    all_epochs = th.zeros((number_of_masks, 1))
    iou_heatmap = th.zeros((number_of_masks, number_of_masks, num_layers))
    sign_heatmap = th.zeros((number_of_masks, number_of_masks, num_layers))

    for mask_no, discovered_iteration in enumerate(masks):
        if mask_no == -1:
            continue
        path = osp.join(
            savedir, f"seed_{which_seed}_mask_{discovered_iteration}_step_finetune.pth"
        )
        data = th.load(path)
        if "test_acc" not in data:
            print(f"skipping {path}")
            continue
        weighted_init_characteristics[mask_no, 0] = data["initial_characteristics"][
            "weighted_imdb"
        ]
        weighted_init_characteristics[mask_no, 1] = data["initial_characteristics"][
            "full_weighted_imdb"
        ]
        grad_init_characteristics[mask_no, 0] = data["initial_characteristics"][
            "grad_imdb"
        ]
        grad_init_characteristics[mask_no, 1] = data["initial_characteristics"][
            "full_grad_imdb"
        ]
        # weighted_partial_characteristics[mask_no, 0] = data["partial_characteristics"][
        # "weighted_imdb"
        # ]
        # weighted_partial_characteristics[mask_no, 1] = data["partial_characteristics"][
        # "full_weighted_imdb"
        # ]
        # weighted_anchor_characteristics[mask_no, 0] = data["anchor_characteristics"][
        # "weighted_imdb"
        # ]
        # weighted_anchor_characteristics[mask_no, 1] = data["anchor_characteristics"][
        # "full_weighted_imdb"
        # ]

        mask_characteristics[mask_no, 0] = data["initial_characteristics"]["imdb"]
        mask_characteristics[mask_no, 1] = data["initial_characteristics"]["full_imdb"]

        narc_characteristics[mask_no, 0] = data["initial_characteristics"]["narc"]
        narc_characteristics[mask_no, 1] = data["initial_characteristics"]["full_narc"]

        all_validation_acc[mask_no] = data["valdiation_acc"]
        # __import__("pdb").set_trace()
        all_test_acc[mask_no] = data["test_acc"]
        all_epochs[mask_no] = discovered_iteration  # data["discovered_epoch"]

    grad_init_characteristics = inf_to_zero(grad_init_characteristics)
    weighted_init_characteristics = inf_to_zero(weighted_init_characteristics)
    mask_characteristics = inf_to_zero(mask_characteristics)
    # __import__("pdb").set_trace()

    # delta_cosine_distance = batch_cosine_sim(
    # weighted_partial_characteristics, weighted_anchor_characteristics
    # )
    # delta_l2_distance = batch_l2_distance(
    # weighted_partial_characteristics, weighted_anchor_characteristics
    # )
    # delta_angular_distance = delta_cosine_distance * delta_l2_distance

    # cosine_distance = batch_cosine_sim(
    # weighted_init_characteristics, weighted_anchor_characteristics
    # )
    # l2_distance = batch_l2_distance(
    # weighted_init_characteristics, weighted_anchor_characteristics
    # )
    # anchor_angular_distance = cosine_distance * l2_distance

    avg_masks_characteristics = mask_characteristics.mean(dim=-1)
    avg_narc_characteristics = narc_characteristics.mean(dim=-1)

    # iou_heatmap = read_iou(masks, iou_heatmap, which_seed)
    # __import__("pdb").set_trace()
    # sign_heatmap = read_sign(masks, sign_heatmap, which_seed)

    full_layer_wise_description = (
        th.cat((mask_characteristics, weighted_init_characteristics), dim=1)
        .norm(dim=1, keepdim=True)
        .mean(dim=-1)
    )
    full_spectrum_description = (
        th.cat(
            (
                mask_characteristics,
                weighted_init_characteristics,
                grad_init_characteristics,
            ),
            dim=1,
        )
        .norm(dim=1, keepdim=True)
        .mean(dim=-1)
    )
    # print(full_layer_wise_description.shape)
    # __import__("pdb").set_trace()

    ret = {
        # "delta": delta_angular_distance,
        # "anchor": anchor_angular_distance,
        "imdb": avg_masks_characteristics,
        "narc": avg_narc_characteristics,
        "test_acc": all_test_acc,
        "epochs": all_epochs,
        "iou_heatmap": iou_heatmap,
        # "sign_heatmap": sign_heatmap,
        # "delta_cos": delta_cosine_distance,
        # "delta_l2": delta_l2_distance,
        # "anchor_cos": cosine_distance,
        # "anchor_l2": l2_distance,
        "weighted_imdb": weighted_init_characteristics.mean(dim=-1),
        "grad_imdb": grad_init_characteristics.mean(dim=-1),
        "full_layer_wise_description": full_layer_wise_description,
        "full_spectrum_description": full_spectrum_description,
    }
    return ret


def create_2d_scatter_plot(
    x,
    y,
    title,
    xtitle,
    ytitle,
    output_dir,
    swapaxis=False,
    horizontal_lines=None,
    trend_line=False,
):
    if ytitle == "test_acc":
        zero_filter = y != 0.0
        x = x[zero_filter]
        y = y[zero_filter]

    if swapaxis:
        x, y = y, x
        xtitle, ytitle = ytitle, xtitle
    # Create scatter plot
    if model_type == "ResNet34":
        fig, ax = create_2d_blank_chart(x, y, xtitle, ytitle, 0.05, (8, 8))
    norm_idx = np.linspace(0, 1, x.size(0))
    cmap = plt.cm.get_cmap("plasma", x.size(0))
    ax.scatter(x, y, c=cmap(norm_idx), marker="o")
    ax.grid(True)

    if horizontal_lines is not None:
        for hl, name, color in horizontal_lines:
            ax.axhline(y=hl, linestyle="-", color=color, label=name)
    # Add labels and titl
    # Save plot to file
    if trend_line:
        m, b = np.polyfit(x, y, 1)
        x_range = np.linspace(min(x), max(x), 100)
        y_range = m * x_range + b
        ax.plot(x_range, y_range, color="snow")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{title}.png")
    plt.tight_layout()
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=x.size(0)))
    )
    if horizontal_lines is not None:
        plt.legend()
    plt.savefig(filename)


def plot_3d_plane(x, y, z, xtitle, ytitle, ztitle, title, output_dir):
    fig, ax = create_3d_blank_chart(x, y, z, xtitle, ytitle, ztitle, 0.001, (12, 12))
    ax.set_title(title)
    resolution = 100
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    # ax.scatter3D(x, y, z.squeeze())
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), rescale=True)

    norm_idx = np.linspace(0, 1, x.size(0))
    cmap = plt.cm.get_cmap("plasma", x.size(0))

    ax.scatter3D(x, y, z, c=cmap(norm_idx), marker="o")

    ######################################
    # def setup_colorbar(fig, surf):
    # ax = surf.axes
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("top", size="5%", pad=0.1)
    # return fig.colorbar(surf, cax=cax)

    ######################################
    def update_view(num, ax):
        if num < 360:
            ax.view_init(30, num)
        else:
            ax.view_init(num - 360, azim=360)

    ######################################
    surf = ax.plot_surface(
        xi, yi, zi, cmap=cm.coolwarm, linewidth=0, alpha=0.8, antialiased=False
    )
    # cbar = setup_colorbar(fig, surf)

    ani = FuncAnimation(
        fig, update_view, frames=np.arange(0, 450, 2), fargs=(ax,), interval=50
    )
    # filename = os.path.join(outputdir, f"{title}.png")
    gifpath = os.path.join(output_dir, title)
    os.makedirs(gifpath, exist_ok=True)

    ani.save(os.path.join(gifpath, f"{title}.gif"), writer="imagemagicks", dpi=80)

    angles = [(30, 0), (30, 90), (30, 180), (30, 270), (90, 0), (-90, 0)]
    names = ["front", "right", "back", "left", "top", "bottom"]

    for ((elev, azim), name) in zip(angles, names):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(os.path.join(gifpath, f"{title}_view_{name}.png"))


def create_heatmap(heatmap, title, output_dir):
    plt.clf()
    heatmap = heatmap + th.transpose(heatmap, -1, -2)
    sns.heatmap(heatmap)
    filename = os.path.join(output_dir, f"{title}.png")
    plt.savefig(filename)


import typing as typ


def generate_scatter_point_series(
    data: dict,
    src_key: str,
    exceptions: typ.List[str],
    outputdir: str,
    swapaxis=False,
    hlines=None,
    trend_line=False,
):
    assert src_key in data
    outputdir = osp.join(outputdir, f"{src_key}_vs")
    exceptions.append(src_key)
    subtitle = ["multi-graphs", "single-graph"]
    for k in data.keys():
        if k in exceptions:
            continue
        else:
            series = data[k]
            shape = series.shape
            if len(shape) == 1:
                create_2d_scatter_plot(
                    data[src_key],
                    series,
                    title=f"{src_key}_vs_{k}",
                    xtitle=src_key,
                    ytitle=k,
                    output_dir=outputdir,
                    swapaxis=swapaxis,
                    horizontal_lines=hlines,
                    trend_line=trend_line,
                )
            else:
                assert shape[-1] == 2
                create_2d_scatter_plot(
                    data[src_key],
                    series[:, 0],
                    title=f"{src_key}_vs_{k}-{subtitle[0]}",
                    xtitle=src_key,
                    ytitle=f"{k}-{subtitle[0]}",
                    output_dir=outputdir,
                    swapaxis=swapaxis,
                    horizontal_lines=hlines,
                    trend_line=trend_line,
                )
                create_2d_scatter_plot(
                    data[src_key],
                    series[:, 1],
                    title=f"{src_key}_vs_{k}-{subtitle[1]}",
                    xtitle=src_key,
                    ytitle=f"{k}-{subtitle[1]}",
                    output_dir=outputdir,
                    swapaxis=swapaxis,
                    horizontal_lines=hlines,
                    trend_line=trend_line,
                )


def generate_imdb_acc_3d_scatter_point_series(
    data: dict, exceptions: typ.List[str], outputdir: str
):
    assert "imdb" in data and "test_acc" in data
    outputdir = osp.join(outputdir, "imdb-acc-vs")
    exceptions.extend(["imdb", "test_acc"])
    subtitle = ["multi-graphs", "single-graph"]
    for k in data.keys():
        if k in exceptions:
            continue
        series = data[k]
        series_shape = series.shape

        for i in range(2):
            imdb = data["imdb"][:, i]
            xtitle = f"imdb-{subtitle[i]}"
            if len(series_shape) > 1:
                _series = series[:, i]
                ytitle = f"{k}-{subtitle[i]}"
            else:
                _series = series
                ytitle = k

            plot_3d_plane(
                imdb,
                _series,
                data["test_acc"],
                xtitle,
                ytitle,
                "accuracy",
                f"{xtitle}_acc_{ytitle}",
                outputdir,
            )


def generate_3d_perspective_series(
    data: dict, src_key, exceptions: typ.List[str], outputdir: str
):
    assert src_key in data and len(data[src_key].shape) == 1
    outputdir = osp.join(outputdir, f"perspective-{src_key}-vs")
    subtitle = ["multi-graphs", "single-graph"]
    for k in data.keys():
        if k in exceptions:
            continue
        series = data[k]
        series_shape = series.shape
        if len(series_shape) == 1:
            continue

        xaxis = data[k][:, 0]
        yaxis = data[k][:, 1]
        xtitle = f"{k}-{subtitle[0]}"
        ytitle = f"{k}-{subtitle[1]}"

        plot_3d_plane(
            xaxis,
            yaxis,
            data[src_key],
            xtitle,
            ytitle,
            src_key,
            f"{xtitle}_{ytitle}_{src_key}",
            outputdir,
        )


import tqdm

if __name__ == "__main__":
    # savedir = "results_v2/random+magnitude+ramanujan/ERK/vgg-c/0.01"
    # outputdir = "anlysis/results_v2/random+magnitude+ramanujan/vgg-c/ERK/0.01"
    # num_layers = 16
    # num_layers = 21
    savedirs = [
        # "./results_lth_v8/SNIP/vgg-d/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/GraSP/vgg-d/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/ERK/vgg-d/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/Rand/vgg-d/0.01/population-3000_sampling-10_iter-2",
        # #
        # "./results_lth_v8/SNIP/ResNet18/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/GraSP/ResNet18/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/ERK/ResNet18/0.01/population-3000_sampling-100_iter-2",
        # "./results_lth_v8/Rand/ResNet18/0.01/population-3000_sampling-10_iter-2",
        # #
        # "./results_lth_v8/SNIP/ResNet34/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/GraSP/ResNet34/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8/ERK/ResNet34/0.01/population-3000_sampling-100_iter-2",
        # "./results_lth_v8/Rand/ResNet34/0.01/population-3000_sampling-100_iter-2",
        # #
        # "./results_lth_v8_Wpretrained/SNIP/vgg-d/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8_Wpretrained/GraSP/vgg-d/0.01/population-1500_sampling-10_iter-2",
        # "./results_lth_v8_Wpretrained/ERK/vgg-d/0.01/population-3000_sampling-100_iter-2",
        # "./results_lth_v8_Wpretrained/Rand/vgg-d/0.01/population-3000_sampling-100_iter-2",
        #
        # "./results_lth_v8_Wpretrained/SNIP/ResNet18/0.01/population-3000_sampling-10_iter-2",
        "./results_lth_v8_Wpretrained/GraSP/ResNet18/0.01/population-1000_sampling-10_iter-2",
        # "./results_lth_v8_Wpretrained/ERK/ResNet18/0.01/population-3000_sampling-100_iter-2",
        # "./results_lth_v8_Wpretrained/Rand/ResNet18/0.01/population-3000_sampling-100_iter-2",
        #
        # "./results_lth_v8_Wpretrained/SNIP/ResNet34/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8_Wpretrained/GraSP/ResNet34/0.01/population-3000_sampling-10_iter-2",
        # "./results_lth_v8_Wpretrained/ERK/ResNet34/0.01/population-3000_sampling-100_iter-2",
        # "./results_lth_v8_Wpretrained/Rand/ResNet34/0.01/population-3000_sampling-100_iter-2",
    ]
    lth_results = {"vgg-d": 0.889, "ResNet18": 0.8943, "ResNet34": 0.9143}
    lth_rewind_results = {"vgg-d": 0.9136, "ResNet18": 0.9122, "ResNet34": 0.9280}
    for savedir in tqdm.tqdm(savedirs, total=len(savedirs)):
        model_type = savedir.split("/")[3]
        if model_type in ("vgg-c", "vgg-d"):
            num_layers = 16
        elif model_type == "ResNet18":
            num_layers = 21
        elif model_type == "ResNet34":
            num_layers = 37
        else:
            raise NotImplementedError

        outputdir = f"lth_analysis_v2/{savedir}"
        os.makedirs(outputdir, exist_ok=True)

        files = list(filter(lambda x: x.endswith("_finetune.pth"), os.listdir(savedir)))
        seeds = set(int(x.split("_")[1]) for x in files)
        for seed in seeds:
            data = {k: v.squeeze() for k, v in read(seed).items()}
            # v[1::] skipping the first mask
            # delta, anchor, imdb, narc, test_acc, epochs, iou_heatmap = data[0:7]
            # delta_cos, delta_l2, anchor_cos, anchor_l2 = data[7::]
            # test_acc = test_acc.squeeze()

            # epoch series
            print("generating epoch series")
            generate_scatter_point_series(
                data, "epochs", ["iou_heatmap", "sign_heatmap"], outputdir
            )
            ## acc series
            print("generating acc series")

            generate_scatter_point_series(
                data,
                "test_acc",
                ["iou_heatmap", "epochs", "sign_heatmap"],
                outputdir,
                swapaxis=True,
                hlines=[
                    (data["test_acc"][0], model_type, "gold"),
                    (lth_results[model_type], "lth", "r"),
                    (lth_rewind_results[model_type], "lth-rewind", "cyan"),
                ],
                # trend_line=True,
            )
            # imdb vs acc vs ....
            # print("generating imdb-v-acc series")
            # generate_imdb_acc_3d_scatter_point_series(
            # data, ["iou_heatmap", "sign_heatmap"], outputdir
            # )

            # # ## persepective series
            # print("generating perspective series")
            # generate_3d_perspective_series(
            # data, "test_acc", ["iou_heatmap", "sign_heatmap"], outputdir
            # )

            # create_heatmap(
            # data["iou_heatmap"].mean(dim=-1), "IoU mask heatmap", outputdir
            # )
            # create_heatmap(
            # data["sign_heatmap"].mean(dim=-1), "Signage heatmap", outputdir
            # )
            plt.close()
