"""
Utilities for manipulating images, rendering images, and rendering videos.
"""
import os
import os.path as osp
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rlf.rl.utils as rutils
import seaborn as sns

try:
    import wandb
except:
    pass


def append_text_to_image(
    image: np.ndarray, lines: List[str], from_bottom: bool = False
) -> np.ndarray:
    """
    Args:
        image: The NxMx3 frame to add the text to.
        lines: The list of strings (new line separated) to add to the image.
    Returns:
        image: (np.array): The modified image with the text appended.
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    if from_bottom:
        y = image.shape[0]
    else:
        y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        if from_bottom:
            y -= textsize[1] + 10
        else:
            y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    final = image + blank_image
    return final


def save_agent_obs(frames, imdim, vid_dir, name):
    use_dir = osp.join(vid_dir, name + "_frames")
    if not osp.exists(use_dir):
        os.makedirs(use_dir)

    if imdim != 1:
        raise ValueError("Only gray scale is supported right now")

    for i in range(frames.shape[0]):
        for frame_j in range(frames.shape[1]):
            fname = f"{i}_{frame_j}.jpg"
            frame = frames[i, frame_j].cpu().numpy()
            cv2.imwrite(osp.join(use_dir, fname), frame)

    print(f"Wrote observation sequence to {use_dir}")


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False, should_print=True):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + ".mp4")
    if osp.exists(vid_file):
        os.remove(vid_file)

    w, h = frames[0].shape[:-1]
    videodims = (h, w)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(vid_file, fourcc, fps, videodims)
    for frame in frames:
        frame = frame[..., 0:3][..., ::-1]
        video.write(frame)
    video.release()
    if should_print:
        print(f"Rendered to {vid_file}")


def plot_traj_data(
    pred: np.ndarray,
    real: np.ndarray,
    save_name: str,
    log_name: str,
    save_path_info: Union[Namespace, str],
    step: int,
    y_axis_name: str = "State %i",
    no_wb: Optional[bool] = None,
    title: str = "",
    ylim=None,
    plot_line=True,
):
    """
    Plots each state dimension of a trajectory comparing a predicted and real trajectory.
    :param pred: Shape [H, D] for a trajectory of length H and state dimension D.
        D plots will be created.
    :param real: Shape [H, D].
    :param save_name: Appended to log_name. This should likely be unique so
        files on the disk are not overriden. Include file extension.
    :param log_name: Has %i in the name to dynamically insert the state dimension.
        Should NOT be unique so the log key is updated.
    :param save_path_info: The save path will either be extracted from the args or the
        path passed as a string.
    :param step: Number of environment steps to log this plot to W&B with. Note
        this is likely different than the number of updates!
    :param y_axis_name: string with %i to dynamically insert state dimension.
    """

    save_name = log_name + "_" + save_name
    if isinstance(save_path_info, str):
        save_path = osp.join(save_path_info, save_name)
    else:
        save_path = osp.join(rutils.get_save_dir(save_path_info), save_name)

    if no_wb is None:
        if not isinstance(save_path_info, Namespace) and "no_wb" not in vars(
            save_path_info
        ):
            raise ValueError(
                f"Could not find property `no_wb` in the passed `save_path_info`"
            )
        no_wb = save_path_info.no_wb

    per_state_mse = np.mean((pred - real) ** 2, axis=0)
    per_state_sqrt_mse = np.sqrt(per_state_mse)

    H, state_dim = real.shape
    for state_i in range(state_dim):
        use_save_path = save_path % state_i
        if plot_line:
            plt.plot(np.arange(H), real[:, state_i], label="Real")
            plt.plot(np.arange(H), pred[:, state_i], label="Pred")
        else:
            plt.scatter(np.arange(H), real[:, state_i], label="Real")
            plt.scatter(np.arange(H), pred[:, state_i], label="Pred")
        plt.grid(b=True, which="major", color="lightgray", linestyle="--")
        plt.xlabel("t")
        plt.ylabel(y_axis_name % state_i)
        if ylim is not None:
            plt.ylim(ylim)

        if isinstance(title, list):
            use_title = title[state_i]
        else:
            use_title = title

        if len(use_title) != 0:
            use_title += "\n"
        use_title += "MSE %.4f, SQRT MSE %.4f" % (
            per_state_mse[state_i],
            per_state_sqrt_mse[state_i],
        )
        plt.title(use_title)
        plt.legend()

        rutils.plt_save(use_save_path)
        if not no_wb:
            use_full_log_name = log_name % state_i
            wandb.log(
                {use_full_log_name: [wandb.Image(use_save_path)]},
                step=step,
            )
    return np.mean(per_state_mse)


def high_res_save(save_path, is_high_quality=True):
    if is_high_quality:
        dpi = 1000
    else:
        dpi = 100

    file_format = save_path.split(".")[-1]
    plt.savefig(save_path, format=file_format, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {save_path}")


def nice_scatter(
    data_points: Dict[str, List[Tuple[float, float]]],
    name_colors: Dict[str, int],
    rename_map: Dict[str, str],
    save_name: str,
    save_dir: str,
    x_axis_name: str,
    y_axis_name: str,
    title_name: str,
    show_legend: bool,
    x_lim: Optional[Tuple[float, float]],
    y_lim: Optional[Tuple[float, float]],
    name_opacities: Dict[str, float] = {},
    fig_size=(4,4),
):
    """
    :param data_points: key is the label, value is the list of X,Y points.
    :param save_dir: Will create the directory if it does not exist.
    """
    os.makedirs(save_dir, exist_ok=True)
    color_pal = sns.color_palette()
    fig, ax = plt.subplots(figsize=fig_size)

    for k in data_points:
        X = []
        Y = []
        colors = []
        for x, y in data_points[k]:
            X.append(x)
            Y.append(y)
            use_color = color_pal[name_colors[k]]
            if k in name_opacities:
                opacity = name_opacities[k]
                use_color = (*use_color, opacity)

            colors.append(use_color)
        ax.scatter(X, Y, c=colors, label=rename_map[k])

    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if show_legend:
        ax.legend()
    ax.grid(True)
    if title_name != "":
        ax.set_title(title_name)

    high_res_save(osp.join(save_dir, save_name + ".pdf"))
    plt.clf()
