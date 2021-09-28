import argparse
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from matplotlib.lines import Line2D
from rlf.exp_mgr import config_mgr
from rlf.exp_mgr.auto_plot import get_data
from rlf.exp_mgr.plotter import high_res_save
from rlf.exp_mgr.wb_data_mgr import get_report_data
from rlf.rl.utils import CacheHelper, human_format_int


def plot_bar(
    plot_df,
    group_key,
    plot_key,
    name_ordering,
    name_colors,
    rename_map,
    show_ticks,
    axis_font_size,
    y_disp_bounds,
):
    df_avg_y = plot_df.groupby(group_key).mean()
    df_std_y = plot_df.groupby(group_key).std()

    avg_y = []
    std_y = []
    for name in name_ordering:
        avg_y.append(df_avg_y.loc[name][plot_key])
        std_y.append(df_std_y.loc[name][plot_key])

    bar_width = 0.35
    bar_darkness = 0.2
    bar_alpha = 0.9
    bar_pad = 0.0
    use_x = np.arange(len(name_ordering))
    colors = [name_colors[x] for x in name_ordering]

    N = len(name_ordering)
    start_x = 0.0
    end_x = round(start_x + N * (bar_width + bar_pad), 3)

    use_x = np.linspace(start_x, end_x, N)
    # plt.figure(figsize=(base_width,6))
    print(use_x)

    plt.bar(
        use_x,
        avg_y,
        width=bar_width,
        color=colors,
        align="center",
        alpha=bar_alpha,
        yerr=std_y,
        edgecolor=(0, 0, 0, 1.0),
        error_kw=dict(
            ecolor=(bar_darkness, bar_darkness, bar_darkness, 1.0),
            lw=2,
            capsize=3,
            capthick=2,
        ),
    )
    if show_ticks:
        xtic_names = [rename_map[x] for x in name_ordering]
    else:
        xtic_names = ["" for x in name_ordering]

    xtic_locs = use_x
    plt.xticks(xtic_locs, xtic_names, rotation=30)
    plt.ylabel(rename_map[plot_key], fontsize=axis_font_size)
    plt.ylim(*y_disp_bounds)


def plot_from_file(plot_cfg_path):
    with open(plot_cfg_path) as f:
        plot_settings = yaml.load(f)

        colors = sns.color_palette()
        group_colors = {
            name: colors[idx] for name, idx in plot_settings["colors"].items()
        }

        def get_setting(local, k, local_override=True, defval=None):
            if local_override:
                if k in local:
                    return local[k]
                elif k in plot_settings:
                    return plot_settings[k]
                else:
                    return defval
            else:
                if k in plot_settings:
                    return plot_settings[k]
                else:
                    return local[k]

        fig = None
        for plot_section in plot_settings["plot_sections"]:
            plot_key = plot_section.get("plot_key", plot_settings.get("plot_key", None))
            match_pat = plot_section.get(
                "name_match_pat", plot_settings.get("name_match_pat", None)
            )
            print(f"Getting data for {plot_section['report_name']}")
            should_combine = get_setting(plot_section, "should_combine", defval=False)
            other_plot_keys = plot_settings.get("other_plot_keys", [])
            other_fetch_fields = []
            fetch_std = get_setting(plot_section, "fetch_std", defval=False)
            if should_combine:
                other_fetch_fields.append("prefix")

            if fetch_std:
                other_plot_keys.append(plot_key + "_std")

            if not (
                "plot_section" not in plot_section
                or len(plot_section["plot_sections"]) == 0
            ):
                raise ValueError("Only line sections supported in bar plotting")

            line_plot_key = get_setting(plot_section, "line_plot_key")
            take_operation = get_setting(plot_section, "line_op")
            line_val_key = get_setting(plot_section, "line_val_key")
            if line_plot_key != line_val_key:
                fetch_keys = [line_plot_key, line_val_key]
            else:
                fetch_keys = [line_plot_key]
            if fetch_std:
                fetch_keys.append(line_plot_key + "_std")
            if len(fetch_keys) == 1:
                fetch_keys = fetch_keys[0]

            line_is_tb = plot_section.get("is_tb", False)
            if "line_is_tb" in plot_section:
                line_is_tb = plot_section["line_is_tb"]
            line_report_name = plot_section["report_name"]
            if "line_report_name" in plot_section:
                line_report_name = plot_section["line_report_name"]
            line_match_pat = match_pat
            if "line_match_pat" in plot_section:
                line_match_pat = plot_section["line_match_pat"]
            line_names = plot_section["line_sections"]
            line_df = get_data(
                line_report_name,
                fetch_keys,
                line_names[:],
                get_setting(
                    plot_section, "force_reload", local_override=False, defval=False
                ),
                line_match_pat,
                get_setting(plot_section, "other_line_plot_key", defval=[]),
                plot_settings["config_yaml"],
                line_is_tb,
                other_fetch_fields,
            )
            line_df = line_df.dropna()
            total_df = None
            for group_name, df in line_df.groupby("run"):
                if take_operation == "min":
                    use_idx = np.argmin(df[line_val_key])
                elif take_operation == "max":
                    use_idx = np.argmax(df[line_val_key])
                elif take_operation == "final":
                    use_idx = -1
                else:
                    raise ValueError(f"Unrecognized line reduce {take_operation}")
                use_val = df.iloc[np.array([use_idx])]
                if total_df is None:
                    total_df = use_val
                else:
                    total_df = pd.concat([use_val, total_df])

            local_renames = {}
            if "renames" in plot_section:
                local_renames = plot_section["renames"]
            colors = sns.color_palette()
            group_colors = {
                name: colors[idx] for name, idx in plot_settings["colors"].items()
            }

            def get_nums_from_str(s):
                return [float(x) for x in s.split(",")]

            plot_bar(
                total_df,
                "method",
                line_plot_key,
                line_names,
                group_colors,
                {
                    **plot_settings["global_renames"],
                    **local_renames,
                },
                get_setting(plot_section, "show_ticks", defval=False),
                get_setting(plot_section, "axis_font_size", defval="x-large"),
                get_nums_from_str(get_setting(plot_section, "y_disp_bounds")),
            )
            if plot_section.get("should_save", True):
                save_loc = plot_settings["save_loc"]
                if not osp.exists(save_loc):
                    os.makedirs(save_loc)
                save_path = osp.join(save_loc, plot_section["save_name"] + ".pdf")
                high_res_save(save_path)
                plt.clf()
                fig = None


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-cfg", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    plot_from_file(args.plot_cfg)
