import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import seaborn as sns
import numpy as np

# Taken from the answer here https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth_arr(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def smooth_data(df, smooth, value, gp_keys=['method', 'run']):
    gp_df = df.groupby(gp_keys)

    ret_dfs = []
    for sub_df in [gp_df.get_group(k) for k in gp_df.indices]:
        use_df = sub_df.copy()
        use_df = use_df.dropna()
        use_df[value] = smooth_arr(use_df[value].tolist(), smooth)

        ret_dfs.append(use_df)

    data = pd.concat(ret_dfs, ignore_index=True)
    return data


def uncert_plot(plot_df, ax, x_name, y_name, avg_key, group_key, smooth_factor,
                y_bounds=None, y_disp_bounds=None, x_disp_bounds=None,
                group_colors=None, xtick_fn=None, ytick_fn=None, legend=False,
               rename_map={}, title=None, axes_font_size=14, title_font_size=18,
                tight=False):
    plot_df = plot_df.copy()
    if tight:
        plt.tight_layout(pad=2.2)
    if group_colors is None:
        methods = plot_df[group_key].unique()
        colors = sns.color_palette()
        group_colors = {method: color for method, color in zip(methods, colors)}

    # Smooth each method and run independently
    plot_df = smooth_data(plot_df, smooth_factor, y_name, [group_key, avg_key])

    avg_y_df = plot_df.groupby([group_key, x_name]).mean()
    std_y_df = plot_df.groupby([group_key, x_name]).std()
    avg_y_df['std'] = std_y_df[y_name]
    lines = []
    for name, sub_df in avg_y_df.groupby(level=0):
        x_vals = sub_df.index.get_level_values(x_name).to_numpy()
        y_vals = sub_df[y_name].to_numpy()
        y_std = sub_df['std'].fillna(0).to_numpy()
        l = ax.plot(x_vals, y_vals, label=name)
        lines.append(l[0])

        plt.setp(l, linewidth=2, color=group_colors[name])
        min_y_fill = y_vals - y_std
        max_y_fill = y_vals + y_std

        if y_bounds is not None:
            min_y_fill = np.clip(min_y_fill, y_bounds[0], y_bounds[1])
            max_y_fill = np.clip(max_y_fill, y_bounds[0], y_bounds[1])

        ax.fill_between(x_vals,
                        min_y_fill,
                        max_y_fill,
                        alpha=0.2, color=group_colors[name])
    if y_disp_bounds is not None:
        ax.set_ylim(*y_disp_bounds)
    if x_disp_bounds is not None:
        ax.set_xlim(*x_disp_bounds)

    if xtick_fn is not None:
        plt.xticks(ax.get_xticks(), [xtick_fn(t) for t in ax.get_xticks()])
    if ytick_fn is not None:
        plt.yticks(ax.get_yticks(), [ytick_fn(t) for t in ax.get_yticks()])

    if legend:
        labs = [l.get_label() for l in lines]
        plt.legend(lines, labs, fontsize='small')

    ax.grid(b=True, which='major', color='lightgray', linestyle='--')

    ax.set_xlabel(rename_map.get(x_name, x_name), fontsize=axes_font_size)
    ax.set_ylabel(rename_map.get(y_name, y_name), fontsize=axes_font_size)
    if title is not None and title != "":
        ax.set_title(title, fontsize=title_font_size)

def high_res_save(save_path):
    file_format = save_path.split('.')[-1]
    plt.savefig(save_path, format=file_format, dpi=1000)
    print(f"Saved figure to {save_path}")

def gen_fake_data(x_scale, data_key, n_runs=5):
    def create_sigmoid():
        noise = np.random.normal(0, 0.01, 100)
        x = np.linspace(0.0, 8.0, 100)
        y = 1/(1 + np.exp(-x))
        y += noise
        return x,y
    df = None
    for i in range(n_runs):
        x, y = create_sigmoid()
        sub_df = pd.DataFrame({'_step': [int(x_i * x_scale) for x_i in x], data_key: y})
        sub_df['run'] = f"run_{i}"
        if df is None:
            df = sub_df
        else:
            df = pd.concat([df, sub_df])
    df['method'] = 'fake'
    return df

