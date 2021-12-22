from typing import Callable, Dict, List, Optional

import pandas as pd

MISSING_VALUE = 0.2444


def plot_table(
    df: pd.DataFrame,
    col_key: str,
    row_key: str,
    cell_key: str,
    col_order: List[str],
    row_order: List[str],
    renames: Dict[str, str] = {},
    error_scaling=1.0,
    n_decimals=2,
    missing_fill_value=MISSING_VALUE,
    error_fill_value=0.3444,
    auto_wrap=False,
    get_row_highlight: Optional[Callable[[str, pd.DataFrame], Optional[str]]] = None,
    write_to=None,
):
    """
    :param df: The index of the data frame does not matter only the row values and column names.
    :param col_key: A string from the set of columns.
    :param row_key: A string from the set of columns (but this is used to form the rows of the table).
    :param renames: Only used for display name conversions. Does not affect functionality.
    """
    df = df.replace("missing", missing_fill_value)
    df = df.replace("error", error_fill_value)

    rows = {}
    for row_k, row_df in df.groupby(row_key):
        df_avg_y = row_df.groupby(col_key)[cell_key].mean()
        df_std_y = row_df.groupby(col_key)[cell_key].std() * error_scaling

        rows[row_k] = (df_avg_y, df_std_y)

    col_sep = " & "
    row_sep = " \\\\\n"

    all_s = []

    def clean_text(s):
        return s.replace("%", "\\%").replace("_", " ")

    row_str = [""]
    for col_k in col_order:
        row_str.append("\\textbf{%s}" % clean_text(renames.get(col_k, col_k)))
    all_s.append(col_sep.join(row_str))

    for row_k in row_order:
        if row_k == "hline":
            all_s.append("\\hline")
            continue
        row_str = []
        row_str.append("\\textbf{%s}" % clean_text(renames.get(row_k, row_k)))
        row_y, row_std = rows[row_k]

        if get_row_highlight is not None:
            sel_col = get_row_highlight(row_k, row_y)
        else:
            sel_col = None
        for col_k in col_order:
            if col_k not in row_y:
                row_str.append("-")
            else:

                val = row_y.loc[col_k]
                std = row_std.loc[col_k]
                if val == missing_fill_value:
                    row_str.append("-")
                elif val == error_fill_value:
                    row_str.append("E")
                else:
                    if col_k == sel_col:
                        row_str.append(
                            "$ \\mathbf{ "
                            + (f"%.{n_decimals}f \pm %.{n_decimals}f" % (val, std))
                            + " }$"
                        )
                    else:
                        row_str.append(
                            f"$ %.{n_decimals}f \pm %.{n_decimals}f $" % (val, std)
                        )

        all_s.append(col_sep.join(row_str))

    ret_s = ""
    if auto_wrap:
        ret_s += "\\resizebox{\\columnwidth}{!}{\n"
    ret_s += "\\begin{tabular}{%s}\n" % ("c" * (len(col_order) + 1))

    ret_s += row_sep.join(all_s)

    ret_s += "\n\\end{tabular}\n"
    if auto_wrap:
        ret_s += "}\n"

    if write_to is not None:
        with open(write_to, "w") as f:
            f.write(ret_s)
        print(f"Wrote result to {write_to}")
    else:
        print(ret_s)

    return ret_s
