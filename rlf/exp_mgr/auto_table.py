from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from rlf.exp_mgr.wb_query import query

MISSING_VALUE = 0.2444


def get_df_for_table_txt(
    table_txt: str, lookup_k: str
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Extracts a dataframe to use for `plot_table` automatically from text copied
    from excel. You want to include the row and column names in the text.
    Example:
    ```
            mirl train	mirl eval	airl train	airl eval
    100	5TK1	UKGZ	YIGE	GN31
    50	14MT	C0JW	KUOP	OVS2
    ```
    If you are getting eval metrics, `lookup_k` should likely be 'final_train_success'.
    """
    data = []
    row_headers = []
    for line in table_txt.split("\n"):
        if line.strip() == "":
            continue
        line_parts = line.split("\t")
        row_ident = line_parts[0].strip()
        if row_ident == "":
            # These are the column headers.
            col_headers = line_parts[1:]
        else:
            assert len(line_parts[1:]) == len(col_headers)
            row_headers.append(row_ident)
            for group, col in zip(line_parts[1:], col_headers):
                r = query([lookup_k], {"group": group}, use_cached=True)
                if r is None or len(r) == 0:
                    r = [{lookup_k: MISSING_VALUE}]

                for d in r:
                    data.append(
                        {"method": row_ident, "type": col, lookup_k: d[lookup_k]}
                    )
    return pd.DataFrame(data), col_headers, row_headers


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
        :param df: The index of the data frame does not matter, only the row values and column names matter.
        :param col_key: A string from the set of columns.
        :param row_key: A string from the set of columns (but this is used to form the rows of the table).
        :param renames: Only used for display name conversions. Does not affect functionality.
        Example: the data fame might look like
            ```
       democount        type  final_train_success
    0     100  mirl train               0.9800
    1     100  mirl train               0.9900
    3     100   mirl eval               1.0000
    4     100   mirl eval               1.0000
    12     50  mirl train               0.9700
    13     50  mirl train               1.0000
    15     50   mirl eval               1.0000
    16     50   mirl eval               0.7200
            ```
            `col_key='type', row_key='demcount',
            cell_key='final_train_success'` plots the # of demos as rows and
            the type as columns with the final_train_success values as the cell
            values. Duplicate row and columns are automatically grouped
            together.
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

    # Add the column title row.
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
                            "\\textbf{ "
                            + (
                                f"%.{n_decimals}f {{\\scriptsize $\pm$ %.{n_decimals}f }}"
                                % (val, std)
                            )
                            + " }"
                        )
                    else:
                        row_str.append(
                            f" %.{n_decimals}f {{\\scriptsize $\pm$ %.{n_decimals}f }} "
                            % (val, std)
                        )

        all_s.append(col_sep.join(row_str))

    ret_s = ""
    if auto_wrap:
        ret_s += "\\resizebox{\\columnwidth}{!}{\n"
    ret_s += "\\begin{tabular}{%s}\n" % ("c" * (len(col_order) + 1))
    # Line above the table.
    ret_s += "\\toprule\n"

    # Separate the column headers from the rest of the table by a line.
    ret_s += all_s[0] + row_sep
    ret_s += "\\midrule\n"

    ret_s += row_sep.join(all_s[1:]) + row_sep
    # Line below the table.
    ret_s += "\\bottomrule"

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
