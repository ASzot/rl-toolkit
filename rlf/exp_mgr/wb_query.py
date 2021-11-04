try:
    import wandb
except:
    pass
import os
import os.path as osp
import sys
from typing import Dict

from rlf.exp_mgr import config_mgr


def query(
    select_fields, filter_fields: Dict[str, str], cfg="./config.yaml", verbose=True
):
    config_mgr.init(cfg)

    wb_proj_name = config_mgr.get_prop("proj_name")
    wb_entity = config_mgr.get_prop("wb_entity")

    api = wandb.Api()

    query_dict = {}

    for f, v in filter_fields.items():
        if f == "group":
            query_dict["group"] = v
        elif f == "tag":
            query_dict["tags"] = v
        else:
            raise ValueError(f"Filter {f}: {v} not supported")

    def log(s):
        if verbose:
            print(s)

    log("Querying with")
    log(query_dict)

    runs = api.runs(f"{wb_entity}/{wb_proj_name}", query_dict)
    log(f"Returned {len(runs)} runs")

    base_data_dir = config_mgr.get_prop("base_data_dir")
    ret_data = []
    for run in runs:
        dat = {}
        for f in select_fields:
            if f == "last_model":
                env_name = run.config["env_name"]
                model_path = osp.join(base_data_dir, "checkpoints", env_name, run.name)
                if not osp.exists(model_path):
                    raise ValueError(f"Could not locate model path {model_path}")
                model_idxs = [
                    int(model_f.split("_")[1].split(".pt")[0])
                    for model_f in os.listdir(model_path)
                    if model_f.startswith("model_")
                ]
                max_idx = max(model_idxs)
                final_model_f = osp.join(model_path, f"model_{max_idx}.pt")
                v = final_model_f
            elif f == "final_train_success":
                # Will by default get the most recent train success metric, if
                # none exists then will get the most recent eval success metric
                # (useful for methods that are eval only)
                succ_keys = [
                    k
                    for k in list(run.summary.keys())
                    if isinstance(k, str) and "success" in k
                ]
                train_succ_keys = [
                    k for k in succ_keys if "eval_train" not in k and "avg_" in k
                ]
                if len(train_succ_keys) > 0:
                    use_k = train_succ_keys[0]
                else:
                    use_k = succ_keys[0]
                v = run.summary[use_k]
            elif f == "status":
                v = run.state
            else:
                # return None
                raise ValueError(f"Select field {f} not supported")
            dat[f] = v
        ret_data.append(dat)
    log(f"Got data {ret_data}")
    return ret_data


def query_s(query_str, verbose=True):
    select_s, filter_s = query_str.split(" WHERE ")
    select_fields = select_s.replace(" ", "").split(",")

    filter_fields = filter_s.replace(" ", "").split(",")
    filter_fields = [s.split("=") for s in filter_fields]
    filter_fields = {k: v for k, v in filter_fields}
    return query(select_fields, filter_fields, verbose=verbose)


if __name__ == "__main__":
    query_str = " ".join(sys.argv[1:])
    query_s(query_str)
