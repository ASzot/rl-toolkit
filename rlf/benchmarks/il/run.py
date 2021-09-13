import argparse
import os
import os.path as osp
import time

from rlf.benchmarks.il.data_download import check_data
from rlf.exp_mgr.run_exp import execute_from_string


def full_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algs", type=str, default="bco,bc,gaifo,gaifo_s,gail,sqil")
    parser.add_argument("--envs", type=str, default="nav,pick,push,maze2d,ant,hand")
    parser.add_argument("--seed", type=str, default="31,41,51")
    args, other_args = parser.parse_known_args()

    algs = args.algs.split(",")
    envs = args.envs.split(",")

    check_data(envs)
    cur_dir = osp.dirname(osp.abspath(__file__))

    cmd_folders = os.listdir(osp.join(cur_dir, "cmds"))

    for i, alg in enumerate(algs):
        for j, env in enumerate(envs):
            if alg in cmd_folders:
                execute_from_string(
                    ["--cmd", f"{alg}/{env}", "--seed", args.seed, *other_args],
                    osp.join(cur_dir, "cmds"),
                    cur_dir,
                )
            else:
                execute_from_string(
                    ["--cmd", f"{alg}/{env}", "--seed", args.seed, *other_args],
                    None,
                    None,
                )

            if (i * len(envs)) + (j + 1) != len(algs) * len(envs):
                print("")
                print("-" * 10)
                print("Waiting for other jobs to start")
                print("-" * 10)
                print("")
                time.sleep(5)


if __name__ == "__main__":
    full_run()
