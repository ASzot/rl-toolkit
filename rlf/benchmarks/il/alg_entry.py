import rlf.envs.ant
import rlf.envs.fetch
from rlf import RunSettings, run_policy
from rlf.algos import GAIFO, GAIL, SQIL, BaseILAlgo, NestedAlgo
from rlf.benchmarks.il.utils import trim_episodes_trans
from rlf.policies import DistActorCritic
from rlf.rl.loggers import BaseLogger, WbLogger

METHODS = {
    "gail": (GAIL(), DistActorCritic()),
    "gaifo": (GAIFO(), DistActorCritic()),
}


class ILBenchSettings(RunSettings):
    def get_policy(self):
        return METHODS[self.base_args.alg][1]

    def get_algo(self):
        algo = METHODS[self.base_args.alg][0]
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        if isinstance(algo, SQIL):
            algo.il_algo.set_transform_dem_dataset_fn(trim_episodes_trans)
        return algo

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger()

    def get_add_args(self, parser):
        parser.add_argument("--alg")
        parser.add_argument("--env-name")
        parser.add_argument("--no-wb", action="store_true", default=False)


if __name__ == "__main__":
    run_policy(ILBenchSettings())
