from enum import Enum

from .base_logger import BaseLogger
from .plt_logger import PltLogger
from .tb_logger import TbLogger
from .wb_logger import WbLogger


class LoggerChoices(Enum):
    WANDB = "wandb"
    MATPLOTLIB = "matplotlib"
    TB_LOGGER = "tb"
    NONE = "none"


def get_logger(logger_choice: LoggerChoices):
    if logger_choice == LoggerChoices.WANDB:
        return WbLogger()
    elif logger_choice == LoggerChoices.MATPLOTLIB:
        return PltLogger(["avg_r"], "Steps", "Reward", "Reward Curve")
    elif logger_choice == LoggerChoices.TB_LOGGER:
        return TbLogger("./data/tb")
    elif logger_choice == LoggerChoices.NONE:
        return BaseLogger()
    else:
        raise ValueError("Invalid logger selection")
