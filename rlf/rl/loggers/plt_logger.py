from collections import defaultdict
from typing import List, Optional

import rlf.rl.utils as rutils
from rlf.rl.loggers.base_logger import BaseLogger


class PltLogger(BaseLogger):
    def __init__(
        self,
        save_keys: Optional[List[str]],
        x_name: str,
        y_names: Optional[List[str]],
        titles: Optional[List[str]],
    ):
        """
        :param save_keys: If None, everything is plotted.
        :param y_names: If None, y-axis name is the plot key name
        :param titles: If None, the title is the plot key name.
        """
        super().__init__(True)
        self.save_keys = save_keys
        self.x_name = x_name
        self.y_names = y_names
        self.titles = titles
        self.logged_vals = defaultdict(list)
        self.logged_steps = defaultdict(list)

    def init(self, args):
        self.args = args
        super().init(args)

    def log_vals(self, key_vals, step_count):
        for k, v in key_vals.items():
            self.logged_vals[k].append(v)
            self.logged_steps[k].append(step_count)

    def close(self):
        if self.save_keys is None:
            self.save_keys = list(self.logged_vals.keys())
        if self.y_names is None:
            self.y_names = self.save_keys
        if self.titles is None:
            self.titles = self.save_keys

        # Plot everything
        for k, y_name, title in zip(self.save_keys, self.y_names, self.titles):
            rutils.plot_line(
                self.logged_vals[k],
                f"final_{k}",
                rutils.get_save_dir(self.args),
                False,
                x_vals=self.logged_steps[k],
                x_name=self.x_name,
                y_name=y_name,
                title=title,
            )
