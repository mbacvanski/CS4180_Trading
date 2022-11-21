import dataclasses
import time
from datetime import datetime
from typing import List

import dill as pickle
import numpy as np
from dateutil import tz
from matplotlib import pyplot as plt

from agents.agents import AgentConfig
from environment import TradingEnv


@dataclasses.dataclass
class ExperimentResult:
    config: AgentConfig
    final_env: TradingEnv
    profits: List[float]
    max_possible_profits: List[float]
    buy_and_hold_profits: List[float]
    algorithm: str
    timestamp: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.timestamp = datetime.utcfromtimestamp(time.time()).replace(tzinfo=tz.gettz('UTC')) \
            .astimezone(tz=tz.gettz('America/Boston')).strftime('%Y-%m-%d--%H-%M-%S')

    def to_file(self):
        filename = f'data/{self.algorithm}_{self.timestamp}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            return filename

    @staticmethod
    def from_file(path: str) -> 'ExperimentResult':
        with open(path, 'rb') as handle:
            return pickle.load(handle)


def visualize_experiment(filename: str):
    r = ExperimentResult.from_file(filename)
    plot_profits(r)
    plt.show()

    r.final_env.render_together()


def plot_profits(r: ExperimentResult):
    plt.plot(range(0, len(r.profits)), r.profits, 'blue', label='Agent profit')
    # plt.plot(range(0, len(r.profits)), r.max_possible_profits, 'green', label='Maximum possible profits')
    plt.plot(range(0, len(r.buy_and_hold_profits)), r.buy_and_hold_profits, 'green', label='Buy and hold profit')

    plt.title(f'Profits per training episode. Avg={np.average(r.profits)}')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.legend()
