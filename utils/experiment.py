import dataclasses
import time
from datetime import datetime
from typing import List

import dill as pickle
from dateutil import tz
from matplotlib import pyplot as plt

from agents.semigradient_sarsa_agent import Config
from environment import TradingEnv


@dataclasses.dataclass
class ExperimentResult:
    config: Config
    final_env: TradingEnv
    profits: List[float]
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
    plt.plot(range(0, len(r.profits)), r.profits)
    plt.show()
