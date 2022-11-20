from typing import List

import numpy as np
from numpy import ndarray

from environment import Action, State


def moving_average(window: int, state: State) -> float:
    # average the `window` most recent close prices
    return np.average(state.history[-window:, 0])


def moving_averages_with_state(windows: List[int], state: State, action: Action) -> ndarray:
    averages = [moving_average(w, state) for w in windows]
    action_onehot = np.zeros(shape=len(Action))
    action_onehot[action.value] = 1
    return np.hstack([averages, action_onehot])
