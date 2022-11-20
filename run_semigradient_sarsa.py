import numpy as np
from easydict import EasyDict

from agents.features import moving_averages_with_state
from agents.semigradient_sarsa_agent import Config, semigradient_sarsa
from environment import StocksEnv
from utils.experiment import ExperimentResult, visualize_experiment


def main():
    config = Config(
        num_episodes=100,
        max_timesteps=300,
        alpha=0.9,
        gamma=0.9,
        epsilon=0.1,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-semigradient_sarsa', "eps_length": 300,
            "window_size": 50, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
        features=[lambda state, action: moving_averages_with_state([10, 50], state, action)],
    )
    final_env, profits = semigradient_sarsa(config)
    result = ExperimentResult(
        config=config,
        final_env=final_env,
        profits=profits,
        algorithm='semigradient_sarsa',
    )
    filename = result.to_file()
    visualize_experiment(filename)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    main()
