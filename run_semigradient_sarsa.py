import numpy as np
from easydict import EasyDict

from agents.features import seven_indicators
from agents.semigradient_sarsa_agent import Config, semigradient_sarsa
from environment import StocksEnv
from utils.experiment import ExperimentResult, visualize_experiment


def main():
    config = Config(
        num_episodes=300,
        max_timesteps=100,
        alpha=0.9,
        gamma=0.9,
        epsilon=0.1,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-semigradient_sarsa', "eps_length": 1000,
            "window_size": 300, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
        features=[seven_indicators]
    )
    config.env.seed(0)
    final_env, profits, max_possible_profits, buy_and_hold_profits = semigradient_sarsa(config)
    result = ExperimentResult(
        config=config,
        final_env=final_env,
        profits=profits,
        max_possible_profits=max_possible_profits,
        buy_and_hold_profits=buy_and_hold_profits,
        algorithm='semigradient_sarsa',
    )
    filename = result.to_file()
    visualize_experiment(filename)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    main()
