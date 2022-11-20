from easydict import EasyDict

from agents.features import moving_averages_with_state
from agents.semigradient_sarsa_agent import Config, semigradient_sarsa
from environment import StocksEnv


def main():
    config = Config(
        num_episodes=1,
        max_timesteps=100,
        alpha=0.9,
        gamma=0.9,
        epsilon=0.1,
        env=StocksEnv(EasyDict({
            "env_id": 'stocks-v0', "eps_length": 300,
            "window_size": 50, "train_range": None, "test_range": None,
            "stocks_data_filename": 'STOCKS_GOOGL'
        })),
        features=[lambda state, action: moving_averages_with_state([10, 50], state, action)]
    )
    result = semigradient_sarsa(config)
    print(result)


if __name__ == '__main__':
    main()
