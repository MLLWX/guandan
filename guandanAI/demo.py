from env.agents import RandomAgent, RLAgent
from env.utils import set_seed, tournament, Logger
from env import Env


def play_demo_random(seed):
    # Make environments
    env = Env(render=True)

    # Seed numpy, torch, random
    set_seed(seed)

    env.set_agents(
        [RLAgent(i, 0) for i in range(env.num_players)])

    while not env.is_terminal():
        env.run()


def main():
    play_demo_random(1)


if __name__ == '__main__':
    main()
