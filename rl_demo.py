import torch

from guandanAI.env.agents import RLAgent, RandomAgent
from guandanAI.env.utils import set_seed
from guandanAI.env import Env, tournament
from guandanAI.dmc import DecisionModel


def play_demo_random(seed):
    # Make environments
    env = Env(render=True)
    # Seed numpy, torch, random
    set_seed(seed)
    # Make model
    model = DecisionModel("cuda:0")
    model.load_state_dict(torch.load("models/model.ckpt", map_location="cuda:0"))
    env.set_agents(
        [RLAgent(0, model), RandomAgent(), RLAgent(2, model), RandomAgent()])

    # win_probs, position_probs = tournament(env, 10, 5)
    # print(f"win probability: {win_probs[0]} - {win_probs[1]}")
    # print(f"position count:")
    # for i in range(4):
    #     print(f"player {i}: {position_probs[i]}")
    while not env.is_terminal():
        env.run()


def main():
    play_demo_random(1)


if __name__ == '__main__':
    main()