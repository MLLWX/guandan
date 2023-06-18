import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..env import Env, RandomAgent, RLAgent, RuleAgent, tournament
from ..dmc import DecisionModel, DecisionModelSimple

WIN_PROBS = []
FRAMES = []

def load_agents(player_list, device):
    agents = []
    for i, player_str in enumerate(player_list):
        if isinstance(player_str, (RandomAgent, RLAgent)):
            player = player_str
        elif player_str == "random":
            player = RandomAgent()
        elif player_str == "base":
            model = DecisionModel(device=device)
            path = "models/base_model.ckpt"
            model.load_state_dict(torch.load(path, map_location=device))
            player = RLAgent(i, model)
        elif player_str == "rule_base":
            player = RuleAgent(i)
        elif player_str == "base_simple":
            model = DecisionModelSimple(device=device)
            path = "models/base_simple_model.ckpt"
            model.load_state_dict(torch.load(path, map_location=device))
            player = RLAgent(i, model)
        else:
            if "base_simple" in player_str:
                model = DecisionModelSimple(device=device)
            else: 
                model = DecisionModel(device=device)
            path = player_str
            model.load_state_dict(torch.load(path, map_location=device))
            player = RLAgent(i, model)
        agents.append(player)
    return agents

def plot_wins(flags, out_dir="imgs"):
    # win_probs 
    plt.plot(FRAMES, np.asarray(WIN_PROBS)[:, 0], label="win_probs")
    plt.axhline(0.5, ls="--", color="k")
    plt.xlabel("frames")
    plt.ylabel("WP")
    plt.title(f"{flags.train_model} vs {flags.eval_model}")
    wins_fig_file_name = os.path.join(out_dir, f"{flags.train_model}_{flags.eval_model}.jpg")
    plt.savefig(wins_fig_file_name, dpi=300)


def evaluate_for_training(learner_model, flags, frame, num_workers, num_epochs, if_plot=False):
    device = flags.eval_device if flags.eval_device=="cpu" else "cuda:" + flags.eval_device 
    train_model = learner_model.__class__(device=device)
    train_model.load_state_dict(learner_model.state_dict())
    eval_model = flags.eval_model
    player_list = [RLAgent(0, train_model), eval_model, RLAgent(2, train_model), eval_model]
    agents = load_agents(player_list, device)
    env = Env(render=False)
    env.set_agents(agents)
    win_probs, position_probs = tournament(env, num_epochs, num_workers)
    WIN_PROBS.append(win_probs)
    FRAMES.append(frame)
    if if_plot:
        plot_wins(flags)
    return win_probs, position_probs

def evaluate(player_paths, num_workers, num_epochs):
    agents = load_agents(player_paths, "cuda:0")
    env = Env(render=False)
    env.set_agents(agents)
    win_probs, position_counts = tournament(env, num_epochs, num_workers)
    print(f"win_probs: {win_probs[0]:.2f} - {win_probs[1]:.2f}")
    for i in range(4):
        print(f"player {i}: {position_counts[i]}")