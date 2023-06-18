import random
import torch.multiprocessing as mp

import numpy as np
import torch

from ..games.base import Card


def set_seed(seed):
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res


def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res

def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣', 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank


def print_card(cards):
    ''' Nicely print a card or list of cards

    Args:
        card (string or list): The card(s) to be printed
    '''
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}       │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}    │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}{}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print('   '.join(line))

def simulate(env, num_epochs, q):
    win_counts = np.array([0, 0])
    position_counts = np.array([[0 for _ in range(4)] for _ in range(4)])
    with torch.no_grad():
        for _ in range(num_epochs):
            env.reset()
            while not env.is_terminal():
                trajectories = env.run()
                winner_id = trajectories["winner_id"]
                for i, player_id in enumerate(winner_id):
                    position_counts[player_id][i] += 1
                for player_id in range(4):
                    if player_id not in winner_id:
                        position_counts[player_id][3] += 1
            win_counts[env.get_result()] += 1
    q.put((win_counts, position_counts))

def tournament(env, num_epochs, num_workers):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (GuandanEnv class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A tuple of probability for each group to win, and each player for each position
    '''
    num_epochs_per_worker = num_epochs // num_workers
    win_counts = np.array([0, 0])
    position_counts = np.array([[0 for _ in range(4)] for _ in range(4)])

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []

    for _ in range(num_workers):
        p = ctx.Process(
                target=simulate,
                args=(env, num_epochs_per_worker, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for _ in range(num_workers):
        result = q.get()
        win_counts += result[0]
        position_counts += result[1]

    win_probs = [win_counts[i] / (num_epochs * num_epochs_per_worker) for i in range(2)]

    return win_probs, list(position_counts)
