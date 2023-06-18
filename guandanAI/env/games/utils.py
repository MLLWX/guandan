import os
import json
from collections import OrderedDict, Counter
from itertools import  chain, product, combinations
import copy

import numpy as np

from .base import Card, CARD_LIST, RANK_INDEX, RANK_LIST


# Read required docs
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# a map of card to its type. Also return both dict and list to accelerate
with open(os.path.join(ROOT_PATH, 'jsondata/card_type.json'), 'r') as file:
    CARD_TYPE = json.load(file, object_pairs_hook=OrderedDict)

# a map of type to its cards
with open(os.path.join(ROOT_PATH, 'jsondata/type_card.json'), 'r') as file:
    TYPE_CARD = json.load(file, object_pairs_hook=OrderedDict)

# rank list of solo character of cards
CARD_RANK_STR = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
                 'A', 'B', 'R']
CARD_RANK_STR_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                       '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10,
                       'K': 11, 'A': 12, 'B': 13, 'R': 14}

CARD_INDEX = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
              '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10,
              'K': 11, 'A': 12, 'B': 14, 'R': 15}

def sub(a, b):
    for item in b:
        if item in a:
            a.remove(item)
    return a

def same_suit(cards_list):
    suit = cards_list[0].suit
    for card in cards_list:
        if card.suit != suit:
            return False
    return True

def is_available_type(cards_str, key):
    if len(cards_str) == len(key) and sorted(cards_str) == sorted(key):
        return True
    return False


def cards2str_with_suit(cards):
    ''' Get the corresponding string representation of cards with suit

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    '''
    return ' '.join([card.suit + card.rank for card in cards])


def cards2str(cards):
    """
    Get the corresponding string representation of cards

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    """
    response = ''
    for card in cards:
        if card.rank == '':
            response += card.suit[0]
        else:
            response += card.rank
    return response

def cards2array(cards):
    """
    Get the corresponding ndarray representation of cards

    Args:
        cards (list): list of Card objects

    Returns:
        rst: ndarray representation of cards
    """
    rst = np.zeros((54, ), dtype=np.int8)
    for card in cards:
        rst += card.get_array()
    return rst

def array2cards(cards_array):
    """
    Get the corresponding list of cards

    Args:
        cards_array (ndarray)

    Returns:
        rst: corresponding list of cards
    """
    rst = []
    for i, num in enumerate(cards_array):
        if i != 52 and i != 53:
            suit, rank = CARD_LIST[i]
        else:
            # RJ or BJ
            suit, rank = CARD_LIST[i], ""
        for _ in range(num):
            rst.append(Card(suit, rank))
    return rst

def ho_array2str(ho_array):
    """
    Get the str the heart offficer array stand for

    Args:
        ho_array (ndarray): shape (13, )

    Returns:
        rst: corresponding str
    """
    rst = ""
    for i, num in enumerate(ho_array):
        rst += RANK_LIST[i]*num
    return rst

def cards2ho_array(cards):
    rst = np.zeros((13, ), dtype=np.int8)
    for card in cards:
        if isinstance(card, Card):
            rst += card.get_ho_array()
        else:
            rst[RANK_INDEX[card]] += 1
    return rst

def action2str(action, officer):
    """
    Get the corresponding card str in card_type.json of action

    Args:
        action (ndarray): action of shape (54+13, )
        officer: current officer

    Returns:
        rst: str representation of action
    """

    h_officer_card = Card("H", officer)
    cards_list = sorted(array2cards(action[:54]))
    h_officer_card_num = np.sum(action[54:])

    if h_officer_card_num == 0:
        cards_str = cards2str(cards_list)
        # 处理 “A” 的顺序
        if cards_str == "2345A":
            cards_str = "A2345"
        if cards_str == "222AAA":
            cards_str = "AAA222"
        if cards_str == "2233AA":
            cards_str = "AA2233"
        # 检查是否同花顺
        cards_type = CARD_TYPE_ROUND[cards_str][0][0]
        if cards_type == "solo_chain_5" and same_suit(cards_list):
            cards_str = "*" + cards_str
    else:
        # 删除Heart officer
        cards_list = [card for card in cards_list if card != h_officer_card]
        cards_str = cards2str(cards_list) + ho_array2str(action[54:])
        for key, value in CARD_TYPE_ROUND.items():
            if is_available_type(cards_str, key):
                cards_str = key
                break
        cards_type = value[0][0]
        if cards_type == "solo_chain_5" and same_suit(cards_list):
            cards_str = "*" + cards_str
    return cards_str

def actions2type(actions_array, officer):
    """
    Get the corresponding card str in card_type.json of action

    Args:
        actions (ndarray): action of shape (N, 54+13)
        officer: current officer

    Returns:
        rst: types of actions
    """
    if np.any(actions_array[0]):
        cards_str = [action2str(actions_array[i], officer) for i in range(len(actions_array))]
        types =  [CARD_TYPE_ROUND[card_str][0] for card_str in cards_str]
    elif len(actions_array) == 1:
        types = [["pass", "0"]]
    else:
        types = [["pass", "0"], *actions2type(actions_array[1:], officer)]

    # types = []
    # for action_array in actions_array:
    #     if np.any(action_array):
    #         card_str = action2str(action_array, officer)
    #         types.append(CARD_TYPE_ROUND[card_str][0])
    #     else:
    #         types.append(["pass", "0"])

    return types


def target2actions(candidate, target, officer, h_officer_num):
    """
    transfer target to actions that can be played with current hand.

    Args:
        candidate (list): A list of the cards of candidate
        target (string): A string representing the number of cards of target
        officer: current officer
        h_officer_num: num of heart officer in current hand

    Returns:
        rst: list of action(ndarray)
    """
    rst = []
    h_officer_card_array = Card("H", officer).get_array()
    candidate_array = cards2array(candidate) - h_officer_num*h_officer_card_array
    if target.startswith("*"):
        # straight_flush
        target = target[1:]
        target_index = np.array([RANK_INDEX[r] for r in target])
        target_expand = []
        for i in range(4):
            target_array = np.zeros((54, ), dtype=np.int8)
            target_array[target_index+i*13] = 1
            target_expand.append(target_array)

        candidate_no_h_officer = array2cards(candidate_array)
        for t in target_expand:
            target_cards = array2cards(t)
            temp_cards = []
            for card in candidate_no_h_officer:
                if card in target_cards and card not in temp_cards:
                    temp_cards.append(card)
            lose_num = len(target_cards) - len(temp_cards)
            
            for num in range(lose_num, h_officer_num+1):
                # 红心参谋替代
                rst_cards = [Card("H", officer)] * num
                remain_num = len(temp_cards) - (num - lose_num)
                to_appended_cards = combinations(temp_cards, remain_num)
                for cards_to_append in to_appended_cards:
                    cards_to_replace = [card for card in target_cards if card not in cards_to_append]
                    cards_array = cards2array(rst_cards+list(cards_to_append))
                    ho_array = cards2ho_array(cards_to_replace)
                    rst.append(np.concatenate((cards_array, ho_array), axis=0))
    else:
        com = []
        stable_cards = []
        lose_num = 0
        for rank, num in Counter(target).items():
            if rank == "B":
                if candidate_array[52] < num:
                    return rst
                else:
                    stable_cards.extend([Card("BJ", "")]*num)
            elif rank == "R":
                if candidate_array[53] < num:
                    return rst
                else:
                    stable_cards.extend([Card("RJ", "")]*num)
            else:
                index = RANK_INDEX[rank]
                rank_candidate = candidate_array[index:52:13]
                candidate_num = np.sum(rank_candidate)
                _lose_num = np.clip((num - candidate_num), 0, None)
                lose_num += _lose_num

                com_num = min(candidate_num, num)
                com_cards = []
                for n, suit in zip(rank_candidate, Card.valid_suit[:4]):
                    com_cards.extend([Card(suit, rank)]*n)
                com.append(set(combinations(com_cards, com_num)))
        if stable_cards:
            # 有大小王
            rst.append(np.concatenate((cards2array(stable_cards), np.zeros((13, ), dtype=np.int8)), axis=0))
        else:
            card_rst = list(product(*com))
            for num in range(lose_num, min(h_officer_num, len(target))+1):
                rst_cards = [Card("H", officer)] * num
                for item in card_rst:
                    temp_cards = list(chain(*item))
                    remain_num = len(temp_cards) - (num - lose_num)
                    if remain_num:
                        to_appended_cards = combinations(temp_cards, remain_num)
                    else:
                        to_appended_cards = [[]]
                    for cards_to_append in to_appended_cards:
                        cards_2_append_rank = [card.rank for card in cards_to_append]
                        cards_to_replace = sub(list(target), cards_2_append_rank)
                        cards_array = cards2array(rst_cards+list(cards_to_append)) 
                        ho_array = cards2ho_array(cards_to_replace)
                        rst.append(np.concatenate((cards_array, ho_array), axis=0))
    return rst

CARD_TYPE_ROUND = copy.deepcopy(CARD_TYPE)
TYPE_CARD_ROUND = copy.deepcopy(TYPE_CARD)

def change_card_type_round(officer):
    global CARD_TYPE_ROUND, TYPE_CARD_ROUND

    CARD_TYPE_ROUND = copy.deepcopy(CARD_TYPE)
    for card, value in CARD_TYPE_ROUND.items():
        if officer in card:
            if value[0][0] == "solo":
                value[0][1] = "13"
            elif value[0][0] == "pair":
                value[0][1] = "13"
            elif value[0][0] == "trio":
                value[0][1] = "13"
            elif value[0][0] == "trio_pair" and card.count(officer) == 3:
                value[0][1] = "13"
            elif value[0][0].startswith("bomb"):
                value[0][1] = "13"

    # 生成 TYPE_CARD_ROUND
    TYPE_CARD_ROUND = {}
    for card, type_weight in CARD_TYPE_ROUND.items():
        card_type, weight = type_weight[0]
        if card_type in TYPE_CARD_ROUND:
            if weight in TYPE_CARD_ROUND[card_type]:
                TYPE_CARD_ROUND[card_type][weight].append(card)
            else:
                TYPE_CARD_ROUND[card_type][weight] = [card]
        else:
            TYPE_CARD_ROUND[card_type] = {}
            TYPE_CARD_ROUND[card_type][weight] = [card]

# 获得比之前玩家出的牌更大的牌
def get_gt_actions(player, greater_player, officer, h_officer_num):
    """
    Provide player's cards which are greater than the ones played by
    previous player in one round

    Args:
        player (Player object): the player waiting to play cards
        greater_player (Player object): the player who played current biggest cards.
        officer: current officer
        h_officer_num: heart officer num in current player's hand card

    Returns:
        list: ndarray of greater cards, shape (n, 54+13)

    Note:
        1. return value contains 'pass'(all zero)
    """
    if officer != "A":
        change_card_type_round(officer)

    # add 'pass' to legal actions
    gt_actions = [np.zeros((67, ), dtype=np.int8)]
    current_hand = player.current_hand  # list of Cards
    target_action = greater_player.prior_action # ndarray of shape(67, )
    target_str = action2str(target_action, officer)

    target_types = CARD_TYPE_ROUND[target_str]
    type_dict = {}
    for card_type, weight in target_types:
        if card_type not in type_dict:
            type_dict[card_type] = weight

    # 如果上个玩家出四大天王，没有牌比它大
    if 'rocket' in type_dict:
        return np.asarray(gt_actions)

    # 炸弹
    type_dict['rocket'] = -1

    for i in range(10, 3, -1):
        if i == 5:
            if "straight_flush" not in type_dict:
                type_dict["straight_flush"] = -1
            else:
                break
        if "bomb_" + str(i) not in type_dict:
            type_dict["bomb_" + str(i)] = -1
        else:
            break

    for card_type, weight in type_dict.items():
        candidate = TYPE_CARD_ROUND[card_type]
        for can_weight, cards_list in candidate.items():
            if int(can_weight) > int(weight):
                for cards in cards_list:
                    actions_contained = target2actions(current_hand, cards, officer, h_officer_num)
                    gt_actions.extend(actions_contained)
    return np.unique(np.asarray(gt_actions, dtype=np.int8), axis=0).reshape(-1, 67)

def init_guandan_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    res = []
    for i in range(2):
        res += [Card(suit, rank) for suit in suit_list for rank in rank_list]
        res.append(Card('BJ', ''))
        res.append(Card('RJ', ''))
    return res


if __name__ == '__main__':
   pass
