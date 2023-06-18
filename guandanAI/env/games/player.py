''' Implement Guandan Player class
'''
import numpy as np

from .utils import get_gt_actions
from .utils import cards2array, array2cards, CARD_INDEX, cards2str
from .base import Card


class GuandanPlayer:
    """
    Player can store cards in the player's hand and the role,
    determine the actions can be made according to the rules,
    and can perform corresponding action
    """

    def __init__(self, player_id):
        ''' Give the player an id in one game
        Args:
            player_id (int): the player_id of a player
        Notes:
            当前轮该玩家出的牌
            1. played_cards: The cards played in one round
            当前玩家手上的牌
            2. _current_hand: The rest of the cards after playing some of them
        '''
        self.player_id = player_id
        self._current_hand = []
        self.prior_action = None
        self.tribute_card = None

    # 当前手上的牌
    @property
    def current_hand(self):
        return self._current_hand

    def set_current_hand(self, value):
        self._current_hand = value

    def get_state(self):
        state = {}
        state['pokers'] = cards2array(self._current_hand)

        return state

    # 获取当前可以进行的动作
    def available_actions(self, officer, h_officer_num, greater_player=None, judger=None):
        """
        Get the actions can be made based on the rules
        Args:
            officer: current officer
            h_officer_num: num of heart officer in current player's hand cards
            greater_player (Player object): player who played current biggest cards.
            judger (Judger object): object of Judger
        Returns:
            list: list of string of actions. Eg: ['pass', '8', '9', 'T', 'J']
        """
        # 之前没有玩家出牌或者自己打出最大牌
        if greater_player is None or greater_player.player_id == self.player_id:
            # 获得当前可以执行的动作
            actions = np.copy(judger.get_playable_actions(self))
        # 获得比上一位玩家更大的牌
        else:
            actions = get_gt_actions(self, greater_player, officer, h_officer_num)
        return actions

    # 出牌
    def play(self, action, greater_player, officer):
        """
        Perfrom action
        Args:
            action (ndarray): action of shape (54+13, )
            greater_player (Player object): The player who played current biggest cards.
        Returns:
            object of Player: If there is a new greater_player, return it, if not, return None
        """
        # 不出牌
        self.prior_action = np.copy(action)
        if ~np.any(action):
            return greater_player
        # 出牌
        else:
            # 历史动作
            _current_hand_array = cards2array(self._current_hand) - action[:54]
            self.set_current_hand(array2cards(_current_hand_array))

            return self

    # 当前手上大王的数量
    def count_RJ(self):
        cnt = 0
        for card in self._current_hand:
            if card.suit == 'RJ':
                cnt += 1
        return cnt
    
    def get_sorted_hand(self, officer):
        def get_index(card):
            card_str = cards2str([card])
            if card_str == officer:
                card_index = 13
            else:
                card_index = CARD_INDEX[card_str]
            return card_index
        return sorted(self._current_hand, key=get_index)

    # 获取当前玩家进贡的牌
    def get_tribute_card(self, officer):
        sorted_cards = self.get_sorted_hand(officer)
        tribute_card_index = -1
        # 进贡除了红心参谋以外最大的牌
        while sorted_cards[tribute_card_index] == Card("H", officer):
            tribute_card_index -= 1
        
        return sorted_cards[tribute_card_index]

    def set_tribute_card(self, card):
        self.tribute_card = card
