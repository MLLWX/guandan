''' Implement Judger class
'''
import numpy as np

from .utils import TYPE_CARD_ROUND
from .utils import cards2array, target2actions


class GuandanJudger:
    """ Determine what cards a player can play
    """

    def __init__(self, players, officer):
        """
        Initilize the Judger class
        """
        # 4位玩家
        # 当前可以出的牌
        self.playable_actions = []
        self.officer = officer
        self.players = players
        for player in self.players:
            # 当前可以出的牌型
            self.playable_actions.append(self.playable_actions_from_hand(player.current_hand))

    # 查找当前可以出的牌
    def playable_actions_from_hand(self, current_hand):
        """ Get playable actions from hand
        Returns:
            playable_actions: ndarray of playable actions, shape (n, 54+13)
        """
        playable_actions = []
        h_officer_num = self.count_heart_officer(current_hand)

        for _, candidate in TYPE_CARD_ROUND.items():
            for _, cards_list in candidate.items():
                for cards in cards_list:
                    actions_contained = target2actions(current_hand, cards, self.officer, h_officer_num)
                    playable_actions.extend(actions_contained)
        return np.unique(np.asarray(playable_actions, dtype=np.int8), axis=0).reshape(-1, 67)

    # 重新计算当前可以出的牌型
    def calc_playable_actions(self, player):
        """
        Recalculate all legal cards the player can play according to his
        current hand.
        Args:
            player (Player object): object of Player
        Returns:
            list: list of string of playable cards
        """
        hand_cards_array = cards2array(player.current_hand)
        playable_actions = self.playable_actions[player.player_id]
        candidate_playerable = playable_actions[~np.any((hand_cards_array - playable_actions[:, :54] < 0), axis=1)]
        self.playable_actions[player.player_id] = candidate_playerable

    # 获取当前玩家可出的牌
    def get_playable_actions(self, player):
        """ Provide all legal cards the player can play according to his
        current hand.
        Args:
            player (Player object): object of Player
        Returns:
            playable_actions: ndarray representation of playable actions, shape(n, 54+13)
        """
        return self.playable_actions[player.player_id]

    # 判断当前玩家是否出完牌
    @staticmethod
    def judge_game(players, player_id):
        """
        Args:
            players (list): list of Player objects
            player_id (int): integer of player's id
        """
        player = players[player_id]
        if len(player.current_hand) == 0:
            return True
        return False

    @staticmethod
    def judge_payoffs(winner_id):
        payoffs = np.array([0, 0, 0, 0])
        winner_group = winner_id[0] % 2
        loser_group = (winner_id[0] + 1) % 2
        # 双上
        if len(winner_id) == 2:
            payoffs[[winner_group, winner_group+2]] = 3
            payoffs[[loser_group, loser_group+2]] = -3
        elif len(winner_id) == 3:
            # 1、3
            if winner_id[0] % 2 == winner_id[2] % 2:
                payoffs[[winner_group, winner_group+2]] = 2
                payoffs[[loser_group, loser_group+2]] = -2
            # 1、4
            else:
                payoffs[[winner_group, winner_group+2]] = 1
                payoffs[[loser_group, loser_group+2]] = -1
        return payoffs
        # 统计红心参谋的数量
    def count_heart_officer(self, cards_list):
        cnt = 0
        for card in cards_list:
            if card.rank == self.officer and card.suit == 'H':
                cnt += 1
        return cnt
