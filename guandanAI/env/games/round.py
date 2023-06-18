""" Implement Guandan Round class
"""
import random
from collections import deque

import numpy as np

from .dealer import GuandanDealer as Dealer
from .utils import CARD_TYPE_ROUND, cards2str

HISTORY_PER_PLAYER_SIZE = 10

class GuandanRound:
    """ Round can call other Classes' functions to keep the game running
    """

    def __init__(self, np_random, played_cards, officer):
        self.np_random = np_random
        self.played_cards = played_cards
        self.officer = officer
        self.trace = [deque(maxlen=HISTORY_PER_PLAYER_SIZE) for _ in range(4)]

        self.greater_player = None
        self.dealer = Dealer(self.np_random, self)
        self.winner_group = None
        self.winners = []
        self.tribute_cards = None
        self.tribute_players = None
        self.detribute_players = None
        self.detribute = False

    def initiate(self, players, winner_group, winners):
        """
        初始化一轮
        Args:
            :param officer:
            :param winners: 上一小局玩家出完牌的顺序
            :param players: 所有玩家列表
            :param winner_group: 上一小局获胜的组
        """
        self.players = players
        self.greater_player = None
        # 上轮赢的组
        self.winner_group = winner_group
        self.winners = winners

        self.dealer.init(self.players)
        # 如果是游戏的第一小局，随机选择一个玩家开始游戏
        if not self.winner_group:
            self.current_player = random.randint(0, 3)
        # 否则进贡
        else:
            self.current_player, self.detribute = self.pay_tribute()

    # 进行一轮
    def proceed_round(self, player, action):
        """
        Call other Classes's functions to keep one round running
        Args:
            player (object): object of Player
            action (ndarray): array representation of cards (shape(54+13, ))
        Returns:
            object of Player: player who played current biggest cards.
        """
        self.trace[player.player_id].append(np.copy(action))
        # 如果当前玩家出牌
        if np.any(action):
            self.played_cards[self.current_player] += action[:54]
        # 出牌
        self.greater_player = player.play(action, self.greater_player, self.officer)
        return self.greater_player

    # 进贡
    def pay_tribute(self):
        """
        :return: 下一步进发出动作的玩家，是否为还贡
        """
        # 单贡
        if len(self.winners) >= 3:
            # 进贡的玩家
            tribute_player = None
            # 还牌的玩家
            detribute_player = self.players[self.winners[0]]
            for player in self.players:
                if player.player_id not in self.winners:
                    tribute_player = player
                    break
            # 如果抓到2个大王，则抗贡
            if tribute_player.count_RJ() == 2:
                # 上游出牌
                self.tribute_players = None
                self.detribute_players = None
                return self.winners[0], False
            else:
                self.tribute_cards = [tribute_player.get_tribute_card(self.officer)]
                self.tribute_players = [tribute_player]
                self.detribute_players = [detribute_player]
                # 上游玩家还贡
                return tribute_player.player_id, True
        # 双上
        elif len(self.winners) == 2:
            tribute_players = []
            for player in self.players:
                if player.player_id not in self.winners:
                    tribute_players.append(player)
            # 双下抓到两张大王，抗贡
            if tribute_players[0].count_RJ() + tribute_players[1].count_RJ() >= 2:
                self.tribute_players = None
                self.detribute_players = None
                return self.winners[0], False
            else:
                self.tribute_cards = [player.get_tribute_card(self.officer) for player in tribute_players]
                self.tribute_players = tribute_players
                self.detribute_players = [self.players[i] for i in self.winners]
                current_player = tribute_players[self.max_tribute_cards_index()].player_id
                if cards2str([self.tribute_cards[0]]) == cards2str([self.tribute_cards[1]]):
                    # 贡牌相等
                    current_player = (self.winners[0] + 1) % 4
                return current_player, True
            
    def max_tribute_cards_index(self):
        """
        Returns:
            index of the max tribute card in a list of two (0 or 1) 
        """
        cards_str_list = [cards2str([card]) for card in self.tribute_cards]
        weights = [int(CARD_TYPE_ROUND[card_str][0][1]) for card_str in cards_str_list]
        return np.argmax(weights)
        
