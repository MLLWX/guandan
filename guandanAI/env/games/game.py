import copy

import numpy as np

from .utils import CARD_RANK_STR_INDEX, CARD_RANK_STR, array2cards
from .player import GuandanPlayer as Player
from .round import GuandanRound as Round
from .judger import GuandanJudger as Judger


class GuandanGame:
    """
    Provide game APIs for env to run guandan and get corresponding state
    information.
    """

    def __init__(self):
        self.np_random = np.random.RandomState()
        self.num_players = 4
        self.winner_id = []

    # 初始化游戏
    def init_game(self):
        """
        Initialize players and state.
        Returns:
            dict: first state in one game
            int: current player's id
        """
        # initialize public variables
        self.winner_group = None
        self.winner_id = []
        # 当前两组玩家的参谋
        self.group_officer = ['2', '2']
        # 当前两组玩家的等级
        self.group_grade = [CARD_RANK_STR_INDEX[officer] for officer in self.group_officer]

        # initialize players
        self.players = [Player(num)
                        for num in range(self.num_players)]

        # 出过的牌
        self.played_cards = [np.zeros((54, ), dtype=np.int8)
                             for _ in range(self.num_players)]
        # 初始化第一局
        self.round = Round(self.np_random, self.played_cards, '2')
        self.round.initiate(self.players, self.winner_group, self.winner_id)

        # 初始化裁判
        self.judger = Judger(self.players, '2')

        # get state of first player
        player_id = self.round.current_player
        # print(player_id)
        self.state = self.get_state(player_id)

        return self.state, player_id
    
    def next_round(self):
        """
        To next round of the whole game
        Returns:
            dict: first state in one round
            int: current player's id
        """
        # 更新两组玩家的等级
        winner_group = self.winner_id[0] % 2
        self.group_grade[winner_group] += self.judger.judge_payoffs(self.winner_id)[winner_group]
        # 保证级数小于等于A
        if self.group_grade[winner_group] > CARD_RANK_STR_INDEX["A"]:
            self.group_grade[winner_group] = CARD_RANK_STR_INDEX["A"]
        # 更新两组玩家的参谋
        self.group_officer = [CARD_RANK_STR[grad] for grad in self.group_grade]
        officer = self.group_officer[winner_group]

        # 初始化player
        for player in self.players:
            player.prior_action = None
            player.prior_played_cards = None

        # 出过的牌
        self.played_cards = [np.zeros((54, ), dtype=np.int8)
                             for _ in range(self.num_players)]
        # 初始化新的一局
        self.round = Round(self.np_random, self.played_cards, officer)
        self.round.initiate(self.players, winner_group, self.winner_id)

        # initialize public variables
        self.winner_id = []

    def init_next_round(self):
        # get state of first player
        player_id = self.round.current_player
        # print(player_id)
         # 初始化裁判
        self.judger = Judger(self.players, self.round.officer)
        self.state = self.get_state(player_id)

        return self.state, player_id
    
    def pay_tribute(self, pay_attribute_cards):
        """
        tribute and detribute
        Args:
            pay_tribute_cards (List[ndarray]): pay tribute cards arrays
        Returns:
            dict: next player's state
            int: next player's id
        """
        if self.round.detribute:
            tribute_cards = self.round.tribute_cards
            tribute_players = self.round.tribute_players
            detribute_players = self.round.detribute_players
            detribute_cards = [array2cards(card_array)[0] for card_array in pay_attribute_cards]

            for i, player in enumerate(tribute_players):
                player._current_hand.remove(tribute_cards[i])
            for i, player in enumerate(detribute_players):
                player._current_hand.remove(detribute_cards[i])
            if len(tribute_cards) == 2:
                max_indx = self.round.max_tribute_cards_index()
                min_indx = (1 - max_indx) % 2

                for player, indx in zip(detribute_players, [max_indx, min_indx]):
                    player._current_hand.append(tribute_cards[indx])
                tribute_players[max_indx]._current_hand.append(detribute_cards[0])
                tribute_players[min_indx]._current_hand.append(detribute_cards[1])
            else:
                detribute_players[0]._current_hand.append(tribute_cards[0])
                tribute_players[0]._current_hand.append(detribute_cards[0])

    def step(self, action):
        """
        Perform one draw of the game
        Args:
            action (ndarray): specific action 
        Returns:
            dict: next player's state
            int: next player's id
        """

        # perfrom action
        player = self.players[self.round.current_player]
        assert ~np.any(action) or np.any(np.all((self.legal_actions - action == 0), axis=1)) # action in legal_actions
        self.round.proceed_round(player, action)

        # 如果出牌
        if np.any(action):
            # 当前可以出的牌
            self.judger.calc_playable_actions(player)
        #
        if self.judger.judge_game(self.players, self.round.current_player):
            self.winner_id.append(self.round.current_player)
        # 轮到下一位玩家出牌
        next_id = (player.player_id + 1) % len(self.players)
        while len(self.players[next_id].current_hand) == 0:
            if next_id not in self.winner_id:
                self.winner_id.append(next_id)
            if self.round.greater_player.player_id == next_id:
                self.round.greater_player = None
                next_id = (next_id + 2) % len(self.players)
                break
            else:
                next_id = (next_id + 1) % len(self.players)

        if self.round.greater_player and self.round.greater_player.player_id == next_id:
            self.round.greater_player = None
        self.round.current_player = next_id

        # 获得下一位玩家的状态
        state = self.get_state(next_id)

        return state, next_id

    # 获取当前玩家的状态
    def get_state(self, player_id):
        """
        Return player's state
        Args:
            player_id (int): player id
        Returns:
            (dict): The state of the player
        """
        player = self.players[player_id]
        # 如果当前小局结束
        if self.is_over():
            # 清空当前动作 (54+13, )
            actions = np.zeros((1, 67), dtype=np.int8)
        # 如果当前小局没有结束
        else:
            actions = player.available_actions(self.round.officer, self.judger.count_heart_officer(player.current_hand),
                                         self.round.greater_player, self.judger)

        # 获得当前状态
        state = player.get_state()
        state["played_cards"] = copy.deepcopy(self.played_cards)
        rest_cards_num = [len(player._current_hand) for player in self.players]
        state["rest_cards_num"] = rest_cards_num
        state["legal_actions"] = actions
        self.legal_actions = np.copy(actions)
        state["winner"] = copy.copy(self.winner_id)
        state["group_grade"] = copy.copy(self.group_grade)
        state["officer_grade"] = copy.copy(CARD_RANK_STR_INDEX[self.round.officer])
        state["history"] = [np.array(_trace, dtype=np.int8).reshape(-1, 67) for _trace in self.round.trace]
        if self.round.greater_player:
            state["greater_action"] = copy.deepcopy(self.round.greater_player.prior_action)
        else:
            state["greater_action"] = np.zeros((67, ), np.int8)

        def round_mid(a, b, c):
            return ((a - b) % 4 + (c - a) % 4) < 4

        partner_id = (player_id + 2) % 4
        if partner_id in self.winner_id:
            partner_action = -np.ones((67, ), dtype=np.int8)
        elif (not self.round.greater_player or self.round.greater_player.player_id == player_id
              or round_mid(player_id, self.round.greater_player.player_id, partner_id)):
            partner_action = np.zeros((67, ), dtype=np.int8)
        else:
            partner_action = np.copy(self.players[partner_id].prior_action)
        state["partner_action"] = partner_action
        return state

    # 获得当前玩家的id
    def get_player_id(self):
        """
        Return current player's id
        Returns:
            int: current player's id
        """
        return self.round.current_player

    # 返回当前玩家的数量
    def get_num_players(self):
        """ Return the number of players in guandan
        Returns:
            int: the number of players in guandan
        """
        return self.num_players

    # 判断当前轮是否结束
    def is_over(self):
        """ Judge whether a game is over
        Returns:
            Bool: True(over) / False(not over)
        """
        # 只有一位玩家出完牌
        if len(self.winner_id) < 2:
            return False
        elif len(self.winner_id) == 2:
            player1 = self.winner_id[0]
            player2 = self.winner_id[1]
            # 双上
            if (player1 % 2) == (player2 % 2):
                return True
            # 不是双上，要等第三位玩家出完牌
            else:
                return False
        # 前三位玩家都已经出完牌，当前小局结束
        elif len(self.winner_id) >= 3:
            return True
        return False
    
    def is_terminal(self):
        """ Judge whether the whole game is over
        Returns:
            Bool: True(over) / False(not over)
        """
        if not self.is_over():
            return False
        winner_group = self.winner_id[0] % 2
        terminal = False
        if self.group_grade[winner_group] == 12 and ((self.winner_id[0] + 2) % 4 in self.winner_id):
            terminal = True
        return terminal
    
    def get_result(self):
        ''' Get the whole game result. 
        Returns:
            result (int): the winner group
        '''
        winner_group = None
        if self.is_terminal():
            winner_group = self.winner_id[0] % 2
        return winner_group