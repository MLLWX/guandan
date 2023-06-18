import random

import numpy as np
from ..games.utils import actions2type, CARD_RANK_STR


class RuleAgent:
    ''' A Rule base agent. 
    '''

    def __init__(self, id):
        ''' Initilize the rule base agent

        Args: 
            id(int): id of the agent 
        '''
        self.id = id
        self._card_type_order = {
            'solo': 0, 'pair': 1, 'trio': 1, 'pair_chain_3': 3,
            'trio_chain_2': 3, 'trio_pair': 5, 'solo_chain_5': 6,
            'bomb_4': 7, 'bomb_5': 8, 'straight_flush': 9, 
            'bomb_6': 10, 'bomb_7': 11, 'bomb_8': 12, 'bomb_9': 13, 
            'bomb_10': 14, 'rocket': 15, 'pass': 16
        }
    
    def reset(self):
        pass
    
    def card_type_order(self, _type):
        return self._card_type_order[_type]

    def extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            extracted_state (dict): state after processing for the strategy
        '''
        
        # 得到legal_actions
        extracted_state = {}
        extracted_state["legal_actions"] = (state["legal_actions"], 
                                            actions2type(state["legal_actions"], CARD_RANK_STR[state["officer_grade"]]))

        # 得到observation
        current_player = self.id
        # 自己的位置、队友位置、上位位置、下位位置
        positions = [current_player, (current_player+2) % 4, 
                     ((current_player-1) % 4), ((current_player+1) % 4)]
        played_cards_position_shift = [state["played_cards"][position] for position in positions]
        extracted_state["played_cards_all"] = sum(played_cards_position_shift)
        extracted_state["played_cards"] = played_cards_position_shift

        # group grade 按自己队伍与敌方队伍排序
        group_grade = state["group_grade"]
        extracted_state["group_grade"] = [CARD_RANK_STR[group_grade[current_player % 2]], 
                                          CARD_RANK_STR[group_grade[(current_player + 1) % 2]]]

        offficer_grade = state["officer_grade"]
        extracted_state["officer"] = CARD_RANK_STR[offficer_grade]

        # 当前未打完牌的player
        active = np.ones((4, ), dtype=np.int8)
        active[state["winner"]] = 0
        extracted_state["active"] = active[positions]

        # greater action
        extracted_state["greater_action"] = actions2type([state["greater_action"]], 
                                                         extracted_state["officer"])[0]
        extracted_state["priority"] = extracted_state["greater_action"][0] == "pass"

        # partner action
        if ~np.any(state["partner_action"] == -1):
            extracted_state["partner_action"] = actions2type([state["partner_action"]], 
                                                            extracted_state["officer"])[0]
        # 每个玩家剩余的牌的数量
        extracted_state["rest_cards_num"] = [state["rest_cards_num"][position] for position in positions]

        return extracted_state

    def step(self, raw_state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (ndarray): The action predicted (randomly chosen) by the random agent
            state: for training to record
        '''
        state = self.extract_state(raw_state)
        action = self.strategy(state)

        return action

    def eval_step(self, raw_state):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (ndarray): The action predicted (randomly chosen) by the random agent
        '''
        
        return self.step(raw_state)
    
    def pay_tribute(self, masked_hand_cards_array, state):
        ''' pay tribute

        Args:
            masked_hand_cards_array (ndarray): An ndarray that represents the current hand cards 
            less than or equal to 10
            state (dict): current state for pay tribute 

        Returns:
            action (ndarray): The card to pay tribute
        '''
        actions = np.where(masked_hand_cards_array)[0]
        index = np.random.choice(actions)
        tribute_array = np.zeros((54, ), dtype=np.int8)
        tribute_array[index] = 1
        return tribute_array
    
    def strategy(self, state):
        if state["priority"]:
            action = self.normal_strategy_with_priority(state)
        else:
            action = self.normal_strategy_without_priority(state)
        return action


    def normal_strategy_with_priority(self, state):
        action = self.win_with_high_prob_strategy(state)
        if action is not None:
            return action
        action = self.help_ally_strategy_with_priority(state)
        if action is not None:
            return action
        action = self.reduce_hand_cards_strategy_with_priority(state)
        if action is not None:
            return action
        action = self.resist_enemy_strategy_with_priority(state)
        if action is not None:
            return action
        legal_action_arrays, legal_actions = state["legal_actions"]
        return legal_action_arrays[self.min_strategy(legal_actions)]

    def win_with_high_prob_strategy(self, state):
        legal_action_arrays, legal_actions = state["legal_actions"]
        legal_action_types = set((legal_action[0] for legal_action in legal_actions))
        if len(legal_action_types) == 1:
            card_type = legal_action_types.pop()
            return legal_action_arrays[self.min_strategy(legal_actions, card_type=card_type)]
        else:
            fire_card = set([legal_action[0] for legal_action in legal_actions
                             if self.is_fire_card(legal_action)])
            non_fire_card = [legal_action[0] for legal_action in legal_actions
                             if not self.is_fire_card(legal_action)]
            if len(fire_card) and len(non_fire_card):
                card_type = random.choice(non_fire_card)
                idx = self.min_strategy(legal_actions, card_type=card_type)
                return legal_action_arrays[idx]
        return None

    def help_ally_strategy_with_priority(self, state):
        legal_action_arrays, legal_actions = state["legal_actions"]
        legal_action_types = set((legal_action[0] for legal_action in legal_actions))
        ally_rest = state["rest_cards_num"][1]
        if ally_rest == 1 and 'solo' in legal_action_types:
            return legal_action_arrays[self.min_strategy(legal_actions, card_type='solo')]
        if ally_rest == 2 and 'pair' in legal_action_types:
            return legal_action_arrays[self.min_strategy(legal_actions, card_type='pair')]
        return None

    def resist_enemy_strategy_with_priority(self, state):
        enemy_rest = min(state["rest_cards_num"][2:])
        legal_action_arrays, legal_actions = state["legal_actions"]
        legal_action_types = set((legal_action[0] for legal_action in legal_actions))
        if enemy_rest == 1 and 'pair' in legal_action_types:
            return legal_action_arrays[self.min_strategy(legal_actions, card_type='pair')]
        if enemy_rest == 2 and 'solo' in legal_action_types:
            return legal_action_arrays[self.min_strategy(legal_actions, card_type='solo')]
        return None

    def reduce_hand_cards_strategy_with_priority(self, state):
        legal_action_arrays, legal_actions = state["legal_actions"]
        legal_action_types = set((legal_action[0] for legal_action in legal_actions))
        if 'trio_pair' in legal_action_types:
            return legal_action_arrays[self.min_strategy(legal_actions, card_type='trio_pair')]
        return None

    def normal_strategy_without_priority(self, state):
        action = self.help_ally_strategy_without_priority(state)
        if action is not None:
            return action

        legal_action_arrays, legal_actions = state["legal_actions"]
        legal_action_types = set([legal_action[0] for legal_action in legal_actions])
        greater_action = state["greater_action"]

        assert greater_action[0] != "pass"

        if greater_action[0] in legal_action_types:
            idx = self.min_strategy(legal_actions, greater_action[0])
            return legal_action_arrays[idx]
        elif len(legal_actions) > 1:
            # 有除了pass之外的选择
            rank = int(greater_action[1])
            if rank > 8:
                idx = self.min_strategy(legal_actions)
                return legal_action_arrays[idx]
        return np.zeros((67, ), dtype=np.int8)

    def help_ally_strategy_without_priority(self, state):
        if not state["active"][1]:
            return None
        ally_action = state["partner_action"]
        if self.is_fire_card(ally_action):
            return np.zeros((67, ), dtype=np.int8)
        if int(ally_action[1]) >= 11:
            return np.zeros((67, ), dtype=np.int8)
        return None

    def min_strategy(self, legal_actions, card_type=None):
        if not card_type:
            all_card_types = [legal_action[0] for legal_action in legal_actions]
            card_type = min(all_card_types, key=self.card_type_order)
        index_rank = [(i, int(rank)) for i, (_type, rank) in enumerate(legal_actions) if _type == card_type]
        return min(index_rank, key=lambda item: item[1])[0]
    
    @staticmethod
    def is_fire_card(action):
        return (action[0] in ["straight_flush", "rocket"] or 
                int(action[1])>=14 or action[0].startswith("bomb"))
    
