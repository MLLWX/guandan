import numpy as np

from ..games import HISTORY_PER_PLAYER_SIZE, CARD_RANK_STR_INDEX

class RLAgentV2:
    ''' A RL agent version 2. 
    '''

    def __init__(self, id, model):
        ''' Initilize the random agent

        Args: 
            model (torch.nn.Module): A model to step
            pay_tribute_model (torch.nn.Module): A model to pay tribute
            
        '''
        self.id = id
        self.model = model
        self.train_tribute = hasattr(self.model, "tribute_model")
        self.tribute_input = None

    def reset(self):
        self.tribute_input = None

    def extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            extracted_state (dict): dict of obs, legal actions and history
            observation shape is (501, )
        '''

        def grade2onehot(grade):
            grades_candidate = np.eye(13, dtype=np.int8)
            if isinstance(grade, (list, tuple)):
                return grades_candidate[grade]
            elif isinstance(grade, int):
                return grades_candidate[grade]
            return None 
        
        # 得到legal_actions
        extracted_state = {}
        extracted_state["legal_actions"] = state["legal_actions"]

        # 得到observation
        current_player = self.id
        # 队友位置、上位位置、下位位置
        positions = [(current_player+2) % 4, 
                     ((current_player-1) % 4), ((current_player+1) % 4)]
        played_cards_position_shift = [state["played_cards"][position] for position in positions]
        played_cards_all = sum(state["played_cards"])
        # group grade 按自己队伍与敌方队伍排序
        group_grade = state["group_grade"]
        group_grade = [group_grade[current_player % 2], group_grade[(current_player + 1) % 2]]
        offficer_grade = state["officer_grade"]
        group_grade_onehot = grade2onehot(group_grade)
        officer_grade_onehot = grade2onehot(offficer_grade)
        # 当前未打完牌的player
        active = np.ones((4, ), dtype=np.int8)
        active[state["winner"]] = 0
        active = active[positions]
        # 各个玩家手里牌的数量，按固定规则排序
        rest_cards_num = [state["rest_cards_num"][position] for position in positions]
        rest_cards_num = np.eye(28, dtype=np.int8)[rest_cards_num].reshape(-1)
        extracted_state["obs"] = np.concatenate([state["pokers"],
                                                 *played_cards_position_shift,
                                                 played_cards_all,
                                                 rest_cards_num,
                                                 state["greater_action"],
                                                 state["partner_action"],
                                                 *group_grade_onehot,
                                                 officer_grade_onehot,
                                                 active,
                                                 ], axis=0)
        
        # 得到history （按positions排序）
        extracted_state["history"] = []
        for position in positions:
            _history = np.zeros((HISTORY_PER_PLAYER_SIZE, 67), dtype=np.int8)
            _history_len = len(state["history"][position])
            _history[:_history_len] = state["history"][position]
            # 为了之后模型处理的方便，最小长度设为1
            if _history_len == 0:
                _history_len = 1
            extracted_state["history"].append((_history_len, _history))
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
        obs = state["obs"]
        history = state["history"]
        legal_actions = state["legal_actions"]
        action_index = self.model.step(obs, legal_actions, history, is_train=True)

        return legal_actions[action_index], state

    def eval_step(self, raw_state):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (ndarray): The action predicted (randomly chosen) by the random agent
        '''
        state = self.extract_state(raw_state)
        obs = state["obs"]
        history = state["history"]
        legal_actions = state["legal_actions"]
        action_index = self.model.step(obs, legal_actions, history, is_train=False)
        return legal_actions[action_index]
    
    def pay_tribute(self, masked_hand_cards_array, state):
        ''' pay tribute

        Args:
            masked_hand_cards_array (ndarray): An ndarray that represents the current hand cards 
            less than or equal to 10
            state (dict): current state for pay tribute 

        Returns:
            action (ndarray): The card to pay tribute
        '''
        if self.train_tribute:
            hand_cards_array = state["poker"]
            officer = state["officer"]
            officer_array = np.zeros((13, ))
            officer_array[CARD_RANK_STR_INDEX[officer]] = 1
            actions = np.where(masked_hand_cards_array)[0]
            actions_index = self.model.tribute_model.pay_tribute(hand_cards_array, officer_array, actions)
            index = actions[actions_index]
        else:
            actions = np.where(masked_hand_cards_array)[0]
            index = np.random.choice(actions)

        tribute_array = np.zeros((54, ), dtype=np.int8)
        tribute_array[index] = 1
        if self.train_tribute:
            self.tribute_input = np.concatenate([hand_cards_array, officer_array, tribute_array], axis=0)
        return tribute_array