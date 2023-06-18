import numpy as np


class HumanAgent:
    ''' A human agent.
    '''

    def __init__(self):
        ''' Initilize the random agent

        Args:
            
        '''
        pass

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (ndarray): The action predicted (randomly chosen) by the random agent
        '''
        

        legal_actions = state['legal_actions']
        action_index = np.random.choice(len(legal_actions))
        return legal_actions[action_index]
    
    def pay_tribute(self, masked_hand_cards_array, hand_cards):
        ''' pay tribute

        Args:
            masked_hand_cards_array (ndarray): An ndarray that represents the current hand cards 
            less than or equal to 10

        Returns:
            action (ndarray): The card to pay tribute
        '''
        index = np.random.choice(np.where(masked_hand_cards_array)[0])
        action = np.zeros((54, ), dtype=np.int8)
        action[index] = 1
        return action