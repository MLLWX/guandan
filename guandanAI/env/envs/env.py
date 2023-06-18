import copy

import numpy as np

from ..games import GuandanGame, Card, SUIT_LIST
from ..games import array2cards, cards2array, ho_array2str


color2num = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "fuchsia": 35, 
    "cyan": 36,
    "white": 37,
}
def colorize(string, color="black", bold=False, highlight = False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: black, red, green, yellow,
    blue, fuchsia, cyan, white
    """
    attr = []
    num = color2num[color]
    if highlight: 
        num += 10
    attr.append(str(num))
    if bold: 
        attr.append('1')
    attrs = ';'.join(attr)
    return '\x1b[%sm%s\x1b[0m' % (attrs, string)

class Env(object):
    num_players: int
    num_actions: int
    '''
    The Env class. 
    '''
    def __init__(self, render=False):
        ''' Initialize the environment
        '''
        self.timestep = 0
        self.render = render
        self.num_players = 4 # 四个人
        self.dim_actions = 54 + 13 # 54种牌 + 13种Heart officer 的选择
        self.state_shape = (0, ) # 处理之后的状态

        self.game = GuandanGame()
        self.round_num = 1
    
    def reset(self):
        self.round_num = 1
        self.game = GuandanGame()

    def next_round(self):
        ''' Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        '''
        if self.round_num == 1:
            state, player_id = self.game.init_game()
        else:
            for agent in self.agents:
                agent.reset()
            self.game.next_round()
            self.pay_tribute()
            state, player_id = self.game.init_next_round()
        return self._extract_state(state), player_id

    def step(self, action):
        ''' Step forward

        Args:
            action (ndarray): The action taken by the current player

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        '''
        self.timestep += 1
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def set_agents(self, agents):
        '''
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        '''
        self.agents = agents

    def run(self, is_training=False):
        '''
        Run a game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            output (dict): Contains trajectories of state and actions and payoff of the game
        '''        
        trajectories = {
            "states": [[] for _ in range(self.num_players)], 
            "actions": [[] for _ in range(self.num_players)],
            "payoffs": [],
        }
        state, player_id = self.next_round()

        if self.render:
            winner_id = []
            out_str = f"ROUND {self.round_num}, Grad {self.game.group_grade[0]} : {self.game.group_grade[1]}, officer {self.game.round.officer}"
            print(colorize(out_str, "red", True, True))
        while not self.is_over():
            # Agent plays
            if not is_training:
                action = self.agents[player_id].eval_step(state)
            else:
                action, extracted_state = self.agents[player_id].step(state)
                # Save action and state
                trajectories["states"][player_id].append(extracted_state)
                trajectories["actions"][player_id].append(action)

            # Environment steps
            state, next_player_id = self.step(action)
            # render
            if self.render:
                print(f"player {player_id}: {sorted(array2cards(action[:54]))}")
                if ho_array2str(action[54:]):
                    h_str = ho_array2str(action[54:])
                    print(f"heart officer as {h_str}")
                if self.game.winner_id != winner_id:
                    print(f"player {player_id} finish hand cards")
                    winner_id = copy.deepcopy(self.game.winner_id)

            player_id = next_player_id

        self.round_num += 1
        # Payoffs
        payoffs = self.get_payoffs()
        trajectories["payoffs"] = payoffs
        trajectories["winner_id"] = self.game.winner_id

        if self.render:
            print(colorize(f"ROUND {self.round_num - 1} END. payoffs: {list(payoffs)}", "blue", True))


        return trajectories
    
    def pay_tribute(self):
        if self.game.round.detribute:
            pay_tribute_cards = []
            for player in self.game.round.detribute_players:
                masked_hand_cards = [card for card in player.current_hand if card < Card("S", "J")]
                for suit in SUIT_LIST:
                    officer_card = Card(suit, self.game.round.officer)
                    if officer_card in masked_hand_cards:
                        masked_hand_cards.remove(officer_card)
                masked_hand_cards_array = cards2array(masked_hand_cards)
                state = {"poker": cards2array(player.current_hand), "officer": self.game.round.officer}
                pay_tribute_card_array = self.agents[player.player_id].pay_tribute(masked_hand_cards_array, state)

                assert np.all(pay_tribute_card_array >= 0) and np.sum(pay_tribute_card_array) == 1
                assert array2cards(pay_tribute_card_array)[0] in masked_hand_cards

                pay_tribute_cards.append(pay_tribute_card_array)
            
            self.game.pay_tribute(pay_tribute_cards)

    def is_over(self):
        ''' Check whether the curent game is over

        Returns:
            (boolean): True if current game is over
        '''
        return self.game.is_over()
    
    def is_terminal(self):
        ''' Check whether the curent game is over
        Returns:
            (boolean): True if current whole game is over
        '''
        return self.game.is_terminal()

    def get_player_id(self):
        ''' Get the current player id

        Returns:
            (int): The id of the current player
        '''
        return self.game.get_player_id()


    def get_state(self, player_id):
        ''' Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (numpy.array): The observed state of the player
        '''
        return self._extract_state(self.game.get_state(player_id))

    def _extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            state (dict): dict of obs and legal actions
        '''
        extracted_state = state

        return extracted_state

    # 获得收益
    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.winner_id)
    
    def get_result(self):
        ''' Get the whole game result. 
        Returns:
            result (int): the winner group
        '''
        return self.game.get_result()
