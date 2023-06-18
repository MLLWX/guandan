import numpy as np

CARD_LIST = [
    "SA", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "ST", "SJ", "SQ", "SK",
    "HA", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "HT", "HJ", "HQ", "HK", 
    "CA", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CT", "CJ", "CQ", "CK",
    "DA", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "DT", "DJ", "DQ", "DK", 
    "BJ", "RJ"
]

RANK_SORT_LIST = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

SUIT_LIST = ['S', 'H', 'C', 'D']

CARD_SORT = [suit+rank for rank in RANK_SORT_LIST for suit in SUIT_LIST]
CARD_SORT.extend(["BJ", "RJ"])

CARD_SORT_INDEX = {key: value for value, key in enumerate(CARD_SORT)}

RANK_LIST = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

RANK_INDEX = {key: value for value, key in enumerate(RANK_LIST)}

class Card:
    '''
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    '''
    suit = None
    rank = None
    valid_suit = ['S', 'H', 'C', 'D', 'BJ', 'RJ']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented
        
    def __lt__(self, other):
        if isinstance(other, Card):
            return CARD_SORT_INDEX[str(self)] < CARD_SORT_INDEX[str(other)]
        else:
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: SA, S2, S3, HA, H2, H3
        '''
        return self.suit + self.rank
    
    def __repr__(self):
        return str(self)

    def get_array(self):
        ''' Get array representation of a card.

        Returns:
            ndarray: array representation of a card of shape (54, ).
        '''
        rst = np.zeros((54, ), dtype=np.int8)
        rst[CARD_LIST.index(str(self))] = 1
        return rst
    
    def get_ho_array(self):
        rst = np.zeros((13, ), dtype=np.int8)
        rst[RANK_INDEX[self.rank]] = 1
        return rst
