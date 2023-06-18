from .utils import init_guandan_deck


class GuandanDealer(object):
    """
    Dealer will shuffle, deal cards, and determine players' roles
    """

    def __init__(self, np_random, round):
        """
        Give dealer the deck
        Notes:
            1. deck with 54*2 cards including black joker and red joker
        """
        self.np_random = np_random
        self.deck = init_guandan_deck()
        self.round = round

    # 洗牌
    def shuffle(self):
        """
        Randomly shuffle the deck
        """
        self.np_random.shuffle(self.deck)

    # 发牌
    def deal_cards(self, players):
        """
        Deal cards to players
        Args:
            players (list): list of Player objects
        """
        hand_num = (len(self.deck)) // len(players)
        for index, player in enumerate(players):
            current_hand = self.deck[index * hand_num:(index + 1) * hand_num]
            player.set_current_hand(current_hand)

    # 初始化
    def init(self, players):
        self.shuffle()
        self.deal_cards(players)
