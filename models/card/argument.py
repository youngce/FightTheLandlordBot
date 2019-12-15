class CornerArgument:
    # b = 56 * 0.7
    # margin = 17 * 0.7
    # d = -5 * .7
    def __init__(self, width_margin: int, width: int, shift: int, height_margin: int = 18, height: int = 135,
                 rank_height: int = 82):
        self.width_margin = width_margin
        self.width = width
        self.shift = shift
        self.height_margin = height_margin
        self.height = height
        self.rank_height = rank_height

    def zoom(self, ratio) -> 'CornerArgument':
        return CornerArgument(int(self.width_margin * ratio),
                              int(self.width * ratio),
                              int(self.shift * ratio),
                              int(self.height_margin * ratio),
                              int(self.height * ratio),
                              int(self.rank_height * ratio))


class CornerArgumentFactory:
    @staticmethod
    def own() -> CornerArgument:
        return CornerArgument(width_margin=17, width=56, shift=-5, height_margin=18, height=135, rank_height=82)

    @staticmethod
    def lord_cards():
        return CornerArgument(width_margin=17, width=40, shift=-7, height_margin=15, height=93, rank_height=55)

    @staticmethod
    def pool():
        return CornerArgument(width_margin=20, width=40, shift=-7, height_margin=15, height=110, rank_height=66)
#     todo left cards
