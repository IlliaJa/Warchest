class EloTracker:
    """
    Tracks Elo ratings for named agents.
    Truncation should be recorded as a draw — both players failed to finish the game.
    """

    def __init__(self, k=32, initial=1000):
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def rating(self, agent: str) -> float:
        return self.ratings.setdefault(agent, float(self.initial))

    def win(self, winner: str, loser: str) -> None:
        ra, rb = self.rating(winner), self.rating(loser)
        ea = self._expected(ra, rb)
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - (1 - ea))

    def draw(self, a: str, b: str) -> None:
        ra, rb = self.rating(a), self.rating(b)
        ea = self._expected(ra, rb)
        self.ratings[a] = ra + self.k * (0.5 - ea)
        self.ratings[b] = rb + self.k * (0.5 - (1 - ea))

    def _expected(self, ra: float, rb: float) -> float:
        return 1 / (1 + 10 ** ((rb - ra) / 400))
