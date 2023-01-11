import numpy as np
from numpy import ndarray


def linear_score_function(n: int) -> ndarray:
    """
    With the linear score function the "points" awarded scale linearly from first place
    through last place. For example, improving from 2nd to 1st place has the same sized
    benefit as improving from 5th to 4th place.

    :param n: number of players
    :return: array of the points to assign to each place (summing to 1)
    """
    return np.array([(n - p) / (n * (n - 1) / 2) for p in range(1, n + 1)])

def exponential_score_function(alpha: float):
    """
    With an exponential score function with alpha > 1, more points are awarded to the top
    finishers and the point distribution is flatter at the bottom. For example, improving
    from 2nd to 1st place is more valuable than improving from 5th place to 4th place. A
    larger alpha value means the scores will be more weighted towards the top finishers.

    :param alpha: alpha for teh exponential score function (> 1)
    :return: a function that takes parameter n for number of players and returns an array
    of the points to assign to each place (summing to 1)
    """
    def _exponential_score_template(n: int, alpha: float) -> ndarray:
        if alpha < 1:
            raise ValueError("alpha must be >= 1")
        if alpha == 1:
            return linear_score_function(n)  # it converges to this as alpha -> 1

        out = np.array([alpha ** (n - p) - 1 for p in range(1, n + 1)])
        return out / sum(out)

    return lambda n: _exponential_score_template(n, alpha)

def expected_scores_function(d: float, log_base: int) -> np.ndarray:
    """
    Get the expected scores for all players given their ratings before the matchup.

    :param d: D parameter in Elo algorithm that determines how much Elo difference affects win
    probability
    :param log_base: base to use for logarithms throughout the Elo algorithm. Traditionally Elo
    uses base-10 logs
    :return: a function that takes parameter ratings for ratings of players and returns an array
    of expected scores for all players
    """
    def _expected_score_template(ratings: np.ndarray, d: float, log_base: int) -> ndarray:
        # get all pairwise differences
        diff_mx = ratings - ratings[:, np.newaxis]

        # get individual contributions to expected score using logistic function
        logistic_mx = 1 / (1 + log_base ** (diff_mx / d))
        np.fill_diagonal(logistic_mx, 0)

        # get each expected score (sum individual contributions, then scale)
        expected_scores = logistic_mx.sum(axis=1)
        n = len(ratings)
        denom = n * (n - 1) / 2  # number of individual head-to-head matchups between n players
        expected_scores = expected_scores / denom

        # this should be guaranteed, but check to make sure
        if not np.allclose(1, sum(expected_scores)):
            raise ValueError("expected scores do not sum to 1")

        return expected_scores

    return lambda ratings: _expected_score_template(ratings, d, log_base)

