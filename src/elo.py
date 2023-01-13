import numpy as np
from typing import Union, List, Callable

from .params import *
from .score_functions import exponential_score_function, expected_scores_function

class Elo:
    def __init__(
        self,
        d_value: float = DEFAULT_D_VALUE,
        score_function_base: float = DEFAULT_SCORING_FUNCTION_BASE,
        custom_k_value_function: Callable = None,
        custom_score_function: Callable = None,
        log_base: int = LOG_BASE
    ):
        """
        :param d_value: D parameter in Elo algorithm that determines how much Elo difference affects win
        probability
        :param score_function_base: base value to use for scoring function; scores are approximately
        multiplied by this value as you improve from one place to the next (minimum allowed value is 1,
        which results in a linear scoring function)
        :param custom_k_value_function: a function that takes K value in Elo algorithm that determines 
        how much ratings increase or decrease after each match
        :param custom_score_function: a function that takes an integer input and returns a numpy array
        of monotonically decreasing values summing to 1
        :param log_base: base to use for logarithms throughout the Elo algorithm. Traditionally Elo
        uses base-10 logs
        """
        self.get_k_value = custom_k_value_function or get_k_value
        self._expected_score_func = expected_scores_function(d_value, log_base)
        self._actual_score_func = custom_score_function or exponential_score_function(alpha=score_function_base)

    def get_new_ratings(
            self,
            initial_ratings: Union[List[float], np.ndarray],
            result_order: List[int] = None,
    ) -> np.ndarray:
        """
        Update ratings based on results. Takes an array of ratings before the matchup and returns an array with
        the updated ratings. Provided array should be ordered by the actual results (first place finisher's
        initial rating first, second place next, and so on).

        Example usage:
        >>> elo = MultiElo()
        >>> elo.get_new_ratings([1200, 1000])
        array([1207.68809835,  992.31190165])
        >>> elo.get_new_ratings([1200, 1000, 1100, 900])
        array([1212.01868209, 1012.15595083, 1087.84404917,  887.98131791])

        :param initial_ratings: array of ratings (float values) in order of actual results
        :param result_order: list where each value indicates the place the player in the same index of
        initial_ratings finished in. Lower is better. Identify ties by entering the same value for players
        that tied. For example, [1, 2, 3] indicates that the first listed player won, the second listed player
        finished 2nd, and the third listed player finished 3rd. [1, 2, 2] would indicate that the second
        and third players tied for 2nd place. (default = range(len(initial_ratings))
        :return: array of updated ratings (float values) in same order as input
        """
        if not isinstance(initial_ratings, np.ndarray):
            initial_ratings = np.array(initial_ratings)
        n = len(initial_ratings)  # number of players
        actual_scores = self.get_actual_scores(n, result_order)
        expected_scores = self.get_expected_scores(initial_ratings)
        scale_factor = np.vectorize(self.get_k_value)(np.where(actual_scores - expected_scores >= 0, -initial_ratings, initial_ratings)) * (n - 1)
        ratings = initial_ratings + scale_factor * (actual_scores - expected_scores)
        # if rating < 0 change the rating = 0
        return np.maximum(ratings, 0)

    def get_actual_scores(self, n: int, result_order: List[int] = None) -> np.ndarray:
        """
        Return the scores to be awarded to the players based on the results.

        :param n: number of players in the matchup
        :param result_order: list indicating order of finish (see docstring for MultiElo.get_new_ratings
        for more details
        :return: array of length n of scores to be assigned to first place, second place, and so on
        """
        # calculate actual scores according to score function, then sort in order of finish
        result_order = result_order or list(range(n))
        scores = self._actual_score_func(n)
        scores = scores[np.argsort(np.argsort(result_order))]

        # if there are ties, average the scores of all tied players
        distinct_results = set(result_order)
        if len(distinct_results) != n:
            for place in distinct_results:
                idx = [i for i, x in enumerate(result_order) if x == place]
                scores[idx] = scores[idx].mean()

        # self._validate_actual_scores(scores, result_order)
        return scores

    def get_expected_scores(self, ratings: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Get the expected scores for all players given their ratings before the matchup.

        :param ratings: array of ratings for each player in a matchup
        :return: array of expected scores for all players
        """
        if not isinstance(ratings, np.ndarray):
            ratings = np.array(ratings)
        if ratings.ndim > 1:
            raise ValueError(f"ratings should be 1-dimensional array (received {ratings.ndim})")

        return self._expected_score_func(ratings)