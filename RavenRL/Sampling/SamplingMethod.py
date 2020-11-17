import numpy as np
from typing import List
from RavenRL.Policy import Policy
from collections import defaultdict
from typing import Tuple


class SamplingMethod:
    """
    The base class of a sampling method
    """
    def __init__(self, dataset: List[defaultdict], behv_policy: Policy, gamma=0.9, n_proc=8):
        """
        Parameters
        ----------
        dataset : List[defaultdict]
            A list of trajectories sampled by the behavioral policy.
            Contains (s, a, r, s') for each timestamp
        behv_policy : Policy
            The behavioral policy that generates the dataset.
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        n_proc : int
            The total number of processes that can be spawned for parallelization
        """

        self.n_proc = n_proc
        self.gamma = gamma
        self.dataset = dataset
        self.behv_policy: Policy = behv_policy

        self.eval_policy = None

    def load_eval_policy(self, eval_policy: Policy) -> None:
        """ Load the evaluation policy into the estimator

        Parameters
        ----------
        eval_policy : Policy
            The evaluation policy object.
        """
        assert isinstance(eval_policy, Policy)
        self.eval_policy: Policy = eval_policy

    def get_episodic_est(self, idx=None) -> float:
        raise NotImplementedError

    def get_importance_weights(self, t_s_a_r: Tuple[int, Tuple[np.ndarray, int, int]]) -> float:
        """ Computes the importance weights pi_e / pi_b

        Parameters
        ----------
        t_s_a_r : Tuple[int, Tuple[np.ndarray, int, int]]
            A tuple contains:
                current time `t`
                current state `s`
                current action `a`
                current reward `r`
        """
        assert isinstance(self.eval_policy, Policy)
        # gamma = self.gamma**t
        t, s_a_r = t_s_a_r
        s, a, r = s_a_r
        return self.eval_policy.pi(s, a) / self.behv_policy.pi(s, a)
