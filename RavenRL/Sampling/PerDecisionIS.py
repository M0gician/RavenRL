import numpy as np
from typing import List
from RavenRL.Policy import Policy
from RavenRL.Sampling import SamplingMethod
from collections import defaultdict
from typing import Tuple


class PDIS(SamplingMethod):
    """
    The implementation of Per-Decision Importance Sampling Estimator
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

        super(PDIS, self).__init__(dataset, behv_policy, gamma, n_proc)

    def get_episodic_est(self, idx=None):
        """ Calculate the PDIS estimate of the evaluation policy over one trajectory

        Parameters
        ----------
        idx : int
            The index of a specific trajectory.
        """
        assert isinstance(self.eval_policy, Policy)
        if idx is None:
            trajectory_idx = int(np.random.randint(len(self.dataset), size=1))
        else:
            assert isinstance(idx, int)
            trajectory_idx = idx
        trajectory = self.dataset[trajectory_idx]

        s_history = np.array(trajectory['s'])
        a_hisotry = np.array(trajectory['a'])
        r_history = np.array(trajectory['r'])

        importance_weights = np.array(list(map(
            self.get_importance_weights, enumerate(zip(s_history, a_hisotry, r_history)))))
        cumulative_importance_weight = np.array(
            [importance_weights[:i].prod() for i in range(importance_weights.size)])
        return np.sum(cumulative_importance_weight * r_history)


if __name__ == "__main__":
    from RavenRL.Utils import generate_dataset
    from RavenRL.Env import setup_env
    from RavenRL.Policy import RandomPolicy

    env_id = "cartpole"
    env, _, _ = setup_env(env_id)
    dataset = generate_dataset(env_id, 5000)

    behv_policy = RandomPolicy(env)
    eval_policy = RandomPolicy(env)

    sampling = PDIS(dataset, behv_policy, gamma=1)
    sampling.load_eval_policy(eval_policy)
    # for _ in range(1000):
    #     print(sampling.get_episodic_est())
    print(np.average(np.array([sampling.get_episodic_est() for _ in range(50)])))
