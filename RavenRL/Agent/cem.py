import gym
import numpy as np
from multiprocessing import Pool

from typing import List

from RavenRL.Env import setup_env
from RavenRL.Policy import setup_policy


class CEM:
    """
    The standard implementation of the Cross Entropy Method learning algorithm.
    """
    def __init__(self, epochs: int, pop_size: int, elite_ratio: float, n_sample: int,
                 gamma=0.9, extra_std=2.0, extra_decay_time=10, n_proc=8):
        """
        Parameters
        ----------
        epochs : int
            The total number of iterations that the algorithm will try to train and imporve on its current policies.
        pop_size : int
            The total number of candidate policies (population) that will be generated at each epoch.
        elite_ratio : float
            The percentage of candidates that we will keep (elites) to improve on the next policy generation.
            The algorithm will keep the top `elite_ratio` percent of the population to update its mean and variance
                parameter.
        n_sample : int
            The total number of trials a candidate will be tested over the environment at each epoch.
            A big `n_sample` tends to estimate the policy performance more accurately, but also has a longer runtime
        gamma : float
            The discount factor. Mostly predefined by the MDP.
        extra_std : float
            A hyperparameter which adds extra variance to the Covariance matrix
        extra_decay_time : float
            A hyperparameter which controls the scale of the extra variance adds to the Covariance matrix
        n_proc : int
            The total number of processes that can be spawned for parallelization
       """
        self.epochs = epochs
        self.pop_size = pop_size
        self.elite_ratio = elite_ratio
        assert 0 < elite_ratio < 1
        self.elite_size = int(self.pop_size * self.elite_ratio)
        self.n_sample = n_sample
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time
        self.n_proc = n_proc
        self.gamma = gamma

        self.env_id = None
        self.sampler = None
        self.obs_shape = None
        self.act_shape = None

        self.theta_dim = None
        self.means = None
        self.stds = None
        self.elites = None

    def load_env(self, env_id: str) -> None:
        """ Load a supported Gym environment into the algorithm

        Parameters
        ----------
        env_id : str
            The name for the customized Gym environment (defined in `Env`)
        """
        self.env_id = env_id
        _, self.obs_shape, self.act_shape = setup_env(env_id)

    def init_params(self) -> None:
        """ Define the policy shape. Initialize the mean and std vector for policy generation of CEM.
        """
        assert all(isinstance(val, int) for val in [self.obs_shape, self.act_shape])
        self.theta_dim = (self.obs_shape + 1) * self.act_shape
        self.means = np.random.uniform(size=self.theta_dim)
        self.stds = np.ones(self.theta_dim)

    def update_params(self, elites: np.ndarray) -> None:
        """ Use a predefined policy matrix to initialize the mean and std vector of CEM.

        Parameters
        ----------
        elites : np.ndarray
            A predefined policy that is previously trained by CEM
        """
        self.means = np.mean(elites, axis=0)
        self.stds = np.std(elites, axis=0)

    def candidate_eval(self, theta: np.ndarray, monitor=False) -> float:
        """ Run the given candidate policy over the environment and returns its average performance.

        Parameters
        ----------
        theta : np.ndarray
            A candidate policy generated by CEM
        monitor : bool
            A flag which enables the build-in animator of the environment
        """
        assert isinstance(self.env_id, str)
        env, _, _ = setup_env(self.env_id)
        policy = setup_policy(env, theta)
        rewards = np.zeros(self.n_sample)

        if monitor:
            env = gym.wrappers.Monitor(env, self.env_id, force=True)

        # Run policy over the env for `n_sample` times and compute its average as the estimated performance
        for i in range(self.n_sample):
            done = False
            gamma = 1
            s = env.reset()

            while not done:
                a = policy.act(s)
                s_prime, r, done, info = env.step(a)
                rewards[i] += gamma * r
                gamma *= self.gamma
                s = s_prime

        return np.average(rewards)

    def get_elite_idx(self, rewards: np.ndarray) -> np.ndarray:
        """ Compute the indices of the candidate policies by their estimated performance (descending order)

        Parameters
        ----------
        rewards : np.ndarray
            A vector of estimated performance for each candidate policy
        """
        return np.argsort(rewards)[::-1][:self.elite_size]

    def train(self) -> None:
        """ Iterate over all candidate policies and update parameters using elite policies.
        """
        # Check if parameters are set
        assert all(isinstance(val, np.ndarray) for val in [self.means, self.stds])
        for epoch in range(self.epochs):

            extra_cov = max(1.0 - epoch / self.extra_decay_time, 0) * self.extra_std**2

            candidates = np.random.multivariate_normal(
                mean=self.means,
                cov=np.diag(np.array(self.stds**2) + extra_cov),
                size=self.pop_size
            )

            with Pool(self.n_proc) as p:
                g_candidates = p.map(self.candidate_eval, candidates)
            g_candidates = np.array(g_candidates).reshape(-1)

            elite_mask = self.get_elite_idx(g_candidates)
            self.elites = candidates[elite_mask]

            self.update_params(self.elites)

    def get_best_candidates(self) -> np.ndarray:
        """ Return the last generated elite policies by CEM
        """
        return self.elites

    def get_best_rewards(self, n: int) -> List[float]:
        """ Return the performance estimation of top-n elite policies.

        Parameters
        ----------
        n : int
            The top-n policies that will be evaluated and monitored.
        """
        return [self.candidate_eval(theta, monitor=True) for theta in self.elites[:n]]