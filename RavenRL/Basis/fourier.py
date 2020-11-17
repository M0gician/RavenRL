import numpy as np
from itertools import product
from RavenRL.Basis import Basis
from typing import List, Tuple


class FourierBasis(Basis):
    """
    The implementation of Fourier Basis
    For details, please check the paper
        "Value function approximation in reinforcement learning using the Fourier basis"
        https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf
    """
    def __init__(self, n_var: int, ranges: List[Tuple[float, float]], order=3, val_max=1e5):
        """
        Parameters
        ----------
        n_var : int
            The number of variables (features) in the state representation
        ranges : List[Tuple[float, float]]
            A list of float tuples contains the lower and upper bound for each feature.
            Used for normalization.
        order : int
            The order of the magnitude for the Fourier Series expansion.
        val_max : float
            A constant number which truncates the feature with extremely large or small number
        """
        super(FourierBasis, self).__init__(n_var, ranges, val_max)
        self.n_term = pow(order + 1.0, n_var)
        combinations = product(range(order + 1), repeat=n_var)
        self.multipliers = np.array([list(map(int, x)) for x in combinations])

    def compute_features(self, features: np.ndarray) -> np.ndarray:
        """ Return the state features after Fourier basis expansion

        Parameters
        ----------
        features : np.ndarray
            A state vector which contains `n` features.
        """
        if features.shape[0] == 0:
            return np.ones((1,))
        og_basis = np.array(list(map(lambda k_v: self.scale(*k_v), enumerate(features))))
        return np.cos(np.pi * (self.multipliers @ og_basis))


if __name__ == "__main__":
    from RavenRL.Env import setup_env

    env_id = "cartpole"
    env, obs_shape, act_shape = setup_env(env_id)
    ranges = list(zip(env.observation_space.low, env.observation_space.high))
    basis = FourierBasis(n_var=obs_shape, ranges=ranges, order=4, val_max=5e1)
    print(f"basis_dim: {basis.get_basis_dim()}")

    features = env.reset()
    rewards = 0
    done = False

    while not done:
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        rewards += r
        f = basis.compute_features(s)
        print(f"original basis: {s}")
        print(f" current basis: {f.shape}")