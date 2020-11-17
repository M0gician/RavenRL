import numpy as np
from typing import List, Tuple


class Basis:
    """
    The base class for basis functions
    """
    def __init__(self, n_var: int, ranges: List[Tuple[float, float]], val_max=1e5):
        """
        Parameters
        ----------
        n_var : int
            The number of variables (features) in the state representation
        ranges : List[Tuple[float, float]]
            A list of float tuples contains the lower and upper bound for each feature.
            Used for normalization.
        val_max : float
            A constant number which truncates the feature with extremely large or small number
        """
        self.n_terms = n_var
        self.ranges = np.array(ranges)
        self.val_max = val_max

    def scale(self, idx: int, value: float) -> float:
        """ Normalize a given feature into the range [-1,1]

        Parameters
        ----------
        idx : int
            The position of the feature.
            Used to find the corresponding min and max value for that feature.
        value : float
            The value of the feature.
        """
        lb, ub = self.ranges[idx]
        lb = max(-self.val_max, lb)
        ub = min(self.val_max, ub)
        if lb == ub:
            return 0.0
        res = (value - lb) / (ub - lb)
        assert -1 <= res <= 1
        return res

    def get_basis_dim(self) -> int:
        """ Return the number of features
        """
        return int(self.n_terms)

    def compute_features(self, features: np.ndarray) -> np.ndarray:
        """ Return the state features after basis expansion

        Parameters
        ----------
        features : np.ndarray
            A state vector which contains `n` features.
        """
        if features.shape[0] == 0:
            return np.ones((1,))
        res = np.array(list(map(lambda k_v: self.scale(*k_v), enumerate(features))))
        return (res - 0.5) * 2


if __name__ == "__main__":
    from RavenRL.Env import setup_env

    env_id = "cartpole"
    env, obs_shape, act_shape = setup_env(env_id)
    ranges = list(zip(env.observation_space.low, env.observation_space.high))
    basis = Basis(n_var=obs_shape, ranges=ranges, val_max=5e1)
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
        print(f" current basis: {f}")

