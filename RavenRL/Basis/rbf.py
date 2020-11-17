import numpy as np
from RavenRL.Basis import Basis
from typing import List, Tuple


class RBFBasis(Basis):
    def __init__(self, n_var: int, ranges: List[Tuple[float, float]], val_max=1e5, n_funcs=10, beta=0.9):
        super(RBFBasis, self).__init__(n_var, ranges, val_max)
        self.beta = beta
        self.n_funcs = n_funcs
        self.ranges[self.ranges <= -val_max] = -val_max
        self.ranges[self.ranges >= val_max] = val_max
        self.centers = np.random.uniform(self.ranges[:, 0], self.ranges[:, 1].T, (self.n_funcs, self.n_terms))

    def get_basis_dim(self) -> int:
        return self.n_funcs

    def compute_features(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 0:
            return np.ones((1,))
        return np.array([np.exp(-self.beta * np.linalg.norm(features - c)**2) for c in self.centers])


if __name__ == "__main__":
    from RavenRL.Env import setup_env

    env_id = "cartpole"
    env, obs_shape, act_shape = setup_env(env_id)
    ranges = list(zip(env.observation_space.low, env.observation_space.high))
    basis = RBFBasis(n_var=obs_shape, ranges=ranges, val_max=5, n_funcs=10, beta=0.9)
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