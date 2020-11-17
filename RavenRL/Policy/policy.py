import gym
import numpy as np
from typing import Union


def setup_policy(env: gym.Env, theta: np.ndarray, env_id=None):
    """ Initialize a policy for a predefined environment.

    Parameters
    ----------
    env : str
        The gym environment object
    theta : np.ndarray
        The weight matrix for generating actions
    env_id : Optional[str]
        The id of the environment for type indication
    """
    if env_id == "simglucose":
        return ContinuousPolicy(env, theta)

    elif env.action_space.type == 'discrete':
        return DiscretePolicy(env, theta)

    elif env.action_space.type == 'continuous':
        return ContinuousPolicy(env, theta)

    else:
        raise ValueError


class Policy:
    """
    An implementation of the base policy for action making
    """
    def __init__(self, env: gym.Env, theta=None, env_id=None):
        """
        Parameters
        ----------
        env : str
            The gym environment object
        theta : np.ndarray
            The weight matrix for generating actions
        env_id : Optional[str]
            The id of the environment for type indication
        """
        self.env = env
        self.obs_shape = env.observation_space.shape[0]
        if env_id == "simglucose":
            self.act_shape = env.action_space.shape[0]
        elif self.env.action_space.type == "discrete":
            self.act_shape = self.env.action_space.n
        elif self.env.action_space.type == "continuous":
            self.act_shape = env.action_space.shape[0]
        else:
            ValueError('env not supported')

        if theta is not None:
            assert len(theta) == (self.obs_shape + 1) * self.act_shape
            self.parameter_dim = self.obs_shape * self.act_shape
            self.b = theta[self.parameter_dim:]
            self.W = theta[:self.parameter_dim].reshape(self.obs_shape, self.act_shape)

    def load_theta(self, theta: np.ndarray):
        """ Load a weight matrix into the Policy object

        Parameters
        ----------
        theta : np.ndarray
            The weight matrix for generating actions
        """
        assert len(theta) == (self.obs_shape + 1) * self.act_shape
        self.parameter_dim = self.obs_shape * self.act_shape
        self.b = theta[self.parameter_dim:]
        self.W = theta[:self.parameter_dim].reshape(self.obs_shape, self.act_shape)

    def act(self, observation: np.ndarray) -> Union[int, float]:
        """ Make an `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """
        raise NotImplementedError

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """
        raise NotImplementedError


class ContinuousPolicy(Policy):
    def __init__(self, env: gym.Env, theta: np.ndarray, env_id=None):
        super(ContinuousPolicy, self).__init__(env, theta, env_id)

    def act(self, observation: np.ndarray) -> Union[int, float]:
        return float(np.clip(
            observation @ self.W + self.b,
            self.env.action_space.low,
            self.env.action_space.high
        ))

    def pi(self, observation: np.ndarray, action: int) -> float:
        raise ValueError


class DiscretePolicy(Policy):
    """
    An implementation of a policy for making discrete actions
    """
    def __init__(self, env: gym.Env, theta: np.ndarray, env_id=None):
        """
        Parameters
        ----------
        env : str
            The gym environment object
        theta : np.ndarray
            The weight matrix for generating actions
        env_id : Optional[str]
            The id of the environment for type indication
        """

        super(DiscretePolicy, self).__init__(env, theta, env_id)

    def act(self, observation: np.ndarray) -> Union[int, float]:
        """ Make an `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """

        y = observation @ self.W + self.b
        action = np.argmax(y)
        return int(action)

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """

        assert action < self.act_shape
        y = observation @ self.W + self.b
        act_prob = np.exp(y) / np.exp(y).sum()
        return act_prob[action]


class RandomPolicy(Policy):
    """
    An implementation of a random policy for making discrete actions
    """
    def __init__(self, env: gym.Env, theta=None, env_id=None):
        """
        Parameters
        ----------
        env : str
            The gym environment object
        theta : np.ndarray
            The weight matrix for generating actions
        env_id : Optional[str]
            The id of the environment for type indication
        """

        super(RandomPolicy, self).__init__(env, theta, env_id)

    def act(self, observation: np.ndarray) -> Union[int, float]:
        """ Make a random `action` (a) given current state/observation

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        """

        return self.env.action_space.sample()

    def pi(self, observation: np.ndarray, action: int) -> float:
        """ Get the probability (density) of taking action `a` in state `s` when using current policy
            ( In literature we use pi(a|s) )

        Parameters
        ----------
        observation : np.ndarray
            A feature vector representing the current state/observation
        action : int
            An int representing a specific action
        """

        return 1 / self.act_shape
