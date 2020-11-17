import gym
import numpy as np
from typing import Optional, Tuple

from gym.envs.registration import register

# register(
#     id='simglucose-adolescent2-v0',
#     entry_point='simglucose.envs:T1DSimEnv',
#     kwargs={'patient_name': 'adolescent#002'}
# )


def setup_env(env_id) -> Tuple[gym.Env, int, int]:
    """ Initialize a predefined environment for experiments

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    """
    if env_id == 'cartpole':
        return setup_cartpole()
    elif env_id == 'pendulum':
        return setup_pendulum()
    elif env_id == 'simglucose':
        return setup_simglucose()
    else:
        raise ValueError('env {} not supported'.format(env_id))


def setup_simglucose() -> Tuple[gym.Env, int, int]:
    env = gym.make('simglucose-adolescent2-v0')
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]

    env.action_space.type = 'continuous'
    return env, obs_shape, act_shape


def setup_pendulum() -> Tuple[gym.Env, int, int]:
    env = gym.make('Pendulum-v0')
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]

    env.action_space.type = 'continuous'
    return env, obs_shape, act_shape


def setup_cartpole() -> Tuple[gym.Env, int, int]:
    env = gym.make('CartPole-v0')
    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.n

    env.action_space.type = 'discrete'

    #   shape is empty tuple in the gym env
    env.action_space.shape = (1,)
    return env, obs_shape, act_shape
