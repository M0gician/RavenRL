import gym
import types
import functools
import numpy as np
from typing import List, Union, Tuple, Callable, DefaultDict
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

from RavenRL.Env import setup_env
from RavenRL.Policy import setup_policy
from RavenRL.Sampling import SamplingMethod




def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def generate_episode(env_id: str, _) -> DefaultDict:
    """ A partial function for multiprocessing the dataset generation
        This function samples every `state` (s), `action` (a), `reward` (r), and `n`ext action` (s')
        using an random policy for one trajectory.

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    _: any
        A placeholder variable for multiprocessing.Pool
    """
    env, _, _ = setup_env(env_id)
    done = False

    s = env.reset()
    if env_id == 'simglucose':
        s = s.CGM

    history = defaultdict(list)

    while not done:
        a = env.action_space.sample()
        s_prime, r, done, info = env.step(a)
        if env_id == 'simglucose':
            s_prime = s_prime.CGM
            a = float(a)
        history['s'].append(s)
        history['a'].append(a)
        history['r'].append(r)
        history['s_prime'].append(s_prime)
        s = s_prime
    return history


def generate_dataset(env_id: str, n: int, n_proc=6):
    """ Generate `n` trajectories of data of a given `env_id` using a random policy.
        [Multiprocessing enabled]

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    n : int
        The number of trajectories that will be generated
    n_proc : int
        [Default =6] The number of processes that will be created
            (#process != #thread)
    """
    with Pool(n_proc) as p:
        dataset = p.map(partial(generate_episode, env_id), range(n))
    return dataset


def safety_test(env_id: str, theta: np.ndarray, sampler: SamplingMethod, ref_size: int,
                ci_ub: Callable, g_funcs: List[Callable], delta=0.05) -> Union[Tuple[float, str], Tuple[float, np.ndarray]]:
    """ Run safety test over the candidate policies.

    Parameters
    ----------
    env_id : str
        !!NOTE only 'cartpole' is supported currently!!
        A string in ['cartpole', 'pendulum', 'simglucose']
    theta : np.ndarray
        A collections of candidate policies generated by the learning agent
        [NOTE: use a significance level `delta/n` to prevent Multiple Comparison Problem]
    sampler : SamplingMethod
        The sampling method for performance estimation [IS/PDIS]
    ref_size : int
        The size of the dataset in the safety test.
        Used in the concentration bound calculation to prevent producing a bound that is overly conservative
    g_funcs : List[Callable]
        A :obj:`list` of user-defined constraint functions for safety test.
    delta : float
        [Default =0.05] The significance level for the safety test to get a high confidence performance lower bound
        of a candidate policy
    """
    assert isinstance(env_id, str)
    assert isinstance(sampler, SamplingMethod)
    env, _, _ = setup_env(env_id)
    policy = setup_policy(env, theta)
    sampler.load_eval_policy(policy)

    rewards = np.array([sampler.get_episodic_est(idx=i) for i in range(ref_size)])

    for g in g_funcs:
        if ci_ub(g(rewards), ref_size=ref_size, correction=1, delta=delta) > 0:
            return -np.infty, "NSF"
    return np.average(rewards), theta
