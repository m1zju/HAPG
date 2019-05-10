import gym
import random
import numpy as np
import theano.tensor as TT
import theano
from garage.baselines import LinearFeatureBaseline
from garage.baselines import ZeroBaseline
from garage.envs import normalize
from garage.envs.mujoco import SwimmerEnv
from garage.theano.algos.capg_re import CAPG
from garage.theano.envs import TheanoEnv
from garage.theano.policies import GaussianMLPPolicy
from garage.theano.misc import tensor_utils
from garage.misc.instrument import run_experiment

env_name = "Swimmer"
hidden_sizes = (32, 32)
env = TheanoEnv(normalize(SwimmerEnv()))
policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=hidden_sizes)
backup_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
mix_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
pos_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)
neg_eps_policy = GaussianMLPPolicy(env.spec, hidden_sizes=hidden_sizes)

observations_var = env.observation_space.new_tensor_variable(
    'observations',
    extra_dims=1
)
actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)
rewards_var = tensor_utils.new_tensor(
            'rewards', ndim=1, dtype=theano.config.floatX)

dist = policy.distribution
dist_info_vars = policy.dist_info_sym(observations_var)
old_dist_info_vars = backup_policy.dist_info_sym(observations_var)
kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
mean_kl = TT.mean(kl)
max_kl = TT.max(kl)

#for test
surr_ll = dist.log_likelihood_sym(actions_var, dist_info_vars)
surr_ll_cumsum = dist.log_likelihood_sym_cumsum(actions_var, dist_info_vars)
surr = TT.sum(surr_ll_cumsum * rewards_var)

f_surr_ll = theano.function(inputs=[observations_var, actions_var],
                            outputs=surr_ll
                            )
f_surr_ll_cumsum = theano.function(inputs=[observations_var, actions_var],
                            outputs=surr_ll_cumsum
                            )

paths = []
observations = []
actions = []
rewards = []
observation = env.reset()
for _ in range(20):
    # policy.get_action() returns a pair of values. The second
    # one returns a dictionary, whose values contains
    # sufficient statistics for the action distribution. It
    # should at least contain entries that would be returned
    # by calling policy.dist_info(), which is the non-symbolic
    # analog of policy.dist_info_sym(). Storing these
    # statistics is useful, e.g., when forming importance
    # sampling ratios. In our case it is not needed.
    action, _ = policy.get_action(observation)
    # Recall that the last entry of the tuple stores diagnostic
    # information about the environment. In our case it is not needed.
    next_observation, reward, terminal, _ = env.step(action)
    observations.append(observation)
    actions.append(action)
    rewards.append(reward)
    observation = next_observation
    if terminal:
        # Finish rollout if terminal state reached
        break

# We need to compute the empirical return for each time step along the
# trajectory
path = dict(
    observations=np.array(observations),
    actions=np.array(actions),
    rewards=np.array(rewards),
)
ll = f_surr_ll(path["observations"][0].reshape(1,-1), path["actions"][0].reshape(1,-1))
ll_cumsum = f_surr_ll_cumsum(path["observations"], path["actions"])

print(ll)
print(ll_cumsum)
