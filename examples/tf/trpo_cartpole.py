from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.tf.algos import TRPO
import gym
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

env = TfEnv(normalize(SwimmerEnv()))

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=500,
    n_itr=200,
    discount=0.99,
    step_size=0.01,
    plot=False)
algo.train()
