import gym

from garage.baselines import LinearFeatureBaseline
from garage.misc.instrument import run_experiment
from garage.envs.mujoco import SwimmerEnv
from garage.envs import normalize
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

# Need to wrap in a tf environment and force_reset to true
# see https://github.com/openai/garage/issues/87#issuecomment-282519288
env = TfEnv(normalize(SwimmerEnv))

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=100,
    max_path_length=500,
    n_itr=100,
    discount=0.995,
    step_size=0.001,
    plot=False
)

run_experiment(algo.train(), n_parallel=8, snapshot_mode="last", seed=1)
