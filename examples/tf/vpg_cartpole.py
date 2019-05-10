from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment


env = TfEnv(normalize(SwimmerEnv()))

policy = GaussianMLPPolicy(
    name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = VPG(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=500,
    n_itr=40,
    discount=0.995,
    optimizer_args=dict(tf_optimizer_args=dict(learning_rate=1e-4, )))

run_experiment(algo.train(), n_parallel=1, snapshot_mode="last", seed=1, use_gpu= True, use_tf=True)
