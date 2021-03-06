"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestTfBaselines(TfGraphTestCase):
    def test_baseline(self):
        """Test the Exp_paper initialization."""
        box_env = TfEnv(DummyBoxEnv())
        deterministic_mlp_baseline = DeterministicMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)
