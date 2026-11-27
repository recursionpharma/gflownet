import pytest

from gflownet.algo.flow_matching import FlowMatching
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.mol_building_env import MolBuildingEnvContext


@pytest.mark.parametrize("fm_flag", [False, True])
def test_flow_matching_init_uses_fm_correct_idempotent(fm_flag):
    cfg = Config()
    cfg.algo.fm.correct_idempotent = fm_flag
    algo = FlowMatching(GraphBuildingEnv(), MolBuildingEnvContext(), cfg)
    assert algo.correct_idempotent is fm_flag
