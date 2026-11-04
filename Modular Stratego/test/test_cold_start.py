"""The cold start from demonstrations (DQfD): recording, batch mixing, the large-margin loss."""

import numpy as np
import pytest
import torch

from environment.action_encoding import NUM_ACTIONS
from network.drqn_agent import pack_legal_mask


def _agent(cpu, **kwargs):
    from network.drqn_agent import RainbowAgent
    kwargs.setdefault('num_envs', 2)
    kwargs.setdefault('buffer_size', 500)
    kwargs.setdefault('batch_size', 8)
    return RainbowAgent(player_id=1, device=cpu, **kwargs)


def test_demonstrations_are_stored_apart_with_the_legal_moves_of_their_states(cpu, tiny_network):
    from environment.curriculum import HeuristicOpponent, TrueRandomOpponent
    from network import create_unified_reward_shaper
    from network.training import record_demonstrations
    agent = _agent(cpu)
    agent.enable_demonstrations(10_000)
    result = record_demonstrations(agent, HeuristicOpponent(cpu, player_id=1), [TrueRandomOpponent(cpu, -1)],
                                   num_games=3, device=cpu, make_shaper=lambda: create_unified_reward_shaper(1),
                                   max_turns=60)
    assert result['games'] == 3 and result['transitions'] == len(agent.demo_memory) > 20
    assert len(agent.memory) == 0  # the replay memory is the agent's own experience only
    assert not agent.recording_demos and all(len(b.buffer) == 0 for b in agent.n_step_buffers)
    for e in agent.demo_memory.buffer:
        legal = np.unpackbits(e.legal, count=NUM_ACTIONS).astype(bool)
        assert legal[e.action]  # the demonstrated move is one of the state's legal moves (flips too)
        assert agent.v_min <= e.reward <= agent.v_max
    assert sum(e.done for e in agent.demo_memory.buffer) >= 2  # n-step tails of the finished games


def _synthetic_demos(agent, n, moves_per_state=5, seed=0):
    """n random states, each with a few legal moves, the first of them demonstrated; every
    transition is terminal with reward 0, so the C51 target cannot tell the moves apart."""
    rng = np.random.default_rng(seed)
    states, legal_sets = [], []
    for _ in range(n):
        s = torch.from_numpy((rng.random((agent.input_channels, 10, 10)) > 0.8).astype(np.float32))
        legal = rng.choice(NUM_ACTIONS, moves_per_state, replace=False)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[legal] = True
        agent.demo_memory.add(s, int(legal[0]), 0.0, s, True, legal=pack_legal_mask(mask))
        states.append(s)
        legal_sets.append(legal)
    return torch.stack(states), legal_sets


def _greedy_is_demonstrated(agent, states, legal_sets):
    with torch.no_grad():
        q = (agent._q_forward_for_acting(states).float().exp() * agent.support).sum(dim=2)
    return np.mean([int(legal[np.argmax(q[i, legal].numpy())] == legal[0]) for i, legal in enumerate(legal_sets)])


@pytest.mark.parametrize("margin_weight, expect_imitation", [(1.0, True), (0.0, False)])
def test_the_margin_loss_makes_the_demonstrated_move_the_greedy_one(cpu, tiny_network, margin_weight,
                                                                    expect_imitation):
    """With equal returns for every move, only the margin loss says which move to prefer."""
    torch.manual_seed(0)
    agent = _agent(cpu, batch_size=16)
    agent.amp_enabled = False
    agent.demo_margin, agent.demo_margin_weight = 0.1, margin_weight
    for g in agent.optimizer.param_groups:
        g['lr'] = 1e-3
    agent.enable_demonstrations(100)
    states, legal_sets = _synthetic_demos(agent, 16)
    for _ in range(150):
        assert agent.replay(demo_only=True) is not None
    rate = _greedy_is_demonstrated(agent, states, legal_sets)
    if expect_imitation:
        assert rate >= 0.9 and agent.last_margin_loss < 0.05
    else:
        assert rate < 0.6  # chance is 1 / 5


def test_a_share_of_each_batch_is_demonstrations_falling_to_none(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=8)
    agent.demo_ratio, agent.demo_decay_episodes = 0.25, 100
    agent.enable_demonstrations(100)
    _synthetic_demos(agent, 20)
    s = torch.zeros(agent.input_channels, 10, 10)
    for i in range(40):
        agent._add_experience(s, i, 0.0, s, True)
    drawn = []
    original = agent.demo_memory.sample
    agent.demo_memory.sample = lambda n: (drawn.append(n), original(n))[1]
    agent.current_episode = 0
    agent.replay()
    agent.current_episode = 50
    agent.replay()
    agent.current_episode = 100
    agent.replay()
    assert drawn == [2, 1]  # 25%, then 12.5% of 8, then none
    assert agent.current_demo_ratio() == 0.0


def test_the_trainer_cold_starts_once(tmp_path, config_override, monkeypatch):
    from .test_trainer_smoke import run_tiny_training
    from network.training import MetricsTracker
    trackers = []
    original_init = MetricsTracker.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trackers.append(self)

    monkeypatch.setattr(MetricsTracker, '__init__', init)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2, COLD_START_ENABLED=True,
                      COLD_START_GAMES=2, COLD_START_PRETRAIN_UPDATES=3, COLD_START_DEMO_CAPACITY=5000)
    cold = trackers[0].metrics['cold_start']
    assert cold['games'] == 2 and cold['transitions'] > 8 and cold['pretrain_updates'] == 3
