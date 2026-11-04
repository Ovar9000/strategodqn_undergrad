"""The asynchronous learner: same number of updates, bounded lag, pauses, errors, the acting copy."""

import threading
import time

import pytest
import torch

from network.training.async_learner import AsyncLearner


class FakeAgent:
    """What AsyncLearner uses of an agent; replay() takes `delay` seconds."""
    history_embedding_size = 0
    device = torch.device('cpu')

    def __init__(self, delay=0.0, fail_at=None):
        self.delay, self.fail_at = delay, fail_at
        self.calls, self.running, self.syncs = 0, False, 0
        self.lock = threading.Lock()

    def enable_async(self):
        pass

    def sync_acting_network(self):
        self.syncs += 1

    def replay(self, episode=None):
        self.running = True
        time.sleep(self.delay)
        self.calls += 1
        self.running = False
        if self.fail_at is not None and self.calls == self.fail_at:
            raise ValueError("boom")
        return 0.5


def test_every_requested_update_runs_and_its_loss_comes_back():
    agent = FakeAgent()
    learner = AsyncLearner(agent, max_lag=3)
    for _ in range(10):
        learner.request(3)
    learner.wait_idle()
    assert agent.calls == agent.syncs == 30
    assert learner.drain_losses() == [0.5] * 30 and learner.drain_losses() == []
    learner.stop()


def test_the_loop_waits_once_more_than_max_lag_updates_are_owed():
    agent = FakeAgent(delay=0.05)
    learner = AsyncLearner(agent, max_lag=2)
    t = time.perf_counter()
    learner.request(6)  # returns once at most 2 are owed: 4 have started, 3 have finished
    assert time.perf_counter() - t >= 0.12
    assert learner._owed <= 2
    learner.stop()
    assert agent.calls == 6  # stop() runs the owed updates


def test_no_update_runs_while_paused():
    agent = FakeAgent(delay=0.01)
    learner = AsyncLearner(agent, max_lag=50)
    learner.request(20)
    with learner.paused():
        assert not agent.running
        before = agent.calls
        time.sleep(0.05)
        assert agent.calls == before and not agent.running
    learner.stop()
    assert agent.calls == 20


def test_a_learner_error_reaches_the_training_loop():
    agent = FakeAgent(fail_at=2)
    learner = AsyncLearner(agent, max_lag=5)
    with pytest.raises(ValueError, match="boom"):
        learner.request(3)
        learner.wait_idle()
    learner.stop(finish=False)


def test_acting_reads_a_copy_that_follows_the_learner(cpu, tiny_network, config_override):
    from network.drqn_agent import RainbowAgent
    config_override(WARMUP_STEPS=0)
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1, buffer_size=200, batch_size=8)
    s = torch.zeros(agent.input_channels, 10, 10)
    for i in range(20):
        agent._add_experience(s, i, 1.0, s, True)
    learner = AsyncLearner(agent, max_lag=2)
    assert agent.acting_network is not agent.q_network
    before = [p.clone() for p in agent.acting_network.parameters()]
    learner.request(2)
    learner.wait_idle()
    after = list(agent.acting_network.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))  # synced after the updates
    assert all(torch.equal(a, q) for a, q in zip(after, agent.q_network.parameters()))
    learner.stop()
