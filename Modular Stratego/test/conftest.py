"""Shared pytest setup: import paths, CPU device, config overrides, slow-test switch."""

import os
import sys

import pytest

# Modular Stratego (the import root) and Python Stratego Game are on sys.path via pytest.ini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import torch

# Under pytest-xdist (`-n N`) every worker is its own process: one torch thread each, so N workers
# don't oversubscribe the cores (N x 4 threads on 16 cores made things slower, not faster).
torch.set_num_threads(1 if os.environ.get('PYTEST_XDIST_WORKER') else min(4, torch.get_num_threads()))

_PROJECT_KEY = os.path.normcase(os.path.realpath(PROJECT_ROOT))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow learning tests")


# Tests that drive the frozen thesis DQN trainer (network/, train_dqn.py): ~80% of the suite's time
# (measured 2026-11-23: ~1,600 of ~2,000 test-seconds). Marked `legacy` so day-to-day work on vec/
# can run `-m "not legacy"`; run them whenever network/, environment/, train_dqn.py or settings/
# change, and always run the full suite before reporting results.
LEGACY_FILES = {
    'test_learner.py', 'test_curriculum.py', 'test_cold_start.py', 'test_turn_handling.py',
    'test_trainer_smoke.py', 'test_aaren_integration.py', 'test_crash_regressions.py',
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if os.path.basename(str(item.fspath)) in LEGACY_FILES:
            item.add_marker(pytest.mark.legacy)
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cpu():
    return torch.device('cpu')


@pytest.fixture(autouse=True)
def _same_random_state_for_every_test():
    """Every test starts from the same global RNG state, whatever ran before it in the process.
    Without this, a test using unseeded setups (env.reset()) passed or failed depending on test
    order, which pytest-xdist changes from run to run (test_learner's random game once ended early)."""
    import random
    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Code that writes relative paths (logs, plots) writes into the test's tmp dir, not the repo."""
    monkeypatch.chdir(tmp_path)


def _project_modules():
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and os.path.normcase(os.path.realpath(f)).startswith(_PROJECT_KEY):
            yield mod


@pytest.fixture
def config_override(monkeypatch):
    """Override training_config constants everywhere they were copied by `from ... import`.

    Modules read config at import time (`from training_config import X`, `from settings import *`),
    so the value is patched on every loaded project module that has the name, plus on
    training_config and settings so later imports pick it up. Undone after the test.
    """
    from settings import training_config
    import settings

    def apply(**values):
        for name, value in values.items():
            monkeypatch.setattr(training_config, name, value, raising=False)
            monkeypatch.setattr(settings, name, value, raising=False)
            for mod in _project_modules():
                if hasattr(mod, name):
                    monkeypatch.setattr(mod, name, value)
    return apply


@pytest.fixture
def tiny_network(config_override):
    """A RainbowDQN small enough for fast CPU tests."""
    config_override(NETWORK_CHANNELS=16, RESNET_BLOCKS=1, ATTENTION_HEADS=2)
