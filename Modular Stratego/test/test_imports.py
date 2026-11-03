"""Modules import through their packages from one root (Modular Stratego), with no sys.path edits:
each module is loaded once, under one name, and the entry points run from a plain interpreter.

(Before, 20 files added the project's subfolders to sys.path and imported modules by bare name,
so a module could be loaded twice: training.MetricsTracker and network.training.MetricsTracker
were different classes.)"""
import glob
import os
import pickle
import re
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = os.path.join(os.path.dirname(ROOT), 'Python Stratego Game')


def project_files():
    for base in (ROOT, GUI):
        for path in glob.glob(os.path.join(base, '**', '*.py'), recursive=True):
            if '__pycache__' not in path:
                yield path


def test_no_code_edits_sys_path_except_the_guis_one_entry():
    edits = re.compile(r"sys\.path\s*(\.\s*(insert|append|extend)|\[|\+?=)|syspath_prepend")
    found = []
    for path in project_files():
        if os.path.samefile(path, __file__):
            continue
        for n, line in enumerate(open(path, encoding='utf-8'), 1):
            if edits.search(line.split('#')[0]):
                found.append(f"{os.path.relpath(path, os.path.dirname(ROOT))}:{n}")
    assert found == [os.path.join('Python Stratego Game', 'marq_path.py') + ':8'], found


def test_every_project_module_is_loaded_once():
    import train_dqn  # noqa: F401
    import dqn_bot_logic  # noqa: F401
    import setup_agent_integration  # noqa: F401
    names = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and os.path.realpath(f).startswith(os.path.realpath(ROOT)):
            names.setdefault(os.path.realpath(f), []).append(name)
    twice = {f: n for f, n in names.items() if len(n) > 1}
    assert names and not twice, twice


def run_plain(code, cwd):
    """A fresh interpreter without pytest's pythonpath: only what running a script there gives."""
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['OMP_NUM_THREADS'] = '1'
    out = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_the_entry_points_import_from_a_plain_interpreter():
    # train_dqn.py runs from Modular Stratego; the scripts in subfolders with python -m
    run_plain("import train_dqn, tools.export_onnx, network.training.preflight, "
              "visualizers.visualize_q_values", ROOT)
    # the GUI finds Modular Stratego itself
    assert run_plain("import dqn_bot_logic, setup_agent_integration, stratego\n"
                     "b = stratego.Board()\n"
                     "print(setup_agent_integration.place_army(b, 1, stratego.Piece))", GUI).strip() == 'heuristic'


def test_archives_saved_with_bare_module_names_still_load(tmp_path, monkeypatch):
    """A full-state archive's replay buffer pickles Experience tuples; the old ones say
    'prioritized_memory.Experience'."""
    from network import prioritized_memory
    from network.training import legacy_pickle
    exp = prioritized_memory.Experience(*range(len(prioritized_memory.Experience._fields)))
    monkeypatch.setitem(sys.modules, 'prioritized_memory', prioritized_memory)
    monkeypatch.setattr(prioritized_memory.Experience, '__module__', 'prioritized_memory')
    path = tmp_path / 'old_state.pt'
    torch.save({'memory': [exp]}, path)
    assert b'prioritized_memory' in path.read_bytes() and b'network.prioritized_memory' not in path.read_bytes()
    monkeypatch.undo()
    with pytest.raises(ModuleNotFoundError):
        torch.load(path, weights_only=False)
    state = torch.load(path, weights_only=False, pickle_module=legacy_pickle)
    assert state['memory'][0] == exp and type(state['memory'][0]) is prioritized_memory.Experience
