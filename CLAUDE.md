# Project notes for Claude

- Every change to how the agent trains, and every finding about a training run, is written up
  as a point of learning in the thesis appendix `Latex/chapters/AppendixC.tex`, in the same turn
  as the work. Use the `thesis-learning-log` skill (`.claude/skills/thesis-learning-log/`); it
  says what counts and how an entry is written.
- Tests (run from `Modular Stratego/`, with the Python313 that has torch; pytest-xdist installed):
  - fast tier while working on `vec/`: `python -m pytest -q test -n 8 -m "not legacy"` (~1.5 min)
  - full suite before reporting, or after touching `network/`, `environment/`, `train_dqn.py` or
    `settings/`: `python -m pytest -q test -n 8` (~5.5 min; serial was ~16 min)
  - `legacy` = the 7 files that drive the frozen thesis DQN trainer (list in `test/conftest.py`).
- Strength claims ("A beats B"): check with `python -m vec.verify A.pt B.pt --out <json>` on fresh
  OS-random seeds; the pre-chosen primary mode (sampled, no safety filter) decides, not the best of four.
