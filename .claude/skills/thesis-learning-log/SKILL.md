---
name: thesis-learning-log
description: Record every change to the Stratego DQN training system (network, learner, reward, replay, curriculum or opponent schedule, trainer loop, config defaults, speedups) and every training-run finding or diagnosis as a "point of learning" in the thesis appendix Latex/chapters/AppendixC.tex. Use after making or finishing any such change or diagnosis in this repository, before wrapping up the turn, even when the user did not ask for the write-up.
---

# Thesis learning log

The thesis keeps an implementation log in `Latex/chapters/AppendixC.tex` ("Implementation Log
and Lessons Learned"). Each change to how the agent trains, and each finding about a run, is a
point of learning for the thesis, so it gets an entry there, in the same turn as the work.

## When it applies

Write an entry when the turn:
- changes training behaviour: `Modular Stratego/network/`, `environment/` (rules, observation,
  curriculum or opponent schedule), `train_dqn.py`, or a default in `settings/training_config.py`;
- makes training faster or cheaper in a way that could change results (lanes, replay ratio,
  batching, asynchrony);
- diagnoses a run (a probe, a profile, a crash cause) or reads its results (an evaluation, a
  learning curve) in a way that explains or motivates a change.

Skip pure refactors with no behavioural effect, test-only changes, GUI cosmetics and typo fixes.
If unsure, write the entry: a short entry is cheap, a lost lesson is not.

## How to write an entry

1. Read `Latex/chapters/AppendixC.tex` first. If the change extends or revises an existing entry
   (a follow-up, a new measurement, a reversal), update that entry instead of adding a new one,
   and say what changed and why. Never delete an earlier lesson that turned out wrong: add
   what was learned from it being wrong.
2. Otherwise add a `\section{...}` at the end, before the "Engineering notes" section. Its title
   names the change, not the file.
3. Follow the existing pattern, in thesis prose (full sentences, past tense for what was done):
   - `\paragraph{Change.}` What is different now, in terms of the algorithm or the experiment,
     with the settings that matter (values, not variable names, unless the name is the point).
   - `\paragraph{Why.}` The evidence: the measurement, run, probe or argument that motivated it,
     with the actual numbers (run name, episode, before/after). Only numbers that were measured
     in this repository or are cited; never estimates presented as results.
   - `\paragraph{Result.}` (when measured) What the change did. If it has not been measured yet,
     say so plainly ("not yet measured in a full run") rather than leaving it out.
   - `\paragraph{Lesson.}` The general lesson, one to three sentences, stated so that it would
     help someone building a different agent.
4. Small engineering lessons (a Python pitfall, a tooling surprise) go as an `\item` under
   "Engineering notes" instead of their own section.
5. Cite with `\cite{key}`. Reuse keys from `Latex/references.bib`; add an entry there only for a
   source you can name exactly (authors, title, venue or arXiv id, year). Never invent one.
6. Keep LaTeX safe: escape `_`, `%`, `&`, `#` in text (`formal\_3`), use `\texttt{}` for
   identifiers, math mode for symbols. The appendix is included from `Latex/sp.tex` (and,
   commented out like the other appendices, from `Latex/sptex.tex`).
7. If `latexmk` is available, check that the document still builds:
   `cd Latex && latexmk -pdf -interaction=nonstopmode -halt-on-error sp.tex`
   and fix any error the entry caused. Do not commit build artefacts.
8. In the turn's closing message, tell the user in one line which entry was added or updated.
