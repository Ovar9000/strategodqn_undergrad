"""
Asynchronous learner: the gradient updates of one agent run on a background thread (and their
own CUDA stream) while the training loop acts and steps the environments, as in the actor-learner
split of Ape-X (Horgan et al. 2018) and IMPALA (Espeholt et al. 2018), here with one process.

The replay ratio is unchanged. The loop still owes one update per REPLAY_UPDATE_INTERVAL learner
moves and hands them over with request(); it waits once more than max_lag updates are owed, so
the acting network is at most max_lag + 1 updates behind the learner (synchronously it is never
behind) and the learner can never fall behind for good. Acting reads a copy of the network that
is refreshed after every update (RainbowAgent.enable_async), and anything that reads or replaces
the learner's weights or optimizer (checkpoints, copying them into agent 2, the LR schedule) runs
inside paused().
"""

import contextlib
import threading
from typing import List, Optional

import torch


class AsyncLearner:
    """Runs agent.replay() on its own thread; see the module docstring."""

    def __init__(self, agent, max_lag: int = 2):
        if agent.history_embedding_size:
            raise ValueError("the asynchronous learner does not run with AAREN (its supervised "
                             "training shares the network): set ASYNC_LEARNER = False or AAREN_ENABLED = False")
        agent.enable_async()
        self.agent = agent
        self.max_lag = max(1, int(max_lag))
        self._cond = threading.Condition()
        self._owed = 0          # updates requested and not started
        self._busy = False      # an update is running
        self._paused = False
        self._stop = False
        self._episode = None
        self._losses: List[float] = []
        self._error: Optional[BaseException] = None
        self.updates = 0        # updates run (losses or not)
        self._thread = threading.Thread(target=self._run, name='async-learner', daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- training loop side
    def request(self, n: int, episode=None):
        """`n` more updates are owed (episode: for PER beta annealing). Blocks while more than
        max_lag are owed; raises the learner thread's error, if it had one."""
        with self._cond:
            self._raise_error()
            self._owed += n
            self._episode = episode
            self._cond.notify_all()
            while self._owed > self.max_lag and self._error is None:
                self._cond.wait()
            self._raise_error()

    def drain_losses(self) -> List[float]:
        """Losses of the updates finished since the last call."""
        with self._cond:
            self._raise_error()
            losses, self._losses = self._losses, []
            return losses

    def wait_idle(self):
        """Wait until every owed update has run (the synchronous trainer's state)."""
        with self._cond:
            while (self._owed or self._busy) and self._error is None:
                self._cond.wait()
            self._raise_error()

    @contextlib.contextmanager
    def paused(self):
        """No update runs inside: the learner's weights, optimizer and memory hold still."""
        with self._cond:
            self._paused = True
            while self._busy and self._error is None:
                self._cond.wait()
        try:
            with self._cond:
                self._raise_error()
            yield
        finally:
            with self._cond:
                self._paused = False
                self._cond.notify_all()

    def stop(self, finish: bool = True):
        """Run the owed updates (finish) or drop them, then end the thread."""
        failure = None
        if finish and self._thread.is_alive():
            try:
                self.wait_idle()
            except BaseException as e:  # noqa: BLE001 - raised once the thread is joined
                failure = e
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join()
        if failure is not None:
            raise failure
        with self._cond:
            self._raise_error()

    def _raise_error(self):
        if self._error is not None:
            error, self._error = self._error, None
            self._stop = True
            raise error  # the learner thread's own exception, with its traceback

    # ---------------------------------------------------------------- learner thread
    def _run(self):
        agent = self.agent
        stream = torch.cuda.Stream(device=agent.device) if agent.device.type == 'cuda' else None
        while True:
            with self._cond:
                while not self._stop and (self._paused or not self._owed):
                    self._cond.wait()
                if self._stop:
                    return
                self._owed -= 1
                self._busy = True
                episode = self._episode
                self._cond.notify_all()  # request() may be waiting for the backlog to shrink
            loss, error = None, None
            try:
                with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
                    loss = agent.replay(episode=episode)
                    agent.sync_acting_network()
                    if stream is not None:
                        stream.synchronize()
                # Diagnostics the loop reads: plain floats, not tensors of this stream
                for name in ('_cached_dqn_grad_norm', '_cached_aaren_grad_norm', '_last_margin_loss'):
                    value = getattr(agent, name, None)
                    if isinstance(value, torch.Tensor):
                        setattr(agent, name, float(value))
            except BaseException as e:  # noqa: BLE001 - handed to the training loop
                error = e
            with self._cond:
                self._busy = False
                self.updates += 1
                if loss:
                    self._losses.append(loss)
                if error is not None:
                    self._error = error
                    self._stop = True
                self._cond.notify_all()
                if self._stop:
                    return
