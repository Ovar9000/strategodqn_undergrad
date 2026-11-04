"""
Demonstrations for the cold start (DQfD, Hester et al. 2018): games in which a scripted player
plays the learning agent's side, recorded exactly as the agent's own games are (a transition is a
move plus the opponent's reply, rewarded with the PBRS shaping plus the outcome, through the
agent's n-step buffers and flip augmentation), into the agent's demonstration buffer.
"""

from typing import Callable, Dict, List, Optional

from environment import StrategoEnvironment
from environment.heuristic_setup import HeuristicSetupAgent
from .transitions import TurnTracker, outcome_reward


def record_demonstrations(agent, demonstrator, opponents: List, num_games: int, device,
                          make_shaper: Callable, max_turns: Optional[int] = None,
                          no_battle_draw_plies: Optional[int] = None) -> Dict[str, int]:
    """
    `demonstrator` plays Player 1 in `num_games` games, game g against opponents[g % len(opponents)]
    with the heuristic setup on both sides and the starting player alternating. The games run side
    by side in waves of one game per n-step buffer of the agent. Every completed Player-1
    transition goes to agent.demo_memory (enable_demonstrations first) with the legal moves of its
    state, which the margin loss compares the demonstrated move against. `make_shaper()` returns a
    fresh Player-1 reward shaper; one is made per game.

    Returns games, wins, losses, draws (the demonstrator's) and transitions (stored, flips included).
    """
    if agent.demo_memory is None:
        raise ValueError("enable_demonstrations() first")
    if agent.history_embedding_size:
        # A demonstration has no AAREN history of its own to store, and zero planes would teach
        # the network that a game without history looks like this one
        raise ValueError("the cold start records no AAREN history: set AAREN_ENABLED = False")
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}
    wave = len(agent.n_step_buffers) if agent.n_step_buffers else 16
    stored_before = len(agent.demo_memory)
    winners = []
    agent.recording_demos = True
    try:
        for first in range(0, num_games, wave):
            games = list(range(first, min(num_games, first + wave)))  # lane = g - first
            envs, states, shapers = {}, {}, {}
            for g in games:
                env = StrategoEnvironment(device, full_observability=False, max_turns=max_turns,
                                          no_battle_draw_plies=no_battle_draw_plies)
                placements = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p))
                              for p in (1, -1)}
                states[g] = env.reset(placements[1], placements[-1], starting_player=1 if g % 2 == 0 else -1)
                envs[g], shapers[g] = env, make_shaper()
                shapers[g].reset()
            for lane in range(len(games)):
                if agent.n_step_buffers:
                    agent.n_step_buffers[lane].reset()
            tracker = TurnTracker(len(games))
            legal_at_move = {}  # lane -> Player 1's legal moves when it made its open move
            moves = {g: envs[g].get_valid_moves() for g in games}
            active = list(games)
            while active:
                for g in [g for g in active if not moves[g]]:
                    active.remove(g)  # the environment ends a game without moves; guard against a stuck loop
                    winners.append(0)
                by_side = {side: [g for g in active if states[g].current_player == side] for side in (1, -1)}
                groups = [(demonstrator, by_side[1])]
                for opp in {id(o): o for o in opponents}.values():
                    groups.append((opp, [g for g in by_side[-1] if opponents[g % len(opponents)] is opp]))
                completed = []
                for actor, group in groups:
                    if not group:
                        continue
                    actions = actor.act_batch([states[g].board for g in group], [moves[g] for g in group],
                                              [states[g] for g in group], env_indices=[None] * len(group),
                                              full_observability=False)
                    for g, action in zip(group, actions):
                        lane, player = g - first, states[g].current_player
                        if player == 1:
                            legal_at_move[lane] = moves[g]
                        next_state, _, done, info = envs[g].step(action)
                        moves[g] = [] if done else envs[g].get_valid_moves()
                        for t in tracker.record(lane, player, states[g], action, next_state, done, info,
                                                next_moves=moves[g]):
                            if t['player'] != 1:
                                continue
                            reward = shapers[g](previous_state=t['state'], action=t['action'],
                                                current_state=t['next_state'], done=t['done'],
                                                winner=t['winner'], info=t['info'])
                            if t['done']:
                                reward += outcome_reward(t['winner'], 1)
                            completed.append((t, reward, legal_at_move[lane]))
                        states[g] = next_state
                        if done:
                            active.remove(g)
                            winners.append(info.get('winner', 0))
                if completed:
                    agent.remember_batch(
                        [t['state'].board for t, _, _ in completed], [t['action'] for t, _, _ in completed],
                        [r for _, r, _ in completed], [t['next_state'].board for t, _, _ in completed],
                        [t['done'] for t, _, _ in completed], [True] * len(completed),
                        [t['state'] for t, _, _ in completed], [t['next_state'] for t, _, _ in completed],
                        infos=[t['info'] for t, _, _ in completed], lane_ids=[t['lane'] for t, _, _ in completed],
                        history_snapshots=[None] * len(completed),
                        next_legal_moves=[t['next_moves'] for t, _, _ in completed],
                        legal_moves=[legal for _, _, legal in completed])
    finally:
        agent.recording_demos = False
        for buf in agent.n_step_buffers or []:
            buf.reset()  # nothing of a demonstration may run into a training game's n-step return
    wins, losses = winners.count(1), winners.count(-1)
    return {'games': len(winners), 'wins': wins, 'losses': losses, 'draws': len(winners) - wins - losses,
            'transitions': len(agent.demo_memory) - stored_before}
