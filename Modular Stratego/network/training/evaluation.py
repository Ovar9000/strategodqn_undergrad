"""
Evaluation games: the learning agent plays greedily (NoisyNet off) against a fixed opponent,
outside the training lanes. The curriculum advances on these results, not on training games
(which mix in exploration noise and accumulate over the whole phase).
"""

from typing import Dict, Optional

from environment import StrategoEnvironment
from environment.battle import BattleResolver
from environment.board import HIDDEN_PIECE
from environment.heuristic_setup import HeuristicSetupAgent
from environment.piece import PieceType
from settings.training_config import EVAL_NO_BATTLE_DRAW_PLIES


def play_evaluation_games(agent, opponent, num_games: int, device, full_observability=False,
                          max_turns: Optional[int] = None, no_battle_draw_plies: int = EVAL_NO_BATTLE_DRAW_PLIES
                          ) -> Dict[str, float]:
    """
    `agent` (Player 1) plays `num_games` games against `opponent` (Player 2). Both sides use the
    heuristic setup, as in training; the starting player alternates. The agent acts greedily on
    its mean weights and reads no AAREN history (a lane's history belongs to its training game).

    The games run side by side: each round, every game with the agent to move goes into one
    act_batch call and every game with the opponent to move into another. (They used to run one
    after another, one position per network call: 160-280 s per evaluation.) Neither side keeps
    per-game state here, so batching does not change a game.

    Returns wins, losses, draws, games and win_rate = wins / games (draws count as non-wins), and
    the agent's attacks on enemy pieces whose rank its view shows: revealed_attacks and how many
    it won / lost by the battle rule (revealed_won, revealed_lost; the rest tied). The share lost
    measures the first skill the game asks for, comparing two known ranks, in the agent's own
    choices (formal_4 to formal_7 lost about two-thirds of them).
    """
    setups = {1: HeuristicSetupAgent(player_id=1), -1: HeuristicSetupAgent(player_id=-1)}
    envs, states = [], []
    for game in range(num_games):
        env = StrategoEnvironment(device, full_observability=full_observability, max_turns=max_turns,
                                  no_battle_draw_plies=no_battle_draw_plies)
        placements = {p: setups[p].place_pieces(env._generate_pieces(), env.get_valid_placement_positions(p))
                      for p in (1, -1)}
        states.append(env.reset(placements[1], placements[-1], starting_player=1 if game % 2 == 0 else -1))
        envs.append(env)
    if hasattr(opponent, 'reset_history'):
        opponent.reset_history()

    noisy = getattr(agent, 'noisy_acting', False)
    agent.noisy_acting = False
    winners = [0] * num_games  # 0 = draw, also for a game that ends without moves
    resolver = BattleResolver()
    revealed = {'revealed_attacks': 0, 'revealed_won': 0, 'revealed_lost': 0}
    active = list(range(num_games))
    try:
        while active:
            moves = {g: envs[g].get_valid_moves() for g in active}
            for g in [g for g in active if not moves[g]]:
                active.remove(g)  # the environment ends a game without moves; guard against a stuck loop
            # Split before stepping: a game the agent moves in this round waits for the next round
            by_side = {side: [g for g in active if states[g].current_player == side] for side in (1, -1)}
            for actor, side in ((agent, 1), (opponent, -1)):
                games = by_side[side]
                if not games:
                    continue
                actions = actor.act_batch([states[g].board for g in games], [moves[g] for g in games],
                                          [states[g] for g in games], env_indices=[None] * len(games),
                                          full_observability=full_observability)
                for g, action in zip(games, actions):
                    if side == 1 and action is not None:
                        _count_revealed_attack(states[g].board, action, resolver, revealed)
                    states[g], _, done, info = envs[g].step(action)
                    if done:
                        winners[g] = info.get('winner', 0)
                        active.remove(g)
    finally:
        agent.noisy_acting = noisy
    wins, losses = winners.count(1), winners.count(-1)
    return {'wins': wins, 'losses': losses, 'draws': num_games - wins - losses, 'games': num_games,
            'win_rate': wins / max(1, num_games), **revealed}


def _count_revealed_attack(view, action, resolver, counts):
    """Add Player 1's move `action` to `counts` if it attacks an enemy piece of known rank on
    `view` (Player 1's board: enemy pieces negative, unseen ones HIDDEN_PIECE)."""
    (r0, c0), (r1, c1) = action
    target, mover = int(view[r1, c1]), int(view[r0, c0])
    if target == HIDDEN_PIECE or not -12 <= target <= -1 or not 1 <= mover <= 12:
        return
    result = resolver.resolve_battle(PieceType(mover), PieceType(-target))
    counts['revealed_attacks'] += 1
    counts['revealed_won'] += result == 1
    counts['revealed_lost'] += result == -1
