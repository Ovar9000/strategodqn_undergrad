"""Step 7: curriculum (start, evaluation gates, caps, PFSP) and the scripted opponents' rank tables."""

import importlib
import random
from collections import Counter

import pytest
import torch

from environment.piece import PieceType as P

HIDDEN = -20


def _curriculum(tmp_path, phase=1):
    from environment.curriculum import CurriculumManager
    return CurriculumManager(start_phase=phase, save_dir=str(tmp_path))


# ---------------------------------------------------------------- start and advancement

def test_training_starts_at_phase_one():
    from settings.training_config import CURRICULUM_START_PHASE
    assert CURRICULUM_START_PHASE == 1


def test_draws_do_not_count_as_wins(tmp_path):
    """The thesis gate used decisive games only: 30 wins and 70 draws in 100 games passed 70%."""
    c = _curriculum(tmp_path)
    for opp in ('random', 'heuristic'):
        for k in range(100):
            c.update_metrics({'winner': 1 if k < 30 else 0, 'opponent_type': opp})
    assert not c.check_phase_transition()
    c.record_evaluation({'random': 0.3, 'heuristic': 0.3})
    assert not c.check_phase_transition()


def test_training_games_alone_never_advance_a_phase(tmp_path):
    c = _curriculum(tmp_path)
    for k in range(500):
        c.update_metrics({'winner': 1, 'opponent_type': 'random' if k % 2 else 'heuristic'})
    assert not c.check_phase_transition()


def test_every_phase_is_reachable_by_evaluation(tmp_path):
    from environment.curriculum import TrainingPhase
    c = _curriculum(tmp_path)
    for phase in (1, 2, 3):
        assert c.current_phase == phase
        gates = c.evaluation_gates()
        assert gates
        c.record_evaluation({opp: threshold - 0.01 for opp, threshold in gates.items()})
        assert not c.check_phase_transition()
        c.record_evaluation({opp: threshold for opp, threshold in gates.items()})
        assert c.check_phase_transition() and c.advance_phase()
    assert c.current_phase == TrainingPhase.LEAGUE_TRAINING
    assert c.evaluation_gates() == {} and not c.check_phase_transition()


def test_an_evaluation_from_the_previous_phase_does_not_carry_over(tmp_path):
    c = _curriculum(tmp_path)
    c.record_evaluation({opp: 1.0 for opp in c.evaluation_gates()})
    assert c.check_phase_transition() and c.advance_phase()
    assert not c.check_phase_transition()


def test_a_phase_ends_at_its_episode_cap(tmp_path, config_override):
    config_override(PHASE_1_MAX_EPISODES=10)
    c = _curriculum(tmp_path)
    for _ in range(9):
        c.update_metrics({'winner': -1, 'opponent_type': 'random'})
    assert not c.check_phase_transition()
    c.update_metrics({'winner': -1, 'opponent_type': 'random'})
    assert c.check_phase_transition()


# ---------------------------------------------------------------- PFSP

def test_phase_one_trains_against_the_gate_opponents_from_the_start(tmp_path):
    c = _curriculum(tmp_path)
    assert set(c.get_opponent_distribution()) == {'random', 'heuristic'}
    for _ in range(40):
        c.update_metrics({'winner': 1, 'opponent_type': 'heuristic'})
    assert 'smart_heuristic' in c.get_opponent_distribution()


def test_opponents_are_sampled_by_their_pfsp_weights(tmp_path, monkeypatch):
    """The thesis trainer cycled through the sorted opponent types in fixed blocks."""
    from environment.curriculum import CurriculumManager
    c = _curriculum(tmp_path)
    monkeypatch.setattr(CurriculumManager, 'get_opponent_distribution',
                        lambda self: {'random': 0.9, 'smart_heuristic': 0.1})
    rng = random.Random(0)
    counts = Counter(c.sample_opponent(rng) for _ in range(4000))
    assert counts['random'] / 4000 == pytest.approx(0.9, abs=0.02)


def test_pfsp_weights_follow_the_recent_win_rate(tmp_path):
    c = _curriculum(tmp_path, phase=2)
    for _ in range(200):  # an old losing streak ...
        c.update_metrics({'winner': -1, 'opponent_type': 'frozen_heuristic'})
    for _ in range(100):  # ... then mastery
        c.update_metrics({'winner': 1, 'opponent_type': 'frozen_heuristic'})
    for k in range(100):
        c.update_metrics({'winner': 1 if k < 40 else -1, 'opponent_type': 'smart_heuristic'})
    metrics = c.metrics[c.current_phase]
    assert metrics.win_rate_vs('frozen_heuristic') == 1.0
    assert metrics.win_rate_vs('smart_heuristic') == pytest.approx(0.4)
    weights = c.get_opponent_distribution()
    assert weights['smart_heuristic'] > weights['frozen_heuristic']


# ---------------------------------------------------------------- battle rules and rank tables

def test_battle_from_values_matches_the_environment_rules():
    from environment.battle import BattleResolver, battle_from_values
    resolver = BattleResolver()
    for attacker in P:
        for defender in P:
            expected = resolver.resolve_battle(attacker, defender)
            assert battle_from_values(attacker.value, -defender.value) == expected
            assert battle_from_values(-attacker.value, defender.value) == expected
    assert battle_from_values(P.MARSHAL.value, HIDDEN) is None


def test_is_enemy_value():
    from environment.battle import is_enemy_value
    assert is_enemy_value(-5, 1) and is_enemy_value(5, -1)
    assert is_enemy_value(HIDDEN, 1) and is_enemy_value(HIDDEN, -1)  # the marker on either side's view
    assert not is_enemy_value(-13, 1) and not is_enemy_value(0, 1) and not is_enemy_value(5, 1)


def _board(setup):
    b = [[0] * 10 for _ in range(10)]
    for (r, c), v in setup.items():
        b[r][c] = v
    return b


def _scores(opponent, attacker, defender, score):
    """Score of P2's `attacker` (at (3, 0)) attacking P1's `defender` (at (4, 0))."""
    board = _board({(3, 0): -attacker, (4, 0): defender})
    return score(opponent, ((3, 0), (4, 0)), board)


SCORERS = {
    'heuristic': (lambda: importlib.import_module('environment.curriculum').HeuristicOpponent('cpu', -1),
                  lambda o, m, b: o._score_move(m, b, -1)),
    'smart_heuristic': (lambda: importlib.import_module('environment.curriculum').SmartHeuristicOpponent('cpu', -1),
                        lambda o, m, b: o._score_move(m, b, [m], -1)),
    'greedy': (lambda: importlib.import_module('network.opponents').GreedyAgent(player_id=-1),
               lambda o, m, b: o._score_move(m, b, -1)),
}


@pytest.mark.parametrize('name', sorted(SCORERS))
def test_scripted_opponents_know_the_special_battles(name):
    """Board values are PieceType values (Flag 1, Spy 2, Scout 3, Miner 4 ... Marshal 11, Bomb 12);
    the thesis tables read them as ranks 0-11, so the Spy and Miner never took their targets."""
    make, score = SCORERS[name]
    opponent = make()
    spy_takes_marshal = _scores(opponent, P.SPY.value, P.MARSHAL.value, score)
    spy_attacks_general = _scores(opponent, P.SPY.value, P.GENERAL.value, score)
    miner_takes_bomb = _scores(opponent, P.MINER.value, P.BOMB.value, score)
    major_hits_bomb = _scores(opponent, P.MAJOR.value, P.BOMB.value, score)
    assert spy_takes_marshal > spy_attacks_general
    assert miner_takes_bomb > major_hits_bomb


@pytest.mark.parametrize('name', ['heuristic', 'smart_heuristic'])
def test_a_hidden_piece_is_not_scored_as_a_certain_loss(name):
    """HIDDEN (-20) read as rank 20, so an attack on an unseen piece always looked lost."""
    make, score = SCORERS[name]
    opponent = make()
    hidden = score(opponent, ((3, 0), (4, 0)), _board({(3, 0): -P.SERGEANT.value, (4, 0): HIDDEN}))
    certain_loss = _scores(opponent, P.SERGEANT.value, P.MARSHAL.value, score)
    assert hidden > certain_loss


@pytest.mark.parametrize('name', ['heuristic', 'smart_heuristic'])
def test_cheap_pieces_probe_hidden_pieces_and_the_spy_does_not(name):
    """With the thesis weights a probe scored below a sideways move, so these opponents only
    shuffled until the no-battle draw."""
    make, score = SCORERS[name]
    opponent = make()
    def probe(attacker):
        return score(opponent, ((3, 0), (4, 0)), _board({(3, 0): -attacker, (4, 0): HIDDEN}))
    sideways = score(opponent, ((3, 0), (3, 1)), _board({(3, 0): -P.SCOUT.value}))
    assert probe(P.SCOUT.value) > sideways
    assert probe(P.SCOUT.value) > probe(P.MARSHAL.value)
    assert probe(P.SPY.value) < probe(P.SCOUT.value)


def test_later_phases_are_gated_on_the_stronger_scripted_opponent():
    from environment.curriculum import phase_gates, TrainingPhase
    assert set(phase_gates(TrainingPhase.MEMORY_GAP)) == {'heuristic'}
    assert set(phase_gates(TrainingPhase.SELF_PLAY)) == {'heuristic'}


def test_smart_heuristic_values_its_own_pieces_by_type():
    from environment.curriculum import SmartHeuristicOpponent
    values = SmartHeuristicOpponent('cpu', -1).piece_values
    assert values[P.MARSHAL.value] == 100 and values[P.GENERAL.value] == 80 and values[P.FLAG.value] == 200
    assert values[P.BOMB.value] == 0


def test_smart_heuristic_does_not_take_lakes_for_enemies():
    from environment.curriculum import SmartHeuristicOpponent
    board = _board({(3, 2): P.CAPTAIN.value, (4, 2): -13, (2, 2): -P.SCOUT.value})
    assert SmartHeuristicOpponent('cpu', 1)._check_adjacent_enemies(3, 2, board, 1) == 1


# ---------------------------------------------------------------- evaluation games

def test_evaluation_games_are_counted_and_greedy(cpu, tiny_network):
    from network.drqn_agent import RainbowAgent
    from environment.curriculum import TrueRandomOpponent
    from network.training.evaluation import play_evaluation_games
    agent = RainbowAgent(player_id=1, device=cpu, num_envs=1)
    seen = []
    original = agent._q_forward_for_acting

    def spy(x):
        seen.append(agent.noisy_acting)
        return original(x)

    agent._q_forward_for_acting = spy
    result = play_evaluation_games(agent, TrueRandomOpponent(cpu), num_games=2, device=cpu, max_turns=30)
    assert result['wins'] + result['losses'] + result['draws'] == result['games'] == 2
    assert result['win_rate'] == result['wins'] / 2
    assert seen and not any(seen)
    assert agent.noisy_acting  # restored afterwards


def test_evaluation_games_run_side_by_side_and_each_side_moves_only_on_its_turn(cpu, monkeypatch):
    from environment.curriculum import TrueRandomOpponent
    from network.training import evaluation
    calls = []

    class Side(TrueRandomOpponent):
        def act_batch(self, boards, valid_moves_list, game_states=None, **kwargs):
            assert all(s.current_player == self.player_id for s in game_states)
            calls.append((self.player_id, len(boards)))
            return super().act_batch(boards, valid_moves_list, game_states)

    original_step = evaluation.StrategoEnvironment.step

    def legal_step(env, action):
        assert tuple(map(tuple, action)) in {tuple(map(tuple, m)) for m in env.get_valid_moves()}
        return original_step(env, action)

    monkeypatch.setattr(evaluation.StrategoEnvironment, 'step', legal_step)
    result = evaluation.play_evaluation_games(Side(cpu, 1), Side(cpu, -1), num_games=6, device=cpu, max_turns=40)
    assert result['wins'] + result['losses'] + result['draws'] == 6
    assert max(n for _, n in calls) == 3  # the starting player alternates: 3 games each side to move first
    assert {p for p, _ in calls} == {1, -1}


# ---------------------------------------------------------------- trainer wiring

def test_trainer_evaluates_and_moves_through_the_phases(tmp_path, config_override, monkeypatch):
    """With tiny caps the run must leave Phase 1 (the thesis config started at Phase 4)."""
    import json
    from .test_trainer_smoke import run_tiny_training
    from environment import CurriculumManager  # the class the trainer uses
    evaluations = []
    original = CurriculumManager.record_evaluation

    def record(self, results):
        evaluations.append((self.current_phase.value, dict(results)))
        return original(self, results)

    monkeypatch.setattr(CurriculumManager, 'record_evaluation', record)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=12,
                      EVAL_INTERVAL=4, EVAL_GAMES=1, PHASE_1_MAX_EPISODES=4, PHASE_2_MAX_EPISODES=4)
    assert evaluations and set(evaluations[0][1]) == {'random', 'heuristic', 'expert'}
    state = json.load(open(tmp_path / 'models' / 'curriculum_state.json'))
    assert state['current_phase'] >= 2


def test_the_stages_change_only_the_opponents(tmp_path, config_override, monkeypatch):
    """The thesis curriculum gave Phase 1 full observability, its own turn limit and a per-phase
    shaping weight, so each transition changed the game under the network. Every stage, and every
    evaluation, now plays fog of war with the one turn limit."""
    from .test_trainer_smoke import run_tiny_training
    import train_dqn
    from environment.parallel_environment import ParallelStrategoEnvironment
    settings = []
    for name in ('set_full_observability', 'set_max_turns'):
        original = getattr(ParallelStrategoEnvironment, name)
        monkeypatch.setattr(ParallelStrategoEnvironment, name,
                            lambda self, v, _n=name, _o=original: (settings.append((_n, v)), _o(self, v))[1])
    evals = []
    original_eval = train_dqn.play_evaluation_games

    def play(*args, **kwargs):
        evals.append((kwargs['full_observability'], kwargs['max_turns']))
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(train_dqn, 'play_evaluation_games', play)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=12,
                      EVAL_INTERVAL=4, EVAL_GAMES=1, PHASE_1_MAX_EPISODES=4, PHASE_2_MAX_EPISODES=4)
    assert settings and set(settings) == {('set_full_observability', False), ('set_max_turns', 40)}
    assert evals and set(evals) == {(False, 40)}


def test_self_play_does_not_overwrite_the_learning_agent2(tmp_path, config_override, monkeypatch):
    """Phase 4: the thesis trainer loaded agent1's weights (or a league checkpoint) into agent2,
    the second learner, every time it was picked as the opponent."""
    from .test_trainer_smoke import run_tiny_training
    from network import RainbowAgent
    from environment import CurriculumManager  # the class the trainer uses
    monkeypatch.setattr(CurriculumManager, 'get_opponent_distribution', lambda self: {'self': 0.5, 'league': 0.5})
    agents = []
    original_init = RainbowAgent.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        agents.append(self)

    monkeypatch.setattr(RainbowAgent, '__init__', init)
    overwrites = []
    original_load = torch.nn.Module.load_state_dict

    def load(self, *args, **kwargs):
        if any(self is a.q_network and a.player_id == -1 and not a.inference_only for a in agents):
            overwrites.append(self)
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', load)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=8, CURRICULUM_START_PHASE=4)
    assert not overwrites
    assert any(a.player_id == -1 and a.inference_only for a in agents)  # the frozen snapshot opponent


def test_agent2_and_the_snapshot_read_aaren_history_from_phase_one(tmp_path, config_override, monkeypatch):
    """With AAREN on, agent2 got no history before Phase 4 although its network reads the AAREN
    planes (and agent1's weights, which expect them, were loaded into it)."""
    from .test_trainer_smoke import run_tiny_training
    from network import RainbowAgent
    agents = []
    original_init = RainbowAgent.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        agents.append(self)

    monkeypatch.setattr(RainbowAgent, '__init__', init)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2, AAREN_ENABLED=True)
    p2_agents = [a for a in agents if a.player_id == -1]
    assert len(p2_agents) == 2  # the learner agent2 and the frozen snapshot opponent
    assert all(a.history is not None and len(a.history_instances) == 2 for a in p2_agents)


def test_greedy_play_never_imitates_the_heuristic():
    """Imitation moves are for training only: an agent acting greedily plays its own argmax."""
    import torch
    from network import RainbowAgent
    from environment import StrategoEnvironment
    env = StrategoEnvironment(torch.device('cpu'))
    state = env.reset()
    agent = RainbowAgent(player_id=1, device=torch.device('cpu'), use_pbs=False, inference_only=True)
    calls = []
    agent.move_filter.get_filtered_actions = lambda *a, **k: calls.append(1) or []
    for _ in range(200):
        agent.act_batch([state.board], [env.get_valid_moves()], [state], env_indices=[None])
    assert not calls


def test_later_stages_keep_the_stage_one_opponents(tmp_path):
    """formal_6: after the stage-1 cap the pool held only opponents it drew or lost against, and
    the agent stopped attacking. Stages 2 and 3 keep random and the heuristic in the pool."""
    for phase in (2, 3):
        weights = _curriculum(tmp_path, phase=phase).get_opponent_distribution()
        assert {'random', 'heuristic'} <= set(weights)
        assert weights['random'] >= 0.15 - 1e-9
        assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- heuristic setup & GUI bot logic

def test_heuristic_setup_places_all_40_pieces_with_correct_groups():
    """smart_heuristic_setup must partition all 40 Stratego pieces with no duplicates or omissions,
    and place Colonels/Majors (not Captains) in the protected back rows."""
    from environment.heuristic_setup import smart_heuristic_setup
    from vec.compat import all_pieces

    army = all_pieces()
    expected_counts = Counter(army)
    colonel_in_back = 0
    colonel_total = 0

    for seed in range(25):
        random.seed(seed)
        for player_id, rows, back_rows in ((1, range(6, 10), {8, 9}), (-1, range(0, 4), {0, 1})):
            valid_pos = [(r, c) for r in rows for c in range(10)]
            placement = smart_heuristic_setup(list(army), valid_pos, player_id)
            assert len(placement) == 40
            positions = [pos for _, pos in placement]
            assert len(set(positions)) == 40
            assert set(positions) == set(valid_pos)
            assert Counter(p for p, _ in placement) == expected_counts
            for p, (r, _) in placement:
                if p == P.COLONEL:
                    colonel_total += 1
                    colonel_in_back += int(r in back_rows)
    # Colonels are explicitly placed in back rows (2 rows of 10 = 20 spots >= 18 placed before step 8)
    assert colonel_in_back == colonel_total


def test_gui_expectamax_utility_full_chance_node_and_policy_value_net(tmp_path):
    """GUI Expectamax: losing higher-rank pieces must hurt more than losing Scouts, chance nodes
    must evaluate all positive-probability ranks (no top-3 pruning), and DQNBotLogic auto-detects
    PolicyValueNet checkpoints."""
    import dqn_bot_logic as D
    import stratego as S
    from network.history_aggregator import HistoryAggregator
    from vec.net import PolicyValueNet, save as save_pv

    class StubAgent:
        device = torch.device('cpu')
        history = None

    search = D.ExpectamaxSearch(StubAgent(), max_depth=1, max_time=10.0)
    # 1. Loss utility: losing General hurts more than losing Scout
    loss_scout = search._calculate_outcome_utility(P.SCOUT, P.MARSHAL)
    loss_general = search._calculate_outcome_utility(P.GENERAL, P.MARSHAL)
    assert loss_general < loss_scout
    assert loss_scout == pytest.approx(-0.5 - (P.SCOUT.value / 12.0) * 0.5)
    assert loss_general == pytest.approx(-0.5 - (P.GENERAL.value / 12.0) * 0.5)

    # 2. Chance node evaluates ALL positive-probability ranks (12 on default prior, not top-3)
    b = S.Board()
    b.set((5, 0), S.Piece(owner=2, rank=6))  # Captain
    b.set((6, 0), S.Piece(owner=1, rank=2, revealed=False))  # unrevealed enemy
    seen_ranks = []
    orig_combat = search._simulate_combat

    def spy_combat(board, move, attacker_rank, defender_rank):
        seen_ranks.append(defender_rank)
        return orig_combat(board, move, attacker_rank, defender_rank)

    search._simulate_combat = spy_combat

    class DummyAdapter:
        def get_state_evaluation(self, board, player_id):
            return 0.0

    import time
    search._expectamax_recursive(b, ((5, 0), (6, 0)), player=2, depth=1, start_time=time.time(),
                                 prior_q=0.0, DQN_adapter=DummyAdapter())
    expected_nonzero = {pt for pt, p in HistoryAggregator.get_default_piece_prior().items() if p > 0}
    assert len(expected_nonzero) == 12
    assert set(seen_ranks) == expected_nonzero

    # 3. DQNBotLogic auto-detects a PolicyValueNet checkpoint and plays legally without leaking hidden pieces
    pv_path = str(tmp_path / 'pv_test.pt')
    save_pv(PolicyValueNet(channels=8, blocks=1, in_planes=59), pv_path)
    bot = D.DQNBotLogic(pv_path, player_id=2, device='cpu')
    assert bot.pv_net is not None and bot.agent is None
    random.seed(0)
    full = S.Board()
    S.auto_setup(full, 1)
    S.auto_setup(full, 2)
    obs = bot._get_state_representation(full, 2)
    assert obs.shape == (59, 10, 10)
    move = bot.choose_move(full, 2)
    assert move in bot._get_gui_legal_moves(full, 2)
    val = bot.get_state_evaluation(full, 2)
    assert -1.0 <= val <= 1.0
    recs = bot.get_multi_recommendations(full, 2, top_n=3)
    assert len(recs) == 3 and recs[0]['label'] == 'Best'
    probs = bot.get_rank_probabilities((6, 0))
    assert probs is not None and sum(probs.values()) > 0.5
    # Regression: evaluating & recommending for the OTHER side (player_id != bot.player_id) must work cleanly
    opp_val = bot.get_state_evaluation(full, 1)
    assert -1.0 <= opp_val <= 1.0
    opp_recs = bot.get_multi_recommendations(full, 1, top_n=3)
    assert len(opp_recs) == 3 and opp_recs[0]['label'] == 'Best'
    assert all(r['move'] in bot._get_gui_legal_moves(full, 1) for r in opp_recs)
    opp_move = bot.get_move_recommendation(full, 1)
    assert opp_move in bot._get_gui_legal_moves(full, 1)
    assert bot._last_moves[1] is None  # recommendation must not overwrite last-move history
