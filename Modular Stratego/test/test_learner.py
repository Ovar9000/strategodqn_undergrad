"""Step 5: the learner (C51 target, masked Double DQN, one gamma, target copies, PER, NoisyNet acting)."""

import numpy as np
import pytest
import torch

from environment.action_encoding import NUM_ACTIONS, encode_move, legal_action_mask, mirror_action
from environment import StrategoEnvironment


def _agent(cpu, **kwargs):
    from network.drqn_agent import RainbowAgent
    kwargs.setdefault('num_envs', 1)
    kwargs.setdefault('buffer_size', 200)
    return RainbowAgent(player_id=1, device=cpu, **kwargs)


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------- C51 projection

def test_projection_keeps_all_the_mass_for_terminal_and_bootstrapped_targets(cpu, tiny_network):
    """A terminal draw (r = 0 lands exactly on the middle atom) got an all-zero target."""
    agent = _agent(cpu)
    rewards = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.4, -10.0, 10.0, 0.123])
    dones = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    probs = torch.softmax(torch.randn(len(rewards), agent.num_atoms), dim=1)
    m = agent._project_distribution(probs, rewards, dones, agent.gamma_n)
    assert torch.allclose(m.sum(dim=1), torch.ones(len(rewards)), atol=1e-5)
    middle = agent.num_atoms // 2
    assert m[2, middle].item() == pytest.approx(1.0)                     # draw: all mass on z = 0
    assert m[5, 0].item() == pytest.approx(1.0) and m[6, -1].item() == pytest.approx(1.0)  # support ends


def test_projection_of_a_terminal_reward_has_the_reward_as_its_mean(cpu, tiny_network):
    agent = _agent(cpu)
    probs = torch.full((3, agent.num_atoms), 1.0 / agent.num_atoms)
    rewards = torch.tensor([0.3, -0.7, 0.0])
    m = agent._project_distribution(probs, rewards, torch.ones(3), agent.gamma_n)
    assert torch.allclose((m * agent.support).sum(dim=1), rewards, atol=1e-5)


# ---------------------------------------------------------------- masked Double DQN

def test_double_dqn_picks_the_best_legal_next_action(cpu, tiny_network):
    agent = _agent(cpu)
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(4, NUM_ACTIONS, agent.num_atoms), dim=2)
    legal = torch.rand(4, NUM_ACTIONS) < 0.01
    chosen = agent._double_dqn_next_actions(log_probs, legal)
    q = (log_probs.exp() * agent.support).sum(dim=2)
    for i in range(4):
        assert legal[i, chosen[i]]
        assert q[i, chosen[i]] == q[i][legal[i]].max()
    unmasked = agent._double_dqn_next_actions(log_probs, None)
    assert torch.equal(unmasked, q.argmax(dim=1))


def _steps(cpu, n):
    """n consecutive (state, move, next_state, next legal moves) of a random game, Player 1's view."""
    env = StrategoEnvironment(cpu)
    state = env.reset()
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        moves = env.get_valid_moves()
        move = moves[rng.integers(len(moves))]
        next_state, _, done, _ = env.step(move)
        assert not done
        reply = env.get_valid_moves()
        after_reply, _, done, _ = env.step(reply[rng.integers(len(reply))])
        assert not done
        out.append((state, move, after_reply, env.get_valid_moves()))
        state = after_reply
    return out


def test_next_state_legal_moves_reach_the_replay_buffer_through_n_steps_and_the_flip(cpu, tiny_network, config_override):
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"])
    agent = _agent(cpu)
    n = agent.n_steps
    steps = _steps(cpu, n)
    for state, move, next_state, next_moves in steps:
        agent.remember_batch([state.board], [move], [0.0], [next_state.board], [False], [True], [state], [next_state],
                             infos=[{}], next_legal_moves=[next_moves])
    stored = [agent.memory.tree.data[i] for i in range(len(agent.memory))]
    assert len(stored) == 2  # one n-step transition and its mirror image
    original, flipped = stored
    expected = legal_action_mask(steps[-1][3], 1)  # the legal moves after the n-th step
    assert np.array_equal(np.unpackbits(original.next_legal, count=NUM_ACTIONS).astype(bool), expected)
    mirrored = np.zeros(NUM_ACTIONS, dtype=bool)
    mirrored[[mirror_action(a) for a in np.flatnonzero(expected)]] = True
    assert np.array_equal(np.unpackbits(flipped.next_legal, count=NUM_ACTIONS).astype(bool), mirrored)
    assert flipped.action == mirror_action(original.action) and original.action == encode_move(steps[0][1], 1)

    sample = agent.memory.sample(4)
    next_legal = sample[-1]
    assert next_legal.dtype == torch.bool and next_legal.shape == (len(sample[0]), NUM_ACTIONS)


def test_terminal_transitions_store_no_mask_and_allow_every_action(cpu, tiny_network):
    agent = _agent(cpu)
    state, move, next_state, next_moves = _steps(cpu, 1)[0]
    agent.remember_batch([state.board], [move], [1.0], [next_state.board], [True], [True], [state], [next_state],
                         infos=[{}], next_legal_moves=[next_moves])
    assert all(agent.memory.tree.data[i].next_legal is None for i in range(len(agent.memory)))
    assert agent.memory.sample(2)[-1] is None


def test_replay_masks_the_double_dqn_action(cpu, tiny_network, config_override, monkeypatch):
    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=4)
    for state, move, next_state, next_moves in _steps(cpu, agent.n_steps + 4):
        agent.remember_batch([state.board], [move], [0.0], [next_state.board], [False], [True], [state], [next_state],
                             infos=[{}], next_legal_moves=[next_moves])
    seen = []
    original = agent._double_dqn_next_actions

    def spy(log_probs, next_legal):
        actions = original(log_probs, next_legal)
        seen.append((next_legal, actions))
        return actions

    monkeypatch.setattr(agent, '_double_dqn_next_actions', spy)
    assert agent.replay() is not None
    [(next_legal, actions)] = seen
    assert next_legal is not None
    assert next_legal[torch.arange(len(actions)), actions].all()


def test_per_priority_is_the_kl_not_the_cross_entropy(cpu, tiny_network, config_override, monkeypatch):
    """Rainbow prioritises by KL(m || p). The cross-entropy adds H(m), about the same for every
    transition, which flattened the priorities (formal_4 probe: terminal transitions got no more
    replay than quiet ones)."""
    from network.drqn_agent import RainbowAgent
    m = torch.softmax(torch.randn(3, 51), dim=1)
    kl = RainbowAgent._per_priorities(m, m.log())
    assert torch.allclose(kl, torch.zeros(3), atol=1e-5)  # the cross-entropy would be H(m) > 0
    p = torch.full((1, 51), 1 / 51)
    one_hot = torch.zeros(1, 51); one_hot[0, 7] = 1
    assert RainbowAgent._per_priorities(one_hot, p.log()).item() == pytest.approx(np.log(51), rel=1e-5)

    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=4)
    _fill(agent)
    given, used = [], []
    original = agent._per_priorities
    monkeypatch.setattr(agent, '_per_priorities', lambda m, lp: used.append(original(m, lp)) or used[-1])
    monkeypatch.setattr(agent.memory, 'update_priorities', lambda idx, td: given.append(np.asarray(td)))
    assert agent.replay() is not None
    assert len(given) == 1 and np.allclose(given[0], used[0].numpy())


def test_target_network_has_no_dropout_and_keeps_its_noise(cpu, tiny_network, config_override):
    from network.networks.noisy_linear import NoisyLinear
    config_override(WARMUP_STEPS=0, TARGET_UPDATE_INTERVAL=1)
    agent = _agent(cpu, batch_size=8)
    _fill(agent)
    assert agent.replay() is not None  # copies the target again
    target = agent.target_network
    assert not target.training
    assert all(mod.noisy for mod in target.modules() if isinstance(mod, NoisyLinear))
    x = torch.rand(2, agent.input_channels, 10, 10)
    with torch.no_grad():
        assert torch.equal(target(x), target(x))  # same noise sample, no dropout
        before = target(x)
        target.reset_noise()
        assert not torch.equal(target(x), before)  # NoisyNet noise is still on


def test_online_network_in_train_mode_is_deterministic_for_fixed_noise(cpu, tiny_network):
    """replay() runs the online network in train mode (for NoisyNet); with attention dropout the
    Double DQN argmax and the demonstration losses were jittered every update."""
    agent = _agent(cpu, batch_size=8)
    net = agent.q_network
    assert net.training
    x = torch.rand(2, agent.input_channels, 10, 10)
    with torch.no_grad():
        assert torch.equal(net(x), net(x))


def test_no_reference_network_without_the_kl_term(cpu, tiny_network, config_override):
    config_override(KL_REG_ENABLED=False)
    assert _agent(cpu).reference_network is None
    config_override(KL_REG_ENABLED=True)
    assert _agent(cpu).reference_network is not None


def test_n_step_battle_flag_is_the_first_steps():
    from network.prioritized_memory import NStepBuffer
    buf = NStepBuffer(n_steps=3, gamma=0.5)
    out = []
    for t in range(5):
        r = buf.add(t, t, 1.0, t + 1, t == 4, is_battle=(t == 2))
        if r:
            out.append(r)
    out += buf.flush()
    assert [o[1] for o in out] == [0, 1, 2, 3, 4]
    assert [o[-1] for o in out] == [False, False, True, False, False]


def test_battle_boost_goes_to_the_battle_move_and_its_mirror(cpu, tiny_network, config_override):
    """The boost goes with the n-step window that starts with the battle move, original and
    mirror. (The thesis code boosted the window that started n-1 moves earlier and never the
    mirror; a battle on the second move of a game was boosted nowhere.)"""
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"])
    agent = _agent(cpu)
    n = agent.n_steps
    for k, (state, move, next_state, next_moves) in enumerate(_steps(cpu, n + 1)):
        info = {'revealed_in_step': [((0, 0), None), ((0, 1), None)]} if k == 1 else {}
        agent.remember_batch([state.board], [move], [0.0], [next_state.board], [False], [True], [state], [next_state],
                             infos=[info], next_legal_moves=[next_moves])
    leaves = agent.memory.tree.tree[agent.memory.capacity - 1:][:len(agent.memory)]
    assert np.allclose(leaves, [1.0, 1.0, 2.5, 2.5])  # windows from move 0, its mirror, move 1, its mirror


# ---------------------------------------------------------------- one gamma

def test_one_discount_factor(cpu, tiny_network):
    from settings.training_config import GAMMA, N_STEPS
    agent = _agent(cpu)
    assert agent.gamma == GAMMA
    assert agent.n_step_buffers[0].gamma == GAMMA
    assert agent.gamma_n == pytest.approx(GAMMA ** N_STEPS)


# ---------------------------------------------------------------- target network

def _fill(agent, n=40):
    c = agent.input_channels
    for i in range(n):
        agent.memory.add(torch.rand(c, 10, 10), i % agent.action_size, 0.1, torch.rand(c, 10, 10), i % 7 == 0)


def test_target_starts_as_a_copy_and_is_hard_copied_every_n_updates(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0, TARGET_UPDATE_INTERVAL=3)
    agent = _agent(cpu, batch_size=8)
    assert _same_weights(agent.q_network, agent.target_network)
    _fill(agent)
    for _ in range(2):
        assert agent.replay() is not None
    assert not _same_weights(agent.q_network, agent.target_network)
    assert agent.replay() is not None
    assert agent.gradient_updates == 3
    assert _same_weights(agent.q_network, agent.target_network)


def test_average_q_is_the_q_of_the_actions_taken(cpu, tiny_network):
    """Not the max over all 3600 actions: nearly all are illegal and untrained, and that max read
    about +2 in formal_2 while the returns were about -1."""
    agent = _agent(cpu, batch_size=8)
    state = (torch.rand(agent.input_channels, 10, 10) > 0.5).float()
    for _ in range(20):
        agent.memory.add(state, 7, 0.0, state, True)
    agent.q_network.eval()
    with torch.no_grad():
        q = (agent.q_network(state.unsqueeze(0)).exp() * agent.support).sum(dim=2)[0]
    assert agent.get_average_q() == pytest.approx(q[7].item(), abs=1e-5)
    assert q[7].item() != pytest.approx(q.max().item(), abs=1e-5)


def test_the_group_norm_is_instance_norm():
    from network.networks.rainbow_dqn import instance_norm
    x = torch.randn(5, 64, 10, 10) * 3 + 1
    assert torch.allclose(instance_norm(64)(x), torch.nn.functional.instance_norm(x), atol=1e-5)
    assert not list(instance_norm(64).parameters()) and not list(instance_norm(64).buffers())


def test_softmax_after_the_gather_is_the_full_distribution_of_that_action(cpu, tiny_network):
    """The learner normalizes only the actions it gathers; that equals gathering from forward()."""
    agent = _agent(cpu)
    net = agent.q_network.eval()
    x = (torch.rand(4, agent.input_channels, 10, 10) > 0.7).float()
    actions = torch.tensor([0, 17, 1234, 3599])
    idx = actions.view(-1, 1, 1).expand(-1, -1, agent.num_atoms)
    with torch.no_grad():
        full = net(x).gather(1, idx).squeeze(1)
        gathered = torch.nn.functional.log_softmax(net.forward_logits(x).gather(1, idx).squeeze(1), dim=1)
    assert torch.allclose(full, gathered, atol=1e-5)


# ---------------------------------------------------------------- PER

def test_new_samples_get_the_largest_priority_so_far():
    from network.prioritized_memory import PrioritizedReplayBuffer
    buf = PrioritizedReplayBuffer(100, device='cpu', alpha=0.6)
    x = torch.zeros(32, 10, 10)
    buf.add(x, 0, 0.0, x, False)
    first = buf.capacity - 1
    buf.update_priorities([first], [4.0])
    buf.add(x, 1, 0.0, x, False)
    assert buf.tree.tree[first] == pytest.approx(4.0 ** 0.6)
    assert buf.tree.tree[first + 1] == pytest.approx(4.0 ** 0.6)  # the thesis code gave 4 ** 0.36


# ---------------------------------------------------------------- NoisyNet acting

def test_acting_uses_fresh_noise_and_greedy_acting_uses_the_mean_weights(cpu, tiny_network):
    agent = _agent(cpu)
    state = StrategoEnvironment(cpu).reset()
    x = agent.get_batch_state_representation([state.board], [state])
    a, b = agent._q_forward_for_acting(x), agent._q_forward_for_acting(x)
    assert not torch.equal(a, b)
    assert agent.q_network.training and not agent.q_network.advantage_out.noisy

    agent.noisy_acting = False
    c, d = agent._q_forward_for_acting(x), agent._q_forward_for_acting(x)
    assert torch.equal(c, d)
    agent.q_network.eval()
    with torch.no_grad():
        assert torch.allclose(agent.q_network(x), c)


def test_act_batch_resamples_the_noise(cpu, tiny_network, config_override):
    config_override(IMITATION_ENABLED=False)
    agent = _agent(cpu)
    env = StrategoEnvironment(cpu)
    state = env.reset()
    before = agent.q_network.advantage_out.weight_epsilon.clone()
    assert agent.act_batch([state.board], [env.get_valid_moves()], [state], env_indices=[0])[0] in env.get_valid_moves()
    assert not torch.equal(before, agent.q_network.advantage_out.weight_epsilon)


def test_replay_resamples_the_noise(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=8)
    _fill(agent)
    before = agent.target_network.advantage_out.weight_epsilon.clone()
    agent.replay()
    assert not torch.equal(before, agent.target_network.advantage_out.weight_epsilon)


# ---------------------------------------------------------------- defaults

def test_plain_rainbow_loss_and_exact_legal_argmax_by_default(cpu, tiny_network):
    agent = _agent(cpu)
    assert not agent.kl_reg_enabled and not agent.entropy_reg_enabled and not agent.dynamic_damping
    assert not agent.use_heuristic_filter


def test_rotate_augmentation_is_rejected(cpu, tiny_network, config_override):
    """Rotating the canonical frame 180 degrees puts your pieces on the opponent's side."""
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip", "rotate"])
    with pytest.raises(ValueError, match="rotate"):
        _agent(cpu)


# ---------------------------------------------------------------- uint8 storage

def test_observations_are_stored_as_uint8_and_read_back(cpu, tiny_network):
    from network.prioritized_memory import pack_state, unpack_states
    agent = _agent(cpu)
    state = StrategoEnvironment(cpu).reset()
    x = agent.get_batch_state_representation([state.board], [state])[0]
    x[27:30] = torch.rand(3, 10, 10)  # the fraction planes
    x[30:32] = (torch.rand(2, 10, 10) < 0.5).float()  # the binary moved planes
    x[32:] = torch.rand(x.size(0) - 32, 10, 10)  # the battle-outlook probabilities
    packed = pack_state(x, 'cpu', torch.float32)
    assert packed.dtype == torch.uint8
    back = unpack_states([packed], 'cpu')[0]
    assert torch.equal(back[:27], x[:27]) and torch.equal(back[30:32], x[30:32])
    assert (back - x).abs().max() <= 1 / 510 + 1e-6
    y = torch.randn(32, 10, 10)
    assert torch.equal(unpack_states([pack_state(y, 'cpu', torch.float32)], 'cpu')[0], y)


# ---------------------------------------------------------------- turn tracker

def test_turn_tracker_hands_over_the_movers_next_legal_moves():
    from network import TurnTracker
    tt = TurnTracker(1)
    assert tt.record(0, 1, 's1', 'a', 's2', False, {}, next_moves=['p2 moves']) == []
    [t] = tt.record(0, -1, 's2', 'b', 's3', False, {}, next_moves=['p1 moves'])
    assert t['player'] == 1 and t['next_moves'] == ['p1 moves']
    ts = tt.record(0, 1, 's3', 'c', 's4', True, {'winner': 1}, next_moves=[])
    assert [t['next_moves'] for t in ts] == [None, None]


def test_trainer_stores_next_legal_moves_of_the_learner(tmp_path, config_override, monkeypatch):
    from network import RainbowAgent
    from .test_trainer_smoke import run_tiny_training
    calls = []
    original = RainbowAgent.remember_batch

    def spy(self, *args, **kwargs):
        if self.player_id == 1:
            calls.append((args[3], args[4], kwargs.get('next_legal_moves')))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RainbowAgent, 'remember_batch', spy)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2)
    checked = 0
    for next_boards, dones, next_moves in calls:
        assert next_moves is not None
        for board, done, moves in zip(next_boards, dones, next_moves):
            if done:
                assert moves is None
                continue
            assert moves
            for (r, c), _ in moves:
                assert board[r][c] > 0  # Player 1's own piece
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------- revealed attacks and the mean loss

def test_a_persistent_boost_survives_priority_updates_and_is_not_corrected():
    from network.prioritized_memory import PrioritizedReplayBuffer
    buf = PrioritizedReplayBuffer(8, device='cpu', alpha=1.0, beta_start=1.0)
    x = torch.zeros(32, 10, 10)
    buf.add(x, 0, 0.0, x, False)
    buf.add(x, 1, 0.0, x, False, boost=4.0)
    leaf = buf.capacity - 1
    assert buf.tree.tree[leaf + 1] == pytest.approx(4 * buf.tree.tree[leaf])
    buf.update_priorities([leaf, leaf + 1], [0.5, 0.5])
    assert buf.tree.tree[leaf + 1] == pytest.approx(4 * buf.tree.tree[leaf])  # kept on update
    weights, indices = None, []
    for _ in range(20):
        sample = buf.sample(2)
        indices, weights = sample[5], sample[6]
        if len(set(indices)) == 2:
            break
    assert len(set(indices)) == 2
    assert torch.allclose(weights, torch.ones_like(weights))  # same raw priority, same weight


def test_attacks_on_revealed_pieces_are_boosted_with_their_mirror(cpu, tiny_network, config_override):
    from environment.piece import PieceType
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"], N_STEPS=1,
                    REVEALED_ATTACK_PRIORITY=4.0)
    agent = _agent(cpu)
    state = torch.zeros(32, 10, 10)
    state[PieceType.CAPTAIN.value - 1, 6, 3] = 1          # own Captain
    state[12 + PieceType.SCOUT.value - 1, 5, 3] = 1       # revealed enemy Scout above it
    state[24, 6, 4] = 1                                   # hidden enemy to its right
    attack_revealed = encode_move(((6, 3), (5, 3)), 1)
    attack_hidden = encode_move(((6, 3), (6, 4)), 1)
    assert agent._attacks_revealed_piece(state, attack_revealed)
    assert not agent._attacks_revealed_piece(state, attack_hidden)
    agent._add_experience(state, attack_revealed, 0.0, state, False)
    agent._add_experience(state, attack_hidden, 0.0, state, False)
    leaves = agent.memory.tree.tree[agent.memory.capacity - 1:][:len(agent.memory)]
    assert np.allclose(leaves, [4.0, 4.0, 1.0, 1.0])  # original and mirror of each


def test_mean_loss_is_added_to_the_cross_entropy(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0)
    losses = {}
    for weight in (0.0, 1.0):
        config_override(MEAN_LOSS_WEIGHT=weight)
        torch.manual_seed(0)
        agent = _agent(cpu, batch_size=8)
        torch.manual_seed(1)
        _fill(agent)
        import random
        random.seed(2)
        agent.memory.sample = (lambda s: lambda n: s)(agent.memory.sample(8))  # the same batch
        losses[weight] = agent.replay()
    assert losses[1.0] > losses[0.0]


# ---------------------------------------------------------------- horizontal flip & battle-outlook parity

def _mirror_game_state(state):
    """Horizontally mirror (col c -> 9 - c) a GameState in board coordinates."""
    mirrored = state.clone()
    mirrored.board = torch.flip(state.board, [1])
    if state.own_revealed is not None:
        mirrored.own_revealed = torch.flip(state.own_revealed, [1])
    if state.moved is not None:
        mirrored.moved = torch.flip(state.moved, [1])
    if state.long_moved is not None:
        mirrored.long_moved = torch.flip(state.long_moved, [1])
    return mirrored


def test_horizontal_flip_matches_mirrored_observation_all_41_planes(cpu, tiny_network, config_override):
    """_flip_state_horizontal must swap directional battle-outlook planes 34<->35 and 38<->39 so
    that flipping an encoded observation matches encoding the horizontally mirrored position."""
    from environment.board import HIDDEN_PIECE, LAKE_SQUARE
    from environment.game_state import GameState
    from environment.observation import encode_observation, NUM_OBS_PLANES
    from environment.piece import PieceType as P
    from network.drqn_agent import _flip_state_horizontal

    board = torch.zeros(10, 10, dtype=torch.int16)
    for r, c in ((4, 2), (4, 3), (5, 2), (5, 3), (4, 6), (4, 7), (5, 6), (5, 7)):
        board[r, c] = LAKE_SQUARE
    # Asymmetric own & enemy pieces creating distinct LEFT vs RIGHT P(win) and P(lose)
    board[6, 1] = P.CAPTAIN.value          # adjacent to revealed Scout on left (wins), Marshal on right (loses)
    board[6, 0] = -P.SCOUT.value
    board[6, 2] = -P.MARSHAL.value
    board[7, 4] = P.SCOUT.value            # Scout ray: left hits hidden moved enemy at (7, 1), right hits General at (7, 8)
    board[7, 1] = HIDDEN_PIECE
    board[7, 8] = -P.GENERAL.value
    board[8, 5] = P.SPY.value              # Spy next to revealed Marshal on left, hidden unmoved on right
    board[8, 4] = -P.MARSHAL.value
    board[8, 6] = HIDDEN_PIECE
    board[2, 3] = P.MINER.value            # Miner next to revealed Bomb on right, hidden long-moved Scout on left
    board[2, 4] = -P.BOMB.value
    board[2, 2] = HIDDEN_PIECE
    board[9, 0] = P.FLAG.value
    board[0, 9] = -P.FLAG.value

    own_revealed = torch.zeros(10, 10, dtype=torch.bool)
    own_revealed[6, 1] = True
    moved = torch.zeros(10, 10, dtype=torch.bool)
    moved[7, 1] = True
    moved[2, 2] = True
    long_moved = torch.zeros(10, 10, dtype=torch.bool)
    long_moved[2, 2] = True
    hidden_pool = torch.tensor([0, 1, 2, 1, 1, 0, 1, 0, 1, 0, 0, 2], dtype=torch.float32)

    state = GameState(
        board=board, current_player=1, turn_count=37, game_over=False, winner=None,
        move_history=[], uncertainty_mask=torch.zeros(10, 10),
        revealed_pieces_p1={}, revealed_pieces_p2={},
        plies_since_battle=19, no_battle_draw_plies=200, max_turns=1500, observability=0.0,
        own_revealed=own_revealed, moved=moved, long_moved=long_moved, hidden_pool=hidden_pool,
    )
    mirrored = _mirror_game_state(state)

    for player in (1, -1):
        obs = encode_observation(state, player, device=cpu)
        obs_mirrored = encode_observation(mirrored, player, device=cpu)
        assert obs.shape == (NUM_OBS_PLANES, 10, 10)

        flipped = _flip_state_horizontal(obs)
        assert torch.allclose(flipped, obs_mirrored, atol=1e-6), (
            f"player={player} mismatched planes: "
            f"{[c for c in range(NUM_OBS_PLANES) if not torch.allclose(flipped[c], obs_mirrored[c], atol=1e-6)]}"
        )
        # Confirm naive flip without channel swap actually fails on planes 34/35/38/39
        naive = torch.flip(obs, [2])
        assert not torch.allclose(naive[[34, 35, 38, 39]], obs_mirrored[[34, 35, 38, 39]], atol=1e-4)

    # Also verify replay buffer and demo buffer store the channel-swapped flipped state
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"], N_STEPS=1)
    agent = _agent(cpu)
    obs_p1 = encode_observation(state, 1, device=cpu)
    obs_p1_mir = encode_observation(mirrored, 1, device=cpu)
    move = ((6, 1), (6, 0))
    agent._add_experience(obs_p1, move, 0.0, obs_p1, True)
    stored_flip = agent.memory.tree.data[1].state.float() / 255.0
    assert torch.allclose(stored_flip, obs_p1_mir, atol=1 / 255.0 + 1e-5)

    agent.enable_demonstrations(16)
    agent.recording_demos = True
    agent._add_experience(obs_p1, move, 0.0, obs_p1, True)
    agent.recording_demos = False
    demo_flip = agent.demo_memory.buffer[1].state.float() / 255.0
    assert torch.allclose(demo_flip, obs_p1_mir, atol=1 / 255.0 + 1e-5)


# ---------------------------------------------------------------- PBRS-aware PER win boost

def test_pbrs_dominant_win_gets_per_win_boost_from_unshaped_outcome(cpu, tiny_network, config_override):
    """Terminal PBRS subtracts Phi(s_prev), pulling dominant wins below 0.8 (and deep losses above
    0.8). Win detection must use unshaped winner info when provided, falling back to reward >= 0.8."""
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"], N_STEPS=1,
                    REVEALED_ATTACK_PRIORITY=1.0)
    agent = _agent(cpu)
    s = torch.zeros(agent.input_channels, 10, 10)
    base_leaf = agent.memory.capacity - 1

    # 1. Dominant win: shaped terminal reward is -0.5 (< 0.8), unshaped winner == player_id (1) -> 10x boost
    agent._add_experience(s, 0, -0.5, s, True, info={'winner': 1})
    # 2. Same via 'game_winner' key -> 10x boost
    agent._add_experience(s, 1, -1.2, s, True, info={'game_winner': 1})
    # 3. Deep loss whose shaped terminal reward is +1.0 (>= 0.8), unshaped winner == -1 -> NO 10x boost
    agent._add_experience(s, 2, 1.0, s, True, info={'winner': -1})
    # 4. Draw with shaped reward >= 0.8, unshaped winner == 0 -> NO 10x boost
    agent._add_experience(s, 3, 0.9, s, True, info={'winner': 0})
    # 5. Fallback when no unshaped winner is provided: reward >= 0.8 -> 10x boost; reward < 0.8 -> 1x
    agent._add_experience(s, 4, 1.0, s, True)
    agent._add_experience(s, 5, 0.2, s, True)

    leaves = agent.memory.tree.tree[base_leaf:base_leaf + len(agent.memory)]
    # Each call writes original + horizontal flip with identical priority
    expected = [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0]
    assert np.allclose(leaves, expected)

    # 6. remember() and remember_batch() (with N_STEPS=3) propagate unshaped winner from info or next_game_state
    config_override(ENABLE_DATA_AUGMENTATION=False, N_STEPS=3, REVEALED_ATTACK_PRIORITY=1.0)
    agent_n = _agent(cpu)
    steps = _steps(cpu, 3)
    for k, (st, mv, nxt, nxt_moves) in enumerate(steps):
        is_last = (k == 2)
        nxt_copy = nxt.clone()
        if is_last:
            nxt_copy.winner = 1  # simulate TurnTracker closing on opponent's move (info lacks winner)
        agent_n.remember_batch([st.board], [mv], [-0.4 if is_last else 0.0], [nxt_copy.board],
                               [is_last], [True], [st], [nxt_copy],
                               infos=[{}], next_legal_moves=[None if is_last else nxt_moves])
    base_n = agent_n.memory.capacity - 1
    leaves_n = agent_n.memory.tree.tree[base_n:base_n + len(agent_n.memory)]
    assert len(leaves_n) == 3 and np.allclose(leaves_n, [10.0, 10.0, 10.0])

    # 7. remember() accepts info dict with unshaped winner
    agent_r = _agent(cpu)
    st, mv, nxt, _ = steps[0]
    agent_r.remember(st, mv, -0.5, nxt, True, info={'winner': 1})
    assert agent_r.memory.tree.tree[agent_r.memory.capacity - 1] == pytest.approx(10.0)


# ---------------------------------------------------------------- action-policy entropy & dynamic damping

def test_policy_entropy_and_dynamic_damping_use_action_policy_not_c51_atoms(cpu, tiny_network, config_override):
    """ENTROPY_REG_ENABLED, KL_REG_ENABLED, and DYNAMIC_DAMPING_ENABLED must regularize the
    softmax policy over (legal) actions derived from expected Q-values, not the 51 C51 atoms."""
    from network.drqn_agent import RainbowAgent

    # 1. Exact unit check of _policy_from_q on masked legal actions
    q = torch.zeros(2, NUM_ACTIONS, requires_grad=True)
    legal = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
    legal[0, :5] = True
    legal[1, 10:18] = True
    pi, safe_log_pi, eff_mask = RainbowAgent._policy_from_q(q, legal)
    ent = -(pi * safe_log_pi).sum(dim=1)
    assert torch.allclose(ent, torch.tensor([np.log(5.0), np.log(8.0)], dtype=torch.float32), atol=1e-5)
    assert torch.allclose(pi[~legal], torch.zeros_like(pi[~legal]))
    kl_uniform = torch.log(eff_mask.sum(dim=1).float()) - ent
    assert torch.allclose(kl_uniform, torch.zeros(2), atol=1e-5)
    # Gradients through masked entropy must be finite and zero on illegal actions
    (ent.sum()).backward()
    assert torch.isfinite(q.grad).all()
    assert torch.equal(q.grad[~legal], torch.zeros_like(q.grad[~legal]))

    # 2. End-to-end replay() with all three policy regularizers enabled (with & without demo legal masks)
    config_override(WARMUP_STEPS=0, KL_REG_ENABLED=True, ENTROPY_REG_ENABLED=True, DYNAMIC_DAMPING_ENABLED=True)
    agent = _agent(cpu, batch_size=8)
    agent.amp_enabled = False
    _fill(agent, n=24)
    loss_per = agent.replay(episode=10)
    assert loss_per is not None and np.isfinite(loss_per)
    for p in agent.q_network.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    # With demo_only (provides state_legal masks per row)
    agent.enable_demonstrations(32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[0, 1, 2, 3, 4]] = True
    from network.drqn_agent import pack_legal_mask
    packed = pack_legal_mask(mask)
    s = torch.zeros(agent.input_channels, 10, 10)
    for i in range(16):
        agent.demo_memory.add(s, i % 5, 0.0, s, True, legal=packed)
    loss_demo = agent.replay(episode=10, demo_only=True)
    assert loss_demo is not None and np.isfinite(loss_demo)
