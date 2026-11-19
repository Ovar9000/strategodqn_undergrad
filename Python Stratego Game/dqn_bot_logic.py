import os
import sys
import time
import copy
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import marq_path  # noqa: F401  (Modular Stratego on sys.path)
from network.drqn_agent import RainbowAgent, V_MIN, V_MAX, NUM_ATOMS
from network.history_aggregator import HistoryAggregator
from environment.action_encoding import encode_move
from environment.battle_outlook import INITIAL_COUNTS
from environment.observation import encode_observations, to_canonical
from environment.piece import PieceType, NUM_PIECE_TYPES
from environment.board import LAKE_SQUARE, HIDDEN_PIECE
from settings.training_config import DEFAULT_MAX_TURNS, EVAL_NO_BATTLE_DRAW_PLIES
from stratego import Board

GUI_RANK_TO_PIECETYPE = {
    10: PieceType.MARSHAL,
    9: PieceType.GENERAL,
    8: PieceType.COLONEL,
    7: PieceType.MAJOR,
    6: PieceType.CAPTAIN,
    5: PieceType.LIEUTENANT,
    4: PieceType.SERGEANT,
    3: PieceType.MINER,
    2: PieceType.SCOUT,
    1: PieceType.SPY,
    0: PieceType.BOMB,
    -1: PieceType.FLAG,
}

# Types an unrevealed piece that moves can be
MOVABLE_TYPES = [pt for pt in PieceType if pt not in (PieceType.BOMB, PieceType.FLAG)]
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class GuiObservation:
    """What environment/observation.py reads from a GameState, built from a GUI board: the
    viewer's grid, the counters behind planes 28-29 normalised by the limits the agent is
    evaluated with, the viewer's pieces the other side has seen (plane 26) and the pieces that
    have moved / made a long move (planes 30-31). Observability (plane 27) is 0: the GUI plays
    with fog of war."""
    board: np.ndarray
    own_revealed: np.ndarray
    turn_count: int
    plies_since_battle: int
    moved: Optional[np.ndarray] = None
    long_moved: Optional[np.ndarray] = None
    max_turns: int = DEFAULT_MAX_TURNS
    no_battle_draw_plies: int = EVAL_NO_BATTLE_DRAW_PLIES
    observability: float = 0.0
    hidden_pool: Optional[np.ndarray] = None  # (12,) rank counts of the hidden enemy pieces (battle outlook)

def to_piece_type(val) -> PieceType:
    if isinstance(val, PieceType):
        return val
    if val in GUI_RANK_TO_PIECETYPE:
        return GUI_RANK_TO_PIECETYPE[val]
    try:
        return PieceType(val)
    except (ValueError, KeyError):
        return PieceType.SCOUT

class TimeoutException(Exception):
    pass

class ExpectamaxSearch:
    """
    Formal Expectamax search algorithm for Stratego.
    
    Structure:
    - Max Node: Bot's turn (select best expected move)
    - Chance Node: Target piece identity (sum over AAREN probabilities)
    - Min Node: Opponent's turn (minimize Bot's utility)
    """
    
    def __init__(self, agent: RainbowAgent, max_depth: int = 4, max_time: float = 4.0):
        self.agent = agent
        self.device = agent.device
        self.max_depth = max_depth
        self.max_time = max_time
        
    def _check_timeout(self, start_time: float):
        if time.time() - start_time > self.max_time:
            raise TimeoutException()

    def search(self, gui_board, current_player: int, DQN_adapter) -> Tuple[Optional[Tuple], float]:
        """
        Initiates the recursive Expectamax search.
        Returns the best move and its expected utility.
        """
        start_time = time.time()

        legal_moves = DQN_adapter._get_gui_legal_moves(gui_board, current_player)
        if not legal_moves:
            return None, -float('inf')
            
        # Sort initial moves by prior Q to prioritize promising branches
        move_scores = []
        expected_q = self._get_expected_q(gui_board, DQN_adapter, current_player)
        
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move, current_player)
            score = expected_q[action_idx].item()
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [m[0] for m in move_scores]

        best_move = None
        best_score = -float('inf')
        
        try:
            for move in sorted_moves:
                action_idx = self.agent._move_to_action_index(move, current_player)
                prior_q = expected_q[action_idx].item()
                
                score = self._expectamax_recursive(
                    gui_board, move, current_player, 
                    depth=1, start_time=start_time, 
                    prior_q=prior_q, DQN_adapter=DQN_adapter
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
        except TimeoutException:
            print(f" [Expectamax] Timeout hit at {time.time() - start_time:.2f}s, bounded at depth limits.")
            
        if best_move is None and sorted_moves:
            best_move = sorted_moves[0]
            
        return best_move, best_score

    def _expectamax_recursive(self, board, move: Tuple, player: int, depth: int, 
                              start_time: float, prior_q: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        src, dst = move
        target_piece = board.get(dst)
        
        # 1. Handle non-attacks (simple move)
        if not target_piece:
            sim_board = self._simulate_move_simple(board, move)
            return self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
        # 2. Handle attacks (Chance Node); a revealed defender's type is known
        if target_piece.revealed:
            probabilities = {to_piece_type(target_piece.rank): 1.0}
        else:
            probabilities = self.agent.history.get_piece_predictions(dst) if self.agent.history else None
        if not probabilities:
            probabilities = HistoryAggregator.get_default_piece_prior()
        
        if not probabilities:
            sim_board = self._simulate_move_simple(board, move)
            return self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
        expected_utility = 0.0
        attacker_rank = board.get(src).rank
        
        # Evaluate all ranks with positive probability (at most 12 ranks), highest likelihood first
        active_probs = [(pt, p) for pt, p in sorted(probabilities.items(), key=lambda x: x[1], reverse=True) if p > 0]
        if not active_probs:
            sim_board = self._simulate_move_simple(board, move)
            return self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)

        total_prob = sum(p for _, p in active_probs)

        for p_type, prob in active_probs:
            norm_prob = prob / total_prob
            
            outcome, sim_board = self._simulate_combat(board, move, attacker_rank, p_type)
            combat_utility = self._calculate_outcome_utility(attacker_rank, p_type)
            
            branch_utility = self._min_node(sim_board, 3 - player, depth, start_time, prior_q, DQN_adapter)
            
            # Combine immediate tactical outcome with deeper state utility
            total_branch_utility = 0.5 * combat_utility + 0.5 * branch_utility
            expected_utility += norm_prob * total_branch_utility
            
        return 0.3 * prior_q + 0.7 * expected_utility

    def _min_node(self, board, player: int, depth: int, start_time: float, prior_q: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        if depth >= self.max_depth:
            # Enemy turn evaluated from BOT's perspective (hence 3-player for Bot if player is Enemy)
            return DQN_adapter.get_state_evaluation(board, 3 - player)
            
        # The opponent's replies as the searching side knows them, ranked by the searching side's
        # own evaluation of the position after each (the two worst for it are searched). The
        # thesis code moved the opponent's true pieces, ranked them with the opponent's true army
        # as network input, and let every attack simply overwrite the defender.
        legal_moves = self._opponent_moves(board, player)
        if not legal_moves:
            return V_MAX  # Win: the top of the network's value scale
        sims = [self._simulate_opponent_move(board, move) for move in legal_moves]
        values = DQN_adapter.get_state_evaluations(sims, 3 - player)
        order = sorted(range(len(sims)), key=lambda i: values[i])

        min_utility = float('inf')
        for i in order[:2]:
            sim_board = sims[i]

            utility = self._max_node(sim_board, 3 - player, depth + 1, start_time, DQN_adapter)
            if utility < min_utility:
                min_utility = utility
                
        return min_utility if min_utility != float('inf') else prior_q

    def _max_node(self, board, player: int, depth: int, start_time: float, DQN_adapter) -> float:
        self._check_timeout(start_time)
        
        if depth >= self.max_depth:
            return DQN_adapter.get_state_evaluation(board, player)
            
        legal_moves = DQN_adapter._get_gui_legal_moves(board, player)
        if not legal_moves:
            return V_MIN  # Loss: the bottom of the network's value scale
            
        expected_q = self._get_expected_q(board, DQN_adapter, player)  # the mover's own view
        move_scores = []
        for move in legal_moves:
            action_idx = self.agent._move_to_action_index(move, player)
            score = expected_q[action_idx].item()
            move_scores.append((move, score))
            
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        max_utility = -float('inf')
        for move_tuple in move_scores[:2]:
            move, prior_q = move_tuple
            utility = self._expectamax_recursive(board, move, player, depth + 1, start_time, prior_q, DQN_adapter)
            if utility > max_utility:
                max_utility = utility
                
        return max_utility

    def _get_expected_q(self, board, DQN_adapter, player=None):
        """Expected Q of all 3600 actions for `player` (default: the bot), from `player`'s view."""
        state_tensor = DQN_adapter._get_state_representation(board, player)
        self.agent.q_network.eval()
        with torch.no_grad():
            log_probs = self.agent.q_network(state_tensor.unsqueeze(0))
            probs = log_probs.exp()
            support = self.agent.support if hasattr(self.agent, 'support') else torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(self.device)
            expected_q = (probs * support).sum(dim=2).squeeze(0)
        return expected_q

    def _opponent_moves(self, board, player: int) -> List[Tuple]:
        """Moves of `player` (the side not searching) that the searching side cannot rule out: a
        revealed piece moves by its rules; an unrevealed one may be any movable piece, so it may
        step to any open or enemy-held neighbour (a long Scout move is left out)."""
        moves = []
        for r in range(10):
            for c in range(10):
                piece = board.get((r, c))
                if not piece or piece.owner != player:
                    continue
                if piece.revealed:
                    moves.extend(((r, c), dst) for dst in board.legal_moves_from((r, c)))
                    continue
                for dr, dc in ORTHOGONAL:
                    nr, nc = r + dr, c + dc
                    if board.in_bounds(nr, nc) and not board.is_lake(nr, nc):
                        other = board.get((nr, nc))
                        if other is None or other.owner != player:
                            moves.append(((r, c), (nr, nc)))
        return moves

    def _likely_type(self, pos) -> PieceType:
        """Most likely type of the unrevealed piece that moves from `pos`: the history's
        prediction, else the setup prior, over the movable types."""
        probs = self.agent.history.get_piece_predictions(pos) if self.agent.history else None
        probs = probs or HistoryAggregator.get_default_piece_prior()
        return max(MOVABLE_TYPES, key=lambda pt: probs.get(pt, 0.0))

    def _simulate_opponent_move(self, board, move: Tuple):
        """The opponent's move, with a battle if it attacks: the attacker is its revealed type or
        its most likely type; the defender is the searching side's own piece."""
        src, dst = move
        defender = board.get(dst)
        if defender is None:
            return self._simulate_move_simple(board, move)
        attacker = board.get(src)
        att_type = to_piece_type(attacker.rank) if attacker.revealed else self._likely_type(src)
        return self._simulate_combat(board, move, att_type, defender.rank)[1]

    def _clone_board(self, board):
        new_board = Board()
        new_board.grid = [row[:] for row in board.grid]
        new_board.turn_count = board.turn_count
        new_board.plies_since_battle = board.plies_since_battle
        new_board.max_turns = board.max_turns
        new_board.no_battle_draw_plies = board.no_battle_draw_plies
        return new_board

    def _simulate_move_simple(self, board, move: Tuple):
        sim_board = self._clone_board(board)
        sim_board.turn_count += 1
        sim_board.plies_since_battle += 1
        src, dst = move
        piece = sim_board.get(src)
        sim_board.set(dst, piece)
        sim_board.set(src, None)
        return sim_board
        
    def _simulate_combat(self, board, move: Tuple, attacker_rank, defender_rank):
        sim_board = self._clone_board(board)
        sim_board.turn_count += 1
        sim_board.plies_since_battle = 0
        src, dst = move
        attacker = sim_board.get(src)
        
        att_pt = to_piece_type(attacker_rank)
        def_pt = to_piece_type(defender_rank)
        
        if att_pt == def_pt:
            sim_board.set(src, None)
            sim_board.set(dst, None)
            outcome = "tie"
        elif def_pt == PieceType.FLAG:
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif def_pt == PieceType.BOMB:
            if att_pt == PieceType.MINER:
                sim_board.set(dst, attacker)
                sim_board.set(src, None)
                outcome = "attacker"
            else:
                sim_board.set(src, None)
                outcome = "defender"
        elif att_pt == PieceType.SPY and def_pt == PieceType.MARSHAL:
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif def_pt == PieceType.SPY:
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        elif att_pt.value > def_pt.value:
            sim_board.set(dst, attacker)
            sim_board.set(src, None)
            outcome = "attacker"
        else:
            sim_board.set(src, None)
            outcome = "defender"
            
        return outcome, sim_board

    def _calculate_outcome_utility(self, attacker_rank, defender_rank) -> float:
        att_pt = to_piece_type(attacker_rank)
        def_pt = to_piece_type(defender_rank)
        
        if att_pt == def_pt:
            return -0.1
        if def_pt == PieceType.FLAG:
            return 10.0
        if def_pt == PieceType.BOMB:
            return 1.0 if att_pt == PieceType.MINER else -1.0
        if att_pt == PieceType.SPY and def_pt == PieceType.MARSHAL:
            return 1.0
        if def_pt == PieceType.SPY:
            return 0.8
        if att_pt.value > def_pt.value:
            return 0.5 + (def_pt.value / 12.0) * 0.5
        else:
            return -0.5 - (att_pt.value / 12.0) * 0.5

class DQNBotLogic:
    """
    High-fidelity adapter bridging the Pygame GUI with the MARQ Rainbow DQN
    (with Formal Expectamax) or a vec.net PolicyValueNet PPO checkpoint
    (auto-detected from checkpoint format).
    """
    
    def __init__(self, model_path: str, player_id: int = 2, device: str = None):
        self.player_id = player_id
        self.opponent_id = 1 if player_id == 2 else 2
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pv_net = None
        self._pv_in_planes = 0
        self._last_moves: Dict[int, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]] = {1: None, 2: None}
        self._last_pv_belief: Optional[np.ndarray] = None

        # Resolve checkpoint path (.pt prioritized)
        possible_pt = model_path.replace('.pth', '.pt') if model_path else None
        resolved_path = None
        for path in [possible_pt, model_path]:
            if path and os.path.exists(path):
                resolved_path = path
                break

        # Auto-detect PolicyValueNet checkpoints (e.g. runs/ppo_2/net_final.pt)
        if resolved_path is not None:
            pv_net = self._try_load_policy_value_net(resolved_path)
            if pv_net is not None:
                self.pv_net = pv_net
                self._pv_in_planes = int(pv_net.config.get('in_planes', 59))
                self.agent = None
                self.expectamax = None
                print(f" [MARQ Bot] PolicyValueNet loaded: {os.path.basename(resolved_path)} ({self._pv_in_planes} planes)")
                return

        # Default path: RainbowAgent + Formal Expectamax
        self.agent = RainbowAgent(
            player_id=player_id, 
            device=self.device, 
            use_pbs=True
        )
        
        if resolved_path is not None:
            self.agent.load_model(resolved_path)
            print(f" [MARQ Bot] Model loaded: {os.path.basename(resolved_path)}")
        else:
            print(f" [MARQ Bot] Warning: No valid model found at {model_path} or its .pt variant.")

        # Initialize formal Expectamax engine (4 ply depth bounded by 4 seconds)
        self.expectamax = ExpectamaxSearch(self.agent, max_depth=4, max_time=4.0)

    def _try_load_policy_value_net(self, path: str):
        """Load a vec.net PolicyValueNet if `path` matches its checkpoint schema, else return None."""
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except Exception:
            return None
        if not (isinstance(ck, dict) and 'config' in ck and 'state_dict' in ck and 'q_network_state_dict' not in ck):
            return None
        from vec.net import PolicyValueNet
        net = PolicyValueNet(**ck['config']).to(self.device)
        net.load_state_dict(ck['state_dict'])
        net.eval()
        return net
        
    def reset(self):
        self._last_moves = {1: None, 2: None}
        self._last_pv_belief = None
        if getattr(self, 'agent', None) is not None:
            self.agent.reset_history()

    def _pv_forward(self, gui_board, player_id: int, with_belief: bool = False):
        """Run PolicyValueNet on one GUI board from `player_id`'s canonical frame.
        Always returns a 3-tuple `(logits, value, belief_or_None)`."""
        state = self._get_pv_state_representation(gui_board, player_id).unsqueeze(0).to(self.device)
        self.pv_net.eval()
        with torch.no_grad():
            if with_belief:
                logits, value, belief = self.pv_net(state, with_belief=True)
                if player_id == self.player_id:
                    self._last_pv_belief = torch.softmax(belief[0].float(), dim=0).cpu().numpy()
                return logits[0].float(), float(value[0].item()), belief[0].float()
            logits, value = self.pv_net(state, with_belief=False)
            return logits[0].float(), float(value[0].item()), None

    def _choose_pv_move(self, gui_board, current_player: int, record_move: bool = True) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        legal_moves = self._get_gui_legal_moves(gui_board, current_player)
        if not legal_moves:
            return None
        logits, _, _ = self._pv_forward(gui_board, current_player, with_belief=(current_player == self.player_id))
        best_move = max(legal_moves, key=lambda m: logits[encode_move(m, current_player)].item())
        if record_move:
            if not hasattr(self, '_last_moves'):
                self._last_moves = {1: None, 2: None}
            self._last_moves[current_player] = best_move
        return best_move
        
    def choose_move(self, gui_board, current_player: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if getattr(self, 'pv_net', None) is not None:
            return self._choose_pv_move(gui_board, current_player, record_move=True)

        # Delegate directly to the recursive expectamax search bounded by 4 seconds
        best_move, best_score = self.expectamax.search(gui_board, current_player, self)
        
        if best_move:
            # Diagnostic logging for attacks
            if gui_board.get(best_move[1]):
                print(f" [Expectamax] Attack selected! Score: {best_score:.3f}")
                probs = self.agent.history.get_piece_predictions(best_move[1]) if self.agent.history else None
                if probs:
                    top_rank = max(probs.items(), key=lambda x: x[1])
                    print(f"   Confidence in identity: {top_rank[0].name} ({top_rank[1]*100:.1f}%)")
                    
        return best_move
    
    def get_state_evaluation(self, gui_board, player_id: int) -> float:
        """
        Calculates board advantage using the Dueling Value Head (or PolicyValueNet value head).
        Positive = Advantageous for player_id.
        """
        if getattr(self, 'pv_net', None) is not None:
            _, value, _ = self._pv_forward(gui_board, player_id, with_belief=(player_id == self.player_id))
            return value

        self.agent.q_network.eval()
        with torch.no_grad():
            state = self._get_state_representation(gui_board, player_id).unsqueeze(0).to(self.device)
            # Use the new evaluate_state_value method
            v_probs = self.agent.q_network.evaluate_state_value(state)
            
            # Expected value of the support
            state_value = torch.sum(v_probs * self.agent.support, dim=1).item()
            
            return state_value

    def get_state_evaluations(self, gui_boards, player_id: int) -> List[float]:
        """get_state_evaluation of several boards in one network call."""
        if getattr(self, 'pv_net', None) is not None:
            self.pv_net.eval()
            with torch.no_grad():
                states = torch.stack([self._get_pv_state_representation(b, player_id) for b in gui_boards]).to(self.device)
                _, values = self.pv_net(states, with_belief=False)
                return values.float().tolist()

        self.agent.q_network.eval()
        with torch.no_grad():
            states = torch.stack([self._get_state_representation(b, player_id) for b in gui_boards]).to(self.device)
            v_probs = self.agent.q_network.evaluate_state_value(states)
            return torch.sum(v_probs * self.agent.support, dim=1).tolist()

    def get_multi_recommendations(self, gui_board, player_id: int, top_n: int = 3) -> List[Dict]:
        """
        Returns multiple move suggestions with chess.com-style classifications.
        """
        if getattr(self, 'pv_net', None) is not None:
            legal_moves = self._get_gui_legal_moves(gui_board, player_id)
            if not legal_moves:
                return []
            logits, _, _ = self._pv_forward(gui_board, player_id, with_belief=(player_id == self.player_id))
            legal_indices = [encode_move(m, player_id) for m in legal_moves]
            legal_logits = logits[legal_indices]
            probs = torch.softmax(legal_logits, dim=0)
            move_scores = sorted(
                [(move, float(probs[i].item())) for i, move in enumerate(legal_moves)],
                key=lambda x: x[1],
                reverse=True,
            )
            best_score = move_scores[0][1]
            recommendations = []
            for i, (move, score) in enumerate(move_scores[:top_n]):
                gap = best_score - score
                if i == 0:
                    label, color = "Best", (52, 211, 153)
                elif gap < 0.1:
                    label, color = "Excellent", (96, 165, 250)
                elif gap < 0.3:
                    label, color = "Good", (134, 239, 172)
                else:
                    label, color = "Inaccuracy", (251, 191, 36)
                recommendations.append({'move': move, 'score': score, 'label': label, 'color': color})
            return recommendations

        # Temporarily switch agent ID
        original_id = self.agent.player_id
        self.agent.player_id = player_id
        
        recommendations = []
        try:
            self.agent.q_network.eval()
            with torch.no_grad():
                state = self._get_state_representation(gui_board, player_id).unsqueeze(0).to(self.device)
                log_probs = self.agent.q_network(state)
                expected_q = torch.sum(torch.exp(log_probs) * self.agent.support, dim=2).squeeze(0)
            
            legal_moves = self._get_gui_legal_moves(gui_board, player_id)
            move_scores = []
            
            for move in legal_moves:
                action_idx = self.agent._move_to_action_index(move, player_id)  # lossless, player_id's frame
                score = expected_q[action_idx].item()
                move_scores.append((move, score))
            
            # Sort by score descending
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            if not move_scores:
                return []
                
            best_score = move_scores[0][1]
            
            for i, (move, score) in enumerate(move_scores[:top_n]):
                gap = best_score - score
                
                # Classify based on gap (Chess.com style)
                if i == 0:
                    label = "Best"
                    color = (52, 211, 153) # Emerald
                elif gap < 0.1:
                    label = "Excellent"
                    color = (96, 165, 250) # Blue
                elif gap < 0.3:
                    label = "Good"
                    color = (134, 239, 172) # Light Green
                else:
                    label = "Inaccuracy"
                    color = (251, 191, 36) # Orange
                
                recommendations.append({
                    'move': move,
                    'score': score,
                    'label': label,
                    'color': color
                })
                
        finally:
            self.agent.player_id = original_id
            
        return recommendations

    def get_move_recommendation(self, gui_board, player_id: int) -> Optional[Tuple]:
        """
        Suggests the best move for the human player.
        """
        if getattr(self, 'pv_net', None) is not None:
            return self._choose_pv_move(gui_board, player_id, record_move=False)

        # Temporarily switch agent ID if needed or just use current logic
        original_id = self.agent.player_id
        self.agent.player_id = player_id
        
        try:
            move = self.choose_move(gui_board, player_id)
        finally:
            self.agent.player_id = original_id
            
        return move

    def get_rank_probabilities(self, pos: Tuple[int, int]) -> Optional[Dict]:
        """
        Expose rank probabilities for a specific square (AAREN or PolicyValueNet belief head).
        """
        if getattr(self, 'pv_net', None) is not None:
            belief = getattr(self, '_last_pv_belief', None)
            if belief is None:
                return None
            r, c = pos
            if self.player_id != 1:
                r, c = 9 - r, 9 - c
            probs = belief[:, r * 10 + c]
            return {PieceType(k + 1).name: float(probs[k]) for k in range(NUM_PIECE_TYPES) if probs[k] > 0.01}

        probs = self.agent.history.get_piece_predictions(pos) if self.agent.history else None
        if probs:
            # Convert PieceType keysto simple names/ints for GUI
            return {pt.name: float(p) for pt, p in probs.items() if p > 0.01}
        return None
    
    def update_from_opponent_move(self, gui_board, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """
        Update AAREN history aggregator (or PolicyValueNet last-move/belief state) after opponent move.
        """
        if not hasattr(self, '_last_moves'):
            self._last_moves = {1: None, 2: None}
        self._last_moves[self.opponent_id] = (from_pos, to_pos)
        if getattr(self, 'pv_net', None) is not None:
            self._pv_forward(gui_board, self.player_id, with_belief=True)
            return
        marq_grid = self._gui_board_to_marq_grid(gui_board)
        move = (from_pos, to_pos)
        self.agent.update_history_batch([move], [marq_grid], acting_player=self.opponent_id)

    def _build_gui_observation(self, gui_board, player_id: int) -> GuiObservation:
        return GuiObservation(
            board=self._gui_board_to_marq_grid(gui_board, player_id),
            own_revealed=self._own_revealed_mask(gui_board, player_id),
            moved=self._moved_mask(gui_board, 'moved'),
            long_moved=self._moved_mask(gui_board, 'long_moved'),
            turn_count=gui_board.turn_count,
            plies_since_battle=gui_board.plies_since_battle,
            max_turns=gui_board.max_turns,
            no_battle_draw_plies=gui_board.no_battle_draw_plies,
            hidden_pool=self._hidden_pool(gui_board, player_id),
        )

    def _get_pv_state_representation(self, gui_board, player_id: int = None) -> torch.Tensor:
        """Build the 41- or 59-plane PolicyValueNet observation in `player_id`'s canonical frame,
        matching vec/engine.py:encode."""
        player_id = self.player_id if player_id is None else player_id
        observation = self._build_gui_observation(gui_board, player_id)
        base_obs = encode_observations([observation], player_id, device=self.device)[0]
        in_planes = getattr(self, '_pv_in_planes', 59)
        if in_planes <= base_obs.shape[0]:
            return base_obs[:in_planes]

        # Planes 41-52: hidden enemy pool / initial counts (broadcast over 10x10)
        pool = torch.as_tensor(observation.hidden_pool, dtype=torch.float32, device=self.device)
        initial = torch.tensor(INITIAL_COUNTS, dtype=torch.float32, device=self.device)
        pool_planes = (pool / initial).view(12, 1, 1).expand(12, 10, 10)

        # Planes 53-54: own pieces that have moved / long-moved (canonical frame)
        own_mask = np.zeros((10, 10), dtype=bool)
        for r, c in gui_board.owner_positions(player_id):
            own_mask[r, c] = True
        own_mov = to_canonical(
            torch.from_numpy((observation.moved & own_mask).astype(np.float32)).to(self.device),
            player_id,
        ).unsqueeze(0)
        own_long = to_canonical(
            torch.from_numpy((observation.long_moved & own_mask).astype(np.float32)).to(self.device),
            player_id,
        ).unsqueeze(0)

        # Planes 55-58: enemy last (from, to), own last (from, to) in canonical frame
        last_planes = torch.zeros((4, 10, 10), dtype=torch.float32, device=self.device)
        last_moves = getattr(self, '_last_moves', None) or {}
        enemy_id = (3 - player_id) if player_id in (1, 2) else -player_id
        for from_ch, to_ch, owner in ((0, 1, enemy_id), (2, 3, player_id)):
            mv = last_moves.get(owner)
            if mv is not None:
                (fr, fc), (tr, tc) = mv
                last_planes[from_ch, fr, fc] = 1.0
                last_planes[to_ch, tr, tc] = 1.0
        last_planes = to_canonical(last_planes, player_id)

        full_obs = torch.cat([base_obs, pool_planes, own_mov, own_long, last_planes], dim=0)
        return full_obs[:in_planes]

    def _get_state_representation(self, gui_board, player_id: int = None) -> torch.Tensor:
        """Network input for `player_id` (default: the bot) in that side's canonical frame, with
        the same 41 planes the training environment gives (see GuiObservation), or 59 planes
        when a PolicyValueNet checkpoint is loaded. The bot's AAREN history only describes the
        bot's opponent, so other views get none."""
        player_id = self.player_id if player_id is None else player_id
        if getattr(self, 'pv_net', None) is not None:
            return self._get_pv_state_representation(gui_board, player_id)
        observation = self._build_gui_observation(gui_board, player_id)
        history = self.agent.history if player_id == self.player_id else None
        return self.agent.get_state_representation(observation, pbs_instance=history, player=player_id)

    @staticmethod
    def _hidden_pool(gui_board, viewer: int) -> np.ndarray:
        """(12,) rank counts (PieceType value - 1) of the other side's pieces `viewer` has not
        seen: public, since every captured piece was revealed in its battle."""
        pool = np.zeros(12, dtype=np.float32)
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                if piece and piece.owner != viewer and not piece.revealed:
                    pool[GUI_RANK_TO_PIECETYPE[piece.rank].value - 1] += 1
        return pool

    @staticmethod
    def _moved_mask(gui_board, flag: str) -> np.ndarray:
        """(10, 10) bool, both sides: the pieces that have moved ('moved') or made a long move."""
        mask = np.zeros((10, 10), dtype=bool)
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                mask[r, c] = bool(piece and getattr(piece, flag, False))
        return mask

    @staticmethod
    def _own_revealed_mask(gui_board, viewer: int) -> np.ndarray:
        """(10, 10) bool: the viewer's pieces whose rank the other side has seen in battle."""
        mask = np.zeros((10, 10), dtype=bool)
        for r, c in gui_board.owner_positions(viewer):
            mask[r, c] = gui_board.get((r, c)).revealed
        return mask

    def _gui_board_to_marq_grid(self, gui_board, viewer: int = None) -> np.ndarray:
        """The GUI board in the training environment's values as `viewer` (default: the bot)
        sees it: PieceType values (Flag 1, Spy 2 ... Marshal 11, Bomb 12), Player 1 positive and
        Player 2 negative, and the other side's unrevealed pieces HIDDEN_PIECE. (The thesis code
        copied the GUI ranks, Spy 1 ... Marshal 10, Bomb 0, Flag -1, so every piece read one rank
        low, the bombs vanished and the flags changed sides; and it showed the human's army.)"""
        viewer = self.player_id if viewer is None else viewer
        grid = np.zeros((10, 10), dtype=np.int32)
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                if piece:
                    if piece.owner != viewer and not piece.revealed:
                        grid[r, c] = HIDDEN_PIECE
                    else:
                        val = GUI_RANK_TO_PIECETYPE[piece.rank].value
                        grid[r, c] = val if piece.owner == 1 else -val
                elif gui_board.is_lake(r, c):
                    grid[r, c] = LAKE_SQUARE
        return grid

    def _get_gui_legal_moves(self, gui_board, player_id: int) -> List[Tuple]:
        moves = []
        for r in range(10):
            for c in range(10):
                piece = gui_board.get((r, c))
                if piece and piece.owner == player_id:
                    legal_dsts = gui_board.legal_moves_from((r, c))
                    for dst in legal_dsts:
                        moves.append(((r, c), dst))
        return moves
