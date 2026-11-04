"""
Configuration settings for DQN training.
"""

# =============================================================================
# HARDWARE & TRAINING SCALE CONFIGURATION (Fixed)
# =============================================================================
# Games in parallel, one worker process each (~350 MB of RAM per worker: the limit on a 16 GB
# machine). More lanes batch more positions per network call; the replay ratio does not depend
# on it (REPLAY_UPDATE_INTERVAL counts learner moves).
NUM_LANES = 16
NUM_ENVS = NUM_LANES      # LEGACY alias - always equals NUM_LANES (kept for compatibility)
BATCH_SIZE = 128         # Fits 6GB VRAM
GAMMA = 0.999           # Discount factor - EXTREMELY HIGH for 1500-step long-term flag capture planning in Phase 4
MEMORY_SIZE = 75000      # Replay capacity (transitions), sized for 6GB VRAM
LEARNING_RATE = 0.00003    # Reduced from 0.0001 for stable distributional RL

# Network size (RainbowDQN backbone). Thesis runs used 64 channels, 6 residual blocks, 4 heads.
NETWORK_CHANNELS = 64
RESNET_BLOCKS = 6
ATTENTION_HEADS = 4
ADVANTAGE_HIDDEN = 512    # Per-square advantage head width (shared by all 100 squares)

# AAREN history embeddings as extra input planes (off in Phase A: the agent sees the 32
# observation planes only). On: HISTORY_EMBEDDING_SIZE more planes for both agents.
AAREN_ENABLED = False
NUM_EPISODES = 1000000    # Effectively infinite - train until manually stopped
SAVE_INTERVAL = 1000       # Save model/export agent every N episodes
PLOT_INTERVAL = 500       # Save metrics plots every N episodes
EVAL_INTERVAL = 500       # Evaluation games (curriculum gates) every N episodes
REPLAY_UPDATE_INTERVAL = 8    # Train every 8 steps (balances data diversity vs update frequency)
# Asynchronous learner (network/training/async_learner.py): agent 1's updates run on a background
# thread and CUDA stream while the loop acts and steps the games. The ratio of updates to moves
# is the same; acting may lag the learner by up to ASYNC_MAX_LAG + 1 updates.
ASYNC_LEARNER = True
ASYNC_MAX_LAG = 2
# Gradient updates between hard copies of the online network into the target network (the
# agent counts its own updates). The thesis code soft-updated with tau=0.001 every 2000/5000 env
# steps, from a target that started as a different random network.
TARGET_UPDATE_INTERVAL = 1000
REWARD_SCALE = 1.0        # Scaling factor for reward calculations

# =============================================================================
# EXPLORATION SETTINGS (Noisy Networks Only - True Rainbow DQN)
# =============================================================================
# Epsilon-greedy DISABLED: Noisy Networks handle exploration (state-dependent)
# This follows the original Rainbow DQN paper which replaces ε-greedy with NoisyNets
EXPLORATION_EPSILON_START = 0.0   # Disabled - Noisy Networks handle exploration
EXPLORATION_EPSILON_END = 0.0     # Disabled
EXPLORATION_EPSILON_DECAY = 1     # Not used

# =============================================================================
# GAME LENGTH (one game for every stage, evaluation and the GUI)
# =============================================================================
# The thesis curriculum changed the turn limit between phases (200 / 400 / ... turns; later
# 1000 / 1000 / 1000 / 1500). Every stage now plays the same game: the stages only choose the
# opponents (environment/curriculum.py).
DEFAULT_MAX_TURNS = 1500

# No-battle draw: the game is drawn after this many plies (single moves) without an attack.
# 0 turns the rule off (the thesis runs had no such rule). Training used 100 and evaluation 200,
# so the agent was graded under a different draw rule from the one it learned under.
NO_BATTLE_DRAW_PLIES = 200
EVAL_NO_BATTLE_DRAW_PLIES = NO_BATTLE_DRAW_PLIES

# Terminal reward of a draw (a win is +1, a loss -1). With a free draw (0, as in the thesis) the
# formal_1 run learned to stop attacking: by episode 7250 it made 0.1 attacks per game and every
# Phase-2 game ended in a no-battle draw. A drawn game now costs both sides, however it ends: when
# a no-battle draw cost more (-0.5) than a turn-limit draw (-0.25), one cheap attack every 99 plies
# bought the difference (12 of 12 shuffling games went to the turn limit, ~10 pieces thrown away each).
DRAW_REWARD = -0.5

# Weights of the shaping potential's terms (material, penetration, information, proximity); see
# network/distributional_reward.compute_potential. Material only: an ordinary move then carries no
# shaping, and a won exchange stands out from a lost one (Appendix C, the combat signal of the
# potential). The earlier potential was (0.4, 0.15, 0.1, 0.35). formal_8 scales material by 5: at
# weight 1 a won exchange differed from a lost one by about 0.06 in the C51 target, a twelfth of the
# target's width, and the network learned a fifth of it in 5000 episodes (Appendix C, revealed
# pieces). The support (drqn_agent.V_MIN / V_MAX) is widened to match.
POTENTIAL_WEIGHTS = (5.0, 0.0, 0.0, 0.0)

# Warmup Period (must collect this many experiences before training starts)
WARMUP_STEPS = 1000       # Reduced further for faster Phase 1 start

# =============================================================================
# IMITATION LEARNING SETTINGS (Learn from Heuristic Expert)
# =============================================================================
# Off: replaced by the cold start below. Imitation let the heuristic play 10% of the agent's moves
# in its first 2000 games; those moves went into the replay memory like the agent's own, so the
# C51 loss learned their value but nothing told the network to prefer them.
IMITATION_ENABLED = False     # Use heuristic policy as expert demonstration
IMITATION_RATIO = 0.10        # 10% of actions use heuristic expert (reduced for Phase 1)
IMITATION_EPISODES = 2000     # Only use imitation for first N episodes (then pure RL)
IMITATION_REWARD_BOOST = 1.5  # Multiply reward for imitation actions (encourages learning good moves)

# =============================================================================
# COLD START FROM DEMONSTRATIONS (DQfD, Hester et al. 2018)
# =============================================================================
# Before its first game the agent learns from COLD_START_GAMES games in which the heuristic plays
# its side (against random and the heuristic, under the training rules and rewards): first
# COLD_START_PRETRAIN_UPDATES updates on those transitions alone, then a share of every batch
# (COLD_START_DEMO_RATIO, falling to 0 over COLD_START_DEMO_DECAY_EPISODES episodes). Demonstrated
# transitions also get the large-margin loss, so the demonstrator's move is valued
# COLD_START_MARGIN above the other legal moves. A from-zero network otherwise starts from random
# play, which against random and the heuristic is mostly draws: formal_3 drew 32 of 40 evaluation
# games against random at episode 1000. A resumed run collects the demonstrations again (they are
# not checkpointed) but does not pre-train again.
COLD_START_ENABLED = True
COLD_START_GAMES = 100                 # ~37-39k transitions with the flip, ~3 min to play (measured)
COLD_START_DEMO_CAPACITY = 60000       # transitions kept (uint8 states: ~6 KB each, on the GPU)
COLD_START_PRETRAIN_UPDATES = 5000     # DQfD pre-trained for 750k updates; ~9 min here (~107 ms each)
COLD_START_DEMO_RATIO = 0.10           # share of each batch from the demonstrations ...
COLD_START_DEMO_DECAY_EPISODES = 5000  # ... falling linearly to 0 by this episode (0: constant)
# The demonstrated move must be valued this much above any other legal move (0.1 = 1.25 atoms of
# the +-2 support; a winning attack is worth ~0.05 more than a losing one). DQfD used 0.8 with
# rewards clipped to +-1 and lambda = 1.
COLD_START_MARGIN = 0.1
COLD_START_MARGIN_WEIGHT = 1.0

# Multi-Step Returns (N-Step DQN)
N_STEPS = 5               # 5-step returns for deeper credit assignment in long-horizon games (200-500 moves)
GAMMA_N = GAMMA ** N_STEPS  # Bootstrap discount of an n-step return (the agent uses GAMMA for the whole return)

# Episode-Level Replay Settings (Option B: Trajectory Segment Sampling)
EPISODE_REPLAY_ENABLED = False        # Disable dual-buffer episode replay for speed
EPISODE_REPLAY_MAX_EPISODES = 500     # Max stored complete episodes (FIFO eviction)
EPISODE_REPLAY_SEGMENT_LENGTH = 16    # Contiguous steps per segment sample
EPISODE_REPLAY_MIX_RATIO = 0.25      # 25% of batch from episode segments, 75% from PER


# Learning Rate Scheduler
LR_SCHEDULER_ENABLED = False
LR_SCHEDULER_STEP_SIZE = 5000   # Reduce LR every N episodes
LR_SCHEDULER_GAMMA = 0.9        # Multiply LR by this factor

# Prioritized Experience Replay
PER_ENABLED = True              # Prioritized Experience Replay enabled
PER_ALPHA = 0.6                 # Priority exponent (0 = uniform, 1 = full prioritization)
PER_BETA_START = 0.4            # Initial importance sampling weight
PER_BETA_END = 1.0              # Final importance sampling weight
PER_BETA_ANNEAL_EPISODES = 10000  # Episodes to anneal beta
# Transitions whose first move attacks an enemy piece of known rank keep this many times their
# priority on every update, and the importance-sampling weights do not undo it: they are about 5%
# of the replay, and the rank-swap probe needs them concentrated (Appendix C, revealed pieces)
REVEALED_ATTACK_PRIORITY = 4.0

# Weight of a squared error between the predicted and target means of the C51 distribution,
# added to the cross-entropy. The cross-entropy fits the whole return distribution (s.d. about
# 0.77), so mean differences of a few hundredths, such as winning or losing an exchange, are
# learned very slowly; the probe doubled the learned gap with weight 1. 0 = plain C51.
MEAN_LOSS_WEIGHT = 1.0

# Data Augmentation (State/Action Symmetry)
ENABLE_DATA_AUGMENTATION = True # Also store each transition mirrored left-right
AUGMENTATION_TYPES = ["flip"] # "flip" only: the left-right mirror is the one symmetry of the canonical frame

# League Settings (Setup League)
LEAGUE_INTERVAL = 500 # Run setup league every N episodes
LEAGUE_GENERATIONS = 3 # Number of generations to evolve per interval

# Visualization settings
GENERATE_GIFS = False # Whether to generate GIFs of games
GIF_INTERVAL = 100   # Generate GIF every N episodes (reduced frequency)

# AAREN history (named PBS in the thesis code)
PBS_UPDATE_INTERVAL = 4  # The trainer updates the AAREN histories every N steps

# AAREN Performance Optimizations
AAREN_USE_FP16 = True           # Use half-precision inference (~30% faster)
AAREN_USE_TORCHSCRIPT = False   # Disabled - AAREN uses dynamic ops that don't compile
AAREN_HIDDEN_SIZE = 64          # Keep at 64 for checkpoint compatibility
AAREN_NUM_LAYERS = 3            # Keep at 3 for checkpoint compatibility
AAREN_USE_REVEAL_DATA = True    # When False, AAREN supervised training skips reveal data
                                # (debug flag to test if PieceActionAaren overfits on open info)

# History Aggregator Settings (Simplified PBS replacement)
# Uses AAREN embeddings instead of explicit belief distributions
HISTORY_EMBEDDING_SIZE = 64     # Size of AAREN embedding per position (matches hidden_size)

# Rainbow DQN Performance Optimizations  
USE_TORCH_COMPILE = False     # Disabled on Windows due to missing Triton dependency
TORCH_COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"

# League Training Settings (Opponent Diversity)
LEAGUE_TRAINING_ENABLED = True
LEAGUE_SAVE_INTERVAL = 500      # Save agent to league every N episodes
LEAGUE_MAX_AGENTS = 50          # Max historical agents to keep

# Opponent Selection Probabilities (ADJUSTED for faster learning)
# Phase 1: Mix of easy and medium opponents to learn basics AND winning strategy
OPPONENT_LEAGUE_PROB = 0.0      # 0% historical opponents (disabled for now)
OPPONENT_RANDOM_PROB = 0.85     # 85% random agent (easy to beat)
OPPONENT_GREEDY_PROB = 0.15     # 15% heuristic agent (learning pressure)
OPPONENT_SELF_PROB = 0.0        # 0% self-play (disabled for now)

# Each lane draws its opponent type per game from the curriculum's PFSP weights. The frozen
# snapshot opponent (lagged self / league) reloads a checkpoint at most every
# OPPONENT_CYCLE_INTERVAL episodes. (The thesis trainer locked all lanes to one type per
# OPPONENT_CYCLE_INTERVAL block and cycled through the types, ignoring the weights.)
OPPONENT_CYCLE_INTERVAL = 50

# =============================================================================
# OPPONENT SCHEDULE (the former curriculum)
# =============================================================================
# The stages ("phases") choose only the opponents; every stage plays the same game: fog of war,
# the same turn limit, draw rule and reward shaping. The thesis curriculum also switched from
# full observability to fog of war after Phase 1 and cut the shaping weight per phase
# (1.0 / 0.5 / 0.2 / 0), so the network had to relearn the game at each switch.
CURRICULUM_ENABLED = True       # The opponent schedule (off: the legacy opponent pool)
CURRICULUM_START_PHASE = 1      # Stage 1 -> 4 (the thesis runs started at Phase 4)

# Stage gates: win rate (wins / games, draws count as non-wins) of the greedy agent in EVAL_GAMES
# evaluation games per gate opponent, every EVAL_INTERVAL episodes (curriculum.phase_gates)
PHASE_1_WIN_THRESHOLD_RANDOM = 0.70     # Stage 1 -> 2: vs random ...
PHASE_1_WIN_THRESHOLD_HEURISTIC = 0.50  # ... and vs heuristic
PHASE_2_WIN_THRESHOLD = 0.55            # Phase 2 -> 3: vs heuristic (fog of war)
PHASE_3_WIN_THRESHOLD = 0.70            # Phase 3 -> 4: vs heuristic (fog of war)
EVAL_GAMES = 40                         # evaluation games per gate opponent

# Phase Episode Limits: a phase ends after MAX training episodes even if its gates are not met
# (0 = no cap: only the gates advance a stage). From formal_8 there is no cap. In formal_6 and
# formal_7 the stage-1 cap moved an agent that had passed no gate to opponents it lost to, and
# within 500 episodes it drew 37 of 40 games against the random opponent (Appendix C, formal_7).
PHASE_1_MAX_EPISODES = 0
PHASE_2_MAX_EPISODES = 0
PHASE_3_MAX_EPISODES = 0

# Scenario Drill Settings (Phase 5)
SCENARIO_DRILL_INTERVAL = 1000  # Run scenario drills every N episodes during Phase 4

# =============================================================================
# KL-REGULARIZATION SETTINGS (Anti-Cycling)
# =============================================================================
# Prevents strategy oscillation by anchoring to a reference policy
KL_REG_ENABLED = False           # Off in Phase A: plain Rainbow loss (the thesis runs had it on)
KL_REG_WEIGHT = 0.01             # Weight of KL term in total loss
REF_POLICY_UPDATE_INTERVAL = 50000  # Update reference network every N steps

# =============================================================================
# ENTROPY REGULARIZATION SETTINGS (Bluffing/Mixed Strategies)
# =============================================================================
# Encourages stochastic policies to avoid being exploited
ENTROPY_REG_ENABLED = False      # Off in Phase A (the thesis runs had it on)
ENTROPY_COEFF_START = 0.01       # Small coefficient (keep exploration without chaos)
ENTROPY_COEFF_END = 0.001        # Final entropy coefficient (decays over training)
ENTROPY_ANNEAL_EPISODES = 50000  # Episodes to anneal entropy coefficient

# =============================================================================
# ATARAXOS ADVANTAGE FILTERING (Training Efficiency)
# =============================================================================
# Filter training to high-impact transitions based on TD error magnitude
ADVANTAGE_FILTERING_ENABLED = False  # DISABLED: causes 4× slowdown (2 extra forward passes on 4× batch every 2 steps). Re-enable at Phase 3+ when Q-values are meaningful.
ADVANTAGE_OVERSAMPLE_FACTOR = 4   # Sample 4× batch, keep top 25% by TD error
ADVANTAGE_MIN_BATCH = 64          # Minimum batch size after filtering

# =============================================================================
# ATARAXOS DYNAMIC DAMPING (Magnetic Regularization)
# =============================================================================
# Dual KL regularization with power-law annealing schedule
DYNAMIC_DAMPING_ENABLED = False   # Off in Phase A (the thesis runs had it on)
TOTAL_TRAINING_EPISODES = 100000  # For schedule normalization

# Magnet Policy (toward uniform): Starts high, decays to prevent overconfidence
MAGNET_COEFF_START = 0.1          # α_0: Initial coefficient
MAGNET_COEFF_END = 0.001          # α_T: Final coefficient  
MAGNET_POWER = 2.0                # Quadratic decay: α(t) = α_0 * (1-t)^p

# Target KL (toward target network): Starts low, grows for stability
TARGET_KL_COEFF_START = 0.001     # β_0: Initial coefficient
TARGET_KL_COEFF_END = 0.1         # β_T: Final coefficient
TARGET_KL_POWER = 2.0             # Quadratic growth: β(t) = β_0 + (β_T-β_0) * t^p

# REWARDS ARE NOW CONSOLIDATED IN distributional_reward.py
