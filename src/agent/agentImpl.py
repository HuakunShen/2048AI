import numpy as np
from typing import Union
from pathlib import Path
from src.agent.agent import Player
from src.game.controller.game import Game
from src.game.utils import ARROW_KEYS, UP, DOWN, LEFT, RIGHT
from src.game.model.staticboardImpl import NumpyStaticBoard, TorchStaticBoard
import torch
import random
from collections import deque
from src.agent.model.dqn import DQN
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time


class BacktrackingAIPlayer(Player):
    """
    With search depth=5
    accuracy is around 33%
    in 100 seeds, the following are passing
            score  max_val     runtime
    seed
    6     20328   2048.0  247.989939
    8     20396   2048.0  260.319733
    10    20368   2048.0  243.177021
    14    20328   2048.0  241.503823
    15    20168   2048.0  246.624875
    17    20264   2048.0  238.900709
    23    20288   2048.0  249.540942
    28    20228   2048.0  249.040510
    30    20256   2048.0  245.020730
    32    20300   2048.0  246.836974
    33    20280   2048.0  245.686801
    37    20336   2048.0  256.914075
    40    20332   2048.0  255.554091
    45    20200   2048.0  246.123966
    47    20300   2048.0  250.852761
    56    20256   2048.0  253.074194
    62    20252   2048.0  244.514475
    63    20212   2048.0  245.337596
    65    20440   2048.0  245.588092
    67    20256   2048.0  246.610915
    70    20200   2048.0  237.210614
    71    20484   2048.0  248.832584
    73    20456   2048.0  251.539340
    78    20216   2048.0  247.156880
    79    20196   2048.0  235.604769
    80    20336   2048.0  242.382276
    81    20312   2048.0  252.015181
    83    20436   2048.0  253.451177
    84    20352   2048.0  242.617993
    87    20252   2048.0  243.486014
    88    20200   2048.0  242.957299
    94    20304   2048.0  210.975342
    95    20248   2048.0  177.931139
    """

    def __init__(self, game: Game, search_depth: int = 10, quiet: bool = False,
                 ui: bool = True, torch=False):
        super().__init__(game, quiet, ui)
        if not self._quiet:
            print("Init Backtracking AI Player")
        self.search_depth = search_depth
        self.board = TorchStaticBoard if torch else NumpyStaticBoard

    def recurse_tree(self, matrix: np.ndarray, depth: int) -> int:
        if depth == self.search_depth:
            return 0
        score = 0
        for i, move in enumerate(ARROW_KEYS):
            result_matrix, curr_score, changed = self.board.move(
                matrix=matrix, direction=move, inplace=False)
            self.board.set_random_cell(result_matrix, inplace=True)
            score = max(score, curr_score +
                        self.recurse_tree(result_matrix, depth + 1))
        return score

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                action=first_move, inplace=True)
            if changed:
                scores[first_move_i] += score
            else:
                continue
            score = self.recurse_tree(game_clone1.get_matrix(), 0)
            scores[first_move_i] += score
        return ARROW_KEYS[np.argmax(scores)]


class RandomGuessAIPlayer(Player):
    """
    Around 4% accuracy
    Sample:
    seed_ = 44
    g = Game(seed=seed_)
    player = RandomGuessAIPlayer(
        game=g, searches_per_move=20, search_length=10, ui=False)
    score_, max_val, runtime = player.run()

    Total Time Taken: 0:03:15.620000
    Average Time Taken: 1.96s
        score  max_val    runtime
    seed
    44    20300   2048.0  36.838138
    65    20532   2048.0  34.245618
    72    20324   2048.0  35.080057
    81    20224   2048.0  36.138344
    """

    def __init__(self, game: Game, searches_per_move: int = 20, search_length: int = 10, quiet: bool = False,
                 ui: bool = True):
        super().__init__(game, quiet, ui)
        if not self._quiet:
            print("Init Random guesser AI Player")
        self.search_length = search_length
        self.searches_per_move = searches_per_move

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        scores = np.zeros(4)
        init_game_clone = self._game.clone()
        for first_move_i in range(4):
            game_clone1 = init_game_clone.clone()
            first_move = ARROW_KEYS[first_move_i]
            matrix, score, changed = game_clone1.move(
                action=first_move, inplace=True)
            if changed:
                scores[first_move_i] += score
            else:
                continue
            max_cumulative_score = 0
            for later_moves in range(self.searches_per_move):
                move_number = 1
                game_clone2 = game_clone1.clone()
                changed = True
                score_cumulative = 0
                while changed and move_number < self.search_length:
                    matrix, score, changed = game_clone2.move(
                        inplace=True)  # make a random move
                    if changed:
                        score_cumulative += score
                        # scores[first_move_i] += score
                        move_number += 1
                max_cumulative_score = max(
                    max_cumulative_score, score_cumulative)
            scores[first_move_i] += max_cumulative_score
        return ARROW_KEYS[np.argmax(scores)]


class DQNPlayer(Player):
    """
    Double Deep Q-Learning agent for 2048 (headless-friendly).

    Key design choices (see the DQN model for the one-hot encoding rationale):
      * Action masking - only board-changing moves are ever selected, so the
        game always makes progress (no invalid-move stalls) and every stored
        transition carries a real state change. This removes the need for an
        invalid-move penalty entirely.
      * Double DQN target (select with policy net, evaluate with target net)
        + Huber loss + gradient clipping for stable value estimates - vanilla
        DQN badly over-estimates on 2048.
      * Device split: the game simulation runs on the fast numba-jit
        NumpyStaticBoard on CPU; only the neural net lives on ``device``
        (e.g. Apple MPS). This avoids per-step host<->GPU thrash that would
        otherwise dominate runtime with tiny 4x4 tensor ops.
    """

    def __init__(self, game: Game, quiet: bool = False, ui: bool = False,
                 memory_size: int = 100000, batch_size: int = 256,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay_steps: int = 50000,
                 learning_rate: float = 5e-4, target_update: int = 1000,
                 reward_scale: float = 0.01, train_every: int = 1,
                 empty_weight: float = 0.5, reward_clip: float = 1.0,
                 device: torch.device = None):
        super().__init__(game, quiet, ui)
        self.device = device if device is not None else torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.board = NumpyStaticBoard  # env math backend (masking / simulation)

        # Policy & target networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Hyper-parameters
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update = target_update
        self.reward_scale = reward_scale
        self.train_every = train_every
        self.empty_weight = empty_weight
        self.reward_clip = reward_clip

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.total_steps = 0
        self.writer: Optional[SummaryWriter] = None  # created lazily on train()

    # ------------------------------------------------------------------ #
    # State / action helpers
    # ------------------------------------------------------------------ #
    def _matrix_np(self) -> np.ndarray:
        """Current board as a numpy array regardless of the game backend."""
        m = self._game.get_matrix()
        return m if isinstance(m, np.ndarray) else m.detach().cpu().numpy()

    def valid_moves_mask(self, matrix: np.ndarray = None) -> np.ndarray:
        """Boolean mask over ARROW_KEYS: True where the move changes the board."""
        if matrix is None:
            matrix = self._matrix_np()
        mask = np.zeros(4, dtype=bool)
        for i, direction in enumerate(ARROW_KEYS):
            _, _, changed = self.board.move(matrix, direction, inplace=False)
            mask[i] = changed
        return mask

    def get_state(self) -> torch.Tensor:
        """Raw board as a [1, 4, 4] float tensor on the network device."""
        return torch.as_tensor(
            self._matrix_np(), dtype=torch.float32, device=self.device).unsqueeze(0)

    def _select_action(self, matrix: np.ndarray, greedy: bool = False) -> Tuple[int, np.ndarray]:
        """Return (action_index, valid_mask). Epsilon-greedy over valid moves."""
        mask = self.valid_moves_mask(matrix)
        valid_idx = np.flatnonzero(mask)
        if len(valid_idx) == 0:
            return 0, mask  # terminal - no move helps
        if (not greedy) and random.random() < self.epsilon:
            return int(np.random.choice(valid_idx)), mask
        state = torch.as_tensor(matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state).squeeze(0).cpu().numpy()
        q_masked = np.where(mask, q, -np.inf)
        return int(np.argmax(q_masked)), mask

    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        """Greedy-ish action for the shared Player.run() loop."""
        greedy = self.epsilon <= self.epsilon_min
        idx, _ = self._select_action(self._matrix_np(), greedy=greedy)
        return ARROW_KEYS[idx]

    def calculate_reward(self, score: int, prev_max: int, current_max: int, done: bool) -> float:
        """Merge score (scaled) plus a log2 bonus for reaching a new highest tile."""
        reward = score * self.reward_scale
        if current_max > prev_max:
            reward += float(np.log2(current_max))
        return reward

    # ------------------------------------------------------------------ #
    # Replay / learning
    # ------------------------------------------------------------------ #
    def remember(self, state: torch.Tensor, action: int, reward: float,
                 next_state: torch.Tensor, done: bool):
        self.memory.append(
            (state.detach().cpu(), action, reward, next_state.detach().cpu(), done))

    def replay(self) -> Optional[float]:
        """One Double-DQN gradient step on a random minibatch."""
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: pick next action with policy net, value it with target net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _decay_epsilon(self):
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self, episodes: int = 1000, log_dir: str = None,
              checkpoint_dir: str = None, save_every: int = 0,
              eval_every: int = 0, eval_episodes: int = 20,
              target_update: int = None):
        if target_update is not None:
            self.target_update = target_update
        if self.writer is None:
            self.writer = SummaryWriter(log_dir or f'runs/dqn_2048_{int(time.time())}')

        best_eval_max = 0
        for episode in range(episodes):
            self._game.restart()
            state = self.get_state()
            total_reward = 0.0
            episode_max = 0
            moves = 0
            loss_sum, loss_n = 0.0, 0

            while not self._game.get_is_done():
                matrix = self._matrix_np()
                prev_max = int(matrix.max())
                prev_empty = int((matrix == 0).sum())
                action_idx, _ = self._select_action(matrix, greedy=False)

                _, score, changed = self._game.move(ARROW_KEYS[action_idx])
                next_board = self._matrix_np()
                next_state = torch.as_tensor(
                    next_board, dtype=torch.float32, device=self.device).unsqueeze(0)
                done = self._game.get_is_done()
                current_max = int(self._game.get_max_val())
                reward = self.calculate_reward(score, prev_max, current_max, done)
                # Potential-based empty-cell shaping: F = gamma*phi(s') - phi(s),
                # phi = empty_weight * (#empty cells). Nudges toward keeping the
                # board open (key to surviving in 2048) without changing the
                # optimal policy.
                if self.empty_weight:
                    next_phi = 0.0 if done else self.empty_weight * int((next_board == 0).sum())
                    reward += self.gamma * next_phi - self.empty_weight * prev_empty
                # Reward clipping: bound per-step reward so bootstrapped Q targets
                # stay bounded (~ reward_clip / (1-gamma)). Without this the Q
                # values diverge to ~1e5 and the net collapses to a constant
                # (input-independent) output -> greedy policy becomes ~random.
                if self.reward_clip:
                    reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

                self.remember(state, action_idx, reward, next_state, done)
                state = next_state
                total_reward += reward
                episode_max = max(episode_max, current_max)
                moves += 1
                self.total_steps += 1
                self._decay_epsilon()

                if self.total_steps % self.train_every == 0:
                    loss = self.replay()
                    if loss is not None:
                        loss_sum += loss
                        loss_n += 1
                if self.total_steps % self.target_update == 0:
                    self.update_target_network()

                if self._game.has_won():
                    break

            # ---- per-episode logging ----
            score_ = self._game.get_score()
            self.writer.add_scalar('Score/episode', score_, episode)
            self.writer.add_scalar('MaxTile/episode', episode_max, episode)
            self.writer.add_scalar('Reward/episode', total_reward, episode)
            self.writer.add_scalar('Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Moves/episode', moves, episode)
            if loss_n:
                self.writer.add_scalar('Loss/train', loss_sum / loss_n, episode)

            if not self._quiet and episode % 10 == 0:
                print(f"ep {episode:5d} | score {score_:6d} | max {episode_max:5d} "
                      f"| moves {moves:4d} | eps {self.epsilon:.3f} | steps {self.total_steps}")

            if save_every and checkpoint_dir and (episode + 1) % save_every == 0:
                self.save_model(str(Path(checkpoint_dir) / f"checkpoint_ep{episode + 1}.pth"))

            if eval_every and (episode + 1) % eval_every == 0:
                stats = self.evaluate(episodes=eval_episodes)
                self.writer.add_scalar('Eval/mean_score', stats['mean_score'], episode)
                self.writer.add_scalar('Eval/mean_max_tile', stats['mean_max'], episode)
                self.writer.add_scalar('Eval/best_max_tile', stats['best_max'], episode)
                self.writer.add_scalar('Eval/reach2048_rate', stats['win_rate'], episode)
                if not self._quiet:
                    print(f"  [eval @ep{episode + 1}] mean_score {stats['mean_score']:.0f} "
                          f"mean_max {stats['mean_max']:.0f} best_max {stats['best_max']} "
                          f"reach2048 {stats['win_rate'] * 100:.0f}%")
                if checkpoint_dir and stats['best_max'] >= best_eval_max:
                    best_eval_max = stats['best_max']
                    self.save_model(str(Path(checkpoint_dir) / "best_model.pth"))

        return best_eval_max

    # ------------------------------------------------------------------ #
    # Greedy evaluation (headless, no exploration)
    # ------------------------------------------------------------------ #
    def evaluate(self, episodes: int = 20) -> Dict[str, float]:
        maxes, scores, wins = [], [], 0
        for _ in range(episodes):
            self._game.restart()
            while not self._game.get_is_done():
                idx, mask = self._select_action(self._matrix_np(), greedy=True)
                if not mask.any():
                    break
                self._game.move(ARROW_KEYS[idx])
                if self._game.has_won():
                    wins += 1
                    break
            maxes.append(int(self._game.get_max_val()))
            scores.append(self._game.get_score())
        return {
            'mean_score': float(np.mean(scores)),
            'mean_max': float(np.mean(maxes)),
            'best_max': int(np.max(maxes)),
            'win_rate': wins / episodes,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_model(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, path)

    def load_model(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.total_steps = checkpoint.get('total_steps', 0)
