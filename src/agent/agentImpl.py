import numpy as np
from typing import Union
from src.agent.agent import Player
from src.game.controller.game import Game
from src.game.utils import ARROW_KEYS, UP, DOWN, LEFT, RIGHT
from src.game.model.staticboardImpl import NumpyStaticBoard, TorchStaticBoard
import torch
import random
from collections import deque
from src.agent.model.dqn import DQN
from typing import Dict, List, Tuple
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
    Deep Q-Learning Agent for 2048
    Uses experience replay and target network for stable learning
    """
    def __init__(self, game: Game, quiet: bool = False, ui: bool = True,
                 memory_size: int = 10000, batch_size: int = 64,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001, device: torch.device = None):
        super().__init__(game, quiet, ui)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # DQN networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode_rewards = []
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # TensorBoard writer
        self.writer = SummaryWriter(f'runs/dqn_2048_{int(time.time())}')
        
    def get_state(self) -> torch.Tensor:
        """Convert game matrix to tensor state"""
        return torch.tensor(self._game.get_matrix(), dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def calculate_reward(self, score: int, changed: bool, prev_max: int, current_max: int) -> float:
        """Calculate reward based on multiple factors"""
        reward = 0.0
        
        # Reward for score increase
        reward += score * 0.1  # Scale down the raw score
        
        # Reward for increasing maximum tile
        if current_max > prev_max:
            reward += (current_max - prev_max) * 0.5
        
        # Penalty for invalid moves
        if not changed:
            reward -= 5.0
            
        # Extra reward for reaching milestone tiles
        milestones = {64: 10, 128: 20, 256: 40, 512: 80, 1024: 160, 2048: 500}
        if current_max in milestones and current_max > prev_max:
            reward += milestones[current_max]
            
        return reward
    
    def get_move(self) -> Union[UP, DOWN, LEFT, RIGHT]:
        state = self.get_state()
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return ARROW_KEYS[np.random.randint(0, 4)]
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            # Log Q-values distribution
            self.writer.add_histogram('q_values', q_values, self.total_steps)
            return ARROW_KEYS[q_values.argmax().item()]
    
    def remember(self, state: torch.Tensor, action: int, reward: float, 
                next_state: torch.Tensor, done: bool):
        """Store experience in replay memory"""
        # Ensure tensors are detached from computation graph and on CPU for storage
        state = state.detach().cpu()
        next_state = next_state.detach().cpu()
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors with consistent dimensions
        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Log loss
        self.writer.add_scalar('Loss/train', loss.item(), self.total_steps)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, episodes: int = 1000, target_update: int = 10):
        """Train the DQN agent"""
        self.total_steps = 0
        best_score = 0
        
        for episode in range(episodes):
            self._game.restart()
            state = self.get_state()
            total_reward = 0
            episode_max = 0
            moves_this_episode = 0
            
            while not self._game.get_is_done():
                # Get action
                action = self.get_move()
                action_idx = ARROW_KEYS.index(action)
                
                # Store previous max value
                prev_max = self._game.get_max_val()
                
                # Take action
                matrix, score, changed = self._game.move(action)
                next_state = self.get_state()
                done = self._game.get_is_done()
                
                # Calculate reward with new scheme
                current_max = self._game.get_max_val()
                reward = self.calculate_reward(score, changed, prev_max, current_max)
                episode_max = max(episode_max, current_max)
                
                # Store experience
                self.remember(state, action_idx, reward, next_state, done)
                
                # Train on past experiences
                self.replay()
                
                state = next_state
                total_reward += reward
                moves_this_episode += 1
                self.total_steps += 1
                
                # Decay epsilon based on total steps
                self.epsilon = max(self.epsilon_min, 
                                 self.epsilon_min + 
                                 (self.epsilon - self.epsilon_min) * 
                                 np.exp(-self.total_steps / 10000))
                
                if self._game.has_won():
                    break
            
            # Update target network periodically
            if episode % target_update == 0:
                self.update_target_network()
            
            # Log episode statistics
            current_score = self._game.get_score()
            best_score = max(best_score, current_score)
            
            self.writer.add_scalar('Score/episode', current_score, episode)
            self.writer.add_scalar('MaxTile/episode', episode_max, episode)
            self.writer.add_scalar('Reward/episode', total_reward, episode)
            self.writer.add_scalar('Epsilon/episode', self.epsilon, episode)
            self.writer.add_scalar('Moves/episode', moves_this_episode, episode)
            self.writer.add_scalar('BestScore', best_score, episode)
            
            if not self._quiet:
                print(f"Episode {episode}, Score: {current_score}, "
                      f"Max Value: {episode_max}, Moves: {moves_this_episode}, "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def save_model(self, path: str):
        """Save the policy network"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }, path)
    
    def load_model(self, path: str):
        """Load a saved policy network"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint.get('total_steps', 0)
