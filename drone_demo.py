import pygame
import sys
import random
import time
import math
import numpy as np
from queue import PriorityQueue
from collections import deque, defaultdict

# Grid Setup
GRID_SIZE = 20
CELL_SIZE = 30
FOV_RADIUS = 3  # Drone sensor range

# Colors
COLORS = {
    'grid': (40, 40, 40),
    'unexplored': (20, 20, 40),  # Dark blue for unsurveilled areas
    'explored': (100, 150, 100),  # Green for cleared areas
    'obstacle': (80, 80, 80),  # Buildings/walls
    'path': (255, 255, 0),  # Yellow drone path
    'drone': (0, 100, 255),  # Blue drone
    'fov': (100, 200, 255, 30),  # Light blue surveillance area
    'frontier': (255, 150, 0),  # Orange for patrol points
    'button': (70, 70, 200),
    'button_hover': (100, 100, 255),
    'button_text': (255, 255, 255),
    'threat_detected': (255, 0, 0),  # Red for detected threats
    'threat_hidden': (139, 0, 0),  # Dark red for hidden threats
    'safe_zone': (0, 255, 0, 50),  # Transparent green for safe areas
    'alert': (255, 50, 50),  # Bright red for alerts
    'patrol_route': (255, 200, 0),  # Golden patrol route
    'surveillance_coverage': (0, 255, 100, 40),  # Light green coverage
    'q_learning': (255, 0, 255),  # Magenta for Q-learning path
    'q_values': (128, 0, 128, 100),  # Purple for Q-value visualization
    'sarsa': (255, 100, 0),  # Orange for SARSA
    'double_q': (0, 255, 255),  # Cyan for Double Q-Learning
    'dqn': (255, 255, 0),  # Yellow for DQN
    'learned_path': (255, 100, 255)  # Pink for learned optimal path
}

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, td_error=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(abs(td_error) + 1e-6)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        # Prioritized sampling based on TD-error
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)

class SimpleNeuralNetwork:
    """Simple neural network for DQN implementation"""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.lr = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        # For momentum
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.mW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3
    
    def backward(self, x, y_true, y_pred):
        m = x.shape[0]
        
        # Output layer gradients
        dz3 = y_pred - y_true
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2 gradients
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1 gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update with momentum
        momentum = 0.9
        self.mW3 = momentum * self.mW3 + (1 - momentum) * dW3
        self.mb3 = momentum * self.mb3 + (1 - momentum) * db3
        self.mW2 = momentum * self.mW2 + (1 - momentum) * dW2
        self.mb2 = momentum * self.mb2 + (1 - momentum) * db2
        self.mW1 = momentum * self.mW1 + (1 - momentum) * dW1
        self.mb1 = momentum * self.mb1 + (1 - momentum) * db1
        
        # Apply gradients
        self.W3 -= self.lr * self.mW3
        self.b3 -= self.lr * self.mb3
        self.W2 -= self.lr * self.mW2
        self.b2 -= self.lr * self.mb2
        self.W1 -= self.lr * self.mW1
        self.b1 -= self.lr * self.mb1
    
    def copy_weights(self):
        """Return copy of current weights"""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
            'W3': self.W3.copy(), 'b3': self.b3.copy()
        }
    
    def load_weights(self, weights):
        """Load weights from dictionary"""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()
        self.W3 = weights['W3'].copy()
        self.b3 = weights['b3'].copy()

class OptimizedQLearningAgent:
    """Optimized Q-Learning with multiple improvements"""
    
    def __init__(self, grid_width, grid_height, algorithm_type="optimized_q"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.algorithm_type = algorithm_type
        
        # Improved hyperparameters[20][22]
        self.learning_rate = 0.15  # Slightly higher for faster learning
        self.discount_factor = 0.95  # Good balance for surveillance task
        self.epsilon = 0.9  # Start with high exploration
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.995  # Gradual decay
        
        # Action mappings (8-directional + stay)
        self.actions = [
            (-1, 0),   # 0: Up
            (0, 1),    # 1: Right  
            (1, 0),    # 2: Down
            (0, -1),   # 3: Left
            (-1, 1),   # 4: Up-Right
            (1, 1),    # 5: Down-Right
            (1, -1),   # 6: Down-Left
            (-1, -1),  # 7: Up-Left
            (0, 0)     # 8: Stay (for emergency)
        ]
        
        # Algorithm-specific initialization
        if algorithm_type == "dqn":
            self._init_dqn()
        elif algorithm_type == "double_q":
            self._init_double_q()
        elif algorithm_type == "sarsa":
            self._init_sarsa()
        else:
            self._init_optimized_q()
        
        # Performance tracking
        self.episode = 0
        self.total_reward = 0
        self.episode_rewards = []
        self.learning_progress = []
        
        # Experience tracking for better exploration
        self.visited_states = defaultdict(int)
        self.last_state = None
        self.last_action = None
        self.stuck_counter = 0
        self.exploration_bonus_decay = 0.99
        
    def _init_optimized_q(self):
        """Initialize optimized Q-learning with eligibility traces"""
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.eligibility_traces = defaultdict(lambda: np.zeros(len(self.actions)))
        self.trace_decay = 0.9
        
    def _init_double_q(self):
        """Initialize Double Q-Learning[26]"""
        self.q_table_a = defaultdict(lambda: np.zeros(len(self.actions)))
        self.q_table_b = defaultdict(lambda: np.zeros(len(self.actions)))
        
    def _init_sarsa(self):
        """Initialize SARSA algorithm"""
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.next_action = None
        
    def _init_dqn(self):
        """Initialize Deep Q-Network"""
        input_size = 8  # Compact state representation
        hidden_size = 64
        output_size = len(self.actions)
        
        self.main_network = SimpleNeuralNetwork(input_size, hidden_size, output_size, 0.001)
        self.target_network = SimpleNeuralNetwork(input_size, hidden_size, output_size, 0.001)
        self.replay_buffer = ReplayBuffer(10000)
        
        self.target_update_freq = 100  # Update target network every 100 steps
        self.batch_size = 32
        self.steps_done = 0
        
    def get_compact_state(self, pos, grid):
        """Optimized state representation to reduce complexity[23]"""
        x, y = pos
        
        # Local environment features (3x3 around agent)
        obstacle_count = 0
        surveilled_count = 0
        threat_count = 0
        unsurveilled_count = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if grid.grid[ny][nx]['obstacle']:
                        obstacle_count += 1
                    elif grid.grid[ny][nx]['surveilled']:
                        surveilled_count += 1
                        if grid.grid[ny][nx]['threat_detected']:
                            threat_count += 1
                    else:
                        unsurveilled_count += 1
        
        # Directional features - distance to nearest unsurveilled area
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        dir_features = []
        
        for dx, dy in directions:
            distance = 0
            for step in range(1, 8):  # Look ahead 8 steps
                nx, ny = x + dx * step, y + dy * step
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    break
                if grid.grid[ny][nx]['obstacle']:
                    break
                if not grid.grid[ny][nx]['surveilled']:
                    distance = step
                    break
            dir_features.append(min(distance / 8.0, 1.0))  # Normalize
        
        # Compact state: [obstacle_ratio, surveilled_ratio, threat_ratio, unsurveilled_ratio] + dir_features
        state = (
            obstacle_count / 9.0,
            surveilled_count / 9.0, 
            threat_count / max(1, surveilled_count),
            unsurveilled_count / 9.0,
            *dir_features
        )
        
        return state
        
    def choose_action(self, state, grid, pos):
        """Improved action selection with anti-stuck mechanism"""
        if self.algorithm_type == "dqn":
            return self._choose_action_dqn(state, grid, pos)
        elif self.algorithm_type == "sarsa":
            return self._choose_action_sarsa(state, grid, pos)
        else:
            return self._choose_action_epsilon_greedy(state, grid, pos)
    
    def _choose_action_epsilon_greedy(self, state, grid, pos):
        """Improved epsilon-greedy with exploration bonus"""
        # Anti-stuck mechanism
        if self.stuck_counter > 5:
            valid_actions = self.get_valid_actions(pos, grid)
            if valid_actions:
                return random.choice(valid_actions)
        
        # Epsilon-greedy with exploration bonus
        if random.random() < self.epsilon:
            # Exploration: prefer less visited states
            valid_actions = self.get_valid_actions(pos, grid)
            if not valid_actions:
                return 8 # Stay if no other option
            
            # Weight actions by inverse visit count
            action_weights = []
            for action in valid_actions:
                next_pos = self.get_next_position(pos, action)
                visit_count = self.visited_states[next_pos]
                weight = 1.0 / (visit_count + 1)  # Higher weight for less visited
                action_weights.append(weight)
            
            # Weighted random selection
            total_weight = sum(action_weights)
            if total_weight > 0:
                r = random.random() * total_weight
                cumulative = 0
                for i, weight in enumerate(action_weights):
                    cumulative += weight
                    if r <= cumulative:
                        return valid_actions[i]
            return random.choice(valid_actions)
        else:
            # Exploitation
            if self.algorithm_type == "double_q":
                # Double Q-Learning: average both Q-tables
                q_values_a = self.q_table_a[state]
                q_values_b = self.q_table_b[state]
                combined_q = (q_values_a + q_values_b) / 2
                valid_actions = self.get_valid_actions(pos, grid)
                if not valid_actions:
                    return 8 # Stay
                valid_q_values = [(combined_q[action], action) for action in valid_actions]
                return max(valid_q_values, key=lambda x: x[0])[1]
            else:
                # Regular Q-learning
                q_values = self.q_table[state]
                valid_actions = self.get_valid_actions(pos, grid)
                if not valid_actions:
                    return 8 # Stay
                valid_q_values = [(q_values[action], action) for action in valid_actions]
                return max(valid_q_values, key=lambda x: x[0])[1]
    
    def _choose_action_dqn(self, state, grid, pos):
        """DQN action selection"""
        if random.random() < self.epsilon:
            valid_actions = self.get_valid_actions(pos, grid)
            return random.choice(valid_actions) if valid_actions else 8
        else:
            state_array = np.array(state).reshape(1, -1)
            q_values = self.main_network.forward(state_array)[0]
            valid_actions = self.get_valid_actions(pos, grid)
            if not valid_actions:
                return 8
            valid_q_values = [(q_values[action], action) for action in valid_actions]
            return max(valid_q_values, key=lambda x: x[0])[1]
    
    def _choose_action_sarsa(self, state, grid, pos):
        """SARSA action selection"""
        if self.next_action is not None:
            action = self.next_action
        else:
            action = self._choose_action_epsilon_greedy(state, grid, pos)
        
        # Choose next action for SARSA update
        next_pos = self.get_next_position(pos, action)
        next_state_for_sarsa = self.get_compact_state(next_pos, grid)
        
        if random.random() < self.epsilon:
            valid_actions = self.get_valid_actions(next_pos, grid)
            self.next_action = random.choice(valid_actions) if valid_actions else 8
        else:
            q_values = self.q_table[next_state_for_sarsa]
            valid_actions = self.get_valid_actions(next_pos, grid)
            if valid_actions:
                valid_q_values = [(q_values[a], a) for a in valid_actions]
                self.next_action = max(valid_q_values, key=lambda x: x[0])[1]
            else:
                self.next_action = 8
        
        return action
    
    def get_valid_actions(self, pos, grid):
        """Get list of valid actions from current position"""
        x, y = pos
        valid_actions = []
        
        for action_idx, (dy, dx) in enumerate(self.actions):
            new_x, new_y = x + dx, y + dy
            if grid.is_accessible(new_x, new_y):
                valid_actions.append(action_idx)
        
        # BUG FIX: If no actions are valid, allow 'stay'
        if not valid_actions:
            valid_actions.append(8)

        return valid_actions
    
    def calculate_reward(self, pos, grid, threats_found, prev_coverage, action):
        """Enhanced reward function with multiple objectives"""
        reward = 0
        
        # Coverage improvement reward
        current_coverage = grid.get_surveillance_coverage()
        coverage_improvement = current_coverage - prev_coverage
        reward += coverage_improvement * 200  # Increased weight for coverage
        
        # Threat detection reward (highest priority)
        reward += len(threats_found) * 100
        
        # Exploration reward (less visited areas)
        visit_count = self.visited_states[pos]
        exploration_reward = 5.0 / (visit_count + 1)
        reward += exploration_reward
        
        # Efficiency reward (minimize redundant moves)
        unsurveilled_nearby = grid.count_unsurveilled_neighbors(pos[0], pos[1], radius=FOV_RADIUS)
        reward += unsurveilled_nearby * 3
        
        # Anti-stuck penalty
        if action == 8:  # Stay action
            reward -= 10
        
        # Proximity to unsurveilled areas
        if unsurveilled_nearby == 0 and coverage_improvement <= 0:
            reward -= 2  # Small penalty for being in fully surveilled area with no new coverage
        
        # Risk area bonus
        if grid.is_high_risk_area(pos[0], pos[1]) and unsurveilled_nearby > 0:
            reward += 8
        
        return reward
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-values based on algorithm type"""
        if self.algorithm_type == "dqn":
            self._update_dqn(state, action, reward, next_state)
        elif self.algorithm_type == "double_q":
            self._update_double_q(state, action, reward, next_state)
        elif self.algorithm_type == "sarsa":
            self._update_sarsa(state, action, reward, next_state)
        else:
            self._update_optimized_q(state, action, reward, next_state)
        
        self.total_reward += reward
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_optimized_q(self, state, action, reward, next_state):
        """Optimized Q-learning with eligibility traces"""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        td_error = reward + self.discount_factor * max_next_q - current_q
        
        # Update eligibility traces
        self.eligibility_traces[state][action] += 1.0 # Accumulating trace
        
        # Update all states in eligibility trace
        for s in self.q_table:
            for a in range(len(self.actions)):
                if self.eligibility_traces[s][a] > 0:
                    self.q_table[s][a] += self.learning_rate * td_error * self.eligibility_traces[s][a]
                    self.eligibility_traces[s][a] *= self.discount_factor * self.trace_decay
    
    def _update_double_q(self, state, action, reward, next_state):
        """Double Q-Learning update to reduce overestimation bias"""
        if random.random() < 0.5:
            # Update Q_A using Q_B for next action selection
            current_q = self.q_table_a[state][action]
            best_next_action = np.argmax(self.q_table_a[next_state])
            max_next_q = self.q_table_b[next_state][best_next_action]
            
            td_error = reward + self.discount_factor * max_next_q - current_q
            self.q_table_a[state][action] += self.learning_rate * td_error
        else:
            # Update Q_B using Q_A for next action selection  
            current_q = self.q_table_b[state][action]
            best_next_action = np.argmax(self.q_table_b[next_state])
            max_next_q = self.q_table_a[next_state][best_next_action]
            
            td_error = reward + self.discount_factor * max_next_q - current_q
            self.q_table_b[state][action] += self.learning_rate * td_error
    
    def _update_sarsa(self, state, action, reward, next_state):
        """SARSA update (on-policy)"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][self.next_action] if self.next_action is not None else 0
        
        td_error = reward + self.discount_factor * next_q - current_q
        self.q_table[state][action] += self.learning_rate * td_error
    
    def _update_dqn(self, state, action, reward, next_state):
        """DQN update with experience replay"""
        # Store experience
        done = False  # For surveillance, we don't have terminal states
        state_array = np.array(state)
        next_state_array = np.array(next_state)
        
        # Calculate TD error for prioritization
        current_q = self.main_network.forward(state_array.reshape(1, -1))[0][action]
        next_q = np.max(self.target_network.forward(next_state_array.reshape(1, -1))[0])
        td_error = reward + self.discount_factor * next_q - current_q
        
        self.replay_buffer.push(state_array, action, reward, next_state_array, done, td_error)
        
        # Train network if enough experiences
        if len(self.replay_buffer) >= self.batch_size:
            self._train_dqn()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_weights(self.main_network.copy_weights())
    
    def _train_dqn(self):
        """Train DQN with experience replay"""
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones, indices = batch
        
        # Current Q values
        current_q_values = self.main_network.forward(states)
        
        # Next Q values from target network
        next_q_values = self.target_network.forward(next_states)
        target_q_values = rewards + self.discount_factor * np.max(next_q_values, axis=1) * (1 - dones)
        
        # Update Q values for taken actions
        targets = current_q_values.copy()
        for i, action in enumerate(actions):
            targets[i][int(action)] = target_q_values[i]
        
        # Train network
        self.main_network.backward(states, targets, current_q_values)
        
        # Update priorities in replay buffer
        td_errors = np.abs(target_q_values - current_q_values[np.arange(len(actions)), actions.astype(int)])
        self.replay_buffer.update_priorities(indices, td_errors)
    
    def get_next_position(self, pos, action):
        """Get next position based on action"""
        x, y = pos
        dy, dx = self.actions[action]
        return (x + dx, y + dy)
    
    def update_stuck_counter(self, moved):
        """Update stuck counter for anti-stuck mechanism"""
        if not moved:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
    
    def new_episode(self):
        """Start new learning episode"""
        if self.episode > 0:
            self.episode_rewards.append(self.total_reward)
        
        self.episode += 1
        self.total_reward = 0
        self.visited_states.clear()
        self.stuck_counter = 0
        
        # Clear eligibility traces for next episode
        if self.algorithm_type == "optimized_q":
            self.eligibility_traces.clear()
    
    def get_learning_stats(self):
        """Get learning statistics"""
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        stats = {
            'episode': self.episode,
            'epsilon': self.epsilon,
            'avg_reward': avg_reward,
            'stuck_count': self.stuck_counter,
            'algorithm': self.algorithm_type
        }
        
        if self.algorithm_type == "optimized_q":
            stats['q_table_size'] = len(self.q_table)
        elif self.algorithm_type == "double_q":
            stats['q_table_size'] = len(self.q_table_a) + len(self.q_table_b)
        elif self.algorithm_type == "dqn":
            stats['replay_size'] = len(self.replay_buffer)
            stats['steps'] = self.steps_done
        
        return stats

class SurveillanceGrid:
    def __init__(self, width, height):
        self.grid = [[{'surveilled': False, 'obstacle': False, 'threat': False, 'threat_detected': False} 
                       for _ in range(width)] for _ in range(height)]
        self.width = width
        self.height = height
        self.total_cells = width * height
        self.obstacle_cells = set()
        self.threat_cells = set()
        self.detected_threats = set()
        self.total_threats = 0
        
    def set_grid(self, grid_map):
        """Set up the surveillance environment with buildings and obstacles"""
        for y in range(self.height):
            for x in range(self.width):
                if grid_map[y][x] == '1':
                    self.grid[y][x]['obstacle'] = True
                    self.obstacle_cells.add((x, y))
                else:
                    self.grid[y][x]['obstacle'] = False
    
    def place_threats(self, num_threats=8):
        """Randomly place threats/bandits in the environment"""
        available_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]['obstacle']:
                    available_cells.append((x, y))
        
        if len(available_cells) < num_threats:
            num_threats = len(available_cells)
        
        threat_positions = random.sample(available_cells, num_threats)
        for x, y in threat_positions:
            self.grid[y][x]['threat'] = True
            self.threat_cells.add((x, y))
        
        self.total_threats = num_threats
        print(f"Placed {num_threats} threats in the surveillance area")
        
    def surveillance_scan(self, x, y):
        """Drone scans area and detects threats within FOV"""
        threats_found = []
        newly_surveilled_count = 0
        for dx in range(-FOV_RADIUS, FOV_RADIUS+1):
            for dy in range(-FOV_RADIUS, FOV_RADIUS+1):
                scan_x, scan_y = x + dx, y + dy
                if 0 <= scan_x < self.width and 0 <= scan_y < self.height:
                    # Mark as surveilled
                    if not self.grid[scan_y][scan_x]['surveilled']:
                        self.grid[scan_y][scan_x]['surveilled'] = True
                        newly_surveilled_count +=1
                    
                    # Check for threats
                    if self.grid[scan_y][scan_x]['threat'] and not self.grid[scan_y][scan_x]['threat_detected']:
                        self.grid[scan_y][scan_x]['threat_detected'] = True
                        self.detected_threats.add((scan_x, scan_y))
                        threats_found.append((scan_x, scan_y))
        
        return threats_found
    
    def is_accessible(self, x, y):
        """Check if drone can navigate to this position"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                not self.grid[y][x]['obstacle'])
    
    def get_surveillance_coverage(self):
        """Calculate percentage of area under surveillance"""
        surveilled = sum(sum(cell['surveilled'] for cell in row) for row in self.grid)
        total_surveillable = self.total_cells - len(self.obstacle_cells)
        return surveilled / total_surveillable if total_surveillable > 0 else 0
    
    def get_threat_detection_rate(self):
        """Calculate percentage of threats detected"""
        if self.total_threats == 0:
            return 1.0
        return len(self.detected_threats) / self.total_threats
    
    def is_surveillance_complete(self):
        """Check if all accessible areas have been surveilled"""
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]['obstacle'] and not self.grid[y][x]['surveilled']:
                    return False
        return True
    
    def count_unsurveilled(self):
        """Count unsurveilled accessible areas"""
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]['obstacle'] and not self.grid[y][x]['surveilled']:
                    count += 1
        return count
    
    def is_high_risk_area(self, x, y):
        """Identify high-risk areas (near obstacles, corners, etc.)"""
        obstacle_count = 0
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x+dx, y+dy
            if not (0 <= nx < self.width and 0 <= ny < self.height) or self.grid[ny][nx]['obstacle']:
                obstacle_count += 1
        return obstacle_count >= 2  # Corner or narrow passage
    
    def count_unsurveilled_neighbors(self, x, y, radius=FOV_RADIUS+1):
        """Count unsurveilled cells within radius (for patrol planning)"""
        count = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x+dx, y+dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not self.grid[ny][nx]['obstacle'] and
                    not self.grid[ny][nx]['surveilled']):
                    count += 1
        return count

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def detect_patrol_points(grid, drone_pos):
    """Find optimal patrol points (boundaries between surveilled and unsurveilled areas)"""
    patrol_points = []
    
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x]['surveilled'] and not grid.grid[y][x]['obstacle']:
                # Check if adjacent to unsurveilled area
                is_frontier = False
                for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < grid.width and 0 <= ny < grid.height and
                        not grid.grid[ny][nx]['surveilled'] and
                        not grid.grid[ny][nx]['obstacle']):
                        is_frontier = True
                        break
                if is_frontier:
                    patrol_points.append((x, y))
    
    return patrol_points

def a_star_search(grid, start, goal):
    """A* pathfinding for drone navigation"""
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # 8-directional movement (drones can move diagonally)
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while not frontier.empty():
        current_priority, current = frontier.get()
        
        if current == goal:
            break
            
        for dy, dx in directions:
            next_pos = (current[0]+dx, current[1]+dy)
            
            if not grid.is_accessible(next_pos[0], next_pos[1]):
                continue
                
            # Diagonal movement costs more
            move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            new_cost = cost_so_far[current] + move_cost
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(goal, next_pos)
                frontier.put((priority, next_pos))
                came_from[next_pos] = current

    # Reconstruct path
    path = []
    current = goal
    if goal in came_from or goal == start:
        while current and current != start:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
    
    return path

# --- Traditional Surveillance Algorithms ---
def nearest_patrol_point(grid, drone_pos):
    """Navigate to the nearest patrol point for systematic surveillance"""
    patrol_points = detect_patrol_points(grid, drone_pos)
    
    if not patrol_points:
        return []
    
    # Find closest patrol point
    closest_point = min(patrol_points, key=lambda p: heuristic(drone_pos, p))
    
    # Navigate to closest patrol point
    path = a_star_search(grid, drone_pos, closest_point)
    return path

def risk_based_surveillance(grid, drone_pos):
    """Prioritize high-risk areas for surveillance"""
    patrol_points = detect_patrol_points(grid, drone_pos)
    candidates = []
    
    if not patrol_points:
        return []
    
    # Evaluate risk level for each patrol point
    for point in patrol_points:
        x, y = point
        distance = heuristic(drone_pos, point)
        
        # Calculate surveillance value
        unsurveilled_nearby = grid.count_unsurveilled_neighbors(x, y)
        risk_factor = 2.0 if grid.is_high_risk_area(x, y) else 1.0
        
        # Risk-based score: (surveillance_value * risk_factor) / distance
        score = (unsurveilled_nearby * risk_factor) / max(1, distance**0.5) # De-emphasize distance slightly
        candidates.append((score, point))
    
    if candidates:
        # Sort by score (highest first)
        candidates.sort(reverse=True)
        best_point = candidates[0][1]
        path = a_star_search(grid, drone_pos, best_point)
        return path
    
    return []

def perimeter_patrol(grid, drone_pos):
    """Follow perimeter and building edges for comprehensive surveillance"""
    perimeter_cells = []
    
    # Find perimeter cells (adjacent to obstacles or boundaries)
    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.grid[y][x]['obstacle']:
                # Check if adjacent to obstacle or boundary
                is_perimeter = False
                for dy, dx in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = x+dx, y+dy
                    if not (0 <= nx < grid.width and 0 <= ny < grid.height) or grid.grid[ny][nx]['obstacle']:
                        is_perimeter = True
                        break
                
                if is_perimeter:
                    unsurveilled_nearby = grid.count_unsurveilled_neighbors(x, y, radius=FOV_RADIUS)
                    if unsurveilled_nearby > 0:
                        perimeter_cells.append((x, y))
    
    if not perimeter_cells:
        # Fall back to nearest patrol point
        return nearest_patrol_point(grid, drone_pos)
    
    # Find optimal perimeter point
    candidates = []
    for perimeter_cell in perimeter_cells:
        distance = heuristic(drone_pos, perimeter_cell)
        unsurveilled_nearby = grid.count_unsurveilled_neighbors(perimeter_cell[0], perimeter_cell[1], radius=2)
        score = unsurveilled_nearby / max(1, distance)
        candidates.append((score, perimeter_cell))
    
    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))  # Sort by score (desc), then position
        target_perimeter = candidates[0][1]
        path = a_star_search(grid, drone_pos, target_perimeter)
        return path
    
    return []

# --- Helper Classes & Functions ---
class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        
    def draw(self, screen, font):
        color = COLORS['button_hover'] if self.hovered else COLORS['button']
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (30, 30, 30), self.rect, width=2, border_radius=5)
        
        text_surf = font.render(self.text, True, COLORS['button_text'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
        
    def is_clicked(self, pos, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                if self.action:
                    self.action()
                return True
        return False

class SurveillanceMetrics:
    def __init__(self, grid):
        self.grid = grid
        self.start_time = pygame.time.get_ticks()
        self.reset()
        
    def reset(self):
        self.flight_time = 0
        self.distance_traveled = 0
        self.threats_detected = 0
        self.last_pos = None
        
    def update(self, new_pos, threats_found):
        if self.last_pos:
            self.distance_traveled += heuristic(self.last_pos, new_pos)
        self.last_pos = new_pos
        self.threats_detected += len(threats_found)
        
    def get_metrics(self):
        coverage = self.grid.get_surveillance_coverage()
        threat_detection = self.grid.get_threat_detection_rate()
        flight_time = (pygame.time.get_ticks() - self.start_time) / 1000
        
        return {
            'coverage': coverage,
            'threat_detection': threat_detection,
            'flight_time': flight_time,
            'distance': self.distance_traveled,
            'threats_found': len(self.grid.detected_threats),
            'total_threats': self.grid.total_threats,
            'unsurveilled': self.grid.count_unsurveilled(),
        }

def visualize_surveillance(screen, grid, drone_pos, path, patrol_points, 
                           algorithm_name, metrics, threats_found_this_step, stuck=False, 
                           next_algo_button=None, q_agent=None):
    screen.fill(COLORS['unexplored'])
    
    # Draw Q-values visualization for RL algorithms
    if q_agent:
        max_q_val = 1
        q_values_to_render = {}
        # Pre-calculate max Q value for normalization
        for y in range(grid.height):
            for x in range(grid.width):
                if not grid.grid[y][x]['obstacle']:
                    state = q_agent.get_compact_state((x, y), grid)
                    if algorithm_name == "DQN":
                        state_array = np.array(state).reshape(1, -1)
                        q_vals = q_agent.main_network.forward(state_array)[0]
                    elif algorithm_name == "Double Q-Learning":
                        q_vals_a = q_agent.q_table_a.get(state, np.zeros(len(q_agent.actions)))
                        q_vals_b = q_agent.q_table_b.get(state, np.zeros(len(q_agent.actions)))
                        q_vals = (q_vals_a + q_vals_b) / 2
                    else: # SARSA or Optimized Q
                        q_vals = q_agent.q_table.get(state, np.zeros(len(q_agent.actions)))
                    
                    current_max = np.max(q_vals)
                    if current_max > max_q_val:
                        max_q_val = current_max
                    q_values_to_render[(x,y)] = current_max

        # Render Q-values
        for (x, y), max_q in q_values_to_render.items():
            if max_q > 0:
                intensity = min(255, int((max_q / max_q_val) * 255))
                # BUG FIX: Use algorithm_name (which is in scope) instead of algorithm_type (which is not)
                if algorithm_name == "DQN":
                    color = (intensity, intensity, 0, 100)
                elif algorithm_name == "Double Q-Learning":
                    color = (0, intensity, intensity, 100)
                elif algorithm_name == "SARSA":
                    color = (intensity, intensity // 2, 0, 100)
                else: # Optimized Q-Learning
                    color = (intensity // 2, 0, intensity, 100)
                
                s = pygame.Surface((CELL_SIZE-2, CELL_SIZE-2), pygame.SRCALPHA)
                s.fill(color)
                screen.blit(s, (x*CELL_SIZE+1, y*CELL_SIZE+1))

    # Draw grid cells
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x]['obstacle']:
                color = COLORS['obstacle']
            elif grid.grid[y][x]['surveilled']:
                color = COLORS['explored']
            else:
                color = COLORS['unexplored']
                
            pygame.draw.rect(screen, color, 
                             (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, COLORS['grid'], 
                             (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # Draw hidden threats
    for x, y in grid.threat_cells:
        if not grid.grid[y][x]['threat_detected']:
            pygame.draw.circle(screen, COLORS['threat_hidden'],
                               (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//4)

    # Draw detected threats
    for x, y in grid.detected_threats:
        pygame.draw.circle(screen, COLORS['threat_detected'],
                           (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)
        pygame.draw.circle(screen, COLORS['alert'],
                           (x*CELL_SIZE + CELL_SIZE//2, y*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3 + 2, 2)
    
    # Draw patrol points
    for x, y in patrol_points:
        pygame.draw.rect(screen, COLORS['frontier'],
                         (x*CELL_SIZE+6, y*CELL_SIZE+6, CELL_SIZE-12, CELL_SIZE-12), 2)
    
    # Draw surveillance FOV
    fov_surface = pygame.Surface((CELL_SIZE*(2*FOV_RADIUS+1), 
                                  CELL_SIZE*(2*FOV_RADIUS+1)), pygame.SRCALPHA)
    pygame.draw.circle(fov_surface, COLORS['fov'], 
                       (fov_surface.get_width()//2, fov_surface.get_height()//2), FOV_RADIUS*CELL_SIZE)
    screen.blit(fov_surface, (drone_pos[0]*CELL_SIZE - FOV_RADIUS*CELL_SIZE,
                              drone_pos[1]*CELL_SIZE - FOV_RADIUS*CELL_SIZE))
    
    # Draw flight path
    if path:
        path_color_map = {
            "Risk-Based Surveillance": COLORS['alert'],
            "Perimeter Patrol": COLORS['patrol_route'],
            "Optimized Q-Learning": COLORS['q_learning'],
            "SARSA": COLORS['sarsa'],
            "Double Q-Learning": COLORS['double_q'],
            "DQN": COLORS['dqn']
        }
        path_color = path_color_map.get(algorithm_name, COLORS['path'])
        
        points = [(drone_pos[0]*CELL_SIZE + CELL_SIZE//2, drone_pos[1]*CELL_SIZE + CELL_SIZE//2)]
        points.extend([(p[0]*CELL_SIZE+CELL_SIZE//2, p[1]*CELL_SIZE+CELL_SIZE//2) for p in path])
        if len(points) > 1:
            pygame.draw.lines(screen, path_color, False, points, 2)
    
    # Draw drone
    pygame.draw.circle(screen, COLORS['drone'],
                       (drone_pos[0]*CELL_SIZE + CELL_SIZE//2,
                        drone_pos[1]*CELL_SIZE + CELL_SIZE//2), CELL_SIZE//3)
    
    # Display info panel
    info_panel = pygame.Surface((screen.get_width(), 60))
    info_panel.fill((20, 20, 40))
    screen.blit(info_panel, (0, GRID_SIZE*CELL_SIZE))

    # Display surveillance metrics
    font = pygame.font.SysFont('Arial', 14)
    bold_font = pygame.font.SysFont('Arial', 14, bold=True)
    
    metrics_text = [
        f"ALGORITHM: {algorithm_name}",
        f"Coverage: {metrics['coverage']*100:.1f}%",
        f"Threats Found: {metrics['threats_found']}/{metrics['total_threats']}",
        f"Time: {metrics['flight_time']:.1f}s",
        f"Uninspected: {metrics['unsurveilled']}"
    ]
    
    for i, text in enumerate(metrics_text):
        surf = bold_font.render(text, True, (255, 255, 0)) if i == 0 else font.render(text, True, (255, 255, 255))
        screen.blit(surf, (10, GRID_SIZE*CELL_SIZE + 5 + i*11))

    if q_agent:
        q_stats = q_agent.get_learning_stats()
        rl_metrics = [
            f"Episode: {q_stats['episode']}",
            f"Epsilon: {q_stats['epsilon']:.3f}",
            f"Avg Reward: {q_stats['avg_reward']:.1f}",
            f"Stuck: {q_stats['stuck_count']}"
        ]
        if 'replay_size' in q_stats:
             rl_metrics.append(f"Replay Buf: {q_stats['replay_size']}")
        
        for i, text in enumerate(rl_metrics):
             surf = font.render(text, True, (255, 0, 255))
             screen.blit(surf, (220, GRID_SIZE*CELL_SIZE + 5 + i*11))

    # Show recent threat detections
    if threats_found_this_step:
        alert_text = bold_font.render("THREAT DETECTED!", True, (255, 50, 50))
        screen.blit(alert_text, (screen.get_width() - 250, GRID_SIZE*CELL_SIZE + 40))

    # Draw button
    if next_algo_button:
        next_algo_button.draw(screen, font)
    
    pygame.display.flip()

# Surveillance environment layout
SURVEILLANCE_ENVIRONMENT = [
    "00000000000000000000",
    "01111111101111111110",
    "01000000100000000010",
    "01001110100111001010",
    "01001000100001001010",
    "01001111101111001010",
    "01000000000000001010",
    "01111110011111111010",
    "01000000000000000010",
    "01011111111111111010",
    "01010000000000001010",
    "01010111111110101010",
    "01010100000010101010",
    "01010111111110101010",
    "01010000000000001010",
    "01011111111111111010",
    "01000000000000000010",
    "01111111111111111110",
    "00000000000000000000",
    "00000000000000000000"
]

def run_surveillance_demo():
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE+60))
    pygame.display.set_caption("Optimized RL Surveillance System - Multi-Algorithm Demo")
    clock = pygame.time.Clock()
    
    # Create control button
    next_algo_button = Button(
        GRID_SIZE*CELL_SIZE - 150, GRID_SIZE*CELL_SIZE + 20, 
        140, 30, "Next Algorithm"
    )
    
    # Updated surveillance algorithms
    algorithms = [
        ("Nearest Patrol Point", nearest_patrol_point, None),
        ("Risk-Based Surveillance", risk_based_surveillance, None),  
        ("Perimeter Patrol", perimeter_patrol, None),
        ("Optimized Q-Learning", None, "optimized_q"),
        ("Double Q-Learning", None, "double_q"),
        ("SARSA", None, "sarsa"),
        ("DQN", None, "dqn")
    ]
    
    algo_idx = 0
    while True: # Loop forever through algorithms
        algo_name, algo_func, algorithm_type = algorithms[algo_idx]
        
        # Initialize surveillance grid
        grid = SurveillanceGrid(GRID_SIZE, GRID_SIZE)
        grid.set_grid(SURVEILLANCE_ENVIRONMENT)
        grid.place_threats(num_threats=8)
        
        metrics = SurveillanceMetrics(grid)
        
        # Initialize RL agent if needed
        q_agent = None
        if algorithm_type:
            q_agent = OptimizedQLearningAgent(GRID_SIZE, GRID_SIZE, algorithm_type)
            q_agent.new_episode()
        
        # Start drone at entry point
        drone_pos = (1, 1)
        threats_found_this_step = grid.surveillance_scan(*drone_pos)
        path = []
        
        # Navigation variables
        force_next_algorithm = False
        
        # BUG FIX: Corrected variable name from 'threats' to 'grid.total_threats'
        print(f"\n--- Starting {algo_name} surveillance mission ---")
        if q_agent:
            q_stats = q_agent.get_learning_stats()
            print(f"RL Config: {algorithm_type}, Episode: {q_stats['episode']}, Epsilon: {q_stats['epsilon']:.3f}")
        print(f"Mission: Detect {grid.total_threats} threats in urban environment")
        
        # Main surveillance loop for the current algorithm
        steps = 0
        max_steps = 1000 # Increased steps for better learning
        
        while not force_next_algorithm and steps < max_steps:
            steps += 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                mouse_pos = pygame.mouse.get_pos()
                next_algo_button.check_hover(mouse_pos)
                if next_algo_button.is_clicked(mouse_pos, event):
                    force_next_algorithm = True

            # --- MAJOR BUG FIX: Implemented the learning loop for RL agents ---
            if q_agent:
                # 1. Get current state and other pre-action data
                prev_state = q_agent.get_compact_state(drone_pos, grid)
                prev_coverage = grid.get_surveillance_coverage()

                # 2. Agent chooses an action
                action = q_agent.choose_action(prev_state, grid, drone_pos)
                
                # 3. Determine next position and update drone
                next_pos = q_agent.get_next_position(drone_pos, action)
                moved = (next_pos != drone_pos)
                drone_pos = next_pos
                path = [] # For RL, path is just the next immediate step
                
                # 4. Perform scan, get feedback from environment
                threats_found_this_step = grid.surveillance_scan(*drone_pos)
                
                # 5. Calculate reward
                reward = q_agent.calculate_reward(drone_pos, grid, threats_found_this_step, prev_coverage, action)

                # 6. Get the new state
                next_state = q_agent.get_compact_state(drone_pos, grid)

                # 7. Update the agent (THE ACTUAL LEARNING STEP)
                q_agent.update_q_table(prev_state, action, reward, next_state)

                # 8. Update helper variables
                q_agent.visited_states[drone_pos] += 1
                q_agent.update_stuck_counter(moved)
            else:
                # For traditional algorithms, plan a path
                if not path:
                    path = algo_func(grid, drone_pos)

                if path:
                    drone_pos = path.pop(0)
                
                threats_found_this_step = grid.surveillance_scan(*drone_pos)

            metrics.update(drone_pos, threats_found_this_step)
            
            # Check for mission completion
            if grid.is_surveillance_complete() and len(grid.detected_threats) == grid.total_threats:
                print(f"Mission complete for {algo_name}!")
                force_next_algorithm = True

            # Visualization
            patrol_points = detect_patrol_points(grid, drone_pos)
            visualize_surveillance(screen, grid, drone_pos, path, patrol_points,
                                   algo_name, metrics.get_metrics(), threats_found_this_step,
                                   next_algo_button=next_algo_button, q_agent=q_agent)
            
            clock.tick(20) # Control simulation speed

        # Move to the next algorithm
        algo_idx = (algo_idx + 1) % len(algorithms)

if __name__ == '__main__':
    run_surveillance_demo()
