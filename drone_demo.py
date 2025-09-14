import pygame
import sys
import random
import time
import math
import numpy as np
from queue import PriorityQueue
from collections import deque, defaultdict
import pickle
import os
import copy

# Grid Setup
GRID_SIZE = 20
CELL_SIZE = 30
FOV_RADIUS = 2  # Cells visible in all directions

# Colors
COLORS = {
    'grid': (40, 40, 40),
    'unexplored': (0, 0, 0),
    'explored': (200, 200, 200),
    'obstacle': (100, 100, 100),
    'path': (255, 0, 0),
    'agent': (0, 0, 255),
    'fov': (0, 255, 0, 50),
    'frontier': (100, 255, 100),
    'button': (70, 70, 200),
    'button_hover': (100, 100, 255),
    'button_text': (255, 255, 255),
    'info_gain': (255, 200, 100),
    'wall_follow': (150, 100, 255),
    'q_learning': (255, 100, 100),
    'sarsa': (100, 255, 255),
    'dqn': (255, 150, 255),
    'ucb': (255, 255, 100),
    'thompson': (150, 255, 150),
    'epsilon_greedy': (200, 150, 100),
    # --- ADDED COLORS FOR NEW ALGORITHMS ---
    'random_walk': (200, 200, 0),
    'greedy_bfs': (0, 200, 200),
    'boustrophedon': (255, 0, 255)
}

class ExplorationGrid:
    
    def __init__(self, width, height):
        # CORRECTED LINE: The list comprehension is now properly structured.
        self.grid = [[{'explored': False, 'obstacle': False} for _ in range(width)] for _ in range(height)]
        self.width = width
        self.height = height
        self.total_cells = width * height
        self.obstacle_cells = set()

    def set_grid(self, grid_map):
        for y in range(self.height):
            for x in range(self.width):
                if grid_map[y][x] == '1':
                    self.grid[y][x]['obstacle'] = True
                    self.obstacle_cells.add((x, y))
                else:
                    self.grid[y][x]['obstacle'] = False

    def reveal_area(self, x, y):
        reward = 0
        for dx in range(-FOV_RADIUS, FOV_RADIUS+1):
            for dy in range(-FOV_RADIUS, FOV_RADIUS+1):
                if 0 <= x+dx < self.width and 0 <= y+dy < self.height:
                    if not self.grid[y+dy][x+dx]['explored']:
                        reward += 1
                    self.grid[y+dy][x+dx]['explored'] = True
        return reward

    def is_accessible(self, x, y):
        return (0 <= x < self.width and
                0 <= y < self.height and
                not self.grid[y][x]['obstacle'])

    def get_coverage_percentage(self):
        explored = sum(sum(cell['explored']
                       for cell in row) for row in self.grid)
        total_explorable = self.total_cells - len(self.obstacle_cells)
        return explored / total_explorable if total_explorable > 0 else 0

    def is_fully_explored(self):
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]['obstacle'] and not self.grid[y][x]['explored']:
                    return False
        return True

    def count_unexplored(self):
        unexplored = 0
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]['obstacle'] and not self.grid[y][x]['explored']:
                    unexplored += 1
        return unexplored

    def is_adjacent_to_wall(self, x, y):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x+dx, y+dy
            if (0 <= nx < self.width and 0 <= ny < self.height and
                    self.grid[ny][nx]['obstacle']):
                return True
        return False

    def count_unexplored_neighbors(self, x, y, radius=FOV_RADIUS+1):
        count = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x+dx, y+dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    not self.grid[ny][nx]['obstacle'] and
                        not self.grid[ny][nx]['explored']):
                    count += 1
        return count


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_state_representation(grid, agent_pos):
    x, y = agent_pos
    features = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                if grid.grid[ny][nx]['obstacle']:
                    features.append(1)
                elif grid.grid[ny][nx]['explored']:
                    features.append(0.5)
                else:
                    features.append(0)
            else:
                features.append(1)
    features.append(grid.count_unexplored_neighbors(x, y) / 25.0)
    features.append(grid.get_coverage_percentage())
    features.append(1 if grid.is_adjacent_to_wall(x, y) else 0)
    return np.array(features)


class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = [self.q_table[state_key][action] for action in valid_actions]
        max_q = max(q_values) if q_values else 0
        best_actions = [action for action, q in zip(
            valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, valid_next_actions):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table[state_key][action]
        if valid_next_actions:
            next_q_values = [self.q_table[next_state_key]
                             [a] for a in valid_next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0
        new_q = current_q + self.lr * \
            (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class SARSAAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = [self.q_table[state_key][action] for action in valid_actions]
        max_q = max(q_values) if q_values else 0
        best_actions = [action for action, q in zip(
            valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action]
        new_q = current_q + self.lr * \
            (reward + self.gamma * next_q - current_q)
        self.q_table[state_key][action] = new_q
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class DQNAgent:
    def __init__(self, state_size=28, action_size=8, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        self.q_network = defaultdict(
            lambda: np.random.normal(0, 0.1, action_size))
        self.target_network = defaultdict(
            lambda: np.random.normal(0, 0.1, action_size))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state, valid_actions):
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        state_key = self.get_state_key(state)
        q_values = self.q_network[state_key]
        valid_indices = [self.actions.index(
            action) for action in valid_actions]
        valid_q_values = [q_values[i] for i in valid_indices]
        best_idx = valid_indices[np.argmax(valid_q_values)]
        return self.actions[best_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, min(
            self.batch_size, len(self.memory)))
        for state, action_idx, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            target = reward
            if not done:
                target += 0.95 * np.max(self.target_network[next_state_key])
            target_f = self.q_network[state_key].copy()
            target_f[action_idx] = target
            self.q_network[state_key] += self.learning_rate * \
                (target_f - self.q_network[state_key])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        for key in self.q_network:
            self.target_network[key] = self.q_network[key].copy()


class UCBAgent:
    def __init__(self, c=1.4):
        self.c = c
        self.action_counts = defaultdict(lambda: defaultdict(int))
        self.action_values = defaultdict(lambda: defaultdict(float))
        self.total_counts = defaultdict(int)
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        for action in valid_actions:
            if self.action_counts[state_key][action] == 0:
                return action
        ucb_values = []
        for action in valid_actions:
            avg_reward = self.action_values[state_key][action] / \
                self.action_counts[state_key][action]
            confidence = self.c * math.sqrt(math.log(self.total_counts[state_key]) /
                                            self.action_counts[state_key][action])
            ucb_values.append(avg_reward + confidence)
        best_action_idx = np.argmax(ucb_values)
        return valid_actions[best_action_idx]

    def update(self, state, action, reward):
        state_key = self.get_state_key(state)
        self.action_counts[state_key][action] += 1
        self.action_values[state_key][action] += reward
        self.total_counts[state_key] += 1


class ThompsonSamplingAgent:
    def __init__(self):
        self.alpha = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.beta = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_state_key(self, state):
        return tuple(np.round(state, 2))

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        action_samples = []
        for action in valid_actions:
            sample = np.random.beta(
                self.alpha[state_key][action], self.beta[state_key][action])
            action_samples.append(sample)
        best_action_idx = np.argmax(action_samples)
        return valid_actions[best_action_idx]

    def update(self, state, action, reward):
        state_key = self.get_state_key(state)
        if reward > 0:
            self.alpha[state_key][action] += reward
        else:
            self.beta[state_key][action] += 1


def run_q_learning(grid, agent_pos):
    if not hasattr(run_q_learning, 'agent'):
        run_q_learning.agent = QLearningAgent()
    agent = run_q_learning.agent
    state = get_state_representation(grid, agent_pos)
    valid_actions = [(dx, dy) for dx, dy in agent.actions if grid.is_accessible(
        agent_pos[0] + dx, agent_pos[1] + dy)]
    if not valid_actions:
        return [], set()
    action = agent.choose_action(state, valid_actions)
    next_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])
    temp_grid = copy.deepcopy(grid)
    reward = temp_grid.reveal_area(next_pos[0], next_pos[1])
    reward -= 0.1
    next_state = get_state_representation(temp_grid, next_pos)
    valid_next_actions = [
        (dx, dy) for dx, dy in agent.actions if grid.is_accessible(next_pos[0] + dx, next_pos[1] + dy)]
    agent.update(state, action, reward, next_state, valid_next_actions)
    return [next_pos], {next_pos}


def run_sarsa(grid, agent_pos):
    if not hasattr(run_sarsa, 'agent'):
        run_sarsa.agent = SARSAAgent()
    agent = run_sarsa.agent
    state = get_state_representation(grid, agent_pos)
    valid_actions = [(dx, dy) for dx, dy in agent.actions if grid.is_accessible(
        agent_pos[0] + dx, agent_pos[1] + dy)]
    if not valid_actions:
        return [], set()
    action = agent.choose_action(state, valid_actions)
    next_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])
    temp_grid = copy.deepcopy(grid)
    reward = temp_grid.reveal_area(next_pos[0], next_pos[1])
    reward -= 0.1
    next_state = get_state_representation(temp_grid, next_pos)
    next_valid_actions = [
        (dx, dy) for dx, dy in agent.actions if grid.is_accessible(next_pos[0] + dx, next_pos[1] + dy)]
    next_action = agent.choose_action(
        next_state, next_valid_actions) if next_valid_actions else (0, 0)
    agent.update(state, action, reward, next_state, next_action)
    return [next_pos], {next_pos}


def run_dqn(grid, agent_pos):
    if not hasattr(run_dqn, 'agent'):
        run_dqn.agent = DQNAgent()
    agent = run_dqn.agent
    state = get_state_representation(grid, agent_pos)
    valid_actions = [(dx, dy) for dx, dy in agent.actions if grid.is_accessible(
        agent_pos[0] + dx, agent_pos[1] + dy)]
    if not valid_actions:
        return [], set()
    action = agent.choose_action(state, valid_actions)
    next_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])
    temp_grid = copy.deepcopy(grid)
    reward = temp_grid.reveal_area(next_pos[0], next_pos[1])
    reward -= 0.1
    next_state = get_state_representation(temp_grid, next_pos)
    action_idx = agent.actions.index(action)
    done = temp_grid.is_fully_explored()
    agent.remember(state, action_idx, reward, next_state, done)
    agent.replay()
    return [next_pos], {next_pos}


def run_ucb(grid, agent_pos):
    if not hasattr(run_ucb, 'agent'):
        run_ucb.agent = UCBAgent()
    agent = run_ucb.agent
    state = get_state_representation(grid, agent_pos)
    valid_actions = [(dx, dy) for dx, dy in agent.actions if grid.is_accessible(
        agent_pos[0] + dx, agent_pos[1] + dy)]
    if not valid_actions:
        return [], set()
    action = agent.choose_action(state, valid_actions)
    next_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])
    temp_grid = copy.deepcopy(grid)
    reward = temp_grid.reveal_area(next_pos[0], next_pos[1])
    agent.update(state, action, reward)
    return [next_pos], {next_pos}


def run_thompson_sampling(grid, agent_pos):
    if not hasattr(run_thompson_sampling, 'agent'):
        run_thompson_sampling.agent = ThompsonSamplingAgent()
    agent = run_thompson_sampling.agent
    state = get_state_representation(grid, agent_pos)
    valid_actions = [(dx, dy) for dx, dy in agent.actions if grid.is_accessible(
        agent_pos[0] + dx, agent_pos[1] + dy)]
    if not valid_actions:
        return [], set()
    action = agent.choose_action(state, valid_actions)
    next_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])
    temp_grid = copy.deepcopy(grid)
    reward = temp_grid.reveal_area(next_pos[0], next_pos[1])
    agent.update(state, action, reward)
    return [next_pos], {next_pos}

# --- START OF NEWLY ADDED ALGORITHMS ---

def run_random_walk(grid, agent_pos):
    """Random Walk exploration: move to a random valid neighbor."""
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(actions)
    
    valid_moves = []
    for dx, dy in actions:
        nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
        if grid.is_accessible(nx, ny):
            valid_moves.append((nx, ny))
            
    if not valid_moves:
        return [], set()
        
    next_pos = random.choice(valid_moves)
    return [next_pos], {next_pos}

def greedy_best_first_search(grid, start, goal):
    """Pathfinding that only uses the heuristic (greedy)."""
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    visited = {start}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    while not frontier.empty():
        current = frontier.get()[1]
        
        if current == goal:
            break
            
        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            
            if grid.is_accessible(next_pos[0], next_pos[1]) and next_pos not in visited:
                priority = heuristic(goal, next_pos)
                frontier.put((priority, next_pos))
                came_from[next_pos] = current
                visited.add(next_pos)
                
    path = []
    current = goal
    if goal in came_from or goal == start:
        while current and current != start:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
    return path, visited

def run_greedy_bfs(grid, agent_pos):
    """Greedy Best-First Search: find path to the nearest unexplored cell."""
    target = find_nearest_unexplored(grid, agent_pos, return_path=False)
    if not target:
        return [], set()
        
    path, visited_cells = greedy_best_first_search(grid, agent_pos, target)
    return path, visited_cells

def run_boustrophedon(grid, agent_pos):
    """Boustrophedon Coverage Path (Lawnmower): systematic sweep."""
    if not hasattr(run_boustrophedon, 'state'):
        run_boustrophedon.state = {}

    # Initialize state for the current grid
    if 'target' not in run_boustrophedon.state:
        run_boustrophedon.state = {'y': 1, 'direction': 1, 'target': None}

    state = run_boustrophedon.state
    
    # If agent reached its target or has no target, find a new one
    if agent_pos == state['target'] or state['target'] is None:
        target_found = False
        while state['y'] < grid.height - 1 and not target_found:
            # Determine sweep direction
            if state['direction'] == 1: # Left to right
                x_range = range(1, grid.width - 1)
            else: # Right to left
                x_range = range(grid.width - 2, 0, -1)
                
            # Find the next accessible, unexplored cell in the row
            for x in x_range:
                if grid.is_accessible(x, state['y']) and not grid.grid[state['y']][x]['explored']:
                    state['target'] = (x, state['y'])
                    target_found = True
                    break
            
            # If no target in this row, move to the next and reverse direction
            if not target_found:
                state['y'] += 1
                state['direction'] *= -1
                
        # If no target found after checking all rows, exploration is likely done
        if not target_found:
            return nearest_frontier(grid, agent_pos)

    # Find a path to the current Boustrophedon target
    path, visited = a_star_search(grid, agent_pos, state['target'])
    return path, visited

# --- END OF NEWLY ADDED ALGORITHMS ---

def detect_frontiers(grid, agent_pos):
    frontiers = []
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x]['explored'] and not grid.grid[y][x]['obstacle']:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < grid.width and 0 <= ny < grid.height and
                        not grid.grid[ny][nx]['explored'] and
                            not grid.grid[ny][nx]['obstacle']):
                        frontiers.append((x, y))
                        break
    return frontiers


def a_star_search(grid, start, goal):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while not frontier.empty():
        current = frontier.get()[1]
        if current == goal:
            break
        for dx, dy in directions:
            next_pos = (current[0]+dx, current[1]+dy)
            if not grid.is_accessible(next_pos[0], next_pos[1]):
                continue
            move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            new_cost = cost_so_far[current] + move_cost
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(goal, next_pos)
                frontier.put((priority, next_pos))
                came_from[next_pos] = current
    path = []
    current = goal
    if goal in came_from or goal == start:
        while current and current != start:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
    return path, cost_so_far.keys()


def nearest_frontier(grid, agent_pos):
    frontiers = detect_frontiers(grid, agent_pos)
    visited_cells = set()
    if not frontiers:
        return [], visited_cells
    closest_frontier = min(frontiers, key=lambda f: heuristic(agent_pos, f))
    path, visited = a_star_search(grid, agent_pos, closest_frontier)
    visited_cells.update(visited)
    return path, visited_cells


def information_gain(grid, agent_pos):
    frontiers = detect_frontiers(grid, agent_pos)
    visited_cells = set()
    candidates = []
    if not frontiers:
        return [], visited_cells
    for frontier in frontiers:
        distance = heuristic(agent_pos, frontier)
        gain = grid.count_unexplored_neighbors(frontier[0], frontier[1])
        score = gain / max(1, distance)
        candidates.append((score, frontier))
    candidates.sort(reverse=True)
    if candidates:
        best_frontier = candidates[0][1]
        path, visited = a_star_search(grid, agent_pos, best_frontier)
        visited_cells.update(visited)
        return path, visited_cells
    return [], visited_cells


def wall_following(grid, agent_pos):
    visited_cells = {agent_pos}
    wall_cells = []
    for y in range(grid.height):
        for x in range(grid.width):
            if (not grid.grid[y][x]['obstacle'] and
                grid.grid[y][x]['explored'] and
                    grid.is_adjacent_to_wall(x, y)):
                wall_cells.append((x, y))
    if not wall_cells:
        return nearest_frontier(grid, agent_pos)
    candidates = []
    for wall_cell in wall_cells:
        unexplored_nearby = grid.count_unexplored_neighbors(
            wall_cell[0], wall_cell[1], radius=2)
        if unexplored_nearby > 0:
            distance = heuristic(agent_pos, wall_cell)
            candidates.append((distance, unexplored_nearby, wall_cell))
    if not candidates:
        return nearest_frontier(grid, agent_pos)
    candidates.sort(key=lambda x: (x[0], -x[1]))
    target_wall = candidates[0][2]
    path, visited = a_star_search(grid, agent_pos, target_wall)
    visited_cells.update(visited)
    return path, visited_cells


class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False

    def draw(self, screen, font):
        color = COLORS['button_hover'] if self.hovered else COLORS['button']
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (30, 30, 30),
                         self.rect, width=2, border_radius=5)
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


class ExplorationMetrics:
    def __init__(self, grid):
        self.grid = grid
        self.start_time = pygame.time.get_ticks()
        self.reset()

    def reset(self):
        self.steps = 0
        self.backtracks = 0
        self.path_length = 0
        self.last_positions = deque(maxlen=10)
        self.stucks = 0
        self.exploration_efficiency = []

    def update(self, new_pos):
        self.steps += 1
        self.path_length += 1
        if new_pos in self.last_positions:
            self.backtracks += 1
        self.last_positions.append(new_pos)
        current_coverage = self.grid.get_coverage_percentage()
        self.exploration_efficiency.append(current_coverage)

    def stuck_detected(self):
        self.stucks += 1

    def get_metrics(self):
        explored = sum(sum(cell['explored']
                       for cell in row) for row in self.grid.grid)
        total_explorable = self.grid.total_cells - len(self.grid.obstacle_cells)
        coverage = explored / total_explorable if total_explorable > 0 else 0
        learning_rate = coverage / max(1, self.steps) if self.steps > 0 else 0
        return {
            'coverage': coverage,
            'time': (pygame.time.get_ticks() - self.start_time) / 1000,
            'steps': self.steps,
            'backtracks': self.backtracks,
            'stucks': self.stucks,
            'efficiency': self.path_length / max(1, explored),
            'unexplored': self.grid.count_unexplored(),
            'learning_rate': learning_rate
        }


def find_nearest_unexplored(grid, agent_pos, return_path=True):
    queue = deque([(agent_pos, [])])
    visited = {agent_pos}
    while queue:
        (x, y), path = queue.popleft()
        if not grid.grid[y][x]['explored'] and not grid.grid[y][x]['obstacle']:
            return (x, y) if not return_path else path
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x+dx, y+dy
            if (0 <= nx < grid.width and 0 <= ny < grid.height and
                not grid.grid[ny][nx]['obstacle'] and
                    (nx, ny) not in visited):
                queue.append(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))
    return None


def scan_entire_grid(grid, agent_pos):
    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.grid[y][x]['obstacle'] and not grid.grid[y][x]['explored']:
                path, _ = a_star_search(grid, agent_pos, (x, y))
                if path:
                    return (x, y), path
    return None, []


def handle_stuck_agent(grid, agent_pos):
    temp_radius = FOV_RADIUS + 3
    for dx in range(-temp_radius, temp_radius+1):
        for dy in range(-temp_radius, temp_radius+1):
            nx, ny = agent_pos[0]+dx, agent_pos[1]+dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                grid.grid[ny][nx]['explored'] = True
    target, path = scan_entire_grid(grid, agent_pos)
    if target:
        return target
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(directions)
    for dx, dy in directions:
        nx, ny = agent_pos[0]+dx, agent_pos[1]+dy
        if grid.is_accessible(nx, ny):
            return (nx, ny)
    return None


def visualize(screen, grid, agent_pos, path, visited_cells, frontiers,
              algorithm_name, metrics, stuck=False, next_algo_button=None,
              info_gain_values=None, wall_cells=None):
    screen.fill(COLORS['unexplored'])
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.grid[y][x]['obstacle']:
                color = COLORS['obstacle']
            elif grid.grid[y][x]['explored']:
                color = COLORS['explored']
            else:
                color = COLORS['unexplored']
            pygame.draw.rect(screen, color,
                             (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE-1, CELL_SIZE-1))
    for pos in frontiers:
        pygame.draw.rect(screen, COLORS['frontier'],
                         (pos[0]*CELL_SIZE+5, pos[1]*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10))
    if info_gain_values and algorithm_name == "Information Gain":
        for pos, value in info_gain_values:
            intensity = min(255, int(value * 30))
            color = (255, 255-intensity, 100)
            pygame.draw.rect(screen, color,
                             (pos[0]*CELL_SIZE+8, pos[1]*CELL_SIZE+8, CELL_SIZE-16, CELL_SIZE-16))
    if wall_cells and algorithm_name == "Wall Following":
        for pos in wall_cells:
            pygame.draw.rect(screen, COLORS['wall_follow'],
                             (pos[0]*CELL_SIZE+8, pos[1]*CELL_SIZE+8, CELL_SIZE-16, CELL_SIZE-16))
    for pos in visited_cells:
        if pos != agent_pos:
            alpha_surface = pygame.Surface(
                (CELL_SIZE-16, CELL_SIZE-16), pygame.SRCALPHA)
            alpha_surface.fill((100, 150, 100, 100))
            screen.blit(alpha_surface, (pos[0]*CELL_SIZE+8, pos[1]*CELL_SIZE+8))
    fov_surface = pygame.Surface((CELL_SIZE*(2*FOV_RADIUS+1),
                                  CELL_SIZE*(2*FOV_RADIUS+1)), pygame.SRCALPHA)
    fov_surface.fill(COLORS['fov'])
    screen.blit(fov_surface, (agent_pos[0]*CELL_SIZE - FOV_RADIUS*CELL_SIZE,
                              agent_pos[1]*CELL_SIZE - FOV_RADIUS*CELL_SIZE))
    if path:
        path_color = COLORS.get(algorithm_name.lower().replace(
            ' ', '_'), COLORS['path'])
        for i, pos in enumerate(path[:10]):
            alpha = max(50, 255 - i * 20)
            color_tuple = (*path_color[:3], alpha) if len(path_color) == 4 else (
                path_color[0], path_color[1], path_color[2], alpha)
            pygame.draw.circle(screen, color_tuple,
                               (pos[0]*CELL_SIZE + CELL_SIZE//2,
                                pos[1]*CELL_SIZE + CELL_SIZE//2), 5)
    pygame.draw.circle(screen, COLORS['agent'],
                       (agent_pos[0]*CELL_SIZE + CELL_SIZE//2,
                        agent_pos[1]*CELL_SIZE + CELL_SIZE//2), 10)
    font = pygame.font.SysFont('Arial', 16)
    metrics_text = [
        f"Algorithm: {algorithm_name}",
        f"Coverage: {metrics['coverage']*100:.1f}%",
        f"Time: {metrics['time']:.1f}s",
        f"Steps: {metrics['steps']}",
        f"Efficiency: {metrics['efficiency']:.2f}",
        f"Learning Rate: {metrics['learning_rate']:.4f}",
        f"Cells left: {metrics['unexplored']}"
    ]
    for i, text in enumerate(metrics_text):
        color = (255, 255, 255) if not stuck else (255, 200, 200)
        surf = font.render(text, True, color)
        screen.blit(surf, (10, 10 + i*20))
    if stuck:
        stuck_text = font.render(
            "Agent is stuck! Use 'Next Algorithm' button.", True, (255, 50, 50))
        screen.blit(stuck_text, (10, 160))
    if hasattr(run_q_learning, 'agent') and algorithm_name == "Q-Learning":
        epsilon_text = font.render(
            f"Epsilon: {run_q_learning.agent.epsilon:.3f}", True, (200, 200, 255))
        screen.blit(epsilon_text, (10, 180))
    elif hasattr(run_sarsa, 'agent') and algorithm_name == "SARSA":
        epsilon_text = font.render(
            f"Epsilon: {run_sarsa.agent.epsilon:.3f}", True, (200, 200, 255))
        screen.blit(epsilon_text, (10, 180))
    elif hasattr(run_dqn, 'agent') and algorithm_name == "DQN":
        epsilon_text = font.render(
            f"Epsilon: {run_dqn.agent.epsilon:.3f}", True, (200, 200, 255))
        screen.blit(epsilon_text, (10, 180))
    if next_algo_button:
        next_algo_button.draw(screen, font)
    pygame.display.flip()


PREDEFINED_GRIDS = [
    [
        "00000000000000000000",
        "01111111111111111110",
        "01000000000000000010",
        "01000100100001000010",
        "01000000000000000010",
        "01000000100000100010",
        "01000000000000000010",
        "01001000000010000010",
        "01000000010000000010",
        "01000000000000000010",
        "01001000000000100010",
        "01000000100000000010",
        "01000010000001000010",
        "01000000000000000010",
        "01000100000010000010",
        "01000000010000000010",
        "01000000000000000010",
        "01111111111111111110",
        "00000000000000000000",
        "00000000000000000000"
    ],
    [
        "00000000000000000000",
        "01111111111111111110",
        "01000010000000100010",
        "01011010111110101010",
        "01010000100000101010",
        "01010111111111101010",
        "01010000000000000010",
        "01011111111111111010",
        "01010000000000001010",
        "01010111111111101010",
        "01010100000000001010",
        "01010101111111101010",
        "01010101000000001010",
        "01010101011111111010",
        "01010001000000000010",
        "01011111111111111010",
        "01000000000000000010",
        "01111111111111111110",
        "00000000000000000000",
        "00000000000000000000"
    ],
    [
        "00000000000000000000",
        "01111111111111111110",
        "01000000000000000010",
        "01011111100111111010",
        "01010000000000001010",
        "01010110011001101010",
        "01010100010001001010",
        "01010110011001101010",
        "01010000000000001010",
        "01011111111111111010",
        "01010000000000001010",
        "01010110011001101010",
        "01010100010001001010",
        "01010110011001101010",
        "01010000000000001010",
        "01011111100111111010",
        "01000000000000000010",
        "01111111111111111110",
        "00000000000000000000",
        "00000000000000000000"
    ]
]


def reset_all_agents():
    if hasattr(run_q_learning, 'agent'):
        delattr(run_q_learning, 'agent')
    if hasattr(run_sarsa, 'agent'):
        delattr(run_sarsa, 'agent')
    if hasattr(run_dqn, 'agent'):
        delattr(run_dqn, 'agent')
    if hasattr(run_ucb, 'agent'):
        delattr(run_ucb, 'agent')
    if hasattr(run_thompson_sampling, 'agent'):
        delattr(run_thompson_sampling, 'agent')
    if hasattr(run_boustrophedon, 'state'):
        delattr(run_boustrophedon, 'state')


def run_full_exploration():
    pygame.init()
    screen = pygame.display.set_mode(
        (GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE+100))
    pygame.display.set_caption(
        "Enhanced Exploration: Classical vs ML/RL Algorithms")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    next_algo_button = Button(
        GRID_SIZE*CELL_SIZE - 150, GRID_SIZE*CELL_SIZE + 20, 140, 30, "Next Algorithm")
    reset_button = Button(
        GRID_SIZE*CELL_SIZE - 150, GRID_SIZE*CELL_SIZE + 60, 140, 30, "Reset Agents")
    
    # --- UPDATED ALGORITHM LIST ---
    algorithms = [
        ("Nearest Frontier", nearest_frontier),
        ("Information Gain", information_gain),
        ("Wall Following", wall_following),
        ("Greedy BFS", run_greedy_bfs),
        ("Boustrophedon", run_boustrophedon),
        ("Random Walk", run_random_walk),
        ("Q-Learning", run_q_learning),
        ("SARSA", run_sarsa),
        ("DQN", run_dqn),
        ("UCB", run_ucb),
        ("Thompson Sampling", run_thompson_sampling)
    ]
    
    metrics_data = {grid_idx: {}
                    for grid_idx in range(len(PREDEFINED_GRIDS))}
    for grid_idx, grid_map in enumerate(PREDEFINED_GRIDS):
        print(f"\n=== Testing Grid {grid_idx + 1} ===")
        reset_all_agents()
        for algo_idx, (algo_name, algo_func) in enumerate(algorithms):
            print(f"Running {algo_name}...")
            grid = ExplorationGrid(GRID_SIZE, GRID_SIZE)
            grid.set_grid(grid_map)
            metrics = ExplorationMetrics(grid)
            agent_pos = (1, 1)
            for y in range(1, grid.height-1):
                for x in range(1, grid.width-1):
                    if grid.is_accessible(x, y):
                        agent_pos = (x, y)
                        break
                if agent_pos != (1, 1):
                    break
            grid.reveal_area(*agent_pos)
            visited = {agent_pos}
            path = []
            info_gain_values = []
            wall_cells = []
            stuck_counter = 0
            last_coverage = 0
            stagnation_counter = 0
            force_next_algorithm = False
            running = True
            step_count = 0
            max_steps = 1000
            while running and not force_next_algorithm and step_count < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    mouse_pos = pygame.mouse.get_pos()
                    next_algo_button.check_hover(mouse_pos)
                    reset_button.check_hover(mouse_pos)
                    if next_algo_button.is_clicked(mouse_pos, event):
                        force_next_algorithm = True
                    if reset_button.is_clicked(mouse_pos, event):
                        reset_all_agents()
                frontiers = detect_frontiers(grid, agent_pos)
                current_coverage = grid.get_coverage_percentage()
                info_gain_values = []
                wall_cells = []
                if algo_name == "Information Gain":
                    for frontier in frontiers:
                        gain = grid.count_unexplored_neighbors(
                            frontier[0], frontier[1])
                        info_gain_values.append((frontier, gain))
                elif algo_name == "Wall Following":
                    for y in range(grid.height):
                        for x in range(grid.width):
                            if (not grid.grid[y][x]['obstacle'] and
                                grid.grid[y][x]['explored'] and
                                    grid.is_adjacent_to_wall(x, y)):
                                wall_cells.append((x, y))
                next_pos = agent_pos
                try:
                    path, visited_cells = algo_func(grid, agent_pos)
                    if path:
                        next_pos = path[0]
                    else:
                        target = find_nearest_unexplored(grid, agent_pos)
                        if target:
                            path, visited_cells = a_star_search(
                                grid, agent_pos, target)
                            if path:
                                next_pos = path[0]
                except Exception as e:
                    print(f"Algorithm {algo_name} error: {e}")
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    for dx, dy in directions:
                        nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
                        if grid.is_accessible(nx, ny):
                            next_pos = (nx, ny)
                            break
                if next_pos == agent_pos:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                if abs(current_coverage - last_coverage) < 0.0001:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_coverage = current_coverage
                if stuck_counter > 15 or stagnation_counter > 50:
                    escape_target = handle_stuck_agent(grid, agent_pos)
                    if escape_target:
                        path, visited_cells = a_star_search(
                            grid, agent_pos, escape_target)
                        if path:
                            next_pos = path[0]
                            stuck_counter = 0
                            stagnation_counter = 0
                        else:
                            metrics.stuck_detected()
                    else:
                        metrics.stuck_detected()
                if next_pos != agent_pos:
                    agent_pos = next_pos
                    metrics.update(agent_pos)
                    grid.reveal_area(*agent_pos)
                    visited.add(agent_pos)
                if step_count % 2 == 0 or stuck_counter > 10:
                    visualize(screen, grid, agent_pos, path, visited, frontiers,
                              algo_name, metrics.get_metrics(),
                              stuck=(stuck_counter > 10),
                              next_algo_button=next_algo_button,
                              info_gain_values=info_gain_values if algo_name == "Information Gain" else None,
                              wall_cells=wall_cells if algo_name == "Wall Following" else None)
                if grid.is_fully_explored():
                    print(
                        f"Grid {grid_idx+1} fully explored with {algo_name}!")
                    break
                step_count += 1
                clock.tick(60) # Increased tick rate for faster simulation
            final_metrics = metrics.get_metrics()
            final_metrics['total_steps'] = step_count
            metrics_data[grid_idx][algo_name] = final_metrics
            print(
                f"{algo_name} completed: {final_metrics['coverage']*100:.1f}% coverage in {step_count} steps")
            if algo_name == "Boustrophedon": reset_all_agents() # Reset state for Boustrophedon
            pygame.time.delay(500)
    print("\n" + "="*100)
    print("ENHANCED EXPLORATION ALGORITHM COMPARISON")
    print("="*100)
    print(
        f"{'Grid':<6} {'Algorithm':<18} {'Coverage%':<10} {'Time(s)':<8} {'Steps':<8} {'Efficiency':<10} {'L.Rate':<8}")
    print("-"*100)
    for grid_idx, results in metrics_data.items():
        for algo_name, data in results.items():
            print(f"{grid_idx+1:<6} {algo_name:<18} {data['coverage']*100:<10.1f} "
                  f"{data['time']:<8.1f} {data['total_steps']:<8} {data['efficiency']:<10.2f} "
                  f"{data['learning_rate']:<8.4f}")
    print("\n" + "="*50)
    print("AVERAGE PERFORMANCE ACROSS ALL GRIDS")
    print("="*50)
    algo_averages = {}
    for algo_name, _ in algorithms:
        coverages, efficiencies, learning_rates = [], [], []
        for grid_idx in range(len(PREDEFINED_GRIDS)):
            if algo_name in metrics_data[grid_idx]:
                data = metrics_data[grid_idx][algo_name]
                coverages.append(data['coverage'])
                efficiencies.append(data['efficiency'])
                learning_rates.append(data['learning_rate'])
        if coverages:
            algo_averages[algo_name] = {
                'avg_coverage': np.mean(coverages),
                'avg_efficiency': np.mean(efficiencies),
                'avg_learning_rate': np.mean(learning_rates)
            }
    print(
        f"{'Algorithm':<18} {'Avg Coverage%':<14} {'Avg Efficiency':<14} {'Avg L.Rate':<12}")
    print("-"*70)
    for algo_name, avgs in sorted(algo_averages.items(), key=lambda x: x[1]['avg_coverage'], reverse=True):
        print(f"{algo_name:<18} {avgs['avg_coverage']*100:<14.1f} "
              f"{avgs['avg_efficiency']:<14.2f} {avgs['avg_learning_rate']:<12.4f}")
    screen.fill((0, 0, 0))
    title_font = pygame.font.SysFont('Arial', 20)
    y = 20
    title = title_font.render(
        "Exploration Algorithm Comparison Results", True, (255, 255, 255))
    screen.blit(title, (20, y))
    y += 40
    summary_font = pygame.font.SysFont('Arial', 14)
    for algo_name, avgs in sorted(algo_averages.items(), key=lambda x: x[1]['avg_coverage'], reverse=True):
        color = (100, 255, 100) if avgs['avg_coverage'] > 0.9 else (
            255, 255, 100) if avgs['avg_coverage'] > 0.7 else (255, 200, 200)
        text = f"{algo_name}: {avgs['avg_coverage']*100:.1f}% avg coverage"
        surf = summary_font.render(text, True, color)
        screen.blit(surf, (20, y))
        y += 25
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


if __name__ == "__main__":
    run_full_exploration()
