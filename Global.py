import numpy as np
import random
import pickle
import gym
from gym import spaces
import math
from collections import defaultdict
import gc


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)



def rot90(pattern, size):
    M = 0.5 * (size - 1)
    new_pattern = []
    for (x, y) in pattern:
        new_x = M - (y - M)
        new_y = M + (x - M)
        new_pattern.append((int(new_x), int(new_y)))
    return new_pattern


def rot180(pattern, size):
    M = 0.5 * (size - 1)
    new_pattern = []
    for (x, y) in pattern:
        new_x = M - (x - M)
        new_y = M - (y - M)
        new_pattern.append((int(new_x), int(new_y)))
    return new_pattern


def rot270(pattern, size):
    M = 0.5 * (size - 1)
    new_pattern = []
    for (x, y) in pattern:
        new_x = M + (y - M)
        new_y = M - (x - M)
        new_pattern.append((int(new_x), int(new_y)))
    return new_pattern


def reflection(pattern, size):
    new_pattern = []
    for (x,y) in pattern:
        new_pattern.append((size - x - 1, y))
    return new_pattern


class NTupleApproximator:
    def __init__(self, board_size, patterns, weight=None):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns] if weight is None else weight
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.symmetry_patterns = [self.generate_symmetries(pattern) for pattern in self.patterns]

    def generate_symmetries(self, pattern):
        sym = []
        sym.append(pattern)
        sym.append(rot90(pattern, self.board_size))
        sym.append(rot180(pattern, self.board_size))
        sym.append(rot270(pattern, self.board_size))
        sym.append(reflection(pattern, self.board_size))
        sym.append(reflection(rot90(pattern, self.board_size), self.board_size))
        sym.append(reflection(rot180(pattern, self.board_size), self.board_size))
        sym.append(reflection(rot270(pattern, self.board_size), self.board_size))
        return sym

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        return 0 if tile == 0 else int(math.log(tile, 2))

    def get_feature(self, board, coords): # indeX of the weight table
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board): # function f in the paper
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total_value = 0
        for pattern, weight_table in zip(self.patterns, self.weights):
            sym = self.generate_symmetries(pattern)
            total_value += sum(weight_table[self.get_feature(board, pat)] for pat in sym)
        return total_value / ( len(self.patterns) * len(sym) )

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for pattern, weight_table in zip(self.patterns, self.weights):
            sym = self.generate_symmetries(pattern)
            for pat in sym:
                feature = self.get_feature(board, pat)
                weight_table[feature] += alpha * delta


class TD_MCTS_Node:
    def __init__(self, state, score, parent=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.accumulated_score = score
        # List of untried actions based on the current state's legal moves
        self.untried_actions = board_legal_moves(state)

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0    


class TD_MCTS_After_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = set()
        self.visits = 0
        self.accumulated_score = score


class TD_MCTS:
    def __init__(self, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99, normalization_factor=50000):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.normalization_factor = normalization_factor

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        env = Game2048Env()
        env.board = state.copy()
        env.score = score
        return env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if isinstance(node, TD_MCTS_After_Node):
            print("Error: node is not a TD_MCTS_Node")
            return None, None
        best_action = None
        best_child = None
        best_value = - np.inf
        if node.children == {}:
            print("Error: node has no children")
        for action, child in node.children.items():
            if child.visits == 0:
                return action, child
            child.score = child.accumulated_score / child.visits
            uct_value = child.score + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_action = action
                best_child = child
                best_value = uct_value
        return best_action, best_child

    def rollout(self, board, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        env = self.create_env_from_state(board, 0)
        if env.is_game_over():
            return 0
        state = env.board.copy()
        incremental_reward = 0
        for _ in range(depth):
            legal_moves = [action for action in range(4) if env.is_move_legal(action)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            state, reward, _, _ = env.step(action)
        incremental_reward = env.score
        best_reward = - np.inf
        best_reward = 0
        for action in board_legal_moves(state):
            after_state, reward = deterministic_step(state, action)
            value = reward + self.approximator.value(after_state) * self.gamma
            if value > best_reward:
                best_reward = value
        return ( best_reward + incremental_reward ) / self.normalization_factor

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.accumulated_score += reward
            node = node.parent

    def expand(self, node):
        state = node.state.copy()
        action = node.untried_actions.pop()
        after_state, reward = deterministic_step(state, action)
        after_node = TD_MCTS_After_Node(after_state, ( node.score + reward ) / self.normalization_factor, parent=node, action=action)
        node.children[action] = after_node
        for _ in range(4):
            next_state = board_add_random_tile(after_state.copy())
            next_node = TD_MCTS_Node(next_state, ( node.score + reward ) / self.normalization_factor, parent=after_node)
            after_node.children.add(next_node)
            rollout_reward = self.rollout(next_state, self.rollout_depth)
            self.backpropagate(next_node, rollout_reward)

    def run_simulation(self, root):
        node = root

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children != {}:
            action, after_node = self.select_child(node)
            next_node = np.random.choice(list(after_node.children))
            node = next_node
        state = node.state.copy()

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if not node.fully_expanded():
            self.expand(node)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


def compress_line(line):
    """
    Compresses a single row or column (represented as a list) by sliding all nonzero elements
    to the front and merging equal adjacent numbers. Returns the new line and the instant reward
    gained from merging tiles.
    """
    # Remove zeros
    new_line = [x for x in line if x != 0]
    merged_line = []
    instant_reward = 0
    i = 0
    while i < len(new_line):
        # If the next tile exists and is equal, merge them.
        if i + 1 < len(new_line) and new_line[i] == new_line[i+1]:
            merged_value = new_line[i] * 2
            merged_line.append(merged_value)
            instant_reward += merged_value
            i += 2  # Skip the next element since it was merged.
        else:
            merged_line.append(new_line[i])
            i += 1
    # Pad with zeros to maintain the original line length.
    merged_line += [0] * (len(line) - len(merged_line))
    return merged_line, instant_reward


def board_move_left(board):
    """
    Computes the deterministic result of a left move on the board.
    
    Args:
        board (np.array): A 2D numpy array representing the board.
        
    Returns:
        new_board (np.array): The board after the left move.
        instant_reward (int): The reward (score) obtained by merging tiles during the move.
    """
    new_board = board.copy()
    total_reward = 0
    for i in range(new_board.shape[0]):
        row = list(new_board[i, :])
        new_row, reward = compress_line(row)
        total_reward += reward
        new_board[i, :] = new_row
    return new_board, total_reward


def board_move_right(board):
    """
    Computes the deterministic result of a right move on the board.
    
    Args:
        board (np.array): A 2D numpy array representing the board.
        
    Returns:
        new_board (np.array): The board after the right move.
        instant_reward (int): The reward (score) obtained by merging tiles during the move.
    """
    new_board = board.copy()
    total_reward = 0
    for i in range(new_board.shape[0]):
        # Reverse the row to apply left-move logic.
        row = list(new_board[i, :])[::-1]
        new_row, reward = compress_line(row)
        total_reward += reward
        # Reverse back the result.
        new_board[i, :] = new_row[::-1]
    return new_board, total_reward


def board_move_up(board):
    """
    Computes the deterministic result of an upward move on the board.
    
    Args:
        board (np.array): A 2D numpy array representing the board.
        
    Returns:
        new_board (np.array): The board after the upward move.
        instant_reward (int): The reward (score) obtained by merging tiles during the move.
    """
    new_board = board.copy()
    total_reward = 0
    # Process each column.
    for j in range(new_board.shape[1]):
        col = list(new_board[:, j])
        new_col, reward = compress_line(col)
        total_reward += reward
        new_board[:, j] = new_col
    return new_board, total_reward


def board_move_down(board):
    """
    Computes the deterministic result of a downward move on the board.
    
    Args:
        board (np.array): A 2D numpy array representing the board.
        
    Returns:
        new_board (np.array): The board after the downward move.
        instant_reward (int): The reward (score) obtained by merging tiles during the move.
    """
    new_board = board.copy()
    total_reward = 0
    # Process each column in reverse order.
    for j in range(new_board.shape[1]):
        col = list(new_board[:, j])[::-1]
        new_col, reward = compress_line(col)
        total_reward += reward
        new_board[:, j] = new_col[::-1]
    return new_board, total_reward


def deterministic_step(board, action):
    
    if action == 0:
        moved_board, reward = board_move_up(board)
    elif action == 1:
        moved_board, reward = board_move_down(board)
    elif action == 2:
        moved_board, reward = board_move_left(board)
    elif action == 3:
        moved_board, reward = board_move_right(board)
    else:
        print("Invalid action")

    return moved_board, reward


def board_add_random_tile(board):
    empty_cells = np.argwhere(board == 0)
    if empty_cells.size > 0:
        x, y = random.choice(empty_cells)
        board[x, y] = 2 if random.random() < 0.9 else 4
    return board


def board_legal_moves(board):
    # return a list of legal moves
    moves = []
    for action in range(4):
        temp_board = board.copy()
        new_board, _ = deterministic_step(temp_board, action)
        if not np.array_equal(temp_board, new_board):
            moves.append(action)
    return moves




weights = None
with open("ntuple_weights30000.pkl", "rb") as f:
    weights = pickle.load(f)

patterns = [[(0,0), (0, 1), (0, 2), (1, 0), (1, 1)], [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)], [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)]]

approximator = NTupleApproximator(board_size=4, patterns=patterns, weight=weights)

td_mcts = TD_MCTS(approximator, iterations=50, exploration_constant=0.1, rollout_depth=0, gamma=1)

