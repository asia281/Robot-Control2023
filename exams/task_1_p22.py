import numpy as np

def initialize_belief(grid_size):
    return np.ones((grid_size, grid_size)) / (grid_size * grid_size)

def move(belief, action):
    rows, cols = belief.shape
    new_belief = np.zeros_like(belief)

    for row in range(rows):
        for col in range(cols):
            if action == 'u':
                new_row = max(row - 1, 0)
                new_belief[new_row, col] += 0.8 * belief[row, col]
                new_belief[row, col] += 0.2 * belief[row, col]
            elif action == 'd':
                new_row = min(row + 1, rows - 1)
                new_belief[new_row, col] += 0.8 * belief[row, col]
                new_belief[row, col] += 0.2 * belief[row, col]
            elif action == 'r':
                new_col = min(col + 1, cols - 1)
                new_belief[row, new_col] += 0.8 * belief[row, col]
                new_belief[row, col] += 0.2 * belief[row, col]
            elif action == 'l':
                new_col = max(col - 1, 0)
                new_belief[row, new_col] += 0.8 * belief[row, col]
                new_belief[row, col] += 0.2 * belief[row, col]
    
    return new_belief / new_belief.sum()

def sense(belief, color_observation):
    likelihood = (color_observation == 'R') * 0.5 + (color_observation == 'G') * 0.5
    return belief * likelihood / (belief * likelihood).sum()

def bayes_filter(initial_belief, actions, observations):
    belief = initial_belief

    for i in range(len(actions)):
        belief = move(belief, actions[i])
        belief = sense(belief, observations[i])

    return belief

def print_belief(belief):
    rows, cols = belief.shape
    for row in range(rows):
        for col in range(cols):
            print(f"{belief[row, col]:.4f}", end="\t")
        print()

# Example
grid_size = 10
initial_belief = initialize_belief(grid_size)

actions = ['r', 'r', 'r', 'u', 'u']
observations = ['R', 'G', 'R', 'R', 'G']

final_belief = bayes_filter(initial_belief, actions, observations)

print("Initial Belief:")
print_belief(initial_belief)

print("\nFinal Belief:")
print_belief(final_belief)
