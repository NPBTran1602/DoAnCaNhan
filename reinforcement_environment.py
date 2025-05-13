import random
import numpy as np
from algorithms import find_blank, swap, step, apply_move, convert_states_to_moves
import time

def q_learning(start, goal, episodes=5000, alpha=0.1, gamma=0.95, epsilon=0.2):
    start_time = time.time()
    nodes_expanded = 0
    q_table = {}  # {state: {action: q_value}}

    def state_to_key(state):
        return tuple(tuple(row) for row in state)

    def get_valid_actions(state):
        i, j = find_blank(state)
        actions = []
        for move, (di, dj) in step.items():
            ni, nj = i + di, j + dj
            if 0 <= ni < 3 and 0 <= nj < 3:
                actions.append(move)
        return actions

    def get_reward(state, goal):
        if state == goal:
            return 100
        h = sum(abs((val-1)%3 - j) + abs((val-1)//3 - i)
                for i, row in enumerate(state)
                for j, val in enumerate(row) if val != 0)
        return -h - 1

    def choose_action(state, valid_actions):
        nonlocal nodes_expanded  # Khai báo nonlocal để sử dụng biến từ q_learning
        nodes_expanded += 1  # Tăng nodes_expanded khi chọn hành động
        if random.random() < epsilon:
            return random.choice(valid_actions)
        state_key = state_to_key(state)
        if state_key not in q_table:
            q_table[state_key] = {a: 0 for a in valid_actions}
        q_values = q_table[state_key]
        return max(valid_actions, key=lambda a: q_values.get(a, 0))

    # Training
    for _ in range(episodes):
        current_state = start
        while current_state != goal:
            valid_actions = get_valid_actions(current_state)
            action = choose_action(current_state, valid_actions)
            next_state = apply_move(current_state, action)
            reward = get_reward(next_state, goal)

            current_key = state_to_key(current_state)
            next_key = state_to_key(next_state)
            if current_key not in q_table:
                q_table[current_key] = {a: 0 for a in valid_actions}
            if next_key not in q_table:
                next_valid = get_valid_actions(next_state)
                q_table[next_key] = {a: 0 for a in next_valid}

            # Q-value update
            future_q = max(q_table[next_key].values(), default=0)
            q_table[current_key][action] += alpha * (reward + gamma * future_q - q_table[current_key][action])
            current_state = next_state

    # Policy extraction
    path = []
    current_state = start
    seen = {current_state}
    while current_state != goal and len(path) < 50:  # Prevent infinite loops
        valid_actions = get_valid_actions(current_state)
        state_key = state_to_key(current_state)
        if state_key not in q_table:
            return None, time.time() - start_time, nodes_expanded
        action = max(valid_actions, key=lambda a: q_table[state_key].get(a, 0))
        next_state = apply_move(current_state, action)
        if next_state == current_state or next_state in seen:
            return None, time.time() - start_time, nodes_expanded
        path.append(action)
        current_state = next_state
        seen.add(current_state)

    return path if current_state == goal else None, time.time() - start_time, nodes_expanded