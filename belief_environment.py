import heapq
import random
import time
from algorithms import is_solvable, find_blank, swap, calc_heuristic, step, convert_states_to_moves

def generate_random_states(goal_state, num_states=3):
    def generate_single_state():
        tiles = list(range(9))
        random.shuffle(tiles)
        state = tuple(tuple(tiles[i*3:(i+1)*3]) for i in range(3))
        return state

    states = []
    while len(states) < num_states:
        state = generate_single_state()
        if is_solvable(state, goal_state):
            states.append(state)
    return states

def move_blank(s, action):
    i0, j0 = find_blank(s)
    di, dj = step[action]
    i1, j1 = i0 + di, j0 + dj
    if 0 <= i1 < 3 and 0 <= j1 < 3:
        return swap(s, i0, j0, i1, j1)
    return s

def belief_state(initial_beliefs, goal_state):
    start_time = time.time()
    nodes_expanded = 0
    solvable_beliefs = [s for s in initial_beliefs if is_solvable(s, goal_state)]
    if not solvable_beliefs:
        return None, time.time() - start_time, nodes_expanded

    def h_belief(belief):
        return min(calc_heuristic(st, goal_state) for st in belief)

    best_solution = None
    best_steps = float('inf')

    for start_state in solvable_beliefs:
        start_belief = frozenset([start_state])
        start_rep = start_state
        g0 = 0
        h0 = h_belief(start_belief)
        open_heap = [(g0 + h0, g0, start_belief, start_rep, [start_belief], [start_rep])]
        best_g = {start_belief: 0}

        while open_heap:
            f, g, belief, rep, pathB, pathS = heapq.heappop(open_heap)
            nodes_expanded += 1
            if goal_state in belief:
                steps = len(pathS)
                if steps < best_steps:
                    best_solution = pathS + [goal_state]
                    best_steps = steps
                break
            for act in step:
                nb = frozenset(move_blank(s, act) for s in belief)
                nr = move_blank(rep, act)
                ng = g + 1
                if nb not in best_g or ng < best_g[nb]:
                    best_g[nb] = ng
                    heapq.heappush(open_heap, (ng + h_belief(nb), ng, nb, nr, pathB + [nb], pathS + [nr]))

    return (convert_states_to_moves(best_solution), time.time() - start_time, nodes_expanded) if best_solution else (None, time.time() - start_time, nodes_expanded)

def and_or_search(initial_beliefs, goal_state):
    start_time = time.time()
    nodes_expanded = 0
    solvable_beliefs = [s for s in initial_beliefs if is_solvable(s, goal_state)]
    if not solvable_beliefs:
        return None, time.time() - start_time, nodes_expanded

    best_solution = None
    best_steps = float('inf')

    for start_state in solvable_beliefs:
        visited = set()
        def or_search(belief, path, rep_state):
            nonlocal nodes_expanded
            if goal_state in belief:
                return [rep_state], len(path)
            belief_tuple = frozenset(belief)
            if belief_tuple in visited:
                return None, 0
            visited.add(belief_tuple)
            nodes_expanded += 1

            for move, (di, dj) in step.items():
                new_belief = frozenset(move_blank(s, move) for s in belief)
                new_rep = move_blank(rep_state, move)
                and_result = and_search(new_belief, path + [move], new_rep)
                if and_result is not None:
                    solution, steps = and_result
                    return [rep_state] + solution, steps + 1
            return None, 0

        def and_search(belief, path, rep_state):
            return or_search(belief, path, rep_state)

        start_belief = frozenset([start_state])
        start_rep = start_state
        solution, steps = or_search(start_belief, [], start_rep)
        if solution and steps < best_steps:
            best_solution = solution
            best_steps = steps

    return (convert_states_to_moves(best_solution), time.time() - start_time, nodes_expanded) if best_solution else (None, time.time() - start_time, nodes_expanded)

def search_with_partial_observation(initial_beliefs, goal_state, max_depth=100):
    start_time = time.time()
    nodes_expanded = 0
    solvable_beliefs = [s for s in initial_beliefs if is_solvable(s, goal_state)]
    if not solvable_beliefs:
        return None, time.time() - start_time, nodes_expanded

    best_solution = None
    best_steps = float('inf')

    for start_state in solvable_beliefs:
        visited = set()

        def apply_move_to_belief(belief, move):
            return frozenset(move_blank(s, move) for s in belief)

        def recursive_search(belief, rep_state, sequence, depth):
            nonlocal nodes_expanded
            nodes_expanded += 1
            if depth == 0:
                if goal_state in belief:
                    return [rep_state], sequence
                return None, None

            belief_tuple = frozenset(belief)
            if belief_tuple in visited:
                return None, None
            visited.add(belief_tuple)

            for move in step:
                new_belief = apply_move_to_belief(belief, move)
                new_rep = move_blank(rep_state, move)
                if new_rep == rep_state:
                    continue
                sol_states, sol_moves = recursive_search(new_belief, new_rep, sequence + [move], depth - 1)
                if sol_states is not None:
                    return [rep_state] + sol_states, sol_moves
            return None, None

        for depth in range(1, max_depth + 1):
            visited.clear()
            start_belief = frozenset([start_state])
            start_rep = start_state
            solution_states, solution_moves = recursive_search(start_belief, start_rep, [], depth)
            if solution_states is not None:
                steps = len(solution_moves)
                if steps < best_steps:
                    best_solution = solution_states
                    best_steps = steps
                break

    return (convert_states_to_moves(best_solution), time.time() - start_time, nodes_expanded) if best_solution else (None, time.time() - start_time, nodes_expanded)