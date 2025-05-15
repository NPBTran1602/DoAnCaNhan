from collections import deque
import heapq
import random
import math
import time
from tkinter import messagebox

step = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}

def get_state_from_entries(entries):
    try:
        return tuple(tuple(int(entries[i][j].get() or 0) for j in range(3)) for i in range(3))
    except ValueError:
        messagebox.showerror("Lỗi", "Dữ liệu nhập không hợp lệ! Vui lòng nhập số nguyên.")
        return None

def is_solvable(state1, state2):
    def inversion_parity(flat):
        inv = 0
        for i in range(len(flat)):
            for j in range(i+1, len(flat)):
                if flat[i] > flat[j]:
                    inv += 1
        return inv % 2

    flat1 = [n for row in state1 for n in row if n != 0]
    flat2 = [n for row in state2 for n in row if n != 0]
    return inversion_parity(flat1) == inversion_parity(flat2)

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None

def swap(state, i1, j1, i2, j2):
    new_state = [list(row) for row in state]
    new_state[i1][j1], new_state[i2][j2] = new_state[i2][j2], new_state[i1][j1]
    return tuple(tuple(row) for row in new_state)

def apply_move(state, move):
    i, j = find_blank(state)
    di, dj = step[move]
    new_i, new_j = i + di, j + dj
    if 0 <= new_i < 3 and 0 <= new_j < 3:
        return swap(state, i, j, new_i, new_j)
    else:
        return state

def calc_heuristic(state, result):
    total = 0
    if result is None:
        return total
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                for r in range(3):
                    for c in range(3):
                        if result[r][c] == val:
                            total += abs(r - i) + abs(c - j)
                            break
    return total

def convert_states_to_moves(states):
    if not states or len(states) < 2:
        return []
    moves = []
    for i in range(len(states) - 1):
        s1 = states[i]
        s2 = states[i+1]
        blank1 = find_blank(s1)
        blank2 = find_blank(s2)
        di = blank2[0] - blank1[0]
        dj = blank2[1] - blank1[1]
        for m, (mdi, mdj) in step.items():
            if (mdi, mdj) == (di, dj):
                moves.append(m)
                break
    return moves

def bfs(start, result):
    start_time = time.time()
    nodes_expanded = 0
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        current, path = queue.popleft()
        nodes_expanded += 1
        if current == result:
            return path, time.time() - start_time, nodes_expanded
        blank_i, blank_j = find_blank(current)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current, blank_i, blank_j, new_i, new_j)
                if new_state not in seen:
                    queue.append((new_state, path + [move_]))
                    seen.add(new_state)
    return None, time.time() - start_time, nodes_expanded

def dfs(start, result, max_depth=50):
    start_time = time.time()
    nodes_expanded = 0
    stack = [(start, [])]
    seen = set()
    while stack:
        current, path = stack.pop()
        nodes_expanded += 1
        if len(path) > max_depth:
            continue
        if current == result:
            return path, time.time() - start_time, nodes_expanded
        if current in seen:
            continue
        seen.add(current)
        blank_i, blank_j = find_blank(current)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current, blank_i, blank_j, new_i, new_j)
                stack.append((new_state, path + [move_]))
    return None, time.time() - start_time, nodes_expanded

def ucs(start, result):
    start_time = time.time()
    nodes_expanded = 0
    pq = [(0, start, [])]
    cost_map = {start: 0}
    while pq:
        cost, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        if current == result:
            return path, time.time() - start_time, nodes_expanded
        blank_i, blank_j = find_blank(current)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current, blank_i, blank_j, new_i, new_j)
                new_cost = cost + 1
                if new_state not in cost_map or new_cost < cost_map[new_state]:
                    cost_map[new_state] = new_cost
                    heapq.heappush(pq, (new_cost, new_state, path + [move_]))
    return None, time.time() - start_time, nodes_expanded

def iddfs(start, result, max_depth=50):
    start_time = time.time()
    nodes_expanded = 0
    def dls(state, path, depth, visited):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if state == result:
            return path
        if depth == 0:
            return None
        visited.add(state)
        blank_i, blank_j = find_blank(state)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(state, blank_i, blank_j, new_i, new_j)
                if new_state not in visited:
                    res = dls(new_state, path + [move_], depth - 1, visited)
                    if res:
                        return res
        return None
    for depth in range(1, max_depth + 1):
        solution = dls(start, [], depth, set())
        if solution:
            return solution, time.time() - start_time, nodes_expanded
    return None, time.time() - start_time, nodes_expanded

def greedy(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(abs((value-1) % 3 - j) + abs((value-1) // 3 - i)
                   for i, row in enumerate(state)
                   for j, value in enumerate(row) if value != 0)
    pq = [(heuristic(start), start, [])]
    seen = set()
    while pq:
        _, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        if current == result:
            return path, time.time() - start_time, nodes_expanded
        if current in seen:
            continue
        seen.add(current)
        blank_i, blank_j = find_blank(current)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current, blank_i, blank_j, new_i, new_j)
                if new_state not in seen:
                    heapq.heappush(pq, (heuristic(new_state), new_state, path + [move_]))
    return None, time.time() - start_time, nodes_expanded

def a_search(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(abs((value-1) % 3 - j) + abs((value-1) // 3 - i)
                   for i, row in enumerate(state)
                   for j, value in enumerate(row) if value != 0)
    pq = []
    g_map = {start: 0}
    h0 = heuristic(start)
    heapq.heappush(pq, (h0, 0, start, []))
    while pq:
        f, g, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        if current == result:
            return path, time.time() - start_time, nodes_expanded
        blank_i, blank_j = find_blank(current)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current, blank_i, blank_j, new_i, new_j)
                new_g = g + 1
                new_f = new_g + heuristic(new_state)
                if new_state not in g_map or new_g < g_map[new_state]:
                    g_map[new_state] = new_g
                    heapq.heappush(pq, (new_f, new_g, new_state, path + [move_]))
    return None, time.time() - start_time, nodes_expanded

def ida_search(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(abs((value-1) % 3 - j) + abs((value-1) // 3 - i)
                   for i, row in enumerate(state)
                   for j, value in enumerate(row) if value != 0)
    
    def search(state, path, g, bound, visited):
        nonlocal nodes_expanded
        nodes_expanded += 1
        f = g + heuristic(state)
        if f > bound:
            return f
        if state == result:
            return path
        min_next = float('inf')
        blank_i, blank_j = find_blank(state)
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(state, blank_i, blank_j, new_i, new_j)
                if new_state in visited:
                    continue
                visited.add(new_state)
                t = search(new_state, path + [new_state], g + 1, bound, visited)
                visited.remove(new_state)
                if isinstance(t, list):
                    return t
                if t < min_next:
                    min_next = t
        return min_next
    
    bound = heuristic(start)
    visited = set()
    while True:
        visited.clear()
        visited.add(start)
        t = search(start, [start], 0, bound, visited)
        if isinstance(t, list):
            return convert_states_to_moves(t), time.time() - start_time, nodes_expanded
        if t == float('inf'):
            return None, time.time() - start_time, nodes_expanded
        bound = t

def simple_hill_climbing(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(1 for i in range(3) for j in range(3)
                   if state[i][j] != result[i][j] and state[i][j] != 0)
    current_state = start
    path = []
    while True:
        blank = find_blank(current_state)
        neighbors = [(swap(current_state, *blank, blank[0] + di, blank[1] + dj), move)
                     for move, (di, dj) in step.items()
                     if 0 <= blank[0] + di < 3 and 0 <= blank[1] + dj < 3]
        nodes_expanded += len(neighbors)
        if not neighbors:
            return path if current_state == result else None, time.time() - start_time, nodes_expanded
        current_heuristic = heuristic(current_state)
        for neighbor, move in neighbors:
            if heuristic(neighbor) < current_heuristic:
                current_state = neighbor
                path.append(move)
                break
        else:
            return path if current_state == result else None, time.time() - start_time, nodes_expanded

def steepest_ascent_hill_climbing(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(1 for i in range(3) for j in range(3)
                   if state[i][j] != result[i][j] and state[i][j] != 0)
    current_state = start
    path = []
    while True:
        blank = find_blank(current_state)
        neighbors = [(swap(current_state, *blank, blank[0] + di, blank[1] + dj), move)
                     for move, (di, dj) in step.items()
                     if 0 <= blank[0] + di < 3 and 0 <= blank[1] + dj < 3]
        nodes_expanded += len(neighbors)
        if not neighbors:
            return path if current_state == result else None, time.time() - start_time, nodes_expanded
        best_neighbor = min(neighbors, key=lambda x: heuristic(x[0]))
        if heuristic(best_neighbor[0]) >= heuristic(current_state):
            return path if current_state == result else None, time.time() - start_time, nodes_expanded
        current_state, move = best_neighbor
        path.append(move)

def stochastic_hill_climbing(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(1 for i in range(3) for j in range(3)
                   if state[i][j] != result[i][j] and state[i][j] != 0)
    current_state = start
    path = []
    while True:
        blank = find_blank(current_state)
        neighbors = []
        for move_, (di, dj) in step.items():
            new_i, new_j = blank[0] + di, blank[1] + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current_state, blank[0], blank[1], new_i, new_j)
                neighbors.append((new_state, move_))
        nodes_expanded += len(neighbors)
        if not neighbors:
            return path if current_state == result else None, time.time() - start_time, nodes_expanded
        better_neighbors = [n for n in neighbors if heuristic(n[0]) < heuristic(current_state)]
        if better_neighbors:
            new_state, move_ = random.choice(better_neighbors)
            current_state = new_state
            path.append(move_)
        else:
            return path if current_state == result else None, time.time() - start_time, nodes_expanded

def simulated_annealing(start, result):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(abs((value-1) % 3 - j) + abs((value-1) // 3 - i)
                   for i, row in enumerate(state)
                   for j, value in enumerate(row) if value != 0)
    current_state = start
    current_score = heuristic(current_state)
    path = []
    T = 1000.0
    alpha = 0.95
    max_iterations = 10000
    iteration = 0
    seen = {current_state}
    while T > 0.01 and iteration < max_iterations:
        if current_state == result:
            return path, time.time() - start_time, nodes_expanded
        blank_i, blank_j = find_blank(current_state)
        neighbors = []
        for move_, (di, dj) in step.items():
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_state = swap(current_state, blank_i, blank_j, new_i, new_j)
                neighbors.append((new_state, move_))
        nodes_expanded += len(neighbors)
        if not neighbors:
            break
        new_state, move_ = random.choice(neighbors)
        if new_state in seen:
            continue
        new_score = heuristic(new_state)
        delta = current_score - new_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_state = new_state
            current_score = new_score
            path.append(move_)
            seen.add(new_state)
        T *= alpha
        iteration += 1
    return path if current_state == result else None, time.time() - start_time, nodes_expanded

def beam_search(start, result, beam_width=3):
    start_time = time.time()
    nodes_expanded = 0
    def heuristic(state):
        return sum(1 for i in range(3) for j in range(3)
                   if state[i][j] != result[i][j] and state[i][j] != 0)
    
    beam = [(start, [])]
    while beam:
        if any(state == result for state, _ in beam):
            for state, path in beam:
                if state == result:
                    return path, time.time() - start_time, nodes_expanded
        next_beam = []
        for state, path in beam:
            blank_i, blank_j = find_blank(state)
            neighbors = []
            for move_, (di, dj) in step.items():
                new_i, new_j = blank_i + di, blank_j + dj
                if 0 <= new_i < 3 and 0 <= new_j < 3:
                    new_state = swap(state, blank_i, blank_j, new_i, new_j)
                    neighbors.append((new_state, path + [move_]))
            nodes_expanded += len(neighbors)
            next_beam.extend(neighbors)
        if not next_beam:
            return None, time.time() - start_time, nodes_expanded
        next_beam.sort(key=lambda x: heuristic(x[0]))
        beam = next_beam[:beam_width]
    return None, time.time() - start_time, nodes_expanded

def genetic_algorithm_solver(start, result, pop_size=500, gen_limit=5000, mutation_probability=0.1, max_moves=100):
    start_time = time.time()
    nodes_expanded = 0
    available_moves = list(step.keys())

    def make_random_candidate():
        current = start
        moves = []
        for _ in range(random.randint(10, 30)):
            valid_moves = [m for m, (di, dj) in step.items()
                           if 0 <= find_blank(current)[0] + di < 3 and
                              0 <= find_blank(current)[1] + dj < 3]
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            next_state = apply_move(current, move)
            if next_state == current:
                continue
            moves.append(move)
            current = next_state
        return moves

    def execute_moves(state, moves_seq):
        current = state
        for move in moves_seq:
            new = apply_move(current, move)
            if new == current:
                return None
            current = new
        return current

    def candidate_fitness(moves_seq):
        nonlocal nodes_expanded
        nodes_expanded += 1
        final = execute_moves(start, moves_seq)
        if final is None:
            return float('-inf')
        h_val = calc_heuristic(final, result)
        penalty = len(moves_seq) * 0.5
        return -(h_val + penalty)

    def mate(a, b):
        cut = random.randint(1, min(len(a), len(b)) - 1) if min(len(a), len(b)) > 1 else 1
        child = a[:cut] + b[cut:]
        return child if execute_moves(start, child) is not None else a

    def mutate(cand):
        c = cand.copy()
        for i in range(len(c)):
            if random.random() < mutation_probability:
                c[i] = random.choice(available_moves)
        if random.random() < 0.1 and len(c) < max_moves:
            c.append(random.choice(available_moves))
        elif random.random() < 0.1 and len(c) > 1:
            c.pop(random.randrange(len(c)))
        return c

    def simplify_solution(moves_seq):
        cur = start
        sol = []
        seen = {cur}
        for m in moves_seq:
            nxt = apply_move(cur, m)
            if nxt != cur and nxt not in seen:
                sol.append(m)
                cur = nxt
                seen.add(cur)
            if cur == result:
                break
        return sol

    population = [make_random_candidate() for _ in range(pop_size)]
    solution = None

    for gen in range(gen_limit):
        scored = [(cand, candidate_fitness(cand)) for cand in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        if execute_moves(start, best) == result:
            solution = best
            break
        k = max(1, pop_size // 4)
        elites = [cand for cand, _ in scored[:k]]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = mutate(mate(p1, p2))
            new_pop.append(child)
        population = new_pop

    if solution is None:
        return None, time.time() - start_time, nodes_expanded
    sol = simplify_solution(solution)
    return sol if execute_moves(start, sol) == result else None, time.time() - start_time, nodes_expanded

def searching_with_no_observation(start, result, max_depth=50):
    start_time = time.time()
    nodes_expanded = 0
    available_moves = list(step.keys())

    def apply_moves(state, moves):
        for move in moves:
            blank_i, blank_j = find_blank(state)
            di, dj = step[move]
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                state = swap(state, blank_i, blank_j, new_i, new_j)
            else:
                return None
        return state

    def recursive_search(state, sequence, depth, visited):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if depth == 0:
            if state == result:
                return sequence
            return None

        for move in available_moves:
            blank_i, blank_j = find_blank(state)
            di, dj = step[move]
            new_i, new_j = blank_i + di, blank_j + dj
            if not (0 <= new_i < 3 and 0 <= new_j < 3):
                continue

            new_state = swap(state, blank_i, blank_j, new_i, new_j)
            state_tuple = tuple(map(tuple, new_state))
            if state_tuple in visited:
                continue

            visited.add(state_tuple)
            sol = recursive_search(new_state, sequence + [move], depth - 1, visited)
            if sol is not None:
                return sol
            visited.remove(state_tuple)

        return None

    for depth in range(1, max_depth + 1):
        visited = {tuple(map(tuple, start))}
        sol = recursive_search(start, [], depth, visited)
        if sol is not None:
            return sol, time.time() - start_time, nodes_expanded
    return None, time.time() - start_time, nodes_expanded