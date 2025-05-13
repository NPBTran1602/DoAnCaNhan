import time
from collections import deque
from algorithms import find_blank, swap, step, convert_states_to_moves, is_solvable

def get_valid_moves(state):
    """Tạo danh sách các trạng thái tiếp theo từ trạng thái hiện tại bằng các di chuyển hợp lệ."""
    i, j = find_blank(state)
    valid_states = []
    valid_moves = []
    for move, (di, dj) in step.items():
        ni, nj = i + di, j + dj  # Sửa lỗi: thay nj bằng dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            new_state = swap(state, i, j, ni, nj)
            valid_states.append(new_state)
            valid_moves.append(move)
    return valid_states, valid_moves

def backtracking_search(start, goal):
    start_time = time.time()
    nodes_expanded = 0
    
    def backtrack(state, path, visited, depth, max_depth=50):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if depth > max_depth:  # Giới hạn độ sâu
            return None, None
        if state == goal:
            return path, [state]
        
        visited.add(state)
        next_states, moves = get_valid_moves(state)
        for next_state, move in zip(next_states, moves):
            if next_state not in visited:
                result_path, result_states = backtrack(next_state, path + [move], visited, depth + 1, max_depth)
                if result_path is not None:
                    return result_path, [state] + result_states
        visited.remove(state)
        return None, None

    if not is_solvable(start, goal):
        return None, time.time() - start_time, nodes_expanded

    visited = set()
    path, states = backtrack(start, [], visited, 0)
    if path is None:
        return None, time.time() - start_time, nodes_expanded
    return path, time.time() - start_time, nodes_expanded

def forward_checking(start, goal):
    start_time = time.time()
    nodes_expanded = 0
    
    def is_consistent(next_state, visited):
        """Kiểm tra xem trạng thái tiếp theo có khả thi không."""
        return next_state not in visited and is_solvable(next_state, goal)

    def forward_check(state, visited):
        """Thu hẹp miền giá trị bằng cách kiểm tra tính khả thi của các trạng thái tiếp theo."""
        next_states, moves = get_valid_moves(state)
        valid_next = [(ns, m) for ns, m in zip(next_states, moves) if is_consistent(ns, visited)]
        return valid_next

    def backtrack(state, path, visited, depth, max_depth=50):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if depth > max_depth:  # Giới hạn độ sâu
            return None, None
        if state == goal:
            return path, [state]
        
        visited.add(state)
        valid_next = forward_check(state, visited)
        for next_state, move in valid_next:
            result_path, result_states = backtrack(next_state, path + [move], visited, depth + 1, max_depth)
            if result_path is not None:
                return result_path, [state] + result_states
        visited.remove(state)
        return None, None

    if not is_solvable(start, goal):
        return None, time.time() - start_time, nodes_expanded

    visited = set()
    path, states = backtrack(start, [], visited, 0)
    if path is None:
        return None, time.time() - start_time, nodes_expanded
    return path, time.time() - start_time, nodes_expanded

def ac3(start, goal):
    start_time = time.time()
    nodes_expanded = 0
    
    def revise(xi, xj, domains):
        """Thu hẹp miền giá trị của xi dựa trên ràng buộc với xj."""
        nonlocal nodes_expanded
        nodes_expanded += 1
        revised = False
        valid_states = []
        for state_i in domains[xi]:
            next_states_i, _ = get_valid_moves(state_i)
            consistent = False
            for state_j in domains[xj]:
                if state_j in next_states_i and is_solvable(state_j, goal):
                    consistent = True
                    break
            if consistent:
                valid_states.append(state_i)
            else:
                revised = True
        domains[xi] = valid_states
        return revised

    # Khởi tạo CSP
    max_steps = 100  # Tăng giới hạn số bước
    domains = {i: [] for i in range(max_steps)}
    domains[0] = [start]  # Miền giá trị ban đầu cho bước 0
    visited = set([start])

    # Khởi tạo miền giá trị cho các bước tiếp theo
    for i in range(1, max_steps):
        prev_states = domains[i-1]
        next_states_set = set()
        for prev_state in prev_states:
            next_states, _ = get_valid_moves(prev_state)
            for ns in next_states:
                if ns not in visited and is_solvable(ns, goal):
                    next_states_set.add(ns)
        domains[i] = list(next_states_set)
        visited.update(next_states_set)
        print(f"Step {i}: {len(domains[i])} states in domain")  # Gỡ lỗi
        if not domains[i]:  # Nếu miền rỗng, không có lời giải
            print(f"Domain empty at step {i}")  # Gỡ lỗi
            return None, time.time() - start_time, nodes_expanded

    # Tạo các cung (ràng buộc giữa các bước liên tiếp)
    arcs = [(i, i+1) for i in range(max_steps-1)]
    queue = deque(arcs)
    
    # Thuật toán AC3
    while queue:
        xi, xj = queue.popleft()
        if revise(xi, xj, domains):
            if not domains[xi]:
                print(f"Domain {xi} became empty after revision")  # Gỡ lỗi
                return None, time.time() - start_time, nodes_expanded
            # Thêm các cung liên quan đến xi vào queue
            for k in range(xi):
                if k != xj:
                    queue.append((k, xi))

    def backtrack(assignment, step, depth, max_depth=100):
        nonlocal nodes_expanded
        nodes_expanded += 1
        if depth > max_depth:
            return None, None
        if step >= max_steps:
            return None, None
        if step > 0 and assignment.get(step-1) == goal:
            states = []
            for i in range(step):
                if i in assignment:
                    states.append(assignment[i])
            return states, states
        
        for state in domains[step]:
            if step == 0 or (step-1 in assignment and state in get_valid_moves(assignment[step-1])[0]):
                assignment[step] = state
                result_path, result_states = backtrack(assignment, step + 1, depth + 1, max_depth)
                if result_path is not None:
                    return result_path, result_states
                del assignment[step]
        return None, None

    if not is_solvable(start, goal):
        return None, time.time() - start_time, nodes_expanded

    assignment = {0: start}
    path, states = backtrack(assignment, 1, 0)
    if path is None:
        print("Backtracking failed to find a solution")  # Gỡ lỗi
        return None, time.time() - start_time, nodes_expanded
    return convert_states_to_moves(states), time.time() - start_time, nodes_expanded