import time
import tkinter as tk
from tkinter import messagebox
import random
from ui import create_ui, update_view, add_step_frame
from algorithms import (get_state_from_entries, is_solvable, apply_move,
                       bfs, dfs, ucs, iddfs, greedy, a_search, ida_search,
                       simple_hill_climbing, steepest_ascent_hill_climbing,
                       stochastic_hill_climbing, simulated_annealing, beam_search,
                       genetic_algorithm_solver, searching_with_no_observation, convert_states_to_moves)
from belief_environment import belief_state, and_or_search, search_with_partial_observation, generate_random_states
from constraint_environment import backtracking_search, forward_checking, ac3
from reinforcement_environment import q_learning
from environment_selector import select_environment

delay = 500
paused = False
running = False
result = None
step_frame_count = 0
simulation_result = {}
sim_after_id = None
current_step_index = 0
algorithm_metrics = {}  # Biến toàn cục để lưu số liệu thực thi

algorithms = {
    "BFS": lambda start: bfs(start, result),
    "DFS": lambda start: dfs(start, result),
    "UCS": lambda start: ucs(start, result),
    "IDDFS": lambda start: iddfs(start, result),
    "Greedy": lambda start: greedy(start, result),
    "A* Search": lambda start: a_search(start, result),
    "IDA* Search": lambda start: ida_search(start, result),
    "Simple Hill Climbing": lambda start: simple_hill_climbing(start, result),
    "Steepest Ascent HC": lambda start: steepest_ascent_hill_climbing(start, result),
    "Stochastic HC": lambda start: stochastic_hill_climbing(start, result),
    "Simulated Annealing": lambda start: simulated_annealing(start, result),
    "Beam Search": lambda start: beam_search(start, result),
    "Genetic Algorithm": lambda start: genetic_algorithm_solver(start, result),
    "Searching with No Observation": lambda start: searching_with_no_observation(start, result),
    "Backtracking Search": lambda start: backtracking_search(start, result),
    "Forward Checking": lambda start: forward_checking(start, result),
    "Constraint Propagation (AC3)": lambda start: ac3(start, result),
    "Q-Learning": lambda start: q_learning(start, result),
    "Belief State Search": lambda start: belief_state(start, result),
    "AND-OR Search": lambda start: and_or_search(start, result),
    "Search with Partial Observation": lambda start: search_with_partial_observation(start, result),
}

def get_randomize_params(environment, initial_entries, constraint_initial_entries, belief_entries, goal_entries, belief_goal_entries):
    if environment == "Constraint-Based Search":
        return constraint_initial_entries, []
    elif environment == "Complex Search":
        return belief_entries, belief_goal_entries
    else:
        return initial_entries, goal_entries

def pause_program():
    global paused
    paused = True

def resume_program():
    global paused
    paused = False

def reset_program(steps_container, time_container, view_labels, root, steps_count, time_label, initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, environment):
    global running, paused, step_frame_count, simulation_result, sim_after_id, current_step_index
    running = False
    paused = False
    step_frame_count = 0
    current_step_index = 0
    simulation_result = {}
    if sim_after_id is not None:
        try:
            root.after_cancel(sim_after_id)
        except Exception:
            pass
    sim_after_id = None
    for widget in steps_container.winfo_children():
        widget.destroy()
    for widget in time_container.winfo_children():
        widget.destroy()
    update_view(((0, 0, 0), (0, 0, 0), (0, 0, 0)), view_labels, root)
    steps_count.config(text="Bước: 0")
    time_label.config(text="Thời gian: 00:00")

    # Clear all entry fields
    if environment == "Constraint-Based Search":
        for i in range(3):
            for j in range(3):
                constraint_initial_entries[i][j].delete(0, tk.END)
    elif environment == "Complex Search":
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    belief_entries[k][i][j].delete(0, tk.END)
        for i in range(3):
            for j in range(3):
                belief_goal_entries[i][j].delete(0, tk.END)
                partial_belief_entries[i][j].delete(0, tk.END)
    else:
        for i in range(3):
            for j in range(3):
                initial_entries[i][j].delete(0, tk.END)
                goal_entries[i][j].delete(0, tk.END)

def export_file(simulation_result, environment, selected_algorithm):
    if not simulation_result.get('states'):
        messagebox.showerror("Lỗi", "Chưa có kết quả chạy để xuất file!")
        return

    try:
        total_time = simulation_result.get('total_time')
        if total_time is None:
            total_time = sum(simulation_result.get('step_times', []))

        with open("test.txt", "w", encoding="utf-8") as file:
            file.write("=== 8-Puzzle Solution Steps ===\n\n")
            if environment == "Complex Search" and selected_algorithm in ["Belief State Search", "AND-OR Search", "Search with Partial Observation"]:
                file.write("Trạng thái ban đầu (Niềm Tin):\n")
                for idx, state in enumerate(simulation_result['initial'], 1):
                    file.write(f"Trạng thái {idx}:\n")
                    for row in state:
                        file.write("  " + " ".join(map(str, row)) + "\n")
                    file.write("\n")
            else:
                file.write("Trạng thái ban đầu:\n")
                for row in simulation_result['initial']:
                    file.write("  " + " ".join(map(str, row)) + "\n")
                file.write("\n")
            if 'goal' in simulation_result:
                file.write("Trạng thái đích:\n")
                for row in simulation_result['goal']:
                    file.write("  " + " ".join(map(str, row)) + "\n")
                file.write("\n")
            for i, (move, state, stime) in enumerate(
                zip(
                    simulation_result['steps'],
                    simulation_result['states'][1:],
                    simulation_result['step_times']
                ),
                start=1
            ):
                file.write(f"Bước {i}: {move}\n")
                for row in state:
                    file.write("  " + " ".join(map(str, row)) + "\n")
                file.write(f"  Thời gian bước {i}: {stime:.2f}s\n\n")
            file.write(f"Tổng số bước: {len(simulation_result['steps'])}\n")
            file.write(f"Tổng thời gian: {total_time:.2f}s\n")

        messagebox.showinfo("Xuất file thành công",
                            "Đã xuất kết quả vào file: test.txt")

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi xuất file:\n{e}")

def simulate_solution(steps, current_state, index, all_states, step_times, root, view_labels,
                      steps_container, time_container, steps_count, time_label, max_columns):
    global simulation_result, sim_after_id, step_frame_count, current_step_index

    if index >= len(steps):
        total_time = time.time() - simulation_result['start_time']
        simulation_result['total_time'] = total_time
        time_label.config(text=f"Thời gian: {total_time:.2f}s")
        for w in time_container.winfo_children():
            w.destroy()
        for idx, t in enumerate(simulation_result['step_times'], start=1):
            tk.Label(
                time_container,
                text=f"Bước {idx}: {t:.2f}s",
                bg="#F3E5F5",
                fg="#4A148C",
                font=("Helvetica", 10, "bold")
            ).pack(padx=5, pady=2, anchor="w")
        messagebox.showinfo(
            "Kết quả",
            f"Đã giải trong {len(simulation_result['steps'])} bước và {total_time:.2f} giây"
        )
        sim_after_id = None
        return

    if paused:
        sim_after_id = root.after(50, lambda: simulate_solution(steps, current_state, index, all_states,
                                                                step_times, root, view_labels, steps_container,
                                                                time_container, steps_count, time_label, max_columns))
        return

    now = time.time()
    last = simulation_result.get('last_timestamp', simulation_result['start_time'])
    curr_step_time = now - last
    simulation_result['last_timestamp'] = now

    move_ = steps[index]
    new_state = apply_move(current_state, move_)
    if new_state == current_state:
        sim_after_id = root.after(delay, lambda: simulate_solution(steps, current_state, index+1, all_states,
                                                                  step_times, root, view_labels, steps_container,
                                                                  time_container, steps_count, time_label, max_columns))
        return

    update_view(new_state, view_labels, root)
    all_states.append(new_state)
    step_times.append(curr_step_time)
    step_frame_count = add_step_frame(steps_container, step_frame_count, max_columns, index, move_, new_state, curr_step_time)

    simulation_result['steps'].append(move_)
    simulation_result['states'].append(new_state)
    simulation_result['step_times'].append(curr_step_time)
    steps_count.config(text=f"Bước: {len(simulation_result['steps'])}")
    current_step_index = index

    sim_after_id = root.after(delay, lambda: simulate_solution(steps, new_state, index+1, all_states,
                                                              step_times, root, view_labels, steps_container,
                                                              time_container, steps_count, time_label, max_columns))

def previous_step(root, view_labels, steps_container, time_container, steps_count, time_label, max_columns):
    global simulation_result, step_frame_count, current_step_index
    if not simulation_result.get('states') or current_step_index <= 0:
        messagebox.showerror("Lỗi", "Không có bước trước đó!")
        return

    current_step_index -= 1
    state = simulation_result['states'][current_step_index]
    update_view(state, view_labels, root)
    steps_count.config(text=f"Bước: {current_step_index}")

    # Clear and rebuild step frames up to current step
    for widget in steps_container.winfo_children():
        widget.destroy()
    for widget in time_container.winfo_children():
        widget.destroy()
    step_frame_count = 0
    for idx in range(current_step_index):
        move_ = simulation_result['steps'][idx]
        state_ = simulation_result['states'][idx + 1]
        step_time = simulation_result['step_times'][idx]
        step_frame_count = add_step_frame(steps_container, step_frame_count, max_columns, idx, move_, state_, step_time)
        tk.Label(
            time_container,
            text=f"Bước {idx + 1}: {step_time:.2f}s",
            bg="#F3E5F5",
            fg="#4A148C",
            font=("Helvetica", 10, "bold")
        ).pack(padx=5, pady=2, anchor="w")

def next_step(root, view_labels, steps_container, time_container, steps_count, time_label, max_columns):
    global simulation_result, step_frame_count, current_step_index
    if not simulation_result.get('states') or current_step_index >= len(simulation_result['steps']):
        messagebox.showerror("Lỗi", "Không có bước tiếp theo!")
        return

    current_step_index += 1
    if current_step_index >= len(simulation_result['steps']):
        messagebox.showerror("Lỗi", "Không có bước tiếp theo!")
        return

    move_ = simulation_result['steps'][current_step_index]
    state = simulation_result['states'][current_step_index + 1]
    step_time = simulation_result['step_times'][current_step_index]

    update_view(state, view_labels, root)
    step_frame_count = add_step_frame(steps_container, step_frame_count, max_columns, current_step_index, move_, state, step_time)
    steps_count.config(text=f"Bước: {current_step_index + 1}")

    tk.Label(
        time_container,
        text=f"Bước {current_step_index + 1}: {step_time:.2f}s",
        bg="#F3E5F5",
        fg="#4A148C",
        font=("Helvetica", 10, "bold")
    ).pack(padx=5, pady=2, anchor="w")

    if current_step_index == len(simulation_result['steps']) - 1:
        total_time = sum(simulation_result['step_times'])
        time_label.config(text=f"Thời gian: {total_time:.2f}s")

def randomize_belief_states(entries, goal_entries):
    goal_state = get_state_from_entries(goal_entries) if goal_entries else ((1, 2, 3), (4, 5, 6), (7, 8, 0))
    if isinstance(entries[0][0], list):  # Complex Search: multiple belief states
        random_states = generate_random_states(goal_state, num_states=3)
        for k, state in enumerate(random_states):
            for i in range(3):
                for j in range(3):
                    entries[k][i][j].delete(0, tk.END)
                    value = state[i][j]
                    entries[k][i][j].insert(0, str(value) if value != 0 else "")
    else:  # Single state for other environments
        random_states = generate_random_states(goal_state, num_states=1)
        state = random_states[0]
        for i in range(3):
            for j in range(3):
                entries[i][j].delete(0, tk.END)
                value = state[i][j]
                entries[i][j].insert(0, str(value) if value != 0 else "")

def apply_partial_belief(partial_belief_entries, belief_goal_entries, belief_entries):
    # Read partial belief state
    partial_state = []
    used_values = set()
    try:
        for i in range(3):
            row = []
            for j in range(3):
                val = partial_belief_entries[i][j].get().strip()
                if val:
                    num = int(val)
                    if num < 0 or num > 8 or num in used_values:
                        raise ValueError("Invalid or duplicate value")
                    used_values.add(num)
                    row.append(num)
                else:
                    row.append(None)
            partial_state.append(row)
    except ValueError:
        messagebox.showerror("Lỗi", "Dữ liệu niềm tin 1 phần không hợp lệ! Vui lòng nhập số nguyên từ 0-8, không trùng lặp.")
        return

    # Get initial belief states to check solvability
    initial_states = []
    for k in range(3):
        state = get_state_from_entries(belief_entries[k])
        if state is None:
            messagebox.showerror("Lỗi", "Vui lòng nhập đầy đủ các trạng thái niềm 8-Puzzle không hợp lệ!")
            return
        initial_states.append(state)

    # Generate goal state
    available_values = [i for i in range(9) if i not in used_values]
    goal_state = [[None for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if partial_state[i][j] is not None:
                goal_state[i][j] = partial_state[i][j]
            else:
                if available_values:
                    goal_state[i][j] = available_values.pop(0)
                else:
                    messagebox.showerror("Lỗi", "Không đủ giá trị để tạo trạng thái đích!")
                    return

    # Convert to tuple for solvability check
    goal_state_tuple = tuple(tuple(row) for row in goal_state)

    # Check solvability
    solvable = all(is_solvable(state, goal_state_tuple) for state in initial_states)
    if not solvable:
        # Try shuffling available values to find a solvable configuration
        attempts = 100
        while attempts > 0:
            available_values = [i for i in range(9) if i not in used_values]
            random.shuffle(available_values)
            idx = 0
            temp_goal = [[partial_state[i][j] if partial_state[i][j] is not None else available_values[idx] for j in range(3)] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    if partial_state[i][j] is None:
                        temp_goal[i][j] = available_values[idx]
                        idx += 1
            temp_goal_tuple = tuple(tuple(row) for row in temp_goal)
            if all(is_solvable(state, temp_goal_tuple) for state in initial_states):
                goal_state = temp_goal
                goal_state_tuple = temp_goal_tuple
                solvable = True
                break
            attempts -= 1

    if not solvable:
        messagebox.showerror("Lỗi", "Không thể tạo trạng thái đích khả thi từ niềm tin 1 phần!")
        return

    # Update goal entries
    for i in range(3):
        for j in range(3):
            belief_goal_entries[i][j].delete(0, tk.END)
            value = goal_state[i][j]
            belief_goal_entries[i][j].insert(0, str(value) if value != 0 else "")

def back_to_selector(root):
    root.destroy()
    select_environment(initialize_app)

def solve_puzzle(initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, view_labels,
                 steps_container, time_container, steps_count, time_label, selected_algorithm, root, max_columns, environment):
    global result, paused, running, simulation_result, step_frame_count, sim_after_id, current_step_index, algorithm_metrics
    if sim_after_id is not None:
        try:
            root.after_cancel(sim_after_id)
        except Exception:
            pass
        sim_after_id = None

    for widget in steps_container.winfo_children():
        widget.destroy()
    for widget in time_container.winfo_children():
        widget.destroy()

    paused = False
    running = True
    step_frame_count = 0
    current_step_index = 0
    simulation_result = {}
    simulation_result['states'] = []
    simulation_result['steps'] = []
    simulation_result['step_times'] = []

    steps_count.config(text="Bước: 0")
    time_label.config(text="Thời gian: 00:00")

    alg_name = selected_algorithm.get()
    if environment == "Constraint-Based Search":
        initial_state = get_state_from_entries(constraint_initial_entries)
        if initial_state is None:
            return
        result_state = ((1, 2, 3), (4, 5, 6), (7, 8, 0))  # Default goal state for CSP
        simulation_result['initial'] = initial_state
        initial_states = [initial_state]
    elif environment == "Complex Search":
        initial_states = []
        for k in range(3):
            state = get_state_from_entries(belief_entries[k])
            if state is None:
                return
            initial_states.append(state)
        result_state = get_state_from_entries(belief_goal_entries)
        if result_state is None:
            return
        simulation_result['initial'] = initial_states
        initial_state = initial_states[0]  # For display
    else:
        initial_state = get_state_from_entries(initial_entries)
        if initial_state is None:
            return
        result_state = get_state_from_entries(goal_entries)
        if result_state is None:
            return
        simulation_result['initial'] = initial_state
        initial_states = [initial_state]

    # Check solvability (skip for Constraint-Based Search as CSP ensures solvability)
    if environment != "Constraint-Based Search":
        for state in initial_states:
            if not is_solvable(state, result_state):
                messagebox.showerror("Lỗi", "Puzzle không khả thi!")
                return

    update_view(initial_state, view_labels, root)
    global result
    result = result_state
    simulation_result['goal'] = result_state
    simulation_result['start_time'] = time.time()
    simulation_result['last_timestamp'] = simulation_result['start_time']

    # Khởi tạo algorithm_metrics cho môi trường nếu chưa tồn tại
    if environment not in algorithm_metrics:
        algorithm_metrics[environment] = {}

    if environment == "Complex Search":
        if alg_name == "Belief State Search":
            solution, execution_time, nodes_expanded = belief_state(initial_states, result_state)
        elif alg_name == "AND-OR Search":
            solution, execution_time, nodes_expanded = and_or_search(initial_states, result_state)
        else:  # Search with Partial Observation
            solution, execution_time, nodes_expanded = search_with_partial_observation(initial_states, result_state)
    else:
        algorithm = algorithms.get(alg_name)
        solution, execution_time, nodes_expanded = algorithm(initial_state)

    # Lưu số liệu thực thi
    algorithm_metrics[environment][alg_name] = {
        "execution_time": execution_time,
        "nodes_expanded": nodes_expanded
    }

    if solution:
        simulation_result['states'].append(initial_state)
        sim_after_id = root.after(100, lambda: simulate_solution(solution, initial_state, 0, [initial_state], [],
                                                                root, view_labels, steps_container, time_container,
                                                                steps_count, time_label, max_columns))
    else:
        messagebox.showerror("Thất bại", f"Thuật toán {alg_name} không tìm thấy lời giải!")

def initialize_app(allowed_algorithms=None, environment=None):
    if allowed_algorithms is None:
        allowed_algorithms = list(algorithms.keys())
    filtered_algorithms = {k: v for k, v in algorithms.items() if k in allowed_algorithms}
    (root, initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, view_labels,
     steps_container, time_container, steps_count, time_label, selected_algorithm, max_columns) = create_ui(
        lambda: solve_puzzle(initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, view_labels,
                             steps_container, time_container, steps_count, time_label, selected_algorithm, root, max_columns, environment),
        pause_program,
        resume_program,
        lambda: reset_program(steps_container, time_container, view_labels, root, steps_count, time_label, initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, environment),
        lambda: export_file(simulation_result, environment, selected_algorithm.get()),
        lambda: randomize_belief_states(*get_randomize_params(environment, initial_entries, constraint_initial_entries, belief_entries, goal_entries, belief_goal_entries)),
        lambda: previous_step(root, view_labels, steps_container, time_container, steps_count, time_label, max_columns),
        lambda: next_step(root, view_labels, steps_container, time_container, steps_count, time_label, max_columns),
        lambda: apply_partial_belief(partial_belief_entries, belief_goal_entries, belief_entries),
        lambda: back_to_selector(root),
        filtered_algorithms,
        environment
    )
    return root