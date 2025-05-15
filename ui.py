import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Color and style constants
BG_MAIN       = "#F3E5F5"
FRAME_BG      = "#FCE4EC"
TEXT_COLOR    = "#4A148C"
ACCENT_COLOR  = "#BA68C8"
BTN_SOLVE_BG  = "#CE93D8"
BTN_PAUSE_BG  = "#F48FB1"
BTN_RESUME_BG = "#CE93D8"
BTN_RESET_BG  = "#F06292"
BTN_EXPORT_BG = "#BA68C8"
BTN_RANDOM_BG = "#EC407A"
BTN_STEP_BG   = "#D81B60"
BTN_BACK_BG   = "#AB47BC"
BTN_COMPARE_BG = "#9575CD"  # Thêm màu cho nút so sánh
BTN_FG        = TEXT_COLOR

STEP_FRAME_SIZE = 160
max_columns = 3

def update_view(state, view_labels, root):
    for i in range(3):
        for j in range(3):
            tile = view_labels[i][j]
            value = state[i][j]
            tile.config(text=str(value) if value != 0 else "", bg=BG_MAIN, fg=TEXT_COLOR)
    root.update_idletasks()

def create_state_grid(parent, state_3x3, cell_size=1):
    for i in range(3):
        for j in range(3):
            val = state_3x3[i][j]
            bg_color = "#F8BBD0" if val == 0 else FRAME_BG
            lbl = tk.Label(
                parent,
                text=str(val) if val != 0 else "",
                width=cell_size,
                height=cell_size,
                font=("Helvetica", 10),
                bg=bg_color,
                fg=TEXT_COLOR,
                bd=1,
                relief="ridge"
            )
            lbl.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")

def add_step_frame(steps_container, step_frame_count, max_columns, index, move_, state_3x3, step_time):
    col = step_frame_count % max_columns
    row = step_frame_count // max_columns

    frame = tk.Frame(
        steps_container,
        bg=BG_MAIN,
        bd=1,
        relief="solid",
        width=STEP_FRAME_SIZE,
        height=STEP_FRAME_SIZE
    )
    frame.grid(row=row, column=col, padx=5, pady=5)
    frame.grid_propagate(False)

    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(0, weight=0)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=0)

    lbl_title = tk.Label(
        frame,
        text=f"Bước {index} - {move_}",
        font=("Helvetica", 10, "bold"),
        bg=BG_MAIN,
        fg=TEXT_COLOR
    )
    lbl_title.grid(row=0, column=0, sticky="n", pady=(4,2))

    puzzle_frame = tk.Frame(frame, bg=BG_MAIN)
    puzzle_frame.grid(row=1, column=0, sticky="ns", padx=5)
    create_state_grid(puzzle_frame, state_3x3, cell_size=1)

    time_lbl = tk.Label(
        frame,
        text=f"{step_time:.2f}s",
        font=("Helvetica", 9),
        bg=BG_MAIN,
        fg=ACCENT_COLOR
    )
    time_lbl.grid(row=2, column=0, sticky="s", pady=(2,4))

    return step_frame_count + 1

def show_comparison(algorithms, environment):
    from app import algorithm_metrics
    
    # Tạo cửa sổ mới
    compare_window = tk.Toplevel()
    compare_window.title(f"So sánh thuật toán - {environment}")
    compare_window.geometry("900x700")
    compare_window.configure(bg=BG_MAIN)

    # Lấy dữ liệu từ algorithm_metrics
    metrics = algorithm_metrics.get(environment, {})
    algo_names = []
    execution_times = []
    nodes_expanded = []

    for algo in algorithms:
        execution_time = metrics.get(algo, {}).get("execution_time", None)
        nodes = metrics.get(algo, {}).get("nodes_expanded", None)
        if execution_time is not None and nodes is not None:
            algo_names.append(algo)
            # Chuyển thời gian sang mili-giây nếu quá nhỏ
            execution_times.append(execution_time * 1000 if execution_time < 0.1 else execution_time)
            nodes_expanded.append(nodes)

    if not algo_names:
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu để so sánh. Vui lòng chạy ít nhất một thuật toán.")
        compare_window.destroy()
        return

    # Tạo frame chính
    main_frame = tk.Frame(compare_window, bg=BG_MAIN)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Tạo biểu đồ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Biểu đồ thời gian thực thi
    bars1 = ax1.bar(algo_names, execution_times, color='#BA68C8')
    ax1.set_title("Thời gian thực thi")
    ax1.set_ylabel("Mili-giây" if any(t < 0.1 for t in execution_times) else "Giây")
    ax1.tick_params(axis='x', rotation=45)

    # Thêm giá trị trên cột
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # Biểu đồ số node mở rộng
    bars2 = ax2.bar(algo_names, nodes_expanded, color='#9575CD')
    ax2.set_title("Số node mở rộng")
    ax2.set_ylabel("Số node")
    ax2.tick_params(axis='x', rotation=45)

    # Thêm giá trị trên cột
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Tùy chỉnh layout
    plt.tight_layout()

    # Nhúng biểu đồ vào Tkinter
    canvas = FigureCanvasTkAgg(fig, master=main_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Thêm bảng số liệu
    table_frame = tk.LabelFrame(main_frame, text="Bảng số liệu", font=("Helvetica", 10, "bold"),
                                bg=FRAME_BG, fg=ACCENT_COLOR, bd=2, relief="groove")
    table_frame.pack(fill="x", pady=10)

    # Tiêu đề bảng
    tk.Label(table_frame, text="Thuật toán", font=("Helvetica", 10, "bold"), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=0, column=0, padx=5, pady=2, sticky="w")
    tk.Label(table_frame, text="Thời gian (ms)", font=("Helvetica", 10, "bold"), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=0, column=1, padx=5, pady=2, sticky="w")
    tk.Label(table_frame, text="Node mở rộng", font=("Helvetica", 10, "bold"), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=0, column=2, padx=5, pady=2, sticky="w")

    # Dữ liệu bảng
    for i, (algo, time_ms, nodes) in enumerate(zip(algo_names, execution_times, nodes_expanded), 1):
        tk.Label(table_frame, text=algo, font=("Helvetica", 9), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=i, column=0, padx=5, pady=2, sticky="w")
        tk.Label(table_frame, text=f"{time_ms:.2f}", font=("Helvetica", 9), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=i, column=1, padx=5, pady=2, sticky="w")
        tk.Label(table_frame, text=str(int(nodes)), font=("Helvetica", 9), bg=FRAME_BG, fg=TEXT_COLOR).grid(row=i, column=2, padx=5, pady=2, sticky="w")

    # Đóng cửa sổ khi đóng biểu đồ
    compare_window.protocol("WM_DELETE_WINDOW", lambda: [plt.close(fig), compare_window.destroy()])

def create_ui(solve_puzzle, pause_program, resume_program, reset_program, export_file, randomize_belief_states, previous_step, next_step, apply_partial_belief, back_to_selector, algorithms, environment):
    root = tk.Tk()
    root.title("8-Puzzle Solver - Nguyễn Phạm Bảo Trân - 23110348")
    root.geometry("1200x750")
    root.configure(bg=BG_MAIN)

    root.grid_columnconfigure(0, weight=1, uniform="group1")
    root.grid_columnconfigure(1, weight=0, uniform="group1")
    root.grid_columnconfigure(2, weight=2, uniform="group1")

    # Input Frame
    frame_input = tk.Frame(root, bg=BG_MAIN)
    frame_input.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

    # Normal Frame (for Informed, Uninformed, Local, Reinforcement, and AND-OR Search)
    frame_normal = tk.Frame(frame_input, bg=BG_MAIN)
    # Constraint Frame (for Constraint-Based Search)
    frame_constraint = tk.Frame(frame_input, bg=BG_MAIN)
    # Belief Frame (for Belief State Search, AND-OR Search, Search with Partial Observation)
    frame_belief = tk.Frame(frame_input, bg=BG_MAIN)

    # Initial State Frame (Normal and Constraint)
    frame_initial = tk.LabelFrame(
        frame_normal,
        text="Trạng Thái Xuất Phát",
        font=("Helvetica", 12, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge"
    )
    frame_initial.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    for i in range(3):
        frame_initial.grid_rowconfigure(i, weight=1)
    for j in range(3):
        frame_initial.grid_columnconfigure(j, weight=1)

    initial_entries = []
    for i in range(3):
        row = []
        for j in range(3):
            e = tk.Entry(frame_initial, width=4, font=("Helvetica", 12, "bold"),
                         justify="center", bg=FRAME_BG, bd=1, relief="ridge")
            e.grid(row=i, column=j, padx=3, pady=3, ipady=5, sticky="nsew")
            row.append(e)
        initial_entries.append(row)

    # Randomize Button for Normal Frame
    btn_randomize_normal = tk.Button(
        frame_initial,
        text="Ngẫu Nhiên",
        font=("Helvetica", 10),
        bg=BTN_RANDOM_BG,
        fg=BTN_FG,
        command=randomize_belief_states,
        width=12,
        height=1
    )
    btn_randomize_normal.grid(row=3, column=0, columnspan=3, pady=3)

    # Goal State Frame (Normal only)
    frame_goal = tk.LabelFrame(
        frame_normal,
        text="Trạng Thái Đích",
        font=("Helvetica", 12, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge"
    )
    frame_goal.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    for i in range(3):
        frame_goal.grid_rowconfigure(i, weight=1)
    for j in range(3):
        frame_goal.grid_columnconfigure(j, weight=1)

    goal_entries = []
    for i in range(3):
        row = []
        for j in range(3):
            e = tk.Entry(frame_goal, width=4, font=("Helvetica", 12, "bold"),
                         justify="center", bg=FRAME_BG, bd=1, relief="ridge")
            e.grid(row=i, column=j, padx=3, pady=3, ipady=5, sticky="nsew")
            row.append(e)
        goal_entries.append(row)

    # Initial State Frame (Constraint)
    frame_constraint_initial = tk.LabelFrame(
        frame_constraint,
        text="Trạng Thái Xuất Phát",
        font=("Helvetica", 12, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge"
    )
    frame_constraint_initial.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    for i in range(3):
        frame_constraint_initial.grid_rowconfigure(i, weight=1)
    for j in range(3):
        frame_constraint_initial.grid_columnconfigure(j, weight=1)

    constraint_initial_entries = []
    for i in range(3):
        row = []
        for j in range(3):
            e = tk.Entry(frame_constraint_initial, width=4, font=("Helvetica", 12, "bold"),
                         justify="center", bg=FRAME_BG, bd=1, relief="ridge")
            e.grid(row=i, column=j, padx=3, pady=3, ipady=5, sticky="nsew")
            row.append(e)
        constraint_initial_entries.append(row)

    # Randomize Button for Constraint Frame
    btn_randomize_constraint = tk.Button(
        frame_constraint_initial,
        text="Ngẫu Nhiên",
        font=("Helvetica", 10),
        bg=BTN_RANDOM_BG,
        fg=BTN_FG,
        command=randomize_belief_states,
        width=12,
        height=1
    )
    btn_randomize_constraint.grid(row=3, column=0, columnspan=3, pady=3)

    # Belief Initial States Frame
    frame_belief_initial = tk.LabelFrame(
        frame_belief,
        text="Trạng Thái Xuất Phát (Niềm Tin)",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge"
    )
    frame_belief_initial.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    belief_entries = []
    for k in range(3):
        subframe = tk.Frame(frame_belief_initial, bg=FRAME_BG)
        subframe.grid(row=k, column=0, padx=3, pady=3)
        tk.Label(subframe, text=f"Trạng thái {k+1}", bg=FRAME_BG, fg=TEXT_COLOR,
                 font=("Helvetica", 9)).grid(row=0, column=0, columnspan=3)
        entries = []
        for i in range(3):
            row = []
            for j in range(3):
                e = tk.Entry(subframe, width=4, font=("Helvetica", 10),
                             justify="center", bg=FRAME_BG, bd=1, relief="ridge")
                e.grid(row=i+1, column=j, padx=1, pady=1, ipady=3, sticky="nsew")
                row.append(e)
            entries.append(row)
        belief_entries.append(entries)

    # Randomize Button for Belief Frame
    btn_randomize = tk.Button(
        frame_belief_initial,
        text="Ngẫu Nhiên",
        font=("Helvetica", 10),
        bg=BTN_RANDOM_BG,
        fg=BTN_FG,
        command=randomize_belief_states,
        width=12,
        height=1
    )
    btn_randomize.grid(row=3, column=0, pady=3)

    # Belief Goal State Frame
    frame_belief_goal = tk.LabelFrame(
        frame_belief,
        text="Trạng Thái Đích",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge"
    )
    frame_belief_goal.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    for i in range(3):
        frame_belief_goal.grid_rowconfigure(i, weight=1)
    for j in range(3):
        frame_belief_goal.grid_columnconfigure(j, weight=1)

    belief_goal_entries = []
    for i in range(3):
        row = []
        for j in range(3):
            e = tk.Entry(frame_belief_goal, width=4, font=("Helvetica", 10),
                         justify="center", bg=FRAME_BG, bd=1, relief="ridge")
            e.grid(row=i, column=j, padx=1, pady=1, ipady=3, sticky="nsew")
            row.append(e)
        belief_goal_entries.append(row)

    # Partial Belief State Frame
    frame_partial_belief = tk.LabelFrame(
        frame_belief,
        text="Niềm Tin 1 Phần",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="ridge",
        padx=5,
        pady=5
    )
    frame_partial_belief.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
    for i in range(4):  # 4 rows: 3 for grid, 1 for button
        frame_partial_belief.grid_rowconfigure(i, weight=1)
    for j in range(3):
        frame_partial_belief.grid_columnconfigure(j, weight=1)

    partial_belief_entries = []
    for i in range(3):
        row = []
        for j in range(3):
            e = tk.Entry(frame_partial_belief, width=3, font=("Helvetica", 10),
                         justify="center", bg=FRAME_BG, bd=1, relief="ridge")
            e.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")
            row.append(e)
        partial_belief_entries.append(row)

    # Apply Button for Partial Belief
    btn_apply = tk.Button(
        frame_partial_belief,
        text="Áp Dụng",
        font=("Helvetica", 10),
        bg=BTN_RANDOM_BG,
        fg=BTN_FG,
        command=apply_partial_belief,
        width=10,
        height=1
    )
    btn_apply.grid(row=3, column=0, columnspan=3, pady=5, sticky="nsew")

    # Puzzle View Frame
    frame_view = tk.LabelFrame(
        frame_input,
        text="Puzzle Hiện Tại",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="groove"
    )
    frame_view.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

    info_frame = tk.Frame(frame_view, bg=FRAME_BG)
    info_frame.pack(pady=3)

    steps_count = tk.Label(info_frame, text="Bước: 0", font=("Helvetica", 9), bg=FRAME_BG, fg=TEXT_COLOR)
    steps_count.pack(side="left", padx=5)

    time_label = tk.Label(info_frame, text="Thời gian: 00:00", font=("Helvetica", 9), bg=FRAME_BG, fg=TEXT_COLOR)
    time_label.pack(side="left", padx=5)

    grid_frame = tk.Frame(frame_view, bg=FRAME_BG)
    grid_frame.pack(padx=5, pady=5)

    view_labels = []
    for i in range(3):
        row = []
        for j in range(3):
            lbl = tk.Label(
                grid_frame,
                text="",
                width=4,
                height=2,
                font=("Helvetica", 12, "bold"),
                bg=FRAME_BG,
                fg=TEXT_COLOR,
                bd=1,
                relief="ridge"
            )
            lbl.grid(row=i, column=j, padx=3, pady=3, sticky="nsew")
            row.append(lbl)
        view_labels.append(row)

    # Control Frame
    frame_control = tk.Frame(root, bg=BG_MAIN)
    frame_control.grid(row=0, column=1, sticky="n", padx=10, pady=10)

    # Algorithm Selection Frame
    alg_frame = tk.LabelFrame(
        frame_control,
        text="Chọn Thuật Toán",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="groove"
    )
    alg_frame.pack(padx=5, pady=5, fill="x")

    selected_algorithm = tk.StringVar()
    selected_algorithm.set(list(algorithms.keys())[0] if algorithms else "BFS")
    alg_dropdown = ttk.Combobox(
        alg_frame,
        textvariable=selected_algorithm,
        state="readonly",
        values=list(algorithms.keys()),
        font=("Helvetica", 10)
    )
    alg_dropdown.pack(padx=5, pady=5, fill="x")

    # Switch UI for Complex Search
    def switch_ui(event=None):
        if environment == "Complex Search":
            frame_normal.grid_forget()
            frame_belief.grid(row=0, column=0, sticky="nsew")

    if environment == "Complex Search":
        alg_dropdown.bind("<<ComboboxSelected>>", switch_ui)
        switch_ui()  # Initialize with belief UI for Complex Search

    # Display appropriate frame based on environment
    if environment == "Constraint-Based Search":
        frame_constraint.grid(row=0, column=0, sticky="nsew")
    elif environment == "Complex Search":
        frame_belief.grid(row=0, column=0, sticky="nsew")
    else:
        frame_normal.grid(row=0, column=0, sticky="nsew")

    # Control Buttons Frame
    ctrl_frame = tk.LabelFrame(
        frame_control,
        text="Điều Khiển",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="groove"
    )
    ctrl_frame.pack(padx=5, pady=5, fill="x")

    BUTTON_WIDTH = 12
    BUTTON_HEIGHT = 1

    btn_solve = tk.Button(
        ctrl_frame,
        text="Giải",
        font=("Helvetica", 10),
        bg=BTN_SOLVE_BG,
        fg=BTN_FG,
        command=solve_puzzle,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_solve.pack(padx=5, pady=3, fill="x")

    btn_pause = tk.Button(
        ctrl_frame,
        text="Tạm Dừng",
        font=("Helvetica", 10),
        bg=BTN_PAUSE_BG,
        fg=BTN_FG,
        command=pause_program,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_pause.pack(padx=5, pady=3, fill="x")

    btn_resume = tk.Button(
        ctrl_frame,
        text="Tiếp Tục",
        font=("Helvetica", 10),
        bg=BTN_RESUME_BG,
        fg=BTN_FG,
        command=resume_program,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_resume.pack(padx=5, pady=3, fill="x")

    btn_previous_step = tk.Button(
        ctrl_frame,
        text="Bước Trước",
        font=("Helvetica", 10),
        bg=BTN_STEP_BG,
        fg=BTN_FG,
        command=previous_step,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_previous_step.pack(padx=5, pady=3, fill="x")

    btn_next_step = tk.Button(
        ctrl_frame,
        text="Bước Sau",
        font=("Helvetica", 10),
        bg=BTN_STEP_BG,
        fg=BTN_FG,
        command=next_step,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_next_step.pack(padx=5, pady=3, fill="x")

    btn_reset = tk.Button(
        ctrl_frame,
        text="Reset",
        font=("Helvetica", 10),
        bg=BTN_RESET_BG,
        fg=BTN_FG,
        command=reset_program,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_reset.pack(padx=5, pady=3, fill="x")

    btn_export = tk.Button(
        ctrl_frame,
        text="Xuất File",
        font=("Helvetica", 10),
        bg=BTN_EXPORT_BG,
        fg=BTN_FG,
        command=export_file,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_export.pack(padx=5, pady=3, fill="x")

    btn_compare = tk.Button(
        ctrl_frame,
        text="So sánh thuật toán",
        font=("Helvetica", 10),
        bg=BTN_COMPARE_BG,
        fg=BTN_FG,
        command=lambda: show_comparison(algorithms, environment),
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_compare.pack(padx=5, pady=3, fill="x")

    btn_back = tk.Button(
        ctrl_frame,
        text="Quay Lại",
        font=("Helvetica", 10),
        bg=BTN_BACK_BG,
        fg=BTN_FG,
        command=back_to_selector,
        width=BUTTON_WIDTH,
        height=BUTTON_HEIGHT
    )
    btn_back.pack(padx=5, pady=3, fill="x")

    # Time Display Frame
    frame_time = tk.LabelFrame(
        ctrl_frame,
        text="Thời Gian Chạy",
        font=("Helvetica", 10, "bold"),
        bg=FRAME_BG,
        fg=ACCENT_COLOR,
        bd=2,
        relief="groove"
    )
    frame_time.pack(padx=5, pady=5, fill="both", expand=True)

    time_canvas = tk.Canvas(frame_time, bg=FRAME_BG)
    time_canvas.pack(side="left", fill="both", expand=True)

    time_scrollbar = tk.Scrollbar(frame_time, orient="vertical", command=time_canvas.yview)
    time_scrollbar.pack(side="right", fill="y")
    time_canvas.configure(yscrollcommand=time_scrollbar.set)

    time_container = tk.Frame(time_canvas, bg=FRAME_BG)
    time_canvas.create_window((0, 0), window=time_container, anchor="nw")

    def on_time_configure(event):
        time_canvas.configure(scrollregion=time_canvas.bbox("all"))
    time_container.bind("<Configure>", on_time_configure)

    # Steps Display Frame
    bottom_frame = tk.LabelFrame(
        root,
        text="Các Bước Giải",
        font=("Helvetica", 10, "bold"),
        bg=BG_MAIN,
        fg=ACCENT_COLOR,
        bd=2,
        relief="groove"
    )
    bottom_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

    canvas = tk.Canvas(bottom_frame, bg=BG_MAIN)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(bottom_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    steps_container = tk.Frame(canvas, bg=BG_MAIN)
    canvas.create_window((0, 0), window=steps_container, anchor="nw")

    for col in range(max_columns):
        steps_container.grid_columnconfigure(col, minsize=STEP_FRAME_SIZE + 10)

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    steps_container.bind("<Configure>", on_configure)

    return (root, initial_entries, goal_entries, constraint_initial_entries, belief_entries, belief_goal_entries, partial_belief_entries, view_labels,
            steps_container, time_container, steps_count, time_label, selected_algorithm, max_columns)