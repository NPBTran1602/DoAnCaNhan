import tkinter as tk

ENVIRONMENTS = {
    "Informed Search": ["A* Search", "Greedy", "IDA* Search"],
    "Uninformed Search": ["BFS", "DFS", "UCS", "IDDFS"],
    "Constraint-Based Search": ["Backtracking Search", "Forward Checking", "Constraint Propagation (AC3)"],
    "Local Search": ["Simple Hill Climbing", "Beam Search", "Simulated Annealing", "Stochastic HC", "Steepest Ascent HC", "Genetic Algorithm"],
    "Complex Search": ["AND-OR Search", "Belief State Search", "Search with Partial Observation"],
    "Reinforcement Learning Search": ["Q-Learning"]
}

def select_environment(initialize_app):
    root = tk.Tk()
    root.title("8-Puzzle Environment Selector")
    root.geometry("600x400")
    root.configure(bg="#F3E5F5")

    # Main frame to hold all content
    main_frame = tk.Frame(root, bg="#F3E5F5")
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)

    # Title label
    label = tk.Label(
        main_frame,
        text="Select Search Environment",
        font=("Helvetica", 16, "bold"),
        bg="#F3E5F5",
        fg="#4A148C"
    )
    label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # Function to launch the selected environment
    def launch_environment(env_name):
        root.destroy()
        initialize_app(allowed_algorithms=ENVIRONMENTS[env_name], environment=env_name)

    # Create buttons for each environment in a 2-column grid
    for idx, env_name in enumerate(ENVIRONMENTS):
        row = (idx // 2) + 1  # Start from row 1 to leave space for the title
        col = idx % 2
        btn = tk.Button(
            main_frame,
            text=env_name,
            font=("Helvetica", 12),
            bg="#CE93D8",
            fg="#4A148C",
            activebackground="#BA68C8",
            activeforeground="#4A148C",
            command=lambda e=env_name: launch_environment(e),
            width=25,
            height=2,
            relief="raised",
            bd=2
        )
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

    # Configure grid weights for responsiveness
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)
    for i in range(1, 4):  # Up to 4 rows to accommodate 6 buttons
        main_frame.grid_rowconfigure(i, weight=1)

    root.mainloop()