from run import run
from visualize_grid import plot_path_length
import time

results = {
    "layout_1": {},
    "layout_2": {},
    "layout_3": {},
}

runtime_layouts = {

}

target_1 = (0,16)
obstacles_1 = ()

target_2 = (6, 12)
obstacles_2 = (
    (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11),
    (4, 14), (4, 15), (4, 16)
)


target_3 = (9,12)

obstacles_3 = (
    (5, 6),
    (7,7), (7,8),
    (5, 5), (6,5), (7, 5), (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11), (4,12),
    (4, 14), (4, 15), (4, 16)
)

epsilon_values = [0.3, 0.5, 0.7]

for epsilon in epsilon_values:
    print("Running episodes for epsilon:", epsilon)
    runtime_layouts[epsilon] = {}

    start = time.perf_counter()
    starting_pos_lengths_layout1 = run("Layout_1", target_1, obstacles_1, epsilon)
    end = time.perf_counter()
    results["layout_1"]["path_lengths"] = starting_pos_lengths_layout1
    runtime_layouts[epsilon]["layout_1"]= f"{end - start:0.4f}"

    start = time.perf_counter()
    starting_pos_lengths_layout2 = run("Layout_2", target_2, obstacles_2, epsilon)
    end = time.perf_counter()
    results["layout_2"]["path_lengths"] = starting_pos_lengths_layout2
    runtime_layouts[epsilon]["layout_2"] = f"{end - start:0.4f}"

    start = time.perf_counter()
    starting_pos_lengths_layout3 = run("Layout_3", target_3, obstacles_3, epsilon)
    end = time.perf_counter()
    results["layout_3"]["path_lengths"] = starting_pos_lengths_layout3
    runtime_layouts[epsilon]["layout_3"] = f"{end - start:0.4f}"

    plot_path_length(results, epsilon)
    
for ep in runtime_layouts.keys():
    print("Runtimes using epsilon: ", ep)
    print("Runtime for layout 1:", runtime_layouts[ep]["layout_1"])
    print("Runtime for layout 2:", runtime_layouts[ep]["layout_2"])
    print("Runtime for layout 3:",  runtime_layouts[ep]["layout_3"])