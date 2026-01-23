from run import run
from visualize_grid import plot_path_length
import time

results = {
    "layout_1": {},
    "layout_2": {},
    "layout_3": {},
}

target_1 = (6, 12)
obstacles_1 = (
    (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11),
    (4, 14), (4, 15), (4, 16)
)


target_2 = (0,16)
obstacles_2 = ()


target_3 = (9,12)
'''
obstacles_3 = (
    (6, 11), (7, 11), (8, 11), (9,11),
    (6,11), (6,12), (6,13)
)
'''

obstacles_3 = (
    (5, 6),
    (5, 5), (6,5), (7, 5), (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11), (4,12),
    (4, 14), (4, 15), (4, 16)
)


start = time.perf_counter()
starting_pos_lengths_layout1 = run("Layout_1", target_1, obstacles_1, 0.3)
end = time.perf_counter()
results["layout_1"]["path_lengths"] = starting_pos_lengths_layout1
results["layout_1"]["runtime"] = f"{end - start:0.4f}"

start = time.perf_counter()
starting_pos_lengths_layout2 = run("Layout_2", target_2, obstacles_2, 0.3)
end = time.perf_counter()
results["layout_2"]["path_lengths"] = starting_pos_lengths_layout2
results["layout_2"]["runtime"] = f"{end - start:0.4f}"

start = time.perf_counter()
starting_pos_lengths_layout3 = run("Layout_3", target_3, obstacles_3, 0.3)
end = time.perf_counter()
results["layout_3"]["path_lengths"] = starting_pos_lengths_layout3
results["layout_3"]["runtime"] = f"{end - start:0.4f}"

print(results)
plot_path_length(results)