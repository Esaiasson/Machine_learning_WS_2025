from run import run
import time



def calc_mean_path(layout, results, path_lengths):
    sum = 0
    for key in path_lengths:
        sum = sum  + path_lengths[key]
    results[layout]["mean_path"] = sum/len(path_lengths)

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
obstacles_3 = (
    (6, 11), (7, 11), (8, 11), (9,11),
    (4,15), (5,15), (6,15),
    (6,11), (6,12), (6,13), (6,14)
    
)


start = time.perf_counter()
starting_pos_lengths_layout1 = run("Layout_1", target_1, obstacles_1, 0.3)
end = time.perf_counter()
results["layout_1"]["runtime"] = f"{end - start:0.4f}"
calc_mean_path("layout_1", results, starting_pos_lengths_layout1)

start = time.perf_counter()
starting_pos_lengths_layout2 = run("Layout_2", target_2, obstacles_2, 0.3)
end = time.perf_counter()
results["layout_2"]["runtime"] = f"{end - start:0.4f}"
calc_mean_path("layout_2", results, starting_pos_lengths_layout2)

start = time.perf_counter()
starting_pos_lengths_layout3 = run("Layout_3", target_3, obstacles_3, 0.3)
end = time.perf_counter()
results["layout_3"]["runtime"] = f"{end - start:0.4f}"
calc_mean_path("layout_3", results, starting_pos_lengths_layout3)

print(results)