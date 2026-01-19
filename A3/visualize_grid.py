from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


obstacles = (
    (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11),
    (4, 14), (4, 15), (4, 16)
)

target = (6, 12)

path = [(3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (4,12), (5,12), (6,12)]


def show_grid(obstacle_tuples, target, rows, cols, path):
    
    grid = np.zeros((rows, cols))    

    for tuple in obstacle_tuples:
        grid[tuple[1], tuple[0]] = 1
    
    for tuple in path:
        grid[tuple[1], tuple[0]] = 3
        
    grid[target[1], target[0]] = 2


    cmap = ListedColormap([
        (1, 1, 1, 0.0),   
        (0, 0, 0, 1.0), 
        (0, 1, 0, 1), 
        (0, 0, 1, 0.7)
    ])
    plt.figure(figsize=(rows, cols))
    plt.imshow(grid, origin="lower", cmap=cmap)
    plt.grid(True, which="both", color="black", linewidth=1)
    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.show()
    
    

show_grid(obstacles, target, 17, 10, path)