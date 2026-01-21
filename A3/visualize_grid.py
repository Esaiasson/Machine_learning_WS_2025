from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os


cwd = os.getcwd() 

obstacles = (
    (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11),
    (4, 14), (4, 15), (4, 16)
)

target = (6, 12)

visited = [(3,0), (3,1), (3,3), (3,6), (3,9), (3,11), (4,12), (6,12)]

path = [(3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (4,12), (5,12), (6,12)]

    
def calculate_path(visited):
    path = []

    for i, state in enumerate(visited):
        if i+1 != len(visited):
            initial_position = state
            final_position = visited[i+1]
            
            start_h = min(initial_position[0], final_position[0])
            end_h = max(initial_position[0], final_position[0])

            h_moves = range(start_h, end_h+1)  # horizonatal possible moves
            for move in h_moves:
                state = (move, initial_position[1])
                path.append(state)

            start_v = min(initial_position[1], final_position[1])
            end_v = max(initial_position[1], final_position[1])

            v_moves = range(start_v, end_v+1)
            for move in v_moves:
                state = (final_position[0], move)
                path.append(state)

    return path



def show_grid(ax, obstacle_tuples, target, rows, cols, visited, plotname=None, save=False, folder=None):
    
    path = calculate_path(visited)
    traversed = list(set(path) - set(visited))
    
    grid = np.zeros((rows, cols))    

    for tuple in obstacle_tuples:
        grid[tuple[1], tuple[0]] = 1
    
    for tuple in visited:
        grid[tuple[1], tuple[0]] = 3
        
    for tuple in traversed:
        grid[tuple[1], tuple[0]] = 4
        
    grid[target[1], target[0]] = 2



    cmap = ListedColormap([
        (1, 1, 1, 0.0),  #Empty cells 
        (0, 0, 0, 1.0), # Obstacles 
        (0, 1, 0, 1), # Target
        (0, 0, 1, 0.7), # Visited
        (0, 0, 1, 0.3) #Traversed
    ])

    plt.ion()
    #fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.clear()
    ax.imshow(grid, origin="lower", cmap=cmap)
    
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))

    
    ax.grid(True, which="both", color="black", linewidth=1)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(plotname)
    plt.show()

    if save == True:
        plt.savefig(f'{cwd}/result_graphs/{folder}/{plotname}.png')
    
    

#show_grid(obstacles, target, 17, 10, visited, path)