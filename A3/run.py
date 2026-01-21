import learning as learn
import game_environment as game
import time
import visualize_grid as vis
from matplotlib import pyplot as plt


'''
for episodes in range(1,100):
    s_a_pairs, route = game.episode(q_table, 0.1, target, obstacles, explore=True)
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)        
'''



def run(run_name, target, obstacles, gamma):

    starting_pos_lengths={}
    
    q_table = learn.intialize_q_table()


    plt.ion()
    fig, ax = plt.subplots()

    for episodes in range(1,50000):
        s_a_pairs, route = game.episode(q_table, gamma, target, obstacles, explore=False)
        print(f"Episode: {episodes}, current path length: {len(route)}")
        g_values = learn.calculate_g(s_a_pairs)
        q_table = learn.update_q_table(q_table, g_values)
        if episodes in [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000]:
            s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False)
            vis.show_grid(ax, obstacles,target, 17, 10, route, f"Episode: {episodes}")
            plt.pause(0.1)




    for i in range(0,10):
        s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False, starting_pos=(i,0))
        vis.show_grid(ax, obstacles,target, 17, 10, route, f"Optimal_policy_for_start_({i},0)_gamma_{gamma}", True, run_name)
        starting_pos_lengths[(i,0)] = len(route)

    plt.ioff()
    plt.close(fig)
    
    return starting_pos_lengths





