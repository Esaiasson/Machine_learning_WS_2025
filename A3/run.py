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
def calc_mean_path(episode, episode_lengths, path_lengths):
    sum = 0
    for key in path_lengths:
        sum = sum  + path_lengths[key]
    episode_lengths[episode] = sum/len(path_lengths)
    return episode_lengths


def run(run_name, target, obstacles, gamma):

    episode_lengths={}
    
    q_table = learn.intialize_q_table()


    plt.ion()
    fig, ax = plt.subplots()

    for episodes in range(1,50001):
        s_a_pairs, route = game.episode(q_table, gamma, target, obstacles, explore=False)
        print(f"Episode: {episodes}, current path length: {len(route)}")
        g_values = learn.calculate_g(s_a_pairs)
        q_table = learn.update_q_table(q_table, g_values)
        if (episodes % 1000 == 0) or (episodes == 1):
            if episodes != 1:
                s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False)
                vis.show_grid(ax, obstacles,target, 17, 10, route, f"Episode: {episodes}")
                plt.pause(0.1)

            for i in range(0,10):
                starting_pos_lengths={}
                s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False, starting_pos=(i,0))
                starting_pos_lengths[(i,0)] = len(route)
            episode_lengths = calc_mean_path(episodes, episode_lengths, starting_pos_lengths)



    for i in range(0,10):
        s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False, starting_pos=(i,0))
        vis.show_grid(ax, obstacles,target, 17, 10, route, f"Optimal_policy_for_start_({i},0)_gamma_{gamma}", True, run_name)

    plt.ioff()
    plt.close(fig)
    
    return episode_lengths





