import learning as learn
import game_environment as game

q_table = learn.intialize_q_table()

for episodes in range(1,4):
    s_a_pairs = game.episode()
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)
