import learning as learn
import game_environment as game
import time

q_table = learn.intialize_q_table()

for episodes in range(1,1000):
    s_a_pairs, route = game.episode(q_table, 0.1,explore=True)
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)



print("EXPLORING DONE")
time.sleep(3)


for episodes in range(1,1000):
    s_a_pairs, route = game.episode(q_table, 0.1, explore=False)
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)


s_a_pairs, route = game.episode(q_table, 0, explore=False)
print(route)

print("EPISODE DONE!!!!")  
print(q_table[((0,0), (0,0))])
print(q_table[((1,0), (0,0))])
print(q_table[((2,0), (0,0))])
print(q_table[((3,0), (0,0))])
print(q_table[((4,0), (0,0))])
print(q_table[((5,0), (0,0))])
print(q_table[((6,0), (0,0))])
print(q_table[((7,0), (0,0))])
print(q_table[((8,0), (0,0))])
print(q_table[((9,0), (0,0))])






