import learning as learn
import game_environment as game
import time
import visualize_grid as vis
from matplotlib import pyplot as plt


results = {}

target = (6, 12)


obstacles = (
    (8, 5), (9, 5),
    (4, 10), (5, 10), (6, 10), (7, 10), (4, 11),
    (4, 14), (4, 15), (4, 16)
)

q_table = learn.intialize_q_table()


for episodes in range(1,10):
    s_a_pairs, route = game.episode(q_table, 0.1, target, obstacles, explore=True)
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)        




print("EXPLORING DONE")
time.sleep(3)

plt.ion()
fig, ax = plt.subplots()

for episodes in range(1,10000):
    s_a_pairs, route = game.episode(q_table, 0.3, target, obstacles, explore=False)
    g_values = learn.calculate_g(s_a_pairs)
    q_table = learn.update_q_table(q_table, g_values)
    if episodes in [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
        s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False)
        vis.show_grid(ax, obstacles,target, 17, 10, route)
        plt.pause(0.1)




s_a_pairs, route = game.episode(q_table, 0, target, obstacles, explore=False)
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



vis.show_grid(ax, obstacles,target, 17, 10, route)

plt.ioff()
plt.show()





