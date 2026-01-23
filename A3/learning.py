import itertools

def intialize_q_table():
    
    q_table = {}

    horisontal = range(0,10)
    vertical = range(0,17)
    velocities = [-2, -1, 0, 1, 2]
    action_space = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))

    all_positions = list(itertools.product(horisontal, vertical))
    all_velocities = list(itertools.product(velocities, velocities))
    
    all_states = list(itertools.product(all_positions, all_velocities))

    for state in all_states:
        q_table[state] = {}
        for action in action_space:
            q_table[state][action] = {"sum": 0, "visits": 0, "mean": 0}
            
    return q_table

def calculate_g_old(s_a_pairs):
    gamma = 0.9
    g_values = {}
    
    for i, s_a in enumerate(s_a_pairs):
        if s_a not in s_a_pairs[0:i]: #First pass method
            n = len(s_a_pairs[i+1:])
            if i+1 < len(s_a_pairs): #i +1 since enumerate starts with 0
                g = -((1 - (gamma**n))/(1-gamma)) # Simplified calculation since we have a constant reward
                g_values[s_a] = g
            else:
                g = 0 #If the state, action pair is the last in the sequence we know that the reward is 0 since the target is the only terminal node 
                g_values[s_a] = g
    # print(g_values)
    return g_values


def calculate_g(s_a_pairs):
    gamma = 0.9
    g_values = {}
    g = 0    
    visited = set()

    for i, s_a in enumerate(reversed(s_a_pairs)):
        if i != 0:
            g = gamma*g - 1
            if s_a not in visited:
                visited.add(s_a)
                g_values[s_a] = g
    return g_values

        



def update_q_table(q_table, g_values):
    for key, g in g_values.items():
        state = key[0]
        action = key[1]
        q_table[state][action]["sum"] += g
        q_table[state][action]["visits"] += 1
        q_table[state][action]["mean"] = q_table[state][action]["sum"]/q_table[state][action]["visits"]

    return q_table

q_table = intialize_q_table()


