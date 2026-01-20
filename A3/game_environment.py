import random
import numpy as np
import learning as test


# Environment Setting
# grid_size:  10 x 17
walls = (list(range(0, 10)), list(range(0, 17)))
path = (list(range(0, 10)), list(range(0, 17)))



reward_state = -1

action_space = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


def generate_start():
    v_pos = 0
    h_pos = random.randrange(0, 10)

    return (h_pos, v_pos)


def change_velocity(velocity, action):

    vel = (velocity[0] + action[0], velocity[1] + action[1])
    return vel


def path_validity(velocity, initial_position, final_position, obstacles):
    path = []

    start_h = min(initial_position[0], final_position[0])
    end_h = max(initial_position[0], final_position[0])

    h_moves = range(start_h, end_h+1)  # horizonatal possible moves
    for move in h_moves:
        state = (move, initial_position[1])
        if state in obstacles:
            #print('Oh no! obstacle in the path')
            return False
        path.append(state)

    start_v = min(initial_position[1], final_position[1])
    end_v = max(initial_position[1], final_position[1])

    v_moves = range(start_v, end_v+1)
    for move in v_moves:
        state = (final_position[0], move)
        if state in obstacles:
            #print('Oh no! obstacle in the path')
            return False
        path.append(state)

    #print("Path:", path)
    return True

def validate_action(velocity,action):
    
    potential_vel = change_velocity(velocity, action)

    if ((abs(potential_vel[0]) > 2) or (abs(potential_vel[1]) > 2)) or (potential_vel[0] == 0 and potential_vel[1] == 0):
        return False
    else:
        return True


def select_next_action(state, q_table, epsilon):
    '''
    Docstring for select_next_action
    
    :param state: Description
    :param q_table: Description
    :param epsilon: Description
    '''
    best_expected_return = float('-inf') 
    all_actions = []
    possible_actions = q_table[state] 
    allowed_actions = {}
    
    
    for action in possible_actions.keys():
        if validate_action(state[1], action) == True:
            allowed_actions[action] = possible_actions[action]
        else:
            #Action is not possible to perform in the current state, without validating constraints
            pass


    for action in allowed_actions.keys():
        if allowed_actions[action]["mean"] > best_expected_return:
            best_expected_return = allowed_actions[action]["mean"]
            all_actions.insert(0, action)
        elif allowed_actions[action]["mean"] == best_expected_return:
            result = random.choice([0,1])
            if result == 0:
                #The alredy existing best action "won" the tie
                all_actions.append(action)
            else: 
                #The new action "won" the tie
                best_expected_return = allowed_actions[action]["mean"]
                all_actions.insert(0, action)

        else: 
            all_actions.append(action)


    probability_best = 1 - epsilon
    nbr_of_possible_actions = len(all_actions)
    probability_rest = [(epsilon/(nbr_of_possible_actions-1))]
    probability_weight = probability_rest*(nbr_of_possible_actions-1)

    probability_weight.insert(0,probability_best)

    action = random.choices(all_actions, k=1, weights=probability_weight)
    return action[0]


    


def initial_state_action():
    
    # item0: v_velocity, item1: h_velocity
    intial_velocity = (0, 0)
    # item0: v_position, item1: h_position
    intial_position = generate_start()

    #print("Current position: ", intial_position)
    # initialize potential next position and velocity
    potential_action = random.sample(action_space, 1)[0]
    potential_vel = change_velocity(intial_velocity, potential_action)
    

    # check both
    while ((abs(potential_vel[0]) > 2) or (abs(potential_vel[1]) > 2)) or (potential_vel[0] == 0 and potential_vel[1] == 0):
        # check max velocity < 2
        # check next velocity not null
        potential_action = random.sample(action_space, 1)[0]
        potential_vel = change_velocity(intial_velocity, potential_action)

    # update velocity
    action = potential_action
    state = (intial_position, intial_velocity)
    return ((state), action)


def explore_action(state):
    
    potential_action = random.sample(action_space, 1)[0]
    potential_vel = change_velocity(state[1], potential_action)

    # check both
    while ((abs(potential_vel[0]) > 2) or (abs(potential_vel[1]) > 2)) or (potential_vel[0] == 0 and potential_vel[1] == 0):
        # check max velocity < 2
        # check next velocity not null
        potential_action = random.sample(action_space, 1)[0]
        potential_vel = change_velocity(state[1], potential_action)
        
    return potential_action


def episode(q_table, epsilon, target, obstacles, explore=True):

    state_actions = []
    route = []   

    intial_state_action = initial_state_action()
    state_actions.append(intial_state_action)
    
    position = intial_state_action[0][0]
    route.append(position)
    velocity = intial_state_action[0][1]
    action = intial_state_action[1]
    
    iteration = 0
    while True:
        
        velocity = change_velocity(velocity, action)

        potential_position = (position[0] + velocity[0], position[1] + velocity[1])
        #print("Potential position: ", potential_position)

        # Validity path ckecks
        wall_check = (potential_position[0] in walls[0]) and (potential_position[1] in walls[1])
        obstacles_check = potential_position not in obstacles
        path_check = path_validity(velocity, position, potential_position, obstacles)

        if (wall_check and obstacles_check and path_check):
            # update position
            position = potential_position

            # update travelled_states
            #state = (position, velocity)
            #travelled_states.append(state)


        else:
            # print('hpos',potential_position[0], 'vpos',potential_position[1])
            #print('Rejected Position: ', potential_position)
            #print('wall start', walls[0])
            #print("ohh no wall!, going back")

            # if invalid path => reset
            position = generate_start()
            velocity = (0, 0)
            #state = (position, velocity)
            #travelled_states.append(state)

        #print('Accepted position: ', position)

        if (position == target):
            # target not hit => end
            print('Yup! Hit the target :)')
            route.append(position)
            break
        
        state = (position, velocity)
        if explore == True:
            action = explore_action(state)
        else: 
            action = select_next_action(state, q_table, epsilon)

        route.append(position)
        state_actions.append((state, action))
        iteration += 1 
        print((state, action), "itt:", iteration)

    return state_actions, route


#episode()

#state=((0,0),(2,2))
#q_table = test.intialize_q_table()

#select_next_action(state, q_table, 0.1)