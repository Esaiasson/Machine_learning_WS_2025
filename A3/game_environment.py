import random

# grid_size:  10 x 17
walls = [list(range(0, 10)), list(range(0, 17))]
path = [list(range(0, 10)), list(range(0, 17))]

obstacles = [
    [8, 5], [9, 5],
    [4, 10], [5, 10], [6, 10], [7, 10], [4, 11],
    [4, 14], [4, 15], [4, 16]
]

target = [12, 6]

reward_state = -1
action_space = [[-1, -1], [-1, 0], [-1, 1], [0, -1],
                [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]


def generate_start():
    v_pos = 0
    h_pos = random.randrange(0, 10)

    return [h_pos, v_pos]


def change_velocity(velocity, action):
    vel = [velocity[0] + action[0], velocity[1] + action[1]]
    return vel


def path_validity(velocity, initial_position, final_position, obstacles):
    path = []

    start_h = min(initial_position[0], final_position[0])
    end_h = max(initial_position[0], final_position[0])

    h_moves = range(start_h, end_h+1)  # horizonatal possible moves
    for move in h_moves:
        state = [move, initial_position[1]]
        if state in obstacles:
            print('Oh no! obstacle in the path')
            return False
        path.append(state)

    start_v = min(initial_position[1], final_position[1])
    end_v = max(initial_position[1], final_position[1])

    v_moves = range(start_v, end_v+1)
    for move in v_moves:
        state = [final_position[0], move]
        if state in obstacles:
            print('Oh no! obstacle in the path')
            return False
        path.append(state)

    print("Path:", path)
    return True


def episode():

    # First item in the list is vertical velocity and second is horizontal velocity, i.e. item0: v_velocity, item1: h_velocity
    velocity = [0, 0]
    # First item in the list is vertical position and second is horizontal position, i.e. item0: v_position, item1: h_position
    position = generate_start()
    path = [()]

    iteration = 0
    while iteration < 20:
        print("Current position: ", position)
        potential_action = random.sample(action_space, 1)[0]
        potential_vel = change_velocity(velocity, potential_action)
        print("Potential action: ", potential_action)
        print("Potential velocity: ", potential_vel)

        while ((abs(potential_vel[0]) > 2) or (abs(potential_vel[1]) > 2)) or (potential_vel[0] == 0 and potential_vel[1] == 0):
            potential_action = random.sample(action_space, 1)[0]
            potential_vel = change_velocity(velocity, potential_action)

        velocity = potential_vel

        print("Accepted action: ", potential_action)
        print("Accepted velocity: ", velocity)

        potential_position = [position[0] +
                              velocity[0], position[1] + velocity[1]]
        print("Potential position: ", potential_position)

        # Validity path ckecks
        wall_check = (potential_position[0] in walls[0]) and (potential_position[1] in walls[1])
        obstacles_check = potential_position not in obstacles
        path_check = path_validity(velocity, position, potential_position, obstacles)

        if (wall_check and obstacles_check and path_check):
            position = potential_position
            path.append(position[0])
            path.append(position[1])

        else:
            # print('hpos',potential_position[0], 'vpos',potential_position[1])
            print('Rejected Position: ', potential_position)
            print('wall start', walls[0])
            print("ohh no wall!, going back")
            position = generate_start()
            velocity = [0, 0]

        print('Accepted position: ', position)

        if (position == target):
            print('Yup! Hit the target :)')
            break
        iteration = iteration+1

    print("Path:")
    for i in range(len(path)):
        print(f"X: {path[i]}, ")


episode()
