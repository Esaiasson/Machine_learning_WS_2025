import random

# grid_size:  10 x 17
walls = [[range(0,10)], [range(0,17)]]
 
obstacles = [
    [[8,5], [9,5]],
    [[4,10], [5,10], [6,10], [7,10], [4,11]],
    [[4,14],[4,15],[4,16]]
]

target = [12, 6]

reward_state = -1 
action_space = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]


def generate_start():
    v_pos = 0
    h_pos = random.randrange(0, 10)
    
    return [h_pos, v_pos]


def change_velocity(velocity, action):
    vel = [velocity[0] + action[0], velocity[1] + action[1]]
    return vel

visited_states = [(0,1), (1,2), ]



def episode():
    
    velocity = [0,0] #First item in the list is vertical velocity and second is horizontal velocity, i.e. item0: v_velocity, item1: h_velocity
    position = generate_start() #First item in the list is vertical position and second is horizontal position, i.e. item0: v_position, item1: h_position

    iteration= 0
    while iteration< 20:
        print("Current position: ", position)
        potential_action = random.sample(action_space, 1)[0]
        potential_vel = change_velocity(velocity, potential_action)
        print("Potential action: ", potential_action)
        print("Potential velocity: ", potential_vel)

        while (potential_vel[0] > abs(2) or potential_vel[1] > abs(2)) or (potential_vel[0] == 0 and potential_vel[1] == 0): 
            potential_action = random.sample(action_space, 1)[0]
            potential_vel = change_velocity(velocity, potential_action)

        velocity = potential_vel
        
        print("Accepted action: ", potential_action)
        print("Accepted velocity: ", velocity)
        
        
        potential_position = [position[0] + velocity[0], position[1] + velocity[1]]
        print("Potential position: ", potential_position)

        if((potential_position[0] not in walls[0]) or (potential_position[1] not in walls[1])):
            print("ohh no wall!, going back")
            position = generate_start()
        else:
            position = potential_action
        
        print(position)
        iteration= iteration+1
        
episode()
    
    
        
        
        




