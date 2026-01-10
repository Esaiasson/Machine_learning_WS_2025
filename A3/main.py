import random

# grid_size:  17 x 10
borders = {
    "v_start": 0,
    "v_end": 16, 
    "h_start": 0,
    "h_end": 9   
}


obstacles = {
    "obs1": {
        "v_start":5,
        "v_end": 5,
        "h_start": 8,
        "h_end": 9
    },
    "obs2": {
        "v_start": 10,
        "v_end": 10,
        "h_start": 4,
        "h_end": 7 
    },
    "obs3": {
        "v_start": 11,
        "v_end": 11,
        "h_start": 4,
        "h_end": 4
    },
    "obs4": {
        "v_start": 14,
        "v_end": 16,
        "h_start": 4,
        "h_end": 4
    }
}



target = (12, 6)

action_space = [-1, 0, 1]
reward_state = -1 
action_space = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]



def generate_start():
    v_pos = 0
    h_pos = random.randrange(0, 10)
    
    return [v_pos, h_pos]


def change_velocity(velocity, action):
    vel = [velocity[0] + action[0], velocity[1] + action[1]]
    return vel



def episode():
    
    velocity = [0,0] #First item in the list is vertical velocity and second is horizontal velocity, i.e. item0: v_velocity, item1: h_velocity
    position = generate_start() #First item in the list is vertical position and second is horizontal position, i.e. item0: v_position, item1: h_position

    i = 0
    while i < 20:
        potential_action = random.sample(action_space, 1)[0]
        potential_vel = change_velocity(velocity, potential_action)
        print("Potential action: ", potential_action)
        print("Potential velocity: ", potential_vel)

        while (potential_vel[0] > 2 or potential_vel[1] > 2) or (potential_vel[0] == 0 and potential_vel[1] == 0): 
            potential_action = random.sample(action_space, 1)[0]
            potential_vel = change_velocity(velocity, potential_action)

        velocity = potential_vel
        
        print("Accepted action: ", potential_action)
        print("Accepted velocity: ", velocity)
        i = i+1
        
episode()
    
    
        
        
        




