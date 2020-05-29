# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

#########################################################################################################################################################################################
class CabDriver():

    #####################################################################################################################################################################################
    def __init__(self):
        """initialise your state and define your action space and state space"""

        """There’ll never be requests of the sort where pickup and drop locations are the same. So, the action space A will be: (𝑚−1)∗𝑚 + 1 for m locations. 
           Each action will be a tuple of size 2. You can define action space as below:
            • pick up and drop locations (𝑝,𝑞) where p and q both take a value between 1 and m (considered 0 and m-1);
            • (0,0) tuple that represents ’no-ride’ action."""

        # Consider p and q both take a value between 0 and m-1 rather than 1 and m to keep it in sync with state space
        self.action_space = [(0,0)] + [(i, j) for i in range(m) for j in range(m) if i != j]

        """The state space is defined by the driver’s current location along with the time components (hour-of-the-day and the day-of-the-week). A state is defined by three variables:
            𝑠=𝑋𝑖𝑇𝑗𝐷𝑘 𝑤ℎ𝑒𝑟𝑒 𝑖=0…𝑚−1;𝑗=0….𝑡−1;𝑘=0…..𝑑−1
            Where 𝑋𝑖 represents a driver’s current location, 𝑇𝑗 represents time component (more specifically hour of the day), 𝐷𝑘 represents the day of the week"""

        self.state_space = [(x, y, z) for x in range(m) for y in range(t) for z in range(d)]

        """ Take a radom choice from state space to initialize """
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

    #####################################################################################################################################################################################
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]] = 1 # Fetch location in the city
        state_encod[m+state[1]] = 1 # Fetch time of the day
        state_encod[m+t+state[2]] = 1 # Fetch day of the week

        return state_encod

    #####################################################################################################################################################################################
    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod

    #####################################################################################################################################################################################
    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""

        location = state[0]

        location_lambda = [2, 12, 4, 7, 8]
        requests = np.random.poisson(location_lambda[location])

        # if location == 0:
        #     requests = np.random.poisson(2)

        # The upper limit on these customers’ requests (𝑝,𝑞) is 15.
        if requests >15:
            requests =15

        """ Apart from these requests, the driver always has the option to go ‘offline’ (accept no ride). The no-ride action just moves the time component by 1 hour. 
            So, you need to append (0,0) action to the customer requests."""

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request but it is a possible action (no ride)
        
        actions = [self.action_space[i] for i in possible_actions_index]

        # actions.append([0,0])

        return possible_actions_index, actions   

    #####################################################################################################################################################################################
    # def reward_func(self, state, action, Time_matrix):
    #     """Takes in state, action and Time-matrix and returns the reward"""
    #     return reward

    #####################################################################################################################################################################################
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""

        next_state = []
        
        curr_loc = state[0]     # Current location in the city
        curr_time = state[1]    # Current time of the day
        curr_day = state[2]     # Current day of the week
        pickup_loc = action[0]  # Pickup location
        drop_loc = action[1]    # Drop location

        total_time   = 0
        time_currloc_to_pickuploc = 0
        wait_time = 0
        ride_time = 0

        # If no-ride action, this means move time component by 1 hour meaning 1 hour wait time till next request and next_loc is same as curr_loc
        if ((pickup_loc == 0) and (drop_loc == 0)):
            wait_time = 1       # Wait time is 1 hour
            next_loc = curr_loc # Location remains same

        # If pickup location is same as current location
        elif (curr_loc == pickup_loc): 
            # Get time to travel from pickup location to drop location considering current time and day
            ride_time = Time_matrix[pickup_loc][drop_loc][curr_time][curr_day]
            
            next_loc = drop_loc # Drop location becomes next location

        # If pickup location is different from current location
        else:
            # Get time taken to travel from current location to pickup location
            time_currloc_to_pickuploc = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]

            # Get revised time and day
            new_time, new_day = self.revise_time_day(curr_time, curr_day, time_currloc_to_pickuploc)

            # Get time to travel from pickup location to drop location considering revised time and day
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]

            next_loc  = drop_loc # Drop location becomes next location

        # Calculate total time taken considering wait time, transit time (from current location to pickup location) and ride time
        total_time = (wait_time + time_currloc_to_pickuploc + ride_time)

        # Get revised time and day after total time taken
        next_time, next_day = self.revise_time_day(curr_time, curr_day, total_time)

        next_state = (next_loc, next_time, next_day)
      
        return next_state, ride_time, total_time

    #####################################################################################################################################################################################
    def reward_func(self, ride_time, total_time):
        """Takes in ride_time and total_time (ride time + wait time + travel time from current location to pickup location)"""

        """
            In case of action (0,0) - no ride action, ride_time will be input as 0 and total_time will just be 1 (wait_time)
            In case of an action when driver location and pickup location is same - total_time will just be ride_time
            In case of an action when driver location and pickup location is different - total_time will be ride_time + time_currloc_to_pickuploc
            See next_state_func for more details
        """
        
        reward = (R * ride_time) - (C * total_time)

        return reward

    #####################################################################################################################################################################################
    def step(self, state, action, Time_matrix):
        """ Takes in current state and action as input and return next state, reward and total time for this step."""

        next_state, ride_time, total_time = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(ride_time, total_time)
        
        return next_state, reward, total_time

    #####################################################################################################################################################################################
    def revise_time_day(self, time, day, travel_time):
        """Takes in time, day and travel time either to reach pickup location from current location or to reach drop location from pickup location
           and return new time and new day (if day changes)."""
        
        travel_time = int(travel_time) # Consider only integer part as travel being considered only in hourly intervals

        # If travel within same day
        if (time + travel_time) < 24:
            new_time = time + travel_time
            new_day = day
        # If travel spreading to next day
        else:
            # Convert time between 0-23 range
            new_time = (time + travel_time) % 24
            
            # Get number of days
            num_days = (time + travel_time) // 24
            
            # Convert day between 0-6 range
            new_day = (day + num_days ) % 7

        return new_time, new_day

    #####################################################################################################################################################################################
    def reset(self):
        return self.action_space, self.state_space, self.state_init

#########################################################################################################################################################################################