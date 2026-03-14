import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #   the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
        self.q_index = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    #   state=[snake_head_x, snake_head_y, snake_body[], food_x, food_y]
    def helper_func(self, state):
        # print("IN helper_func", state)
        '''
        * det if wall next to head in the x dir
        * 0 = wall to left
        * 1 = wall to right
        * 2 = no wall to left or right
        '''
        head_x, head_y = state[0], state[1]
        adj_x_wall = self.elem_adj(head_x, helper.BOARD_LIMIT_MIN, helper.BOARD_LIMIT_MAX)
        
        '''
        * det if wall next to head in the y dir
        * 0 = wall to top
        * 1 = wall to bottom
        * 2 = no wall to top or bottom
        '''
        adj_y_wall = self.elem_adj(head_y, helper.BOARD_LIMIT_MIN, helper.BOARD_LIMIT_MAX)


        # distance in the x dir from the food
        # same logic as walls
        # food_dir_x = abs(state[3] - head_x)//40
        food_dir_y, food_dir_x = 2, 2
        food_x_diff = state[3] - head_x
        if food_x_diff < 0:
            food_dir_x = 0 # to left of snake head
        elif food_x_diff > 0:
            food_dir_x = 1 # to right of snake head
        # food_dir_x = self.elem_adj(head_x, state[3])

        # distance in the y dir from the food
        food_y_diff = state[4] - head_x
        if food_y_diff < 0:
            food_dir_y = 0 # to top of snake head
        elif food_y_diff > 0:
            food_dir_y = 1 # to bottom of snake head
        # food_dir_y = self.elem_adj(head_y, state[4])

        # for the next 4, 1 = there is a snake body to dir, 0 = no snake body in that dir
        body = state[2]
        adj_body_top, adj_body_btm, adj_body_left, adj_body_right = 0, 0, 0, 0
        for piece in body:
            x, y = piece
            x_diff, y_diff = x - head_x, y - head_y
            if y_diff == helper.GRID_SIZE:
                adj_body_top = 1
            if y_diff == -helper.GRID_SIZE:
                adj_body_btm = 1
            
            if x_diff == helper.GRID_SIZE:
                adj_body_right = 1
            if x_diff == -helper.GRID_SIZE:
                adj_body_left = 1
            
            # if any any point we are fully surrounded stop looping
            if adj_body_top and adj_body_btm and adj_body_left and adj_body_right:
                break

        return [adj_x_wall, adj_y_wall, food_dir_x, food_dir_y, adj_body_top, adj_body_btm, adj_body_left, adj_body_right]

    '''
    Determine where elem A is in reference to elem B
    * elem C is optional if there is a different after limit. otherwise set to elem b
    Return:
    * 0 if the elem B is right before elem A -> left or above
    * 1 if the elem B is right after elem A -> right or below
    * 2 if neither
    '''
    def elem_adj(self, elem_a, elem_b, elem_c = None):
        elem_c = elem_b if not elem_c else elem_c
        res = 2
        if elem_a - elem_b == helper.GRID_SIZE: # elem B right before elem A
            res = 0
        elif elem_b - elem_a == helper.GRID_SIZE: # elem B right after elem A
            res = 1
        return res

    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    '''
    #   This is the code you need to write.

    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above.

    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make.

    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.

    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.

    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.

    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 

    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    '''
    def agent_action(self, state, points, dead):
        # print("IN AGENT_ACTION")
        q_index = self.helper_func(state) # save to be referenced when updating the model if traning
        # print('qindex', self.q_index)
        q_vals = [self.Q[*q_index, action] for action in range(4)]
        best_q_val = float('-inf')
        action = None
        allEqual = True # checks if all actions have equal q-vals
        for a in range(4):
            if q_vals[a] > best_q_val:
                best_q_val = q_vals[a]
                action = a
                allEqual = False
        
        if self._train:
            self.update_model(q_index, points, dead)

        # save the state and action for training the agent
        self.s = q_index
        self.a = action
        self.points = points
        #UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        if allEqual:
            # choose a random action if all have the same q-vals
            return random.choice([x for x in range(4)])
        return action
        # return random.choices([action, *unoptimal_actions], weights=[90, 10, 10, 10])[0]

    def update_model(self, new_q_idx, points, dead):
        lr = 0.7
        if not self.s:
            q_val = 0
        else:
            q_val = self.Q[*self.s, self.a]
        best_next_val = max([self.Q[*new_q_idx, a] for a in range(4)])
        sample = self.compute_reward(points, dead) + self.gamma*best_next_val
        new_q_val = q_val + lr*(sample - q_val)
        if self.s:
            self.Q[*self.s, self.a] = new_q_val

