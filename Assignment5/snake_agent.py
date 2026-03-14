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
        self.prev_dist_from_food = float('inf')
        self.prev_index = None
        self.s = None
        self.a = None

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
        adj_x_wall = self.wall_adj(head_x)
        
        '''
        * det if wall next to head in the y dir
        * 0 = wall to top
        * 1 = wall to bottom
        * 2 = no wall to top or bottom
        '''
        adj_y_wall = self.wall_adj(head_y)


        # distance in the x dir from the food
        # same logic as walls
        food_dir_y, food_dir_x = 2, 2
        food_x, food_y = state[3], state[4]
        if food_x < head_x:
            food_dir_x = 0 # to left of snake head
        elif food_x > head_x:
            food_dir_x = 1 # to right of snake head

        # distance in the y dir from the food
        if food_y < head_y:
            food_dir_y = 0 # to top of snake head
        elif food_y > head_y:
            food_dir_y = 1 # to bottom of snake head

        # for the next 4, 1 = there is a snake body to dir, 0 = no snake body in that dir
        body = state[2]
        adj_body_top, adj_body_btm, adj_body_left, adj_body_right = 0, 0, 0, 0
        next_top = (head_x, head_y - helper.GRID_SIZE)
        next_bottom = (head_x, head_y + helper.GRID_SIZE)
        next_right = (head_x + helper.GRID_SIZE, head_y)
        next_left = (head_x - helper.GRID_SIZE, head_y)
        if next_top in body:
            adj_body_top = 1
        if next_bottom in body:
            adj_body_btm = 1
        if next_left in body:
            adj_body_left = 1
        if next_right in body:
            adj_body_right = 1
            
        return [adj_x_wall, adj_y_wall, food_dir_x, food_dir_y, adj_body_top, adj_body_btm, adj_body_left, adj_body_right]

    '''
    Determine where wall is in reference to the snake head
    '''
    def wall_adj(self, head_coord):
        res = 2
        if head_coord - helper.BOARD_LIMIT_MIN == helper.GRID_SIZE: # elem B right before elem A
            res = 0
        elif helper.BOARD_LIMIT_MIN - head_coord == helper.GRID_SIZE: # elem B right after elem A
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
        s_index = self.helper_func(state) # save to be referenced when updating the model if traning
        # print('in agent action', s_index, q_vals)
        head_x, head_y, _, food_x, food_y = state
        dist_from_food = abs(food_x - head_x) + abs(food_y - head_y)
        
        # update the model before saving the new state info if this isn't the first time calling agent_action
        if self._train and self.s:
            reward = self.compute_reward(points, dead)
                
            # reward for moving towards the food
            if not dead:
                if dist_from_food < self.prev_dist_from_food:
                    reward += .5
                else:
                    reward -= 1 # didn't move closer
            else:
                reward = -5 # don't reward dying
            
            # update q-table
            lr = self.LPC / (self.LPC + self.N[*self.s, self.a]) # use the previous q-val to help det learning rate
            best_next = 0 if dead else max([self.Q[*s_index, a] for a in range(4)])
            # print(self.Q[self.prev_index])
            self.Q[*self.s, self.a] += lr * (reward + self.gamma*best_next - self.Q[*self.s, self.a])

        self.points = points
        self.prev_dist_from_food = dist_from_food

        # if dead no reason to keep playing
        if dead:
            self.reset()
            return None
        
        # choosing an action
        best_action = None # this will get updated!
        best_val = float('-inf')
        if self._train:
            for action in range(4):
                q_val = self.Q[*s_index, action]
                n_val = self.N[*s_index, action]

                # explore func: f(u, n) = u + k/n where n is the number of time the state is visited
                # going to assume Ne is a threshold for if we should explore
                if n_val < self.Ne:
                    val = 1
                else:
                    val = q_val
                
                if val >= best_val:
                    best_val = val
                    best_action = action

            # only need to update these when training
            self.s = s_index
            self.a = best_action

            # update the n-val to show we've explore this state another time
            self.N[*s_index, best_action] += 1
        else:
            # when in testing mode always choose the best action
            all_q_vals = [self.Q[*s_index, a] for a in range(4)]
            for a in range(4):
                if all_q_vals[a] > best_val:
                    best_val = all_q_vals[a]
                    best_action = a
        
        return best_action

