import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999
MIN_EPSILON = .0001

def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with SARSA

        # while (not done):
        #     action = env.action_space.sample() # currently only performs a random action.
        #     obs,reward,terminated,truncated,info = env.step(action)
        #     episode_reward += reward # update episode reward
            
        #     done = terminated or truncated

        # Choose the initial action using epsilon-greedy policy
        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = max(range(env.action_space.n), key=lambda a: Q_table[(obs, a)])  # Exploit: best action

        ##########################################################
        # SARSA Algorithm
        while not done:
            # Take the action, observe the next state and reward
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # Update episode reward
            done = terminated or truncated

            # Choose the next action using epsilon-greedy policy
            if not done:
                if random.uniform(0, 1) < EPSILON:
                    next_action = env.action_space.sample()  # Explore
                else:
                    next_action = max(range(env.action_space.n), key=lambda a: Q_table[(next_obs, a)])  # Exploit
            else:
                next_action = None  # No next action if done

            # SARSA update rule
            if not done:
                Q_table[(obs, action)] = (
                    (1 - LEARNING_RATE) * Q_table[(obs, action)] +
                    LEARNING_RATE * (reward + DISCOUNT_FACTOR * Q_table[(next_obs, next_action)])
                )
            else:
                Q_table[(obs, action)] = (
                    (1 - LEARNING_RATE) * Q_table[(obs, action)] +
                    LEARNING_RATE * reward
                )

            # Move to the next state and action
            obs = next_obs
            action = next_action
        EPSILON = max(EPSILON * EPSILON_DECAY, MIN_EPSILON)
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open(f'Q_TABLE_SARSA.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################