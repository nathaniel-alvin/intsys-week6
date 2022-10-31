import gym
import numpy as np
import random
from IPython.display import clear_output

"""
Problem Description:
There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions
"""

env = gym.make("Taxi-v3").env
env.render()

"""
This includes:
south, north, east, west, pickup, dropoff
"""
print(f"Number of actions: {env.action_space.n}")
print(f"Number of states: {env.observation_space}")

# initialize q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])


alpha = 0.1  # learning rate
gamma = 0.6  # discount factor
epsilon = 0.3
for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:  # explore
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)
        # env.render()
        # time.sleep(0.5)
        old_value = q_table[state, action]
        next_max = np.max([q_table[next_state]])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
        print(f"Time steps: {epochs}, Penalties: {penalties}")

print("Training finished \n")
