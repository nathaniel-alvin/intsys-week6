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
epsilon = 1
decay = 0.005

NUM_EPISODES = 100000
MAX_STEPS = 99  # per episode

for i in range(NUM_EPISODES):
    state = env.reset()

    penalties, reward = 0, 0
    done = False

    for s in range(MAX_STEPS):
        if random.uniform(0, 1) < epsilon:  # explore
            action = env.action_space.sample()
        else:  # exploit
            action = np.argmax(q_table[state])

        next_state, reward, _, _ = env.step(action)
        # env.render()
        # time.sleep(0.5)
        old_value = q_table[state, action]
        next_max = np.max([q_table[next_state]])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state

        if done == True:
            break

    # Decrease epsilon
    epsilon = np.exp(-decay * i)

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i + 1}")
        print(f"Penalties: {penalties}")
    state = env.reset()
print("Training finished \n")
input("Press Enter to watch trained agent...")

state = env.reset()
done = False
rewards = 0

for s in range(MAX_STEPS):
    print("Trained Agent")
    print(f"Step {s+1}")
    action = np.argmax(q_table[state, :])

    new_state, reward, _, _ = env.step(action)
    rewards += reward
    env.render()
    print(f"score: {rewards}")
    state = new_state

    if done == True:
        break

env.close()
