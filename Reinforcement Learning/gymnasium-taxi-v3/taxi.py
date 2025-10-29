import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
print(env.render()) #Passenger: blue; Destination: purple

# Implement Q-learning
# Initialize Q-table Q, step size (learning rate) 'alpha', discount rate 'gamma' and 'epsilon'
num_train_iters = 20001
Q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.05
gamma = 0.95
epsilon = 0.05
for iter in range(num_train_iters):
    if iter % 1000 == 0:
       print(f'Train iteration: {iter}') 
    # Initialize state s
    state_curr, info = env.reset()
    while True:
        # Choose action a from state using policy derived from Q_table (epsilon-greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(Q_table[state_curr,:])
        # Take action and observe new state and reward
        state_new, reward, terminated, truncated, info = env.step(action)
        # Update Q(s, a) and s
        Q_table[state_curr, action] += alpha * (reward + gamma * np.max(Q_table[state_new, :]) - Q_table[state_curr, action])
        state_curr = state_new
        if terminated:
            break
env.close()

# Test Q-learning implementation
num_test_iters = 10
total_rewards = []
num_actions = []
pi = np.argmax(Q_table, axis=1) # Get action with max value for each state
for iter in range(num_test_iters):
    print(f'Test iteration: {iter}')
    state, info = env.reset()
    print(env.render())
    total_reward_curr = 0
    num_actions_curr = 0
    while True:
        action = pi[state]
        state, reward, terminated, truncated, info = env.step(action)
        print(env.render())
        total_reward_curr += reward
        num_actions_curr += 1
        if terminated:
            break
    total_rewards.append(total_reward_curr)
    num_actions.append(num_actions_curr)
print(f'Total rewards in {num_test_iters} iterations: {total_rewards}')
print(f'Average: {np.mean(total_rewards)}')
print('--------------------------------------------------------------')
print(f'Number of actions in {num_test_iters} iterations: {num_actions}')
print(f'Average: {np.mean(num_actions)}')