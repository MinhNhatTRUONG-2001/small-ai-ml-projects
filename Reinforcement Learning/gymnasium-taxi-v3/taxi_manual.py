import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
print(env.render()) #Passenger: blue; Destination: purple

done = False
truncated = False
while not done and not truncated:
    try:
        action = int(input('0=down, 1=up, 2=right, 3=left, 4=pickup, 5=drop: '))
        state, reward, done, truncated, info = env.step(action)
        print(info)
    except KeyError:
        print('Invalid action number. Please try again.')
        continue
    except ValueError:
        print('Invalid input for action. Please try again.')
        continue
    print(env.render())
    print(f'Observation: State={state}, Reward={reward}, Done={done}')