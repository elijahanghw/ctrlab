from time import time
import numpy as np
import matplotlib.pyplot as plt
from rltools import PPO, load_checkpoint_from_path
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from ctrlab.rl.pendulum import SimplePendulumEnv
from ctrlab.rl.hover import Hover

NUM_ENVIRONMENTS = 1024

seed = np.random.randint(1)
environment_name = 'drone-hover-v0'
# environment_name = 'SimplePendulum-v0'

def env_factory():
    env = gym.make(environment_name)
    env = RescaleAction(env, -1, 1)
    env.reset(seed=None)
    return env

ppo = PPO(env_factory, 
          N_ENVIRONMENTS=NUM_ENVIRONMENTS, 
          TOTAL_STEP_LIMIT=10000000, 
          ON_POLICY_RUNNER_STEPS_PER_ENV=128,
          BATCH_SIZE=1024)

state = ppo.State(seed)
finished = False

while not finished:
    finished = state.step()

# Save policy
with open("policy.h", "w") as f:
    f.write(state.export_policy())

env = gym.make(environment_name)
env.reset(seed=None) and None

# Evaluate policy
def evaluate(environment_name, policy):
    states = []
    actions = []
    ep_reward = 0
    num_steps = 0
    observation = env.env.env.env._get_obs()
    states.append(observation)
    start_time = time()
    finished = False
    while not finished:
        action = state.action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(observation)
        actions.append(action)
        finished = terminated or truncated
        num_steps += 1
        ep_reward += reward
    print(f"Inference time: {time() - start_time}")
    print(f"Inference steps: {num_steps}")
    print(f"Episodic reward: {ep_reward}")
    print(info)
    return np.asarray(states), np.asarray(actions)
        
states, actions = evaluate(environment_name, lambda observation: state.action(observation))
actions = np.clip(actions, -1, 1)
actions = (actions + 1)/2

plt.figure()
plt.plot(states[:-1,0], states[:-1,1])
plt.scatter(states[0,0], states[0,1], marker="o")
plt.scatter(states[-2,0], states[-2,1], marker="x")
plt.scatter(0, 0, marker="o")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.figure()
plt.plot(states[:-1,7])
plt.plot(states[:-1,8])
plt.plot(states[:-1,9])

plt.figure()
plt.plot(actions[:-1,0])
plt.plot(actions[:-1,1])
plt.plot(actions[:-1,2])
plt.plot(actions[:-1,3])
plt.show()
    
# policy = load_checkpoint_from_path("policy.h")