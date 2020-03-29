import gym
import numpy as np
# from gym import envs
# print(envs.registry.all())
# env = gym.make("Humanoid-v1")
env = gym.make("InvertedPendulum-v1")  # 这个比较简单
env = env.unwrapped
observation = env.reset()
#------------查看状态空间范围--------------
print(env.observation_space)
print(env.observation_space.shape[0])
print(env.observation_space.high,env.observation_space.low)
#----------查看动作空间范围---------------
print(env.action_space)
print(env.action_space.shape[0])
print(env.action_space.high,env.action_space.low)
action_bound = np.hstack((env.action_space.low,env.action_space.high))
print(action_bound)
print(action_bound[0],action_bound[1])
for _ in range(1000):
  # env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  print("1: ",env._get_obs())
  # print(reward)
  if done:
    observation = env.reset()
env.close()