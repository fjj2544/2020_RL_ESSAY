from policy_brain import Policy_Net
from liner_env import Planes_Env
from liner_sample import Sample,PolicyBuffer
from variables import save_model_path
from PI2 import Agent
import numpy as np

import tensorflow as tf

# tf.set_random_seed(1)
# np.random.seed(1)

#-----GYM test --------
import gym
if __name__ == '__main__':
    env = gym.make("InvertedPendulum-v1")  # 这个比较简单
    env = env.unwrapped
    # ------------查看状态空间范围--------------
    print(env.observation_space)
    print(env.observation_space.shape[0])
    print(env.observation_space.high, env.observation_space.low)
    ob_dim = env.observation_space.shape[0]
    # ----------查看动作空间范围---------------
    print(env.action_space)
    print(env.action_space.shape[0])
    print(env.action_space.high, env.action_space.low)
    action_bound = np.hstack((env.action_space.low, env.action_space.high))
    ac_dim = env.action_space.shape[0]
    print(action_bound)
    print(action_bound[0], action_bound[1])
    # --------测试采样器
    sampler = Sample(ob_dim=ob_dim,ac_dim=ac_dim)

    main_policy = Policy_Net(observation_dim=ob_dim,
                             action_dim=ac_dim,action_bound=action_bound)
    policy_buffer = PolicyBuffer()
    agent = Agent(action_dim=ac_dim,obs_dim=ob_dim,env=env
                  ,policy_net=main_policy
                  ,sampler=sampler,policybuffer=policy_buffer
                  ,train_times=10,train_steps=3000)
    # cur_obs = env.reset()
    # ob,ac,r_ = sampler.get_episode_reward_with_sample(policy=main_policy,current_env=env,current_obs=cur_obs)
    # print(r_)
    #-----------测试PI2代码
    reward = []
    # agent.mux_train(explore_time=20,use_time=100)
    # agent.training(if_debug=True)
    for _ in range(100):
        agent.differ_train(if_debug=True)
        reward.append(np.sum(agent.batch_reward))
        print(reward)
