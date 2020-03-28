'''
sample for liner_plane_model
现在我可以加经验buffer了
'''
import numpy as np
import copy
import multiprocessing as mp
import time
from variables import GAMA,EXPLORATION_TOTAL_STEPS,roll_outs,main_policy,sim_env
class PolicyBuffer:
    # 初始化
    def __init__(self):
        self.buffer_size = 150000
        self.obs = []
        self.act = []
        self.rwd = []
    # 增经验
    def addExperience(self,obs_batch, act_batch,red_batch):
        if len(obs_batch)+len(self.obs) > self.buffer_size:
            self.rwd[1:len(obs_batch)] = []
            self.obs[1:len(obs_batch)] = []
            self.act[1:len(obs_batch)] = []
        for obs, act, red in zip(obs_batch, act_batch, red_batch):
            self.rwd.append(red)
            self.obs.append(obs)
            self.act.append(act)
    # 采经验
    def getExperience(self,number):
        if number ==0:
            return [],[],[]
        obs = []
        act = []
        reward = []
        index = np.random.choice(range(len(self.act)), number)

        for i in index:
            obs.append(self.obs[i])
            act.append(self.act[i])
            reward.append(self.rwd[i])
        obs = np.reshape(obs, (number,len(obs[0])))
        act = np.reshape(act, (number,len(act[0])))
        reward = np.reshape(reward, (number,len(reward[0])))
        return obs, act, reward
    def buffer_empty(self):
        return len(self.obs)==0
class Sample():
    def __init__(self,env = sim_env):
        self.env = copy.deepcopy(env)
        self.ob_dim = self.env.observation_dim
        self.ac_dim = self.env.action_dim
    def get_episode_reward_with_sample(self
                                       ,policy = main_policy,current_env = sim_env,
                                       sigma_list = np.ones([roll_outs,1]),
                                       total_num = roll_outs,
                                       total_step = EXPLORATION_TOTAL_STEPS):
        # 更新env环境
        self.env = copy.deepcopy(current_env)
        sigma = np.reshape(sigma_list, (total_num, 1))
        # 策略网络
        obs_out = [self.env.state]
        reward_out = []
        # 并行数据
        reward_list = []
        env_list = []
        obs_list = []
        obs_n_list = []
        # 复制得到多份env
        for i in range(total_num):
            env_list.append(copy.deepcopy(self.env))
            reward_list.append([])
            obs_list.append(copy.deepcopy(self.env.state))
        # 记录第一步的状态
        obs_list = np.reshape(obs_list, (total_num, self.ob_dim))
        # 前进1步
        action = policy.policy_sample(sigma, obs_list)
        action = action[0]
        for i in range(total_num):
            observation_next, reward_, done, info = env_list[i].step(action[i])
            obs_n_list.append(observation_next)
            reward_list[i].append(reward_)
        # 第一个动作记录
        action_out = np.reshape(action, [total_num, self.ac_dim])

        # 前进其他步
        for j in range(total_step-1):
            # 更新状态
            obs_list = np.reshape(obs_n_list, (total_num, self.ob_dim))
            # 得到动作
            action = policy.policy_sample(sigma, obs_list)
            action = action[0]
            # 前进一步
            for i in range(total_num):
                observation_next, reward_, done, info = env_list[i].step(action[i])
                obs_n_list[i] = observation_next
                reward_list[i].append(reward_)
        # 计算折扣累计回报
        for id_ in range(total_num):
            for step in range(total_step - 1, 0, -1):
                reward_list[id_][step - 1] += GAMA * reward_list[id_][step]
            reward_out.append(reward_list[id_][0])
        reward_out = np.reshape(reward_out,newshape=[total_num,1])
        return obs_out, action_out, reward_out

if __name__ == '__main__':
    print("START")
    sampler = Sample(sim_env)
    time_now = time.time()
    for i in range(1):
        o,a,r = sampler.get_episode_reward_with_sample(policy=main_policy)
    print(time.time() - time_now)
