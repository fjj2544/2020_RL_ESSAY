'''
sample for liner_plane_model
随机种子很关键,不然肯定会学习炸掉
TODO:环境一定要有随机种子  这个很坑
现在我可以加经验buffer了
TODO: 还是均方误差厉害
'''
import numpy as np
import copy
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import liner_env
import liner_plane_brain
import PI2
import policy_brain
# ===============超参数S===============
policy_list = []
for i in range(mp.cpu_count()-2):
    policy_list.append(policy_brain.Policy_Net(liner_env.Planes_Env(), str(i)))
GAMA = 0.95
TOTAL_STEP = 120
# ===============超参数F===============
# 经验池
class PolicyBuffer:
    # 初始化
    def __init__(self):
        self.buffer_size = 150000
        self.obs = []
        self.act = []
    # 增经验
    def addExperience(self,obs_batch, act_batch):
        if len(obs_batch)+len(self.obs) > self.buffer_size:
            self.obs[1:len(obs_batch)] = []
            self.act[1:len(obs_batch)] = []
        for obs, act in zip(obs_batch, act_batch):
            self.obs.append(obs)
            self.act.append(act)
    # 采经验
    def getExperience(self,number):
        sample = []
        index = np.random.choice(range(len(self.act)), number)
        for i in index:
            sample.append([self.obs[i], self.act])
        return sample

class DynamicBuffer:
    # 初始化
    def __init__(self):
        self.buffer_size = 150000
        self.obs_now = []
        self.obs_det = []
        self.act_now = []
    # 增经验
    def addExperience(self, obs_now_batch, obs_next_batch, act_batch):
        if len(obs_now_batch)+len(self.obs_now) > self.buffer_size:
            self.obs_now[1:len(obs_now_batch)] = []
            self.obs_det[1:len(obs_now_batch)] = []
            self.act_now[1:len(obs_now_batch)] = []
        for obs, obs_, act in zip(obs_now_batch, obs_next_batch, act_batch):
            det = np.array(obs_) - np.array(obs)
            det = det.tolist()
            self.obs_now.append(obs)
            self.obs_det.append(det)
            self.act_now.append(act)
    # 采经验
    def getExperience(self, number):
        sample = []
        index = np.random.choice(range(len(self.obs_now)), number)
        for i in index:
            sample.append([self.obs_now[i], self.obs_det, self.act_now])
        return sample

class Sample():
    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.ob_dim = self.env.observation_dim
        self.ac_dim = self.env.action_dim
        # self.policy = policy_brain.Policy_Net(env, 'policy')
        # self.policy_ = []
        # for i in range(12):
        #     self.policy_.append(policy_brain.Policy_Net(env, str(i)))
    # 更新采样环境
    def update_env(self, env):
        self.env = copy.deepcopy(env)
    # 使用当前策略网络采样，得到一个损失
    def get_one_episodes_reward_with_PID(self, total_step, policy_network, dynamic_net=None,type='policy'):
        batch_rewards = []
        env = copy.deepcopy(self.env)
        observation = env.reset()
        for j in range(total_step):
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [1, self.ob_dim])
            action = policy_network.policy_predict(state)
            # TODO 这里要根据动态网络更改
            if dynamic_net is not None:
                action = np.reshape(action, [1, 1])
                observation = np.reshape(observation, [1, self.ob_dim])
                observation_, reward, done, info = dynamic_net.prediction(np.hstack((observation, action)))
            else:
                observation_, reward, done, info = env.step(action)
            # 存储当前折扣累计回报
            batch_rewards.append(float(reward))
            # 智能体往前推进一步
            observation = observation_
        # 对折扣累计回报进行计算
        for id in range(len(batch_rewards)-1, 0, -1):
            batch_rewards[id-1] += GAMA*batch_rewards[id]
        return batch_rewards[0]

    # 使用策略采样网络采样一条轨迹数据
    # 输出为：
    #   [obs_dim] 状态
    #   [1,total_num] 动作
    #   [total_num] 奖励
    def get_one_episode_reward_with_PID_sub(self, sigma, id_):
        global TOTAL_STEP
        global policy_list
        policy_ = policy_list[id_]
        sigma = np.reshape(sigma,(1,1))
        # 策略网络
        batch_obs_first = copy.deepcopy(self.env.state)
        batch_reward = []
        # 环境
        env = copy.deepcopy(self.env)
        # 状态
        observation = env.state
        observation = np.reshape(observation, (1, self.ob_dim))
        # 第一个动作
        action = policy_.policy_sample(sigma, observation)
        action = action[0]
        # action = 0
        action_first = np.reshape(action, [1, self.ac_dim]).tolist()
        # 从步数的角度进行
        for j in range(TOTAL_STEP):
            observation_list = np.reshape(observation, (1, self.env.observation_dim))
            action = policy_.policy_sample(sigma, observation_list)
            action = action[0]
            # action = 0
            observation_next, reward_, done, info = env.step(action)
            batch_reward.append(reward_)
            observation_n = observation_next
            # 智能体往前推进一步
            observation = observation_n
        # 计算折扣累计回报
        for id_ in range(TOTAL_STEP-1, 0, -1):
            batch_reward[id_-1] += GAMA*batch_reward[id_]
        reward_first = batch_reward[0]
        return batch_obs_first, action_first, reward_first

    def get_episodes_reward_with_PID_mult(self):
        multi_res = [pool.apply_async(self.get_one_episode_reward_with_PID_sub, (1,j,)) for j in range(12) ]
        multi_res[0].get()
        print(len(multi_res))

if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count()-2)
    env = liner_env.Planes_Env()
    policy = policy_brain.Policy_Net(env,'pi')

    sam = Sample(env)
    print("START")
    time_now = time.time()
    for i in range(200):
        sam.get_episodes_reward_with_PID_mult()
        print("======================================")
    print(time.time() - time_now)


