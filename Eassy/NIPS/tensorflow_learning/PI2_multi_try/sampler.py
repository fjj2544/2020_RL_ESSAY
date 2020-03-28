'''------这个类似一个采样器用于并行采样的--------'''
import numpy as np
import copy
import multiprocessing as mp
import time
from variables import N_PROC,explore_policy_list,EXPLORATION_TOTAL_STEPS,GAMA,sim_env,buffer_size,roll_outs
# ===============超参数F===============
# 经验池
class PolicyBuffer:
    # 初始化
    def __init__(self):
        self.buffer_size = buffer_size
        self.obs = []
        self.act = []
    # 增经验
    # 输入必须是一个行向量
    def addExperience(self,obs_batch, act_batch):
        if len(obs_batch)+len(self.obs) > self.buffer_size:
            self.obs[1:len(obs_batch)] = []
            self.act[1:len(obs_batch)] = []
        for obs, act in zip(obs_batch, act_batch):
            self.obs.append(obs)
            # 这样便于提取出数据来
            self.act.append(act)
    # 采经验
    def getExperience(self,number):
        index = np.random.choice(range(len(self.act)), number)
        a_batch = []
        obs_batch = []
        # 为啥是一个self.act
        for i in index:
            obs_batch.append(self.obs[i])
            a_batch.append(self.act[i])
        return obs_batch,a_batch

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

    def get_one_episode_reward_with_PID_sub(self, sigma, id_):
        policy_ = explore_policy_list[id_]
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
        action_first = np.reshape(action, [1, self.ac_dim])
        # 从步数的角度进行
        for j in range(EXPLORATION_TOTAL_STEPS):
            observation_list = np.reshape(observation, (1, self.env.observation_dim))
            action = policy_.policy_sample(sigma, observation_list)
            action = action[0]
            observation_next, reward_, done, info = env.step(action)
            batch_reward.append(reward_)
            observation_n = observation_next
            # 智能体往前推进一步
            observation = observation_n
        # 计算折扣累计回报
        for id_ in range(EXPLORATION_TOTAL_STEPS - 1, 0, -1):
            batch_reward[id_-1] += GAMA*batch_reward[id_]
        reward_first = batch_reward[0]
        return batch_obs_first, action_first, reward_first
    # 按照需求得到训练数据
    def get_episodes_reward_with_PID_mult(self, env = copy.deepcopy(sim_env), sigma=1, roll_outs=roll_outs):
        self.env = copy.deepcopy(env)
        counter = 0
        s = []
        a = []
        r = []
        # 一定不能写成等于
        while counter < roll_outs:
            N_WORKERS = min(roll_outs-counter,N_PROC)
            multi_res = [pool.apply_async(self.get_one_episode_reward_with_PID_sub, (sigma,j,)) for j in range(N_WORKERS) ]
            for _, res in enumerate(multi_res):
                s.append(res.get()[0])
                a.append(res.get()[1])
                r.append(res.get()[2])
            counter += N_WORKERS
        s = np.reshape(s,newshape=[len(s),env.observation_dim])
        a = np.reshape(a,newshape=[len(a),env.action_dim])
        r = np.reshape(r,newshape=[len(r),1])
        return s,a,r
if __name__ == '__main__':
    pool = mp.Pool(N_PROC)
    # 创建采样环境
    sampler = Sample(sim_env)
    print("START")
    #--------------------测试policy_buffer的效果------------
    # 相当于可以get数据了
    f_t = time.time()
    # 差不多大概0.3左右一步
    experience_buffer = PolicyBuffer()
    s = []
    a = []
    r = []
    for i in range(10):
        _s,_a,_r = sampler.get_episodes_reward_with_PID_mult(roll_outs=roll_outs)
        s.append(_s)
        a.append(_a)
        r.append(_r)
        c_t = time.time()
        print(i,"Time cost:",c_t-f_t)
    #-------------测试环境----------------
    print(len(a),len(r),len(s))
    # experience_buffer.addExperience(s,a)
    # # 这个应该有一个划分
    # obs,a_ = experience_buffer.getExperience(10)
    # print(obs,a_)
    # 可以考虑采用batch训练的方法训练一下网络,首先采样50批次训练

