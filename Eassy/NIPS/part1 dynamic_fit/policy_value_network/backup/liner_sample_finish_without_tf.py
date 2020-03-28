'''
sample for liner_plane_model
随机种子很关键,不然肯定会学习炸掉
TODO:环境一定要有随机种子  这个很坑
现在我可以加经验buffer了
TODO: 还是均方误差厉害
'''
import numpy as np
import multiprocessing as mp
from env.plane.liner_env import Planes_Env
import seaborn as sns
import matplotlib.pyplot as plt
import time
from dynamic_network import liner_plane_brain
## 价值网络
## 根据策略网络的采样器
class Sample():
    def __init__(self,env,policy_network=None):
        self.env = env
        self.policy_network = policy_network
        self.ob_dim = self.env.observation_dim
        self.ac_dim = self.env.action_dim
    ## PID采样器,真实数据采样,这里就with Policy了但是比较好写
    def sample_episodes_with_PID(self,K,total_num):
        batch_obs_next = []
        batch_obs = []
        batch_actions = []
        for i in range(total_num):
            observation = self.env.reset()
            ierror = 0
            while True:
                #根据策略网络产生一个动作
                state = np.reshape(observation,[self.ob_dim,1])
                ''' POLICY NETWORK '''
                error = self.env.theta_desired - state[1]
                derror = self.env.dtheta_desired -state[2]
                ierror  = ierror + error * self.env.tau
                action = K[0] * error + K[1] * ierror + K[2] * derror
                '''save data'''
                observation_, reward, done, info = self.env.step(action)
                #存储当前观测
                batch_obs.append(np.reshape(observation,[1,self.ob_dim])[0,:])
                #存储后继观测
                batch_obs_next.append(np.reshape(observation_,[1,self.ob_dim])[0,:])
                #存储当前动作
                batch_actions.append(action)
                if done:
                    break
                #智能体往前推进一步
                observation = observation_
        #reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.env.observation_dim])
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions,[len(batch_actions),1])
        batch_obs_action= np.hstack((batch_obs,batch_actions))
        return batch_obs_action, batch_delta,batch_obs_next
    ## 类似于采样器,这里就是利用PID参数去采集一条样本但是会返回reward
    ## 这个仅仅用来返回reward,采样步长,不再用系统的步长,这里当然可以用局部并行
    def get_one_episodes_reward_with_PID(self, K, total_step):
        batch_obs_next = []
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        ierror = 0
        observation = self.env.reset()
        theta_list = []
        desire_theta_list = []
        loss = 0.0
        for j in range(total_step):
            # 根据策略网络产生一个动作
            state = np.reshape(observation, [self.ob_dim, 1])
            ''' POLICY NETWORK '''
            error = self.env.theta_desired - state[1]
            loss += np.fabs(error)
            derror = self.env.dtheta_desired - state[2]
            ierror = ierror + error * self.env.tau
            action = K[0] * error + K[1] * derror + K[2] * ierror
            '''save data'''
            observation_, reward, done, info = self.env.step(action)
            # 存储当前观测
            batch_obs.append(np.reshape(observation, [1, self.ob_dim])[0, :])
            # 存储后继观测
            batch_obs_next.append(np.reshape(observation_, [1, self.ob_dim])[0, :])
            # 存储当前动作
            theta_list.append(state[1])
            desire_theta_list.append(self.env.theta_desired)
            batch_actions.append(action)
            # 智能体往前推进一步
            observation = observation_
        '''FOR DEBUG'''
        # plt.plot(theta_list,'r')
        # plt.plot(desire_theta_list,'b')
        # plt.show()
        # Overshoot =max(abs(np.array(theta_list))) - max(abs(np.array(desire_theta_list)))
        # 直接用曲线拟合思想好么
        # print(Overshoot)
        batch_rewards.append(loss)
        # reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.env.observation_dim])
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions, [len(batch_actions), 1])
        batch_obs_action = np.hstack((batch_obs, batch_actions))
        batch_rewards = np.reshape(batch_rewards, [len(batch_rewards), 1])
        return batch_obs_action, batch_delta, batch_obs_next, batch_rewards
    # 串行采样
    def get_episodes_reward_with_PID(self, K, total_step, total_num):
        batch_obs_next = []
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        for i in range(total_num):
            ierror = 0
            observation = self.env.reset()
            theta_list  = []
            desire_theta_list = []
            for j in range(total_step):
                # 根据策略网络产生一个动作
                state = np.reshape(observation, [self.ob_dim, 1])
                ''' POLICY NETWORK '''
                error = self.env.theta_desired - state[1]
                derror = self.env.dtheta_desired - state[2]
                ierror = ierror + error * self.env.tau
                action = K[0] * error + K[1] * ierror + K[2] * derror
                '''save data'''
                observation_, reward, done, info = self.env.step(action)
                # 存储当前观测
                batch_obs.append(np.reshape(observation, [1, self.ob_dim])[0, :])
                # 存储后继观测
                batch_obs_next.append(np.reshape(observation_, [1, self.ob_dim])[0, :])
                # 存储当前动作
                theta_list.append(state[1])
                desire_theta_list.append(self.env.dtheta_desired)
                batch_actions.append(action)
                # 智能体往前推进一步
                observation = observation_
            Overshoot = max(abs(np.array(theta_list))) - max(abs(np.array(desire_theta_list)))
            loss = Overshoot if Overshoot > 0 else 0
            # print(Overshoot)
            batch_rewards.append(loss)
        # reshape 观测和回报
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.env.observation_dim])
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions, [len(batch_actions), 1])
        batch_obs_action = np.hstack((batch_obs, batch_actions))
        batch_rewards = np.reshape(batch_rewards,[len(batch_rewards),1])
        return batch_obs_action, batch_delta, batch_obs_next, batch_rewards
    # 并行采样
    # 而且给的是一个K_list
    def get_episodes_reward_with_PID_parall(self, K, total_step, total_num):
        batch_obs_action = []
        batch_delta = []
        batch_obs_next= []
        batch_rewards = []
        if_empty = True
        multi_res = [poll.apply_async(self.get_one_episodes_reward_with_PID,(K[j],total_step,)) for j in range(total_num)]
        for j, res in enumerate(multi_res):
            if if_empty:
                batch_obs_action = res.get()[0]
                batch_delta = res.get()[1]
                batch_obs_next = res.get()[2]
                batch_rewards = res.get()[3]
                if_empty = False
            else:
                batch_obs_action = np.vstack((batch_obs_action,res.get()[0]))
                batch_delta = np.vstack((batch_delta,res.get()[1]))
                batch_obs_next = np.vstack((batch_obs_next,res.get()[2]))
                batch_rewards = np.vstack((batch_rewards,res.get()[3]))
        print(np.mean(batch_rewards))
        return batch_obs_action, batch_delta, batch_obs_next, batch_rewards
## 策略网络
'''上面的代码已经经过测试了'''
## 这里可以采用不同的评价网络,我想的很简单,就是评价网络不一样
## 下面写PI2相关代码
## 相当于可以并行数据采样了
'''
随机采样范围
'''
k_bound = [0,10]
'''
MAX MIN 标准化
'''
def MINMAX(loss):
    res = (loss - loss.min())/(loss.max()-loss.min())
    return res
class PI2():
    def __init__(self,env,sampler,dynamic_network,roll_outs=20,train_time=100):
        self.env = env
        self.random_init = False # 随机初始化
        self.K = np.zeros(shape=(3,1),dtype=np.float64)
        self.sigma = np.zeros(shape=(3,1),dtype=np.float64)
        self.roll_outs = roll_outs
        self.train_time = train_time
        self.sampler = sampler
        self.PI2_coefficient = 30.0 # PI2超参数
        self.K_rolls = np.zeros(shape=(roll_outs,3),dtype=np.float64)
        self.loss_rolls = np.zeros(shape=(roll_outs,1),dtype=np.float64)
        # 考虑到初始值
        self.K_after_training = np.zeros((train_time +1 ,3), dtype=np.float64)
        self.loss_after_training = np.zeros((train_time +1, 1), dtype=np.float64)
        self.current_training = 0 # 记录已经经过几次训练

        self.enhance_interval = 10 # 方差增强间隔
        self.beta = 0.85 # 递增系数

        self.dynamic_net = dynamic_network # 模型逼近器

        self.simulation_time = 1000 # 仿真时间
        pass
    def reset_training_state(self):
        pass
    def set_initial_value(self,INIT_K = [1.5,2.5,0.5],sigma = [1,0.5,0.1]):
        if self.random_init:
            self.K = np.random.uniform(k_bound[0],k_bound[1],(3,1))
        else:
            self.K = np.reshape(INIT_K,[3,1])
        self.sigma = np.reshape(sigma,[3,1])
        self.env.reset()
        self.current_training = 0 # 记录已经经过几次训练
    # 这个地方就可以喂给神经网络去学习了,现在仅学习模型
    # 我不利用所有的去学习,仅仅利用这个一个部分去学习
    # 后期可以利用所有的去学习
    def policy_evl(self):
        for i in range(self.roll_outs):
            self.K_rolls[i] = np.random.normal(self.K,self.sigma).reshape([1,3])
        batch_obs_act,batch_delta,_,self.loss_rolls =  self.sampler.get_episodes_reward_with_PID_parall(self.K_rolls,200,self.roll_outs)
        self.dynamic_net.fit_dynamic(batch_obs_act,batch_delta,epoch = 10)
        '''FOR DEBUG'''
        # plt.plot(self.loss_rolls)
        # plt.show()
    def policy_improve(self):
        exponential_value_loss = np.zeros((self.roll_outs, 1), dtype=np.float64)  #
        probability_weighting = np.zeros((self.roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
        if(self.loss_rolls.max() -self.loss_rolls.min() <= 1e-4):
            probability_weighting[:] = 1.0 / self.roll_outs
        else:
            exponential_value_loss[:] = np.exp(-self.PI2_coefficient * MINMAX(self.loss_rolls)[:] )
            probability_weighting[:] = exponential_value_loss[:] / np.sum(exponential_value_loss)
        '''DEBUG 验证PI2优化过程有效'''
        # plt.plot(probability_weighting,label="pro")
        # plt.plot(self.loss_rolls,label="loss")
        # plt.legend(loc="best")
        # plt.show()
        self.K = np.dot(self.K_rolls.T,probability_weighting)
    def training(self):
        self.K_after_training[self.current_training] = self.K.T
        _,_,_,self.loss_after_training[self.current_training] = self.sampler.get_one_episodes_reward_with_PID(self.K,self.simulation_time)
        while self.current_training < self.train_time:
            self.current_training = self.current_training + 1
            ''' sigma enhance '''
            if self.current_training % self.enhance_interval ==0:
                self.sigma = self.sigma/self.beta
                state,_,_,_ = self.sampler.get_one_episodes_reward_with_PID(self.K,1000)
                plt.plot(state[:,1])
                plt.show()
            self.policy_evl()
            self.policy_improve()
            # 记录参数
            self.K_after_training[self.current_training] =self.K.T
            _,_,_,self.loss_after_training[self.current_training]  = self.sampler.get_one_episodes_reward_with_PID(self.K,self.simulation_time)
        '''测试基本的PI2性能,然后准备添加经验池'''
        plt.plot(self.K_after_training)
        plt.show()
        plt.plot(self.loss_after_training)
        plt.show()
'''测试采样器的性能'''
# env = Planes_Env()
# sampler = Sample(env)
# K = [1.5,2.5,0.5]
# train_x , train_y , real_y= sampler.sample_episodes_with_PID(K,10)
# print(train_x.shape,train_y.shape,real_y.shape)
# cur_state =train_x[:,0:env.observation_dim] # 得到当前状态
# next_state = real_y # 得到下一步的状态
# delta = train_y # 状态改变量
# # print(train_x,train_y,real_y)
'''测试回报器性能 回报器就可以直接这么用'''
# env = Planes_Env()
# sampler = Sample(env)
# mu = np.array([1.5,2.5,0.5])
# sigma = [1,0.5,0.3]
# train_x = None # 相当于训练集是空的
# train_y = None
# real_y = None
# reward_list = None
# if_experince_empty = True# 经验池是否为空
# for i in range(20):
#     K = np.random.normal(mu,sigma)
#     x , y , r_y ,r = sampler.get_spisodes_reward_with_PID(K,200,1)
#     # 第一次采样直接赋值
#     if if_experince_empty:
#         train_x = x
#         train_y = y
#         real_y = r_y
#         reward_list = r
#         if_experince_empty = False
#     else:
#         train_x = np.vstack((train_x,x))
#         train_y = np.vstack((train_y,y))
#         real_y = np.vstack((real_y,r_y))
#         reward_list = np.vstack((reward_list,r))
# print(train_x.shape,train_y.shape,real_y.shape,reward_list.shape)
# plt.plot(reward_list)
# plt.show()

if __name__ == '__main__':
    '''测试并行采样器的性能'''
    poll = mp.Pool(mp.cpu_count())
    env = Planes_Env()
    sampler = Sample(env)
    mu = np.array([1.5,2.5,0.5])
    sigma = [1,0.5,0.3]
    dynamic_net = liner_plane_brain.Dynamic_Net(env)

    # train_x = None # 相当于训练集是空的
    # train_y = None
    # real_y = None
    # reward_list = None
    # if_experince_empty = True# 经验池是否为空
    # for i in range(20):
    #     K = np.random.normal(mu,sigma)
    #     x , y , r_y ,r = sampler.get_spisodes_reward_with_PID(K,200,1)
    #     # 第一次采样直接赋值
    #     if if_experince_empty:
    #         train_x = x
    #         train_y = y
    #         real_y = r_y
    #         reward_list = r
    #         if_experince_empty = False
    #     else:
    #         train_x = np.vstack((train_x,x))
    #         train_y = np.vstack((train_y,y))
    #         real_y = np.vstack((real_y,r_y))
    #         reward_list = np.vstack((reward_list,r))
    '''测试并行通道,查看并行是否可行'''
    # first_t = time.time()
    # K = []
    # total_num = 100
    # for i in range(total_num):
    #    K.append(np.random.normal(mu,sigma))
    # # print("before : ",K)
    # K = np.reshape(K,[len(K),3])
    # # print("after :",K)
    # train_x,train_y,real_y,reward_list = sampler.get_episodes_reward_with_PID_parall(K,200,total_num)
    # print(train_x.shape,train_y.shape,real_y.shape,reward_list.shape)
    # plt.plot(reward_list)
    # plt.show()
    '''测试PI2基本性能 为啥感觉PI2写炸了'''
    RL_BRAIN =PI2(env,sampler,dynamic_net)
    RL_BRAIN.set_initial_value()
    RL_BRAIN.policy_evl()
    RL_BRAIN.policy_improve()
    RL_BRAIN.training()
