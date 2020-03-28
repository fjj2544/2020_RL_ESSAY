import numpy as np
import matplotlib.pyplot as plt

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

        self.enhance_interval = 1 # 方差增强间隔
        self.beta = 0.85 # 递增系数

        self.dynamic_net = dynamic_network # 模型逼近器
        self.simulation_time = 1000 # 仿真时间

        self.buffer_size = 15000
        self.buffer = []

        self.use_dynamic_model = False
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
    # batch_obs_act,batch_delta
    '''
    每一轮训练 roll_outs * sim_time = 20 * 200 = 4000
    每次训练结束 sim_time = 1000
    '''
    def add_experinece(self,experience):
       if len(self.buffer) == 0: # 虽然不正确但是可以用
           self.buffer = experience
       elif self.buffer.shape[0] >= self.buffer_size:
            self.buffer = np.delete(self.buffer,np.arange(0,15000),axis =0) # 删除原来的经验池
            self.buffer = np.vstack((self.buffer, experience))  # 竖着拼接
       else:
           self.buffer = np.vstack((self.buffer,experience)) #竖着拼接
       print(self.current_training,self.buffer.shape)
    def sample_from_experience(self,samples_num = 1000):
        sample_data = np.random.sample(self.buffer,samples_num)
        return sample_data
    def policy_evl(self):
        for i in range(self.roll_outs):
            self.K_rolls[i] = np.random.normal(self.K,self.sigma).reshape([1,3])
        if(self.use_dynamic_model):
            batch_obs_act, batch_delta, batch_next_obs, self.loss_rolls = self.sampler.get_episodes_reward_with_PID(
                self.K_rolls, 200, self.roll_outs,self.dynamic_net)
        else:
            batch_obs_act,batch_delta,batch_next_obs,self.loss_rolls =  self.sampler.get_episodes_reward_with_PID(self.K_rolls,200,self.roll_outs)
        self.add_experinece(np.hstack((batch_obs_act,batch_delta)))
        # print(batch_obs_act.shape,batch_delta.shape,batch_next_obs.shape)
        # self.dynamic_net.fit_dynamic(batch_obs_act,batch_delta,epoch = 10)
        '''测试拟合效果'''
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
        batch_obs_act, batch_delta, batch_next_obs,self.loss_after_training[self.current_training] = self.sampler.get_one_episodes_reward_with_PID(self.K,self.simulation_time)
        self.add_experinece(np.hstack((batch_obs_act,batch_delta)))
        while self.current_training < self.train_time:
            self.current_training = self.current_training + 1
            if self.current_training == 30:
                self.use_dynamic_model = True
            ''' sigma enhance '''
            if self.current_training % self.enhance_interval ==0:
                self.sigma = self.sigma/self.beta
                '''正好测试拟合效果'''
                # print(batch_obs_act.shape,batch_delta.shape)
                if self.use_dynamic_model == False:
                    batch_obs_act, batch_delta = np.hsplit(self.buffer,
                                                           [self.env.observation_dim + self.env.action_dim])
                    self.dynamic_net.fit_dynamic(batch_obs_act,batch_delta,epoch = 10)
                batch_obs_act, batch_delta, batch_next_obs,_ = self.sampler.get_one_episodes_reward_with_PID(self.K,1000)
                self.dynamic_net.prediction(batch_obs_act,batch_next_obs,if_debug = False)

                # plt.plot(state[:,1])
                # plt.show()
            self.policy_evl()
            self.policy_improve()
            # 记录参数
            self.K_after_training[self.current_training] =self.K.T
            batch_obs_act, batch_delta, batch_next_obs,self.loss_after_training[self.current_training]  = self.sampler.get_one_episodes_reward_with_PID(self.K,self.simulation_time)
            self.add_experinece(np.hstack((batch_obs_act, batch_delta)))
        '''测试基本的PI2性能,然后准备添加经验池'''
        plt.plot(self.K_after_training)
        plt.show()
        plt.plot(self.loss_after_training)
        plt.show()
