import matplotlib.pyplot as plt
import time
import math
import numpy as np
import copy
from liner_sample import Sample,PolicyBuffer
from variables import roll_outs,sim_env,EXPLORATION_TOTAL_STEPS,PI2_coefficient,TRAIN_STEPS,TRAIN_TIMES,main_policy
from tools.plot_data import save_figure
'''
我觉得不能仅仅训练一轮,至少训练多轮才能进行训练,或者说训练多轮得到最优基线进行训练,这个时候怎么构建最优基线也是一个问题
1. 每个状态仅仅训练,或者说迭代了一轮,根本达不到所谓的收敛
2. 如果不训练一轮,而是训练多轮,那么这些
3. 首先按照现在的方法根本没有办法得到多轮的数据,

# 方案1 每个点改进10次用最后一次改进的进行学习,或者用最优改进的进行学习       完全静态目标,从原则来讲这个才能叫做PI2，但是这样后面的网络实际上没有更新
简单来说我们原来采用的是方案1,但是由于策略参数化可以立即学到,所以可以逐步改善,那么我们能不能逐步学习?因为现在的方案2我觉得没道理,虽然和方案1差不多
从大数定理来看差不多
# 方案2 每个点每次训练仅仅改进一次,后期通过训练改进10次(相当于边训练边改进)   动态目标,还是得采用这个方案

1. 方差很大


封装好了就可以去测试对应的模型了
那么什么时候应该去封装?
封装有必要解除env么
'''
# MAXMIN标准化
def minmax_nor(loss):
    res = (loss - loss.min())/(loss.max()-loss.min())
    return res
# 应该是三个维度的运算,不是一个维度的运算
class Agent():
    def __init__(self,action_dim,obs_dim,env=sim_env,
                 policy_net = main_policy,sampler = Sample(),
                 roll_outs=roll_outs, explore_steps=EXPLORATION_TOTAL_STEPS,
                 train_times=TRAIN_TIMES, train_steps=TRAIN_STEPS):
        # <<<<<<<<< 关键网络 <<<<<<<<<
        self.env = env
        self.reward_model = sampler
        self.main_policy = policy_net
        #<<<<<<<<<解析环境<<<<<<<<
        self.action_dim = action_dim
        self.observation_dim = obs_dim
        # self.experience_buffer = PolicyBuffer()

        self.use_experience_buffer = PolicyBuffer()
        self.explore_experience_buffer = PolicyBuffer()
        # <<<<<<<<< PI2设置 <<<<<<<<<
        self.roll_outs = roll_outs                                          # 并行宽度
        self.explore_steps = explore_steps                                  # 并行深度
        self.train_time  = train_times                                      # 总策略改进次数
        self.train_steps  = train_steps                                     # 串行交互次数
        self.PI2_coefficient = PI2_coefficient                              # PI2超参数
        self.sigma = np.ones([self.roll_outs,1])                            # 方差
        self.sigma_factor = 0.85                                            # 方差因子
        self.current_steps = 0                                              # 当前step次数
        self.current_training = 0                                           # 训练次数

        self.mini_batch = 50
        self.explore_and_use = 1                                          # 探索利用因子 explore / use 探索/利用
        self.attenuation_factor = 0.99                                       # 探索衰减因子
        #<<<<<<<<PI2数据记录<<<<<<<<<<<
        # 我认为至少每50步训练一次,不至于说每次都训练,但是每50步必须训练一次
        self.batch_action = np.zeros(shape=(self.train_steps +1,self.action_dim),dtype=np.float64)     # 记录每次交互的动作
        self.batch_obs = np.zeros(shape=(self.train_steps +1,self.observation_dim),dtype=np.float64)   # 记录每次交互后的状态
        self.batch_reward = np.zeros(shape=(self.train_steps + 1,1),dtype=np.float64)                  # 计算每一步的loss,感觉没啥用
        self.action_roll  = np.zeros(shape=(self.roll_outs,self.action_dim),dtype=np.float64)
        self.loss_roll = np.zeros(shape=(self.roll_outs,1),dtype=np.float64)                                                   # 记录训练中的loss用于策略改进
        self.loss_after_training =  np.zeros(shape=(self.train_time,1),dtype=np.float64)           # 记录每次更新后的奖励,action应该是拿不到了
        #<<<<<<记录策略网路参数<<<<<<<<<<<,

        self.current_obs  = np.zeros(shape=(1,self.observation_dim),dtype=np.float64)
        self.current_action =  np.zeros(shape=(1,self.action_dim),dtype=np.float64)
        self.current_loss  = np.zeros(shape=(1,1),dtype=np.float64)
        # 绘图记录
    # 我认为一个地方都学不好,凭啥要随机初始化,一定不能随机初始化
    def reset_training_state(self):
        self.sigma = np.ones([self.roll_outs, 1])                            # 采用同样的方差
        self.current_training = 0                 # 重置训练次数
    def reset_step_state(self):
        self.current_obs = self.env.reset()
        self.current_obs = self.current_obs.reshape([1,self.observation_dim])
        self.current_action = self.main_policy.choose_action(self.current_obs)
        self.current_steps = 0
    # 策略评估,记住应该及时的统一相关的env环境
    def policy_evl(self):
        _,self.action_roll,self.loss_roll = self.reward_model.get_episode_reward_with_sample(current_env=self.env,
                                                                                             policy=self.main_policy,sigma_list=self.sigma,
                                                                                             total_step=self.explore_steps,total_num=
                                                                                             self.roll_outs)
    # 策略改进,是否改进一轮就够了?
    # 就算我想要改善多轮,policy必须进行学习
    # 我这个时候真的不应该去更新一下策略网络么
    def policy_improve(self):
        exponential_value_loss = np.zeros((self.roll_outs, 1), dtype=np.float64)  #
        probability_weighting = np.zeros((self.roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
        if (self.loss_roll.max() - self.loss_roll.min() <= 1e-4):
            probability_weighting[:] = 1.0 / self.roll_outs
        else:
            exponential_value_loss[:] = np.exp(-self.PI2_coefficient * minmax_nor(self.loss_roll)[:])
            probability_weighting[:] = exponential_value_loss[:] / np.sum(exponential_value_loss)
        # 验证loss确实小的地方很大
        # plt.plot(probability_weighting,"r+")
        # plt.plot(minmax_nor(self.loss_roll),"b-")
        # plt.show()
        self.current_action = np.dot(self.action_roll.T,probability_weighting)
        # 存储数据进行训练
        # self.experience_buffer.addExperience(np.reshape(self.current_obs,newshape=[1,self.observation_dim]),np.reshape(self.current_action,newshape=[1,self.action_dim]))
        return self.current_action
    # 当前的框架应该是正确的而且可以改变学习频率,但是还是batch学习比较好
    # 框架尽量明确
    def training(self):
        print("start training!")
        self.reset_training_state()
        while self.current_training < self.train_time:
            if self.current_training % 20:
                self.sigma = np.ones([self.roll_outs, 1])  # 采用同样的方差
            self.reset_step_state()
            self.sigma /= self.sigma_factor
            while self.current_steps < self.train_steps:
                self.policy_evl()
                self.policy_improve()
                # 存储标签
                self.batch_obs[self.current_steps] = self.current_obs
                self.batch_action[self.current_steps] = self.current_action
                # 更新网络,这个顺序很关键,记录的顺序不能在这个前面
                self.updateEnv()
                self.batch_reward[self.current_steps] = self.current_loss
                # 为啥要进行批训练
                self.current_steps += 1
            self.explore_experience_buffer.addExperience(self.batch_obs,self.batch_action,self.batch_reward)
            print(self.current_training,time.time()-f_t)
            # 我觉得这样训练也可以
            self.updatepolicy()
            self.current_training +=  1
            self.plot_res(title="Method 1 Train %d epoch"%self.current_training,save_photo=True)

    def differ_train(self):
        print("start training!")
        self.reset_step_state()
        while self.current_steps < self.train_steps:
            self.reset_training_state()
            while self.current_training < self.train_time:
                self.sigma /= self.sigma_factor
                self.policy_evl()
                self.policy_improve()
                self.batch_obs[self.current_steps] = self.current_obs
                self.batch_action[self.current_steps] = self.current_action

                # 类似经验池
                self.main_policy.learn(self.batch_obs[:self.current_steps+1],self.batch_action[:self.current_steps+1])
                self.current_training += 1
            self.updateEnv()
            self.batch_reward[self.current_steps] = self.current_loss
            self.current_steps += 1
            # self.plot_res(title="Train %d steps"%self.current_steps)
            print(self.current_steps,time.time()-f_t)
            self.plot_res(title="Train %d Steps"%self.current_steps)
        self.use_experience_buffer.addExperience(self.batch_obs,self.batch_action,self.batch_reward)
        # self.policy_test()
        self.plot_res(save_photo=True,title="Method 2 Train %d Steps"%self.current_steps)
    # 首先用第一种方法探索
    # 然后用第二种方法训练
    def mux_train(self,explore_time=400,use_time=100):
        self.train_time = explore_time
        self.training()
        for _ in range(use_time):
            self.differ_train()
            self.updatepolicy()
            self.explore_and_use *= self.attenuation_factor
    def policy_test(self,id=None):
        self.reset_step_state()
        while self.current_steps < self.train_steps:
            self.current_action = self.main_policy.choose_action(self.current_obs)
            self.batch_obs[self.current_steps] = self.current_obs
            self.updateEnv()
            self.current_steps += 1
        if id is None:
            title = "Test"
            save_photo = False
        else:
            title = "Test %d Times"%id
            save_photo = True
        self.plot_res(title=title,save_photo=save_photo)
    # 这个地方我还真不能确定他能不能得到
    # 但是我确实没有用所谓的经验池
    def updatepolicy(self):
        # 总共采样次数
        if self.use_experience_buffer.buffer_empty():
            batch_obs, batch_action, _ = self.explore_experience_buffer.getExperience(self.mini_batch)
        else:
            batch_obs,batch_action,_ = self.explore_experience_buffer.getExperience(int(np.clip(self.mini_batch * self.explore_and_use,1,self.mini_batch)))
            _batch_obs,_bat_action,_ = self.use_experience_buffer.getExperience(int(np.clip((1-self.explore_and_use) * self.mini_batch,1,self.mini_batch)))
            batch_obs = np.vstack((batch_obs,_batch_obs))
            batch_action = np.vstack((batch_action,_bat_action))
        self.main_policy.learn(batch_obs=batch_obs,batch_target_act=batch_action)
        # self.main_policy.learn(self.batch_obs,self.batch_action)
        self.main_policy.save_model()
    ## 这个时候应该记录一下对应的东西
    def updateEnv(self):
        self.current_obs,self.current_loss,_,_ = self.env.step(self.current_action)
        self.current_obs = np.reshape(self.current_obs,newshape=[1,self.observation_dim])
    def plot_res(self,title="Train",save_photo=False):
        plt.ion()
        theta = self.batch_obs[:,1]
        theta_desire = self.batch_obs[:,3]
        plt.plot(theta,"b+",label="theta")
        plt.plot(theta_desire,"r^",label="theta_desire")
        plt.legend(loc="best")
        plt.title(title)
        if save_photo:
            save_figure("./photo/",title)
        plt.show()
        # #-------------绘制Loss曲线--
        # plt.plot(self.batch_reward)
        # print("TOTAL LOSS:",np.sum(self.batch_reward))
        # plt.title("batch reward")
        # plt.show()
        # #----------绘制最优的action曲线,检查action对不对
        # plt.plot(self.batch_action)
        # plt.title("TOTAL ACTION!")
        # plt.show()
        # # pass
## 可以逐渐测试每一个部分的功能是否完善
if __name__ == '__main__':
    #------------训练新的模型------------
    f_t = time.time()
    test_ag = Agent(sim_env.action_dim,sim_env.observation_dim)
    # test_ag.mux_train()
    test_ag.differ_train()
    # test_ag.training()
    #-------------测试训练好的模型--------
    # test_ag = Agent()
    # for i in range(100):
    #     test_ag.policy_test(i)