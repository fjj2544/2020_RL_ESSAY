import matplotlib.pyplot as plt
import time
import math
import numpy as np
import copy
from sampler import Sample,PolicyBuffer
from variables import roll_outs,sim_env,EXPLORATION_TOTAL_STEPS,PI2_coefficient,TRAIN_STEPS,TRAIN_TIMES,main_policy,explore_policy_list,main_policy_scope
'''
我觉得不能仅仅训练一轮,至少训练多轮才能进行训练,或者说训练多轮得到最优基线进行训练,这个时候怎么构建最优基线也是一个问题
1. 每个状态仅仅训练,或者说迭代了一轮,根本达不到所谓的收敛
2. 如果不训练一轮,而是训练多轮,那么这些
3. 首先按照现在的方法根本没有办法得到多轮的数据,

# 方案1 每个点改进10次用最后一次改进的进行学习,或者用最优改进的进行学习       完全静态目标,从原则来讲这个才能叫做PI2，但是这样后面的网络实际上没有更新
简单来说我们原来采用的是方案1,但是由于策略参数化可以立即学到,所以可以逐步改善,那么我们能不能逐步学习?因为现在的方案2我觉得没道理,虽然和方案1差不多
从大数定理来看差不多
# 方案2 每个点每次训练仅仅改进一次,后期通过训练改进10次(相当于边训练边改进)   动态目标,还是得采用这个方案

需要检查的:
网络统一
网络冗余
环境统一
是否真正的把网络复制过来了
有没有冗余代码
'''
# ================================== S超参数S ==================================
# 迭代次数
ITER_POLICY_NETWORK = 2000
# 批大小
BATCHSIZE_POLICY_NETWORK = 100
# 折扣参数
GAMA = 0.95
# K限制
# 讲道理K不能为负数,否则不满足稳定性原理
k_bound = [0, 10]
# 并行限制
# 全局变量——实际上初始化的环境都是一样的
ENV_IN_M = None
POLICY_IN_M = None
# ================================== F超参数F ==================================

# MAXMIN标准化
def minmax_nor(loss):
    res = (loss - loss.min())/(loss.max()-loss.min())
    return res

# 应该是三个维度的运算,不是一个维度的运算
class Agent():
    def __init__(self,roll_outs=roll_outs, explore_steps=EXPLORATION_TOTAL_STEPS, train_times=TRAIN_TIMES, train_steps=TRAIN_STEPS):
        # <<<<<<<<< 关键网络 <<<<<<<<<
        self.env = sim_env
        self.reward_model = Sample
        self.experience_buffer = PolicyBuffer()
        self.main_policy = main_policy
        # <<<<<<<<< PI2设置 <<<<<<<<<
        self.roll_outs = roll_outs                                          # 并行宽度
        self.explore_steps = explore_steps                                  # 并行深度
        self.train_time  = train_times                                      # 总策略改进次数
        self.train_steps  = train_steps                                     # 串行交互次数
        self.PI2_coefficient = PI2_coefficient                              # PI2超参数
        self.sigma = 1                                                      # 方差
        self.current_steps = 0                                              # 当前step次数
        self.current_training = 0                                           # 训练次数
        #<<<<<<<<PI2数据记录<<<<<<<<<<<
        # 我认为至少每50步训练一次,不至于说每次都训练,但是每50步必须训练一次
        self.batch_action = np.zeros(shape=(self.train_steps,self.env.action_dim),dtype=np.float64)     # 记录每次交互的动作
        self.batch_obs = np.zeros(shape=(self.train_steps,self.env.observation_dim),dtype=np.float64)   # 记录每次交互后的状态
        self.batch_reward = np.zeros(shape=(self.train_steps,1),dtype=np.float64)                  # 计算每一步的loss,感觉没啥用
        self.action_roll  = np.zeros(shape=(self.roll_outs,self.env.action_dim),dtype=np.float64)
        self.loss_roll = np.zeros(shape=(self.roll_outs,1),dtype=np.float64)                                                   # 记录训练中的loss用于策略改进
        self.loss_after_training =  np.zeros(shape=(self.train_time,1),dtype=np.float64)           # 记录每次更新后的奖励,action应该是拿不到了
        #<<<<<<记录策略网路参数<<<<<<<<<<<,

        self.current_obs  = np.zeros(shape=(self.env.observation_dim,1),dtype=np.float64)
        self.current_action =  np.zeros(shape=(self.env.action_dim,1),dtype=np.float64)
        # 绘图记录
    # 我认为一个地方都学不好,凭啥要随机初始化,一定不能随机初始化
    def reset_training_state(self):
        self.sigma = 1                            # 采用同样的方差
        self.current_training = 0                 # 重置训练次数
    def reset_step_state(self):
        self.current_obs = self.env.reset()
        self.sigma = 1
        self.current_steps = 0
    # 策略评估,记住应该及时的统一相关的env环境
    def policy_evl(self):
        _,self.action_roll,self.loss_roll = self.reward_model.get_episodes_reward_with_PID_mult(self.sigma,self.roll_outs)
    # 策略改进,是否改进一轮就够了?
    # 就算我想要改善多轮,policy必须进行学习
    # 我这个时候真的不应该去更新一下策略网络么
    def policy_improve(self):
        exponential_value_loss = np.zeros((self.roll_outs, 1), dtype=np.float64)  #
        probability_weighting = np.zeros((self.roll_outs, 1), dtype=np.float64)  # probability weighting of each roll
        if (self.loss_rolls.max() - self.loss_rolls.min() <= 1e-4):
            probability_weighting[:] = 1.0 / self.roll_outs
        else:
            exponential_value_loss[:] = np.exp(-self.PI2_coefficient * minmax_nor(self.loss_rolls)[:])
            probability_weighting[:] = exponential_value_loss[:] / np.sum(exponential_value_loss)
        self.current_action = np.dot(self.action_roll.T,probability_weighting)

        # 存储数据进行训练
        # self.experience_buffer.addExperience(np.reshape(self.current_obs,newshape=[1,self.env.observation_dim]),np.reshape(self.current_action,newshape=[1,self.env.action_dim]))
        return self.current_action
    # 当前的框架应该是正确的而且可以改变学习频率,但是还是batch学习比较好
    # 框架尽量明确
    def training(self):
        print("start training!")
        self.reset_training_state()
        while self.current_training < self.train_time:
            self.reset_step_state()
            while self.current_steps < self.train_steps:
                self.policy_evl()
                self.policy_improve()
                self.updateEnv()
                # 存储标签
                self.batch_obs[self.current_steps] = self.current_obs
                self.batch_action[self.current_steps] = self.current_action
                # 为啥要进行批训练
                self.current_steps += 1
            # 我觉得这样训练也可以
            self.updatepolicy()
            self.update_sample_policy()
            self.current_training +=  1
    # 这个地方我还真不能确定他能不能得到
    # 我还真的觉得多个计算图就可以优化这个过程,但是多个计算图咋样训练比较麻烦
    def update_sample_policy(self):
        for explore_policy in explore_policy_list:
            explore_policy.update_model_with_param(main_policy_scope)
    def updatepolicy(self):
        self.main_policy.learn(self.batch_obs,self.batch_action)
        self.main_policy.save_model()
    def updateEnv(self):
        self.current_obs,_,_,_ = self.env.step(self.current_training)
## 可以逐渐测试每一个部分的功能是否完善
if __name__ == '__main__':
    ag = Agent()
    ag.update_sample_policy()