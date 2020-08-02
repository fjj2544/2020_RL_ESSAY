import gym
import time
import random
import sys
sys.path.append("../game/")
from game.grid_game import GridEnv
from game.maze_game import MazeEnv
"""
策略迭代一般流程
for s in States:
    initial v(s) ## 为每一个状态初始化值函数
    initial Action(s) ## 为每一个状态匹配行为,一般动作和状态空间env会给出
    initial PI(s,a)  ## 初始化策略（一般初始化为随机策略）
delta1 = inf
theta1 = eps1 ##计算精度
policy_stable = false
## 大循环是策略改善
while(policy_stable == false)
    ## 策略评估
    while(abs(delta1)>theta1)
        for s in s_space:
            v = v(s) ## 记录更新前值函数便于求delta
            v(s) = 0
            for a in a_space:
                v(s) += r+gmma*v(s') ## s,a->s'
            delta1 += abs(v-v(s))
    ## 策略改善
    policy_stable = True
    for s in s_space:
        old_a = pi(s)
        pi(s) = a | max(r+gmma*v(s')##这里的意思是选择一个最大累计汇报的动作赋值给Pi
        if old_a != pi(s):## 仍然有状态的策略不稳定
            policy_stable = false
    if policy_stable == True:
        retrun v*,pi*

"""


class DP_Policy_Iterator:
    def __init__(self, gym_env):
        ## 超参数
        self.gamma = 0.8
        ##
        self.env = gym_env
        self.env = self.env.unwrapped  ## 只有把他unwrapped之后才能真正的使用。。,虽然不知道为什么，但是老师的代码里面也没有unwrapped掉,所以会报错
        self.env.reset()
        self.states = self.env.observation_space
        self.actions = self.env.action_space
        self.v = dict()
        self.pi = dict()
        ## 初始化随机策略

        for state in self.states:
            self.pi[state] = self.actions[int(random.random() * len(self.actions))]
            self.v[state] = 0.0
    def policy_evaluation(self,iteration_times = 1000, iteration_eps = 1e-6):
        for i in range(iteration_times):
            delta = 0.0
            for state in self.states:
                action = self.pi[state]
                s, r, done, _ = self.env.transform(state, action)
                new_v = r + self.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            if delta < iteration_eps:
                print("dp_policy_evaluation times:", i)
                break

    def policy_improve(self):
        for state in self.states:
            ## 随便选一个合法的
            a1 = self.actions[0]
            s, r, done, _ = self.env.transform(state, a1)
            v1 = r + self.gamma * self.v[s]
            for action in self.actions:
                s, r, done, _ = self.env.transform(state, action)
                if v1 < r + self.gamma * self.v[s]:
                    a1 = action
                    v1 = r + self.gamma * self.v[s]
            self.pi[state] = a1

    def policy_iterator(self,iteration_times = 1000):
        for i in range(iteration_times):
            self.policy_evaluation()
            pi_last = self.pi.copy()
            self.policy_improve()
            if (self.pi == pi_last):
                print("policy iterator times", i)
                break
        return self.pi

    ## 可视化
    def show_env(self):
        ### 可以随机一个state为初始stats
        while (True):
            s = self.env.reset()
            done = False
            while (done == False):
                s, r, done, _ = self.env.step(self.pi[s])
                self.env.render()
                time.sleep(0.25)
        self.env.close()


if __name__ == '__main__':
    ## 创建Gym环境
    gym_env = MazeEnv()
    # gym_env = gym.make("GridGame-v0")
    ## 创建策略迭代类
    dp_policy = DP_Policy_Iterator(gym_env)
    ## 策略迭代
    dp_policy.policy_iterator()
    ## 展示结果
    dp_policy.show_env()
"""
import gym
env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(int(1e4)):
    env.step(env.action_space.sample())
    env.render('human')
env.close()  # https://github.com/openai/gym/issues/893
"""
