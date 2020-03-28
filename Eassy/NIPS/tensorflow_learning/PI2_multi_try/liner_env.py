import numpy as np

###############################################################################################
#                   飞行器模型
#   此为线性模型，没有内部改变的状态变量。但是其中的约束使用强制约束，而不是软约束
#   功能
#       seed(num_seed)：
#           输出：无
#           功能：设置随机种子
#       reset()：
#           输出：新的状态
#           功能：初始化状态
#       step(action)：
#           输出：状态+奖励+完成情况+信息
#           功能：模拟一步，输出
###############################################################################################
class Planes_Env:
    # 初始化
    def __init__(self):
        # self.actions = [0,1]
        self.observation_dim = 3
        self.action_dim = 1
        self.action_bound = [-20,20]
        # 状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]
        self.state = np.array([0.0, 0.0, 0.0])
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        self.max_steps = 400
        self.viewer = None
        # 飞机
        self.b_alpha = 0.073
        self.b_delta_z = -0.0035
        self.a_alpha = 0.7346
        self.a_delta_z = -2.8375
        self.a_q = 3.9779
        self.delta_z_mag = 0.1
        self.tau = 0.02
        self.theta_desired = 10
        self.dtheta_desired = 0
        # 角度阈值 攻角alpha[-1,10];舵偏delta_z[-20,20]
        self.alpha_threshold_max = 10
        self.alpha_threshold_min = -1
        self.delta_z_threhold_max = 20
        self.delta_z_threhold_min = -20
        self.reward = 0.0
        self.cost = 0.0

        self.delta_z = 0.0
        # 最大加速度,dez最大变化量
        self.max_delta_z_change = 1
        self.delta_z_change = 0
        self.last_delta_z = 0.0

    # 更新种子
    def seed(self,num_seed):
        np.random.seed(num_seed)
    def reset(self):
        n = np.random.randint(1, 1000, 1)
        np.random.seed(n)
        self.state = np.zeros((3, 1), dtype=np.float64)
        # self.state = np.random.uniform(-0.5, 0.5, size=(3,))
        self.observation = np.array([0.0, 0.0])
        self.steps_beyond_done = 0
        return self.state

    # 模拟进行一步
    def step(self, action):
        action = action[0]
        state = self.state
        alpha, theta, q = state
        observation_pre = theta - self.theta_desired
        # 动作约束
        self.delta_z = np.clip(action,self.delta_z_threhold_min,self.delta_z_threhold_max)
        # 动力学方程 攻角alpha，俯仰角theta 俯仰角速度q  舵偏delta_z
        alpha_dot = q - self.b_alpha * alpha - self.b_delta_z * self.delta_z
        theta_dot = q
        q_dot = -self.a_alpha * alpha - self.a_q * q - self.a_delta_z * self.delta_z
        # 积分得到状态量
        q = q + self.tau * q_dot
        theta = theta + self.tau * theta_dot
        alpha = np.clip(alpha + self.tau * alpha_dot, self.alpha_threshold_min, self.alpha_threshold_max)
        observation_cur = theta - self.theta_desired

        self.steps_beyond_done += 1
        self.state = np.array([alpha, theta, q])
        # 不满足要求则loss=1
        lose = alpha < self.alpha_threshold_min or alpha > self.alpha_threshold_max
        # TODO：设置回报
        if not lose:
            self.reward = 10*abs(theta - self.theta_desired)
        else:
            if alpha< self.alpha_threshold_min:
                self.reward += 1000*(self.alpha_threshold_min - alpha)
            else:
                self.reward += 1000 * (alpha - self.alpha_threshold_max)
        done = lose or self.steps_beyond_done >= self.max_steps
        self.observation = np.array([observation_pre, observation_cur])
        info = None
        return self.state, self.reward, done,info
