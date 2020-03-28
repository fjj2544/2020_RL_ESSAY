'''
sample for liner_plane_model
随机种子很关键,不然肯定会学习炸掉
TODO:环境一定要有随机种子  这个很坑
现在我可以加经验buffer了
TODO: 还是均方误差厉害
'''
import numpy as np
from env.plane.liner_env import Planes_Env
from dynamic_network import liner_plane_brain
from Algorithm.PI2 import PI2
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
    def get_one_episodes_reward_with_PID(self, K, total_step,dynamic_net = None):
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
            if dynamic_net is not None:
                action = np.reshape(action, [1, 1])
                observation = np.reshape(observation, [1, self.ob_dim])
                observation_ = dynamic_net.prediction(np.hstack((observation, action)))
            else:
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
        batch_rewards.append(loss)
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.observation_dim])
        batch_obs_next = np.reshape(batch_obs_next, [len(batch_obs), self.env.observation_dim])
        batch_delta = batch_obs_next - batch_obs
        batch_actions = np.reshape(batch_actions, [len(batch_actions), 1])
        batch_obs_action = np.hstack((batch_obs, batch_actions))
        batch_rewards = np.reshape(batch_rewards, [len(batch_rewards), 1])
        return batch_obs_action, batch_delta, batch_obs_next, batch_rewards
    # 串行采样
    def get_episodes_reward_with_PID(self, K, total_step, total_num,dynamic_net = None):
        batch_obs_next = []
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        for i in range(total_num):
            ierror = 0
            observation = self.env.reset()
            theta_list  = []
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
                action = K[i][0] * error + K[i][1] * derror + K[i][2] * ierror
                '''save data'''
                if dynamic_net is not None:
                    action = np.reshape(action, [1, 1])
                    observation = np.reshape(observation, [1, self.ob_dim])
                    observation_ = dynamic_net.prediction(np.hstack((observation, action)))
                else:
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
if __name__ == '__main__':
    env = Planes_Env()
    sampler = Sample(env)
    dynamic_net = liner_plane_brain.Dynamic_Net(env)
    RL_BRAIN =PI2(env,sampler,dynamic_net)
    RL_BRAIN.reset_training_state()
    RL_BRAIN.training()
