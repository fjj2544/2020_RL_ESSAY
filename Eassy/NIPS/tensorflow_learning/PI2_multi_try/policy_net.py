'''主要用来构建策略网络和值网络的,后面便于调用'''

import numpy as np
from liner_env import Planes_Env
# ===============超参数S===============
save_model_dir = "./current_best_policy"
# 迭代次数
ITER_POLICY_NETWORK = 2000
# 批大小
BATCHSIZE_POLICY_NETWORK = 100
# ===============超参数F===============

import tensorflow as tf

# 可以学习那个模型进行load,当然也可以不存储直接load
# ===============策略网络===============
class Policy_Net():
    # 初始化网络
    def __init__(self, env, name, lr=0.0001, model_file=None , trainable=True):
        self.n_features = env.observation_dim       # 输入特征维度
        self.n_actions = env.action_dim             # 动作空间维度,特别现在是连续动作
        self.learning_rate = lr                     # 学习率
        self.loss_alpha = 0.95                      # 过去的loss占比
        self.batch = BATCHSIZE_POLICY_NETWORK       # 批大小
        self.iter = ITER_POLICY_NETWORK             # 迭代总次数
        self.action_bound = [-20, 20]                    # 动作限幅
        self.name = name  #网络的名字
        # 采样参数
        self.sigma_exploration = tf.placeholder(tf.float32, shape=[None, 1])
        # 动作网络
        self.action,self.pi_params = self._build_net(name=self.name,trainable=trainable)


        # if trainable:
        #     self._build_train_op()
        # 保证 TENSORFLOW 的训练过程分配足够合理
        # gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction = 0.1)
        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))
        #
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)
        # # 可以恢复模型,当然也可以直接得到模型
        # self.save_path = "./model/a2c/"
        # # 6.定义保存和恢复模型
        # self.saver = tf.train.Saver()
        # if model_file is not None:
        #     self.restore_model(model_file)
    def _build_net(self,name,trainable):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            self.input_layer = tf.placeholder(tf.float32, shape=[None, self.n_features])
            # 2:    100神经元&relu
            self.policy_1 = tf.layers.dense(inputs=self.input_layer, units=100, activation=tf.nn.relu, trainable=trainable)
            # 3:    60神经元&relu
            self.policy_2 = tf.layers.dense(inputs=self.policy_1, units=60, activation=tf.nn.relu, trainable=trainable)
            # 动作
            action = 20 * tf.layers.dense(inputs=self.policy_2, units=self.n_actions, activation=tf.nn.tanh, trainable=trainable)
            # 网络参数
            pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return action,pi_params
    ## 现在类似一种ddpg的写法
    def _build_train_op(self):
        self.target_action = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.losses.mean_squared_error(labels= self.target_action,predictions=self.action)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    # 采样动作
    def policy_sample(self, sigma, observation):
        action = self.sess.run([self.sample], feed_dict={self.input_layer: observation, self.sigma_exploration: sigma})
        return action
    # 实际动作,简单的前向网络
    def policy_predict(self, observation):
        action = self.sess.run([self.action], feed_dict={self.input_layer: observation})
        return action
    # 通过新的参数更新网络
    # 对于每一个网络必须都要更新
    def update_model_with_param(self,param):
        self.sess.run(self.init_op)
        self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(param, self.pi_params)]
        self.sess.run(self.update_oldpi_op)
    # 奇葩的学习错误
    # 主要是更新尺度的问题
    def learn(self, batch_obs, batch_act_best,epoches=10):
        train_obs = []
        train_act = []
        for obs in batch_obs:
            train_obs.append(obs)
        train_obs = np.reshape(train_obs, [len(batch_obs),len(batch_obs[0])])
        for act in batch_act_best:
            train_act.append(act)
        train_act = np.reshape(train_act, [len(batch_act_best),len(batch_act_best[0])])


        for i in range(epoches):
            self.sess.run([self.train_op], feed_dict={self.input_layer: train_obs, self.target_action: train_act})
        loss = self.sess.run([self.loss], feed_dict={self.input_layer: train_obs, self.target_action: train_act})
        return float(loss[0])
    # 保存部分应该学其他人的
    def save_model(self):
        self.saver.save(self.sess, self.save_path)
    # 重新读取对应的模型
    def load_model(self):
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            # 注意全局初始化
            self.sess.run(self.initializer)
            self.saver.restore(self.sess, load_path)
        except Exception as e:
            print(e)
            print("No saved model to load, starting a new model from scratch.")
        else:
            print("Loaded model: {}".format(load_path))
## 局部测试
## 采用局部变量的方法
if __name__ == '__main__':
    # 证明网络覆盖有效果
    env = Planes_Env()
    obs = np.reshape([1,2,3],newshape=[1,3])
    net1 = Policy_Net(env=env,name="main_policy_net",trainable=True)
    res1 = net1.policy_predict(obs)

    net2 = Policy_Net(env=env,name="sample_policy",trainable=False)

    with tf.variable_scope("main_policy_net"):
        print(tf.get_variable_scope().reuse)
        tf.get_variable_scope().reuse_variables()
        print(tf.get_variable_scope().reuse)

    # res2 = net2.policy_predict(obs)

    # net2.update_model_with_param(net1.pi_params)
    #
    # res3 = net2.policy_predict(obs)
    #
    # res4 = net1.policy_predict(obs)
    # print(res1,res2,res3,res4)
