from policy_net import Policy_Net
import liner_env
# -----------------共享参数-------------

N_PROC =  8
GAMA = 0.95
EXPLORATION_TOTAL_STEPS = 120 # 纵向深度,也就是每次纵向深度
TRAIN_STEPS = 200 # 横向深度,类似一个batch的数据,每次大概采集200(s,a)标签,不要随机初始化
TRAIN_TIMES = 50  #每一个训练代表一个policy_improve,batch_improve的思想很重要
roll_outs = 20 ## 探索宽度
sim_env = liner_env.Planes_Env() #环境
buffer_size = 150000
PI2_coefficient = 30

#-----共享网络----------
explore_policy_list = []
for i in range(N_PROC):
    explore_policy_list.append(Policy_Net(env= liner_env.Planes_Env(), name = str(i), trainable=False))
##------------主要训练网络-----
main_policy_scope = "main_policy_net"
main_policy = Policy_Net(env=sim_env, name=main_policy_scope, trainable=True)

