import liner_env
from policy_brain import Policy_Net
# ===============超参数S===============
save_model_dir = "./current_best_policy"

GAMA = 0.95
EXPLORATION_TOTAL_STEPS = 120 # 纵向深度,也就是每次纵向深度
TRAIN_STEPS = 100 # 横向深度,类似一个batch的数据,每次大概采集200(s,a)标签,不要随机初始化
TRAIN_TIMES = 30  #每一个训练代表一个policy_improve,batch_improve的思想很重要  50
roll_outs = 20 ## 探索宽度
sim_env = liner_env.Planes_Env() #环境
buffer_size = 150000
PI2_coefficient = 30

# print(sim_env.reset())
## 主要网络
main_policy = Policy_Net(observation_dim=sim_env.observation_dim, action_dim=sim_env.action_dim,
                         policy_name="main_policy", model_file="./agent_model/policy_net")
# main_policy = Policy_Net(sim_env,"main_policy")