save_model_path = "./agent_model/policy_net"
GAMA = 0.95
EXPLORATION_TOTAL_STEPS = 120  # 纵向深度,也就是每次纵向深度
TRAIN_STEPS = 100  # 横向深度,类似一个batch的数据,每次大概采集200(s,a)标签,不要随机初始化
TRAIN_TIMES = 100  # 每一个训练代表一个policy_improve,batch_improve的思想很重要  50
roll_outs = 20  # 探索宽度
buffer_size = 150000
PI2_coefficient = 30
