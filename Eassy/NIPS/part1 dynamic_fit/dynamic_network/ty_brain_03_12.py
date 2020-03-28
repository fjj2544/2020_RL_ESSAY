import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
class Experience_Buffer(): # 采样器,其实就是一个采样的结构
    def __init__(self,buffer_size = 15000):
        self.buffer = []
        self.buffer_size = buffer_size
    # queen 队列的操作
    # 往队列中添加经验,这里应该是合并吧,怎么可以添加呢？
    def add_experience(self, experience):
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:len(self.buffer)+len(experience)-self.buffer_size]=[]
        ## TODO:可以这么直接添加么,[1,2,3] append ([2,3,4])
        self.buffer.append(experience)
    # TODO:从经验池中随机采样本,现在本来就要时序数据,为什么可以随机采样?随机采样是不正确的吧?
    def sample(self,samples_num, device):
        # 在经验池采样
        sample_data = random.sample(self.buffer, samples_num)
        # print(sample_data)
        # print(len(sample_data))
        state = torch.FloatTensor([x[0] for x in sample_data])
        action = torch.FloatTensor([x[1] for x in sample_data])
        state_ = torch.FloatTensor([x[2] for x in sample_data])
        return state, action, state_
    # TODO:清空经验池的操作为啥没有?或者说随机清空的操作为啥没有,最简单当然是清空经验池,但是结合我们的算法又不应该直接清空经验池
class Dynamic_Net(nn.Module):
    def __init__(self, state_num, action_num):
        super(Dynamic_Net, self).__init__()
        self.fc1 = nn.Linear(state_num+action_num,200)
        # 第二层连接：隐含-输出
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 100)
        self.fc5 = nn.Linear(100, state_num)
    # TODO:为什么拼接有问题？
    '''
    torch.cat((out1, out2), dim=1) 竖着拼接
    out1[0,:]
    out1[1,:]
    out2[0,:]
    ...
    torch.cat((out1, out2), dim=0) 竖着拼接
    out1[:,0] out1[:,1] out2[:,0]     
    '''
    def forward(self, x, y):
        '''连接网络'''

        out1 = self.fc1_1(x)
        out2 = self.fc1_2(y)
        try:
            out = torch.cat((out1, out2), dim=1)
        except:
            out = torch.cat((out1, out2), dim=0)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        out = F.relu(out)
        out = self.fc5(out)
        return out



'''------------------------------动力学模型板块------------------------------'''
class Planes_Env_fake():
    # 状态：altitude, Mach, theta, pitch, arfa, rrange, mass, omega_z
    # 动作：KP, KD, KI
    def __init__(self, buffer_size = 10000, state_num=8, action_num=3):
        self.state = []
        self.altitude = 0,0
        self.Mach = 0.0
        self.theta = 0.0
        self.pitch = 0.0
        self.arfa = 0.0
        self.rrange = 0
        self.mass = 0
        self.omega_z = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer = Experience_Buffer(buffer_size)
        self.network = Dynamic_Net(state_num, action_num).to(self.device)
        self.criterion = nn.L1Loss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.005)

    def init_state(self,env_real):
        self.state = [env_real.altitude, env_real.Mach, env_real.theta, env_real.pitch,
                      env_real.arfa, env_real.rrange, env_real.mass, env_real.omega_z]
        self.altitude = env_real.altitude
        self.Mach = env_real.Mach
        self.theta = env_real.theta
        self.pitch = env_real.pitch
        self.arfa = env_real.arfa
        self.rrange = env_real.rrange
        self.mass = env_real.mass
        self.omega_z = env_real.omega_z

    def learn(self):
        state, action, state_ = self.buffer.sample(256, self.device)   # 获得GPU下的s,a,s'
        torch_dataset = Data.TensorDataset(state, action, state_)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2
        )
        for epoch in range(3):  # 全部的数据使用3遍，train entire dataset 3 times
            for step, (state, action, state_) in enumerate(loader):
                state = state.to(self.device)
                action = action.to(self.device)
                state_ = state_.to(self.device)
                state_ = state_.float()
                eval_state_ = self.network.forward(state, action)
                loss = self.criterion(eval_state_, state_)
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def step(self,action):
        # 动作
        action = torch.FloatTensor(action).to(self.device)
        self.state = torch.FloatTensor(self.state).to(self.device)
        state_ = self.network.forward(self.state, action)
        state_ = state_.data.cpu().numpy()
        state_ = state_.tolist()
        self.state = state_
        self.altitude = self.state[0]
        self.Mach = self.state[1]
        self.theta = self.state[2]
        self.pitch = self.state[3]
        self.arfa = self.state[4]
        self.rrange = self.state[5]
        self.mass = self.state[6]
        self.omega_z = self.state[7]
        return state_

    def save_model(self, i=0):
        # 保存模型
        torch.save(self.network, "model_train_face"+str(i) +".pkl")

    def add_experience(self,state, action, state_):
        # 采样
        experience = [state, action, state_]
        self.buffer.add_experience(experience)

# net1 = Dynamic_Net(2, 3)
# print(net1)