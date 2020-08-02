#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
################################# ######################################
import numpy as np
import pickle
import time
## 记录所有状态的数量
counter = 0
## TODO:如果更改棋盘需要更改这个地方3->4
BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
## TODO:这个主要设置状态，便于后面遍历的时候记录搜索树的所有状态.这个主要是记录棋盘的
class State:
    def __init__(self):
        # the board is represented by an n * n array, N*N阵列
        # 1 represents a chessman of the player who moves first, 先手方
        # -1 represents a chessman of another player 另外一个棋手
        # 0 represents an empty position 空棋盘 data记录了棋盘的情况
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None
    # compute the hash value for one state, it's unique，
    # 确定状态的值，作为最初的估计
    # TODO:这个地方不需要更改，除非减少棋盘的数量，但是其实变成二进制文件之后无所谓,如果一一对应是否可以考虑反向哈希?双表法,如果可以反向查表就可以减少遍历过程
    # TODO:×3是为了保证每一个位置的权重不同
    def hash(self):## 保证每一个棋盘都有不同的编号，能够实现编号
        if self.hash_val is None:## 确定当前棋盘是否有值
            self.hash_val = 0 ## 表示整个棋盘当前的预判值
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):#遍历整个棋盘空间
                if i == -1:##这里的-1 0 1 ->2 0 1
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)
    # check whether a player has won the game, or it's a tie
    # 判断当前棋盘的结果,不同的play的策略是否有区别？
    # 利用冗余简化代码
    # TODO:如果要变成4*4需要改动获胜条件3->4 -3->-4，其他不需要改动
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row 是否一行有三个
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns 是否一列有三个
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        # check diagonals 是否对角线上有三个,对角线方程可能要改
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - 1 - i]

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                #self.print_state()
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                #self.print_state()
                return self.end

        # whether it's a tie ,所有棋盘占满就是平局
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            self.winner = 0
            self.end = True
            #self.print_state()
            return self.end
        # game is still going on,游戏没有结束
        self.end = False
        return self.end
    # TODO:其实第一次遍历的时候我们已经知道了哪些是结束局，哪些是平凡局,平凡局没有必要浪费时间，可以考虑多重表法
    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    # 模仿下棋这个动作,利用symbol模仿下棋这个动作
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state
    # print the board
    # 绘制棋盘,每次都重新绘制棋盘，刷新棋盘,每次都重新绘制棋盘，这个人机交互的时候才有用
    def print_state(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')
## 预先学习，所有内容，基本实现辨识板块
## 这个函数更新棋盘相关状态，实现强化学习的内容
## 这个部分就非常耗时，如果采用基本的hash方法会很慢，可以考虑剪枝
## 这个部分的复杂度是理论上是16的阶乘，但其实带上一个减枝，因为如果棋盘不能再走那么就不需要再走下去,每次提前试探了一步，保证可以提前预知信息。
## 暴搜需要时间，但是查表可能不需要很多时间
## TODO:用一个减枝的暴力搜索遍历所有的可能棋盘,仅仅存储了这个棋盘是否结束
def get_all_states_impl(current_state, current_symbol, all_states ):
    global  counter

    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if current_state.data[i][j] == 0:
                newState = current_state.next_state(i, j, current_symbol) # 更新新的状态
                newHash = newState.hash() # 更新新状态的hash值,只要每一个棋子只有3中状态，那么这个hash就是对的
                if newHash not in all_states.keys(): # 只要保证每一个状态的hash值是不同的就好了
                    isEnd = newState.is_end()  ## 判断是否结束
                    all_states[newHash] = (newState, isEnd)
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)
                    else:
                        counter = counter + 1
## 得到所有状态,symbol确定棋手,采用了数值hash的方案
def get_all_states():
    current_symbol = 1 ## 先手下棋
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
# all possible board configurations
# 获取所有的可能情况和对应的可能结果，但是这个预处理太慢，可能完全无法进行，说白了就是得到了每一个状态的一个hash表
all_states  = get_all_states()
## TODO:建立反向查询表
state_hash =    dict()
for key,state in all_states.items():
    state_hash[state[0]] = key

# all_states = load_obj('test1')
## TODO:用于train和判别的CLASS
class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1 棋盘标记和输赢一样，用双标记法
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol) ##这个仅仅会进行一次，如果仅仅进行一次不是大事情
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    ##  TODO:重置棋盘状态,每一局都会进行
    def reset(self):
        self.p1.reset()
        self.p2.reset()
    ## 实现轮流下棋的过程
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2
    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        while True:
            player = next(alternator)
            if print_state:
                current_state.print_state()
            [i, j, symbol] = player.act()   ### TODO:重点函数，就是电脑怎么下棋的,可以改动的地方
            next_state_hash = current_state.next_state(i, j, symbol).hash()  ## 这个只是记录函数，都没有什么改的
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if is_end:
                if print_state:
                    current_state.print_state()
                return current_state.winner
# AI player TODO:如何保证遍历到了所有的状态？如何才能保证all_state 得到了完全的遍历？
## TODO:关键类,如果要改智能改动这个类
### 基本更新公式 Vn+1 = Vn + step_size*(Vn+1 - Vn)  策略估计更新公式  TODO:如何看出是延迟回报？
class Player:
    # @step_size: the step size to update estimations，更新步长,类似学习率
    # @epsilon: the probability to explore，探索的概率
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict() ## 每一个棋盘对应的值函数
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
    # 重置玩家
    def reset(self):
        self.states = []
        self.greedy = [] ## 记录哪一步贪心，哪一步不是贪心
    # 设置状态
    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)
    # 设置对应的符号,这个地方为啥要更新所有的状态，这个时间也太长了吧,这个是干啥。。
    # 首先会给所有的终局一个VALUE
    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states.keys():
            (state, is_end) = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5
    # 进行值函数迭代,更新策略估计
    # update value estimation
    ### TODO:这个地方可以改迭代算法，根据后面的知识,这个每一局都要进行
    def backup(self):
        # for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print_state()
        ## 每一局结束了逆向更新
        # 必须逆向更新,每一局所有状态的hash值都要重新算一遍,TODO:那我就奇怪了这个地方为啥不查表
        self.states = [state.hash() for state in self.states]
        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]
            td_error = self.greedy[i] * (self.estimations[self.states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error
    ## 确定下一个行为，TODO:重点函数，如果想改最好改这个地方,这个函数一定可以优化

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())
        ## todo:探索的部分不学是否会更好？,下面这个函数返回一个1之内的数据,下面这个action不是固定的数据
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action
        ## 这里是一个贪心行为，保证最大的values的action可以被选中
        values = []
        for hash, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash], pos))
        # to select one of the actions of equal value at random
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action
    # 保存对应的策略方案
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)
    # 加载对应的策略方案
    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
### TODO:这个类仅仅是交互的功能和训练基本没有关系
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        return

    def reset(self):
        return

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        while key=='':
            key = input("Input your position:")
        data = self.keys.index(key)
        i = data // int(BOARD_COLS)
        j = data % BOARD_COLS
        return (i, j, self.symbol)
## todo:训练的主要代码
## todo:动态调参
def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01) # 建立探索和策略模型
    player2 = Player(epsilon=0.01) # 建立探索和策略模型
    judger = Judger(player1, player2) # 建立对应的评价函数
    player1_win = 0.0
    player2_win = 0.0
    ### 一定是最后达到一种均衡的状态才能算作训练完成，下一个棋盘类，优化棋盘类别
    s_time = time.time()
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            e_time = time.time()
            print(e_time - s_time)
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
            s_time = time.time()
        ## TODO:必须理解为什么这个地方backup,这是一个更新学习的过程
        player1.backup() ## 这个地方比较耗时
        player2.backup()
        judger.reset()
    player1.save_policy()
    player2.save_policy()
## todo:这里主要是一个测试集的过程，测试是否完全对抗，当然局部最优可能会陷死。
def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(0, turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie. TODO:讲道理是这样的,0权游戏如果最优必然收敛到平衡点
# So we test whether the AI can guarantee at least a tie if it goes second.
#
def play():
    player1 = HumanPlayer()
    player2 = Player(epsilon=0, step_size=0.3)
    judger = Judger(player1, player2)
    player2.load_policy()
    while True:
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")
        player2.backup()
        judger.reset()
        key = input("end ?")
        if key=='y':
            break
    player2.save_policy()
if __name__ == '__main__':
    print(len(all_states))
    # print(counter)
    # s_time = time.time()
    # train(int(1e4))
    # e_time = time.time()
    # print(e_time - s_time,'S')
    # compete(int(1e3))
    # play()
