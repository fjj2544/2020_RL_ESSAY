import numpy as np
import matplotlib.pyplot as plt
# 辅助模块
linewidth = 1 # 绘图中曲线宽度
fontsize = 5 # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5 #图例中字体大小

'''------------------------------辅助模块------------------------------'''
#按照path创建文件夹 e.g. mkdir("./figure/")
def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

# 存储实验数据 e.g. save_data("./data/","loss.txt",loss)
def save_data(dir,name,data):
    mkdir(dir)
    np.savetxt(dir+name,data)

# 读取实验数据 e.g. save_data("./data/loss.txt")
def read_data(dir):
    data = np.loadtxt(dir)
    return data

# 保存实验图片 e.g. save_figure("./figure/","loss.pdf")
def save_figure(dir,name):
    mkdir(dir)
    plt.savefig(dir+name,bbox_inches = 'tight')

# 绘制控制曲线
def plot_result(alpha_list,delta_z_list,theta_list,theta_desire_list,figure_number = 3):
    '''绘图参数定义'''
    label = ["Optimal Parameters using FR-PI2 ","Adjusted parameters ","Reference parameter","Reference parameter"]
    color = ["r","g","b","k"]
    line_style = ["-","-.",":","--"]
    '''绘制alpha曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("Attack Angle $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(alpha_list[i], label=label[i],color=color[i],linestyle=line_style[i],linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "alpha_Curve.pdf")
    plt.show()
    '''绘制delta_z曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman',fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman',fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("Elevator $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(delta_z_list[i], label=label[i], color=color[i], linestyle=line_style[i],linewidth=linewidth)
    plt.legend(loc='best',  prop={'family':'Times New Roman', 'size':legend_font_size})
    save_figure("./photo/exp1/", "delta_z_Curve.pdf")
    plt.show()
    '''绘制theta曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.005s)$",fontproperties='Times New Roman',fontsize=fontsize)
    plt.ylabel("theta Angle $(Degree)$",fontproperties='Times New Roman',fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(theta_list[i], label=label[i], color=color[i], linestyle=line_style[i],linewidth=linewidth)
    plt.plot(theta_desire_list[0], label="$\\theta_{target}$", linestyle="--",linewidth=linewidth)
    plt.legend(loc='best', prop={'family':'Times New Roman', 'size':legend_font_size})
    save_figure("./photo/exp1/", "theta_Curve.pdf")
    plt.show()

# 对比方法时绘制损失变化曲线
def plot_loss_k(K_after_training_list ,loss_after_training_list,train_time,figure_number = 3):

    label = ["FR-PI2","F-PI2","PI2","Reference parameter"]
    color = ["r", "g", "b", "k"]
    line_style = ["-", "--", ":", "-."]
    marker = ['*', '^', 'h']
    "绘制LOSS曲线"
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Number of Iterations",fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Loss",fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(loss_after_training_list[i][:train_time], label=label[i], color=color[i],
                     linestyle=line_style[i], marker=marker[i],markersize=markersize,linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    # plt.title("Loss Function Curve With Natural PI2", fontdict={'family': 'Times New Roman'})
    save_figure("./photo/exp1/", "loss.pdf")
    plt.show()
