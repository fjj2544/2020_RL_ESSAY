# import imageio
# import os
# import sys
#
# def create_gif(source, name, duration):
# 	"""
#      生成gif的函数，原始图片仅支持png
#      source: 为png图片列表（排好序）
#      name ：生成的文件名称
#      duration: 每张图片之间的时间间隔
# 	"""
# 	frames = []     # 读入缓冲区
# 	for img in source:
# 		frames.append(imageio.imread(img))
# 	imageio.mimsave(name, frames, 'GIF', duration=duration)
# 	print("处理完成")
#
# def main(or_path):
# 	"""
# 	or_path: 目标的文件夹
# 	"""
# 	path = os.chdir(or_path)
# 	pic_list = os.listdir()
# 	gif_name = "result.gif"  # 生成gif文件的名称
# 	duration_time = 0.5
# 	# 生成gif
# 	create_gif(pic_list, gif_name, duration_time)
#
# if __name__ == '__main__':
# 	parm_list = sys.argv
# 	if len(parm_list) != 2:
# 		print("请输入需要处理的文件夹！")
# 	else:
# 		main(parm_list[1])



import matplotlib.pyplot as plt
import imageio,os
from PIL import Image
GIF=[]
filepath="./photo"#文件路径
filenames=os.listdir(filepath)
for filename in os.listdir(filepath):
    GIF.append(imageio.imread(filepath+"\\"+filename))
imageio.mimsave(filepath+"\\"+'result.gif',GIF,duration=0.5)#这个duration是播放速度，数值越小，速度越快

