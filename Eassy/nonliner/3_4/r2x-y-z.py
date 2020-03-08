## 主要用来写作业的
import math
import numpy as np
r = np.array([[0.866,-0.5,0],[0.433,0.75,-0.5],[0.25,0.433,0.866]])
def get_r(row,col):
    return r[row-1,col-1]
def float_equ(a,b):
    if math.fabs(a-b) <= 1e-3:
        return True
    else:
        return False
def check_r():
    a1 = r[:,0]
    a2 = r[:,1]
    a3 = r[:,2].reshape((3,))
    b1 = r[0,:].reshape((3,))
    b2 = r[1,:].reshape((3,))
    b3 = r[2,:].reshape((3,))
    res = []
    # print(type(a1))
    # print(a1.shape)
    # print(a2.shape)
    res.append(np.dot(a1,a2))
    res.append(np.dot(a1,a3))
    res.append(np.dot(a2,a3))
    res.append(np.dot(b1,b2))
    res.append(np.dot(b1,b3))
    res.append(np.dot(b2,b3))
    res7 = np.linalg.det(r)
    flag = True
    for item in res:
        flag &= float_equ(item,0)
        print(flag,item)
    flag &= float_equ(res7,1)
    print(flag)
    return flag
def x_y_z_exp():
    beta = math.atan(-get_r(3,1)/math.sqrt(get_r(1,1)**2+get_r(2,1)**2))
    gmma = math.atan(get_r(3,2)/get_r(3,3))
    alpha = math.atan(get_r(2,1)/get_r(1,1))
    return math.degrees(gmma),math.degrees(beta),math.degrees(alpha)
if check_r():
    print("YES")
else:
    print("FUCK!")
ans = x_y_z_exp()
print(ans)
