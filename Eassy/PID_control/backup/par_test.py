# # # # import multiprocessing as mp
# # # # #import pdb
# # # #
# # # # def job(i):
# # # #     # my_list.append(i)
# # # #     return [i*i,i+2,i+3]
# # # #
# # # # def multicore():
# # # #     pool = mp.Pool(processes=mp.cpu_count())
# # # #     # #pdb.set_trace()
# # # #     # res = pool.map(job, range(10))
# # # #     # for j in range(10):
# # # #     #     print(res[j][0],res[j][1],res[j][2])
# # # #     # res = pool.apply_async(job, (2,))
# # # #     # # 用get获得结果
# # # #     # print(res.get())
# # # #     # # 迭代器，i=0时apply一次，i=1时apply一次等等
# # # #     multi_res = [pool.apply_async(job, (i,)) for i in range(10)]
# # # #     # # 从迭代器中取出
# # # #     x =  [res.get()[0] for res in multi_res]
# # # #     print(x)
# # # # if __name__ == '__main__':
# # # #     multicore()
# # # #     ## 随机是可以并行的，最内层的N可以并行
# # """  CUDA"""
# # # import pycuda.autoinit
# # # import pycuda.driver as drv
# # # import numpy
# # # from pycuda.compiler import SourceModule
# # # mod = SourceModule("""
# # # __global__ void multiply_them(float *dest, float *a, float *b)
# # # {
# # #  const int i = threadIdx.x;
# # #  dest[i] = a[i] * b[i];
# # # }
# # # """)
# # # multiply_them = mod.get_function("multiply_them")
# # # a = numpy.random.randn(400).astype(numpy.float32)
# # # b = numpy.random.randn(400).astype(numpy.float32)
# # # dest = numpy.zeros_like(a)
# # # multiply_them(
# # #   drv.Out(dest), drv.In(a), drv.In(b),
# # #   block=(400,1,1), grid=(1,1))
# # # print(dest-a*b)
# # # #tips: copy from hello_gpu.py in the package.
# # #
# # # """ NUMBA """
# # import numpy as np
# # from numba import jit, vectorize
# # import time
# #
# #
# # # 不使用jit函数修饰器的函数
# # def func1_without_jit(A, C):
# #     for i in range(30):
# #         for j in range(30):
# #             grid = A[10 * i:10 * (i + 1), 10 * j:10 * (j + 1)]
# #             C[i, j] = grid.sum()
# #     return C
# #
# #
# # # 使用jit函数修饰器
# # @jit(nopython=True, nogil=True)
# # def func1_with_jit(A, C):
# #     for i in range(30):
# #         for j in range(30):
# #             grid = A[10 * i:10 * (i + 1), 10 * j:10 * (j + 1)]
# #             C[i, j] = grid.sum()
# #     return C
# #
# #
# # # 使用jit函数修饰器
# # @jit(nopython=True, nogil=True)
# # def func1_with_jit1(A, C):
# #     for i in range(30):
# #         for j in range(30):
# #             grid = A[10 * i:10 * (i + 1), 10 * j:10 * (j + 1)]
# #             C[i, j] = 0
# #             for m in range(10):
# #                 for n in range(10):
# #                     C[i, j] += grid[m, n]
# #     return C
# #
# #
# # if __name__ == '__main__':
# #     A = np.random.rand(90000).astype(np.float32)
# #     A = A.reshape(300, 300)
# #     C = np.zeros([30, 30], dtype=np.float32)
# #     # 先对三个函数预编译一次
# #     C = func1_without_jit(A, C)
# #     C = func1_with_jit(A, C)
# #     C = func1_with_jit1(A, C)
# #
# #     # 计算func1_without_jit运行5000次的时间
# #     t1 = time.time()
# #     for i in range(5000):
# #         C = func1_without_jit(A, C)
# #     print('Time without jit is:' + str(time.time() - t1))
# #
# #     # 计算func1_with_jit运行5000次的时间
# #     t1 = time.time()
# #     for i in range(5000):
# #         C = func1_with_jit(A, C)
# #     print('Time with jit is:' + str(time.time() - t1))
# #
# #     # 计算func1_with_jit1运行5000次的时间
# #     t1 = time.time()
# #     for i in range(5000):
# #         C = func1_with_jit1(A, C)
# #     print('Time with jit1 is:' + str(time.time() - t1))
#
# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy
#
# from pycuda.compiler import SourceModule
#
# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)
#
# multiply_them = mod.get_function("multiply_them")
#
# a = numpy.random.randn(400).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)
#
# dest = numpy.zeros_like(a)
# multiply_them(
#     drv.Out(dest), drv.In(a), drv.In(b),
#     block=(400, 1, 1), grid=(1, 1))
#
# print(dest - a * b)

if __name__ == '__main__':
    x = [1,2,3,4]
    print(x[2:])