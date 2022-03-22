'''
TODO:
    1. this program should recycle the memory immediately(explicitly)
    2. try different reduce order
    3. build the model based on SIMT to explain the result
'''
import numpy as np
from numpy.lib.shape_base import split
import cupy as cp
from cupy import cutensor
from time import perf_counter_ns
import time
import opt_einsum as oe

def generateTestDataDim(p, q, numberOfCell, geometryDim = 3, topologyDim = 3):
    I = q * (q+1) * (q+2) / 6
    K = M = (p+1) * (p+2) * (p+3) / 6
    J = numberOfCell
    L = geometryDim
    return list(map(int,(I,J,K,L,M)))

dtype = np.float64

I,J,K,L,M = generateTestDataDim(3, 3, 100000)
print("Data benchmark:[I,J,K,L,M]=[{},{},{},{},{}]\n".format(I, J, K, L, M))
extent = {'i': I, 'j': J, 'k': K, 'm': M, 'l': L}
mode_a = ('i')
mode_b = ('j', 'i', 'k', 'l')
mode_c = ('j', 'i', 'm', 'l')
mode_d = ('j')
mode_e = ('j', 'k', 'm')
def getSize(mode):
    return tuple(map(lambda x:extent[x], mode))

a = np.random.rand(*getSize(mode_a)).astype(dtype)
b = np.random.rand(*getSize(mode_b)).astype(dtype)
c = np.random.rand(*getSize(mode_c)).astype(dtype)
d = np.random.rand(*getSize(mode_d)).astype(dtype)


test_time = 5
for i in range(test_time):
    split_j = 10
    assert(J % split_j == 0)
    block_j = int(J/split_j)
    res_e_cpu = np.empty(getSize(mode_e)).astype(dtype)
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    for j_1 in range(split_j):
        ta = a
        tb = b[block_j*j_1:block_j*(j_1+1),:,:,:]
        tc = c[block_j*j_1:block_j*(j_1+1),:,:,:]
        td = d[block_j*j_1:block_j*(j_1+1)]
        # e8 = oe.contract('i,ijml,ijkl,j->jkm', a,b,c,d, backend='cupy')
        res_e_cpu[block_j*j_1:block_j*(j_1+1),:,:] = oe.contract('i,jikl,jiml,j->jkm', ta,tb,tc,td, backend='cupy').get()
    e1 = res_e_cpu
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

