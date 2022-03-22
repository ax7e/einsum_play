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




print("cutensor-manual-split")
mode_f = ('i', 'j')
mode_g = ('i', 'j', 'k', 'l')
tmode_a = cutensor.create_mode(*mode_a)
tmode_b = cutensor.create_mode(*mode_b)
tmode_c = cutensor.create_mode(*mode_c)
tmode_d = cutensor.create_mode(*mode_d)
tmode_e = cutensor.create_mode(*mode_e)
tmode_f = cutensor.create_mode(*mode_f)
tmode_g = cutensor.create_mode(*mode_g)

for i in range(test_time):

    split_j = 10
    assert(J % split_j == 0)
    block_j = int(J/split_j)


    res_e_cpu = np.empty(getSize(mode_e)).astype(dtype)
    extent['j']=block_j
    ta = cp.empty(getSize(mode_a)).astype(dtype)
    tb = cp.empty(getSize(mode_b)).astype(dtype)
    tc = cp.empty(getSize(mode_c)).astype(dtype)
    td = cp.empty(getSize(mode_d)).astype(dtype)
    desc_c = cutensor.create_tensor_descriptor(tc)
    desc_b = cutensor.create_tensor_descriptor(tb)
    desc_a = cutensor.create_tensor_descriptor(ta)
    desc_d = cutensor.create_tensor_descriptor(td)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
        
        

    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    for j_1 in range(split_j):
        ta[:] = cp.asarray(a)
        tb[:,:,:,:] = cp.asarray(b[block_j*j_1:block_j*(j_1+1),:,:,:])
        tc[:,:,:,:] = cp.asarray(c[block_j*j_1:block_j*(j_1+1),:,:,:])
        td[:] = cp.asarray(d[block_j*j_1:block_j*(j_1+1)])
        t2_start = perf_counter_ns()
        f_gpu[:,:] = cutensor.contraction(1.0,
                                 ta, desc_a, tmode_a,
                                 td, desc_d, tmode_d,
                                 0.0,
                                 f_gpu, desc_f, tmode_f)
        g_gpu[:,:,:,:] = cutensor.contraction(1.0,
                                 f_gpu, desc_f, tmode_f,
                                 tb, desc_b, tmode_b,
                                 0.0,
                                 g_gpu, desc_g, tmode_g)
        e_gpu[:,:,:] = cutensor.contraction(1.0,
                                 g_gpu, desc_g, tmode_g,
                                 tc, desc_c, tmode_c,
                                 0.0,
                                 e_gpu, desc_e, tmode_e)
        cp.cuda.Stream.null.synchronize()
        t2_end = perf_counter_ns()
        res_e_cpu[block_j*j_1:block_j*(j_1+1),:,:] = e_gpu.get()
        print("Elapsed sub-time", (t2_end-t2_start)/1000000, 'ms')
    extent['j']=J
    e1 = res_e_cpu
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

#print(np.allclose(e0,e1))

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
    for k_1 in range(split_j):
        ta = a
        tb = b[block_j*k_1:block_j*(k_1+1),:,:,:]
        tc = c[block_j*k_1:block_j*(k_1+1),:,:,:]
        td = d[block_j*k_1:block_j*(k_1+1)]
        res_e_cpu[block_j*k_1:block_j*(k_1+1),:,:] = oe.contract('i,jikl,jiml,j->jkm', ta,tb,tc,td, backend='cupy').get()
    e1 = res_e_cpu
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-einsum")
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e0 = oe.contract('i,jikl,jiml,j->jkm', a,b,c,d, backend='cupy').get()
    cp.cuda.Stream.null.synchronize()
    t1_stop = perf_counter_ns()
    print("Elapsed time(cupy-Enisum):", (t1_stop-t1_start)/1000000, 'ms')


print(np.allclose(e0,e1))