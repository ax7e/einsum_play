'''
TODO:
    1. this program should recycle the memory immediately(explicitly)
    2. try different reduce order
    3. build the model based on SIMT to explain the result
'''
import numpy as np
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

dtype = np.float32

I,J,K,L,M = generateTestDataDim(3, 1, 100000)
print("Data benchmark:[I,J,K,L,M]=[{},{},{},{},{}]\n".format(I, J, K, L, M))
extent = {'i': I, 'j': J, 'k': K, 'm': M, 'l': L}
mode_a = ('i')
mode_b = ('k', 'j', 'i', 'l')
mode_c = ('j', 'i', 'l', 'm')
mode_d = ('j')
mode_e = ('j', 'k', 'm')
def getSize(mode):
    return tuple(map(lambda x:extent[x], mode))

a = np.random.rand(*getSize(mode_a)).astype(dtype)
b = np.random.rand(*getSize(mode_b)).astype(dtype)
c = np.random.rand(*getSize(mode_c)).astype(dtype)
d = np.random.rand(*getSize(mode_d)).astype(dtype)


'''
path_info = oe.contract_path('i, ijkl, ijml, j->jkm', a, b, c, d)

print(path_info[0])
print(path_info[1])

'''

a_gpu = cp.asarray(a)
b_gpu = cp.asarray(b)
c_gpu = cp.asarray(c)
d_gpu = cp.asarray(d)
test_time = 5

print("oe auto")
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e8 = oe.contract('i,kjil,jilm,j->jkm', a,b,c,d, backend='cupy')
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(opteinsum-manual):", (t1_end-t1_start)/1000000, 'ms')

extent = {'i': I, 'j': J, 'k': K, 'm': M, 'l': L}
mode_a = ('i')
mode_b = ('i', 'j', 'm', 'l')
mode_c = ('i', 'j', 'k', 'l')
a = np.random.rand(*getSize(mode_a)).astype(dtype)
b = np.random.rand(*getSize(mode_b)).astype(dtype)
c = np.random.rand(*getSize(mode_c)).astype(dtype)
d = np.random.rand(*getSize(mode_d)).astype(dtype)
a_gpu = cp.asarray(a)
b_gpu = cp.asarray(b)
c_gpu = cp.asarray(c)
d_gpu = cp.asarray(d)
test_time = 5

print("oe auto")
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e8 = oe.contract('i,ijml,ijkl,j->jkm', a,b,c,d, backend='cupy')
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(opteinsum-manual):", (t1_end-t1_start)/1000000, 'ms')

# Start the stopwatch / counter
'''

for i in range(1):
    t1_start = perf_counter_ns()
    e0 = np.einsum('i, ijkl, ijml, j->jkm', a, b, c, d)
    t1_stop = perf_counter_ns()
    print("Elapsed time(numpy-Enisum):", (t1_stop-t1_start)/1000000, 'ms')



print("cutensor-einsum")

for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e_gpu = cp.empty((J, K, M)).astype(dtype)
    e_gpu[:, :, :] = cp.einsum('i, ijkl, ijml, j->jkm', a_gpu, b_gpu, c_gpu, d_gpu)
    e1 = e_gpu.get()
    cp.cuda.Stream.null.synchronize()
    t1_stop = perf_counter_ns()
    print("Elapsed time(cupy-Enisum):", (t1_stop-t1_start)/1000000, 'ms')

print("cutensor-manual")
mode_f = ('i', 'j', 'k', 'm')
mode_g = ('i', 'j')
tmode_a = cutensor.create_mode(*mode_a)
tmode_b = cutensor.create_mode(*mode_b)
tmode_c = cutensor.create_mode(*mode_c)
tmode_d = cutensor.create_mode(*mode_d)
tmode_e = cutensor.create_mode(*mode_e)
tmode_f = cutensor.create_mode(*mode_f)
tmode_g = cutensor.create_mode(*mode_g)

for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             b_gpu, desc_b, tmode_b,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             g_gpu, desc_g, tmode_g,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e2 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-manual-4copy")
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
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    d_gpu = cp.asarray(d)
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             g_gpu, desc_g, tmode_g,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e5 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')


print("cutensor-manual-4copy")
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
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    d_gpu = cp.asarray(d)
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             g_gpu, desc_g, tmode_g,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e5 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-manual-2")
mode_f = ('i', 'j', 'k', 'l')
mode_g = ('i', 'j', 'm', 'l')
tmode_a = cutensor.create_mode(*mode_a)
tmode_b = cutensor.create_mode(*mode_b)
tmode_c = cutensor.create_mode(*mode_c)
tmode_d = cutensor.create_mode(*mode_d)
tmode_e = cutensor.create_mode(*mode_e)
tmode_f = cutensor.create_mode(*mode_f)
tmode_g = cutensor.create_mode(*mode_g)

for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             c_gpu, desc_c, tmode_c,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             g_gpu, desc_g, tmode_g,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e3 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-manual-3")
mode_f = ('i', 'j', 'k', 'l')
mode_g = ('j', 'k', 'm')
tmode_a = cutensor.create_mode(*mode_a)
tmode_b = cutensor.create_mode(*mode_b)
tmode_c = cutensor.create_mode(*mode_c)
tmode_d = cutensor.create_mode(*mode_d)
tmode_e = cutensor.create_mode(*mode_e)
tmode_f = cutensor.create_mode(*mode_f)
tmode_g = cutensor.create_mode(*mode_g)

for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             g_gpu, desc_g, tmode_g,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e4 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-manual-4")
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
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    d_gpu = cp.asarray(d)
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             g_gpu, desc_g, tmode_g,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e5 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-manual-4copy")
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
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    d_gpu = cp.asarray(d)
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    desc_a = cutensor.create_tensor_descriptor(a_gpu)
    desc_b = cutensor.create_tensor_descriptor(b_gpu)
    desc_c = cutensor.create_tensor_descriptor(c_gpu)
    desc_d = cutensor.create_tensor_descriptor(d_gpu)
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    desc_g = cutensor.create_tensor_descriptor(g_gpu)
    desc_f = cutensor.create_tensor_descriptor(f_gpu)
    desc_e = cutensor.create_tensor_descriptor(e_gpu)
    f_gpu = cutensor.contraction(1.0,
                             a_gpu, desc_a, tmode_a,
                             d_gpu, desc_d, tmode_d,
                             0.0,
                             f_gpu, desc_f, tmode_f)
    g_gpu = cutensor.contraction(1.0,
                             f_gpu, desc_f, tmode_f,
                             b_gpu, desc_b, tmode_b,
                             0.0,
                             g_gpu, desc_g, tmode_g)
    e_gpu = cutensor.contraction(1.0,
                             g_gpu, desc_g, tmode_g,
                             c_gpu, desc_c, tmode_c,
                             0.0,
                             e_gpu, desc_e, tmode_e)
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    e5 = e_gpu.get()
    print("Elapsed time(numpy-manual):", (t1_end-t1_start)/1000000, 'ms')

print("cutensor-einsum-manual")
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    mode_f = ('i', 'j')
    mode_g = ('i', 'j', 'k', 'l')
    f_gpu = cp.empty(getSize(mode_f)).astype(dtype)
    g_gpu = cp.empty(getSize(mode_g)).astype(dtype)
    e_gpu = cp.empty(getSize(mode_e)).astype(dtype)
    t1_start = perf_counter_ns()
    f_gpu[:,:]=cp.einsum("i, j -> ij",a_gpu,d_gpu)
    #f_gpu[:,:]=cp.tensordot(a_gpu,d_gpu)
    g_gpu[:,:,:,:]=cp.einsum("ij, ijkl -> ijkl",f_gpu,b_gpu)
    e_gpu[:,:,:]=cp.einsum("ijkl,ijml -> jkm",g_gpu,c_gpu)
    e6 = e_gpu.get()
    cp.cuda.Stream.null.synchronize()
    t1_stop = perf_counter_ns()
    print("Elapsed time(cupy-Enisum):", (t1_stop-t1_start)/1000000, 'ms')

print("oe")
expr = oe.contract_expression("i,ijkl,ijml,j->jkm", getSize(mode_a), getSize(mode_b), getSize(mode_c), getSize(mode_d))
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e7 = expr(a,b,c,d,backend='cupy')
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(opteinsum-manual):", (t1_end-t1_start)/1000000, 'ms')

'''
'''
print("oe auto")
for i in range(test_time):
    cp.cuda.Stream.null.synchronize()
    t1_start = perf_counter_ns()
    e8 = oe.contract('i,ijkl,ijml,j->jkm', a,b,c,d, backend='cupy')
    cp.cuda.Stream.null.synchronize()
    t1_end = perf_counter_ns()
    print("Elapsed time(opteinsum-manual):", (t1_end-t1_start)/1000000, 'ms')

'''
print('Check results:')
# print(np.allclose(e0,e3),np.allclose(e0,e1),np.allclose(e0,e2),np.allclose(e0,e4),np.allclose(e0,e5))
# print(np.allclose(e0,e6),np.allclose(e0,e7),np.allclose(e0,e8))
