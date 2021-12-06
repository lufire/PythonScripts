# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:27:06 2021

@author: lukas
"""

import multiprocessing as mp
import numpy as np
import time


def f(x):
    m = 1000
    a = np.random.rand(m, m)
    b = np.random.rand(m, m)
    c = np.matmul(a, b)
    return np.sum(c)

def wait():
    time.sleep(0.001)
    return None

if __name__ == '__main__':
    tstart = time.time()
    
    
    pool = mp.Pool()
    
    tstop = time.time()
    dt_init = tstop - tstart
    
    print('mp.Pool initialization [s]: {:.3f}'.format(dt_init))
    
    n = 500
    
    
    tstart = time.time()
    
    results = pool.map(f, range(n))
    # print(results)
    
    pool.close()
    pool.join()
    
    tstop = time.time()
    dt_par = tstop - tstart
    print('mp.Pool parallelization [s]: {:.3f}'.format(dt_par))
    
    
    # tstart = time.time()
    # pool = mp.Pool()

    # results = [pool.apply(f, args=(x,)) for x in range(n)]
    # print(results)
    
    # pool.close()
    # pool.join()
    
    # tstop = time.time()
    # dt = tstop - tstart
    # print('mp.Pool.apply parallelization [s]: ', dt)
    
    tstart = time.time()
    
    results = [f(i) for i in range(n)]
    # print(results) # [0, 1, 4, 9]
    
    tstop = time.time()
    dt_ser = tstop - tstart
    print('serial [s]: {:.3f}'.format(dt_ser))
    print('speedup [-]: {:.3f}'.format(dt_ser/dt_par))
