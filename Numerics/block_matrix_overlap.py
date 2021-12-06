# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:00:41 2019

@author: lukas
"""
import numpy as np
import scipy as sp
from scipy import linalg


def block_diag_overlap(block_list, overlap):
    m_sblks = [block.shape[0] for block in block_list]
    n_sblks = [block.shape[1] for block in block_list]
    
    n_blocks = len(block_list)
    
    m_ol = overlap[0]
    n_ol = overlap[1]
    
    m_final = np.sum(np.asarray(m_sblks)) - (n_blocks - 1) * m_ol
    n_final = np.sum(np.asarray(n_sblks)) - (n_blocks - 1) * n_ol
    print('m_final')
    print(m_final)
    print(n_final)
    block_array = np.zeros((m_final, n_final))
    
    for i, block in enumerate(block_list):
        print(i)
        print('block')
        print(block)
        print('block_array')
        print(block_array)        

        print(block_array[i*m_sblks[i-1]-i*m_ol:(i+1)*m_sblks[i]-i*m_ol,
                          i*n_sblks[i-1]-i*n_ol:(i+1)*n_sblks[i]-i*n_ol])
        block_array[i*m_sblks[i-1]-i*m_ol:(i+1)*m_sblks[i]-i*m_ol,
                    i*n_sblks[i-1]-i*n_ol:(i+1)*n_sblks[i]-i*n_ol] += \
                    block[:,:]
        print(block_array)
    return block_array

a = np.array([[1.0, 2.0],
              [3.0, 4.0]])
b = np.array([[5.0, 6.0],
              [7.0, 8.0]])
block_1 = sp.linalg.block_diag(a,b)
block_2 = block_1.copy()
block_3 = block_1.copy()


block_list = [block_1, block_2, block_3]
overlap = (2, 2)
                
block_diag_overlap(block_list, overlap)
                