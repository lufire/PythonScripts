import numpy as np
import time

a = np.linspace(0.5, 1.5, 1000)

n_iter = 1000

start = time.time()
for i in range(n_iter):
    b = np.cbrt(a)
finish = time.time()
print('cbrt timing: ', finish - start)

start = time.time()
for i in range(n_iter):
    b = np.power(a, 1.0/3.0)
finish = time.time()
print('func power timing: ', finish - start)

start = time.time()
for i in range(n_iter):
    b = a ** (1.0/3.0)
finish = time.time()
print('literal power timing: ', finish - start)


start = time.time()
for i in range(n_iter):
    b = a ** 0.333
finish = time.time()
print('decimal power timing: ', finish - start)
