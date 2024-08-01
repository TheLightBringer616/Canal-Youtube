import random
import copy

import numpy as np

from numba import jit, njit, cuda, prange

import matplotlib.pyplot as plt

@jit(target_backend='cuda', parallel=True)
def random_walkers(walkers, steps):

    walkers = int(walkers)
    steps = int(steps)
    
    T = [i for i in range(steps + 1)]
    
    random_walks = []
    for j in prange(walkers):
    
        X = [0]
        x = 0
        for i in range(steps):
            rand = random.uniform(0, 1) - 0.5
            rand /= abs(rand)
        
            x += rand
            X.append(x)
        
        random_walks.append(X)

    return random_walks, T

walkers = 1e3
steps = 1e3
X, T = random_walkers(walkers, steps)

plt.figure(figsize = (16, 8))

for i in range(int(walkers)):
    plt.plot(T, X[i], c='k', linewidth=1, alpha=0.02)

plt.show()