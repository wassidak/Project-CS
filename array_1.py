import numpy as np

# print full array 
np.set_printoptions(threshold=np.inf)

load = np.load('Train_30word/wait/0/0.npy')
print(load)
print(np.max(load))
print(np.min(load))
