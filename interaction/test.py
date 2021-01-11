import numpy as np

a = np.array([6, 6, 6, 6, 6, 3, 3, 3, 3, 3], dtype=np.float32)

print('mean: {}, std: {}'.format(np.mean(a), np.std(a)))