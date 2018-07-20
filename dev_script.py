import numpy as np

conv_filter = [[-0.1,  0.5,  2.2],
          [ 0.7 , 0.9,  0.3],
          [-0.2, -0.2,  0.7],
          [ 1.3, -0.1, -1.1]]

area1 = [[ 0.4, -0.8,  2.2],
         [ 0.1,  1.2,  1.5],
         [ 0.2,  0.1, -1.2],
         [-0.2, -0.5,  0.1]]

conv_filter = np.array(conv_filter)
area1 = np.array(area1)

res = conv_filter @ area1.T
print(np.trace(res))

res = conv_filter * area1
print(np.sum(res))