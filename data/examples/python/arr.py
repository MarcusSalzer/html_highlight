import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

y = x[:, 1]
y  # array([2, 5], dtype=int32)
y[0] = 9

print(y)  # array([9, 5], dtype=int32)
print(x)  # array([[1, 9, 3], [4, 5, 6]], dtype=int32)
