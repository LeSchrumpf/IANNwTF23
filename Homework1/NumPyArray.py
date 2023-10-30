import numpy as np

GenericArray = np.random.normal(0, 1, (5, 5))
print("One", GenericArray)
for x in np.nditer(GenericArray, op_flags=['readwrite']):
    if x[...] > 0.09:
        x[...] = x ** 2
    else:
        x[...] = 42
print("Two", GenericArray)
print("Three", GenericArray[4:5])