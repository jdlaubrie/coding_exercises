#testing.py
import timeit
import numpy as np
from boolpy2c import py2c

flags = np.array([True,False,True,True,False], dtype=np.uint8)
py2c(flags)
#cy = timeit.timeit('''example_cython.test(5)''',setup='import example_cython',number=100)
#py = timeit.timeit('''example_original.test(5)''',setup='import example_original',number=100)

#print(cy,py)
#print('Cython is {}x faster'.format(py/cy))
