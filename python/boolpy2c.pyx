cimport cython
cimport numpy as np
import numpy as np

#ctypedef np.uint8_t uint8

def py2c(flags):
    print(flags)
    cdef np.ndarray[int,ndim=1,mode='c'] cflags = flags.astype(np.int32)
    print(cflags)
