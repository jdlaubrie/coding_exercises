# python3
import numpy as np

def GaussJordan(a,b):
    """gaussj: Gauss-Jordan elimination"""

    # a: square matrix to invert. a[n,n]
    # b: matrix containing the "m" right-hand side vector. b[n,m]
    # n: dimension of "a"
    # m: quantity of right-hand side vectors, usually m=1

    if len(a.shape) == 2:
        if a.shape[0] != a.shape[1]:
            raise ValueError("a. must be a matrix (n,n).")
        else:
            n = a.shape[0]
    else:
        raise ValueError("a. must be a matrix of order 2.")

    if len(b.shape) == 1:
        m = 1
        b = np.reshape(b,(b.shape[0],1))
    elif len(b.shape) == 2:
        m = b.shape[1]
    else:
        raise ValueError("b. must be a vector (n) or a matrix (n,m)")

    #INTEGER i,icol,irow,j,k,l,ll
    #INTEGER indxc(n),indxr(n),ipiv(n)
    #REAL big,dum,pivinv

    ipiv = np.zeros((n),dtype=np.int64)
    indxc = np.zeros((n),dtype=np.int64)
    indxr = np.zeros((n),dtype=np.int64)

    for i in range(n): #the main loop over the columns to be reduced
        big = 0.0
        for j in range(n): #the outer loop of the search for the pivot
            if ipiv[j] != 1:
                for k in range(n):
                    if ipiv[k] == 0:
                        if np.abs(a[j,k]) >= big:
                            big = np.abs(a[j,k])
                            irow = j
                            icol = k
                    elif ipiv[k] > 1:
                        raise ValueError("singular matrix in Gauss-Jordan")
        ipiv[icol] += 1
        if irow != icol:
            for l in range(n):
                dum = a[irow,l]
                a[irow,l] = a[icol,l]
                a[icol,l] = dum
            for l in range(m):
                dum = b[irow,l]
                b[irow,l] = b[icol,l]
                b[icol,l] = dum
        indxr[i] = irow  #ready to divide the pivot row by the pivot element
        indxc[i] = icol  #located at irow and icol
        if a[icol,icol] == 0.0:
            raise ValueError("singular matrix in Gauss-Jordan")
        pivinv = 1.0/a[icol,icol]
        a[icol,icol] = 1.0
        for l in range(n):
            a[icol,l] *= pivinv
        for l in range(m):
            b[icol,l] *= pivinv
        for ll in range(n):  # reducing the rows, except for the pivot one
            if ll != icol:
                dum = a[ll,icol]
                a[ll,icol] = 0.0
                for l in range(n):
                    a[ll,l] = a[ll,l] - a[icol,l]*dum
                for l in range(m):
                    b[ll,l] = b[ll,l] - b[icol,l]*dum
    for l in range(n-1,-1,-1):
        if indxr[l] != indxc[l]:
            for k in range(n):
                dum = a[k,indxr[l]]
                a[k,indxr[l]] = a[k,indxc[l]]
                a[k,indxc[l]] = dum
    return a, b

a = np.array([[1.,2.],[3.,4.]])
b = np.array([1.,2.])

a_inv = np.linalg.inv(a)
sol = np.dot(a_inv,b)
print(a_inv)
print(sol)

a_inv,sol = GaussJordan(a,b)
print(a_inv)
print(sol)

