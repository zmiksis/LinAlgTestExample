import numpy as np

#####################################################################################################

def GaussianEliminationPivotVectorized(A):
    
    n = A.shape[0]
    
    # Initialize vector p
    p = np.array(range(n))
    
    # Initialize vector s
    s = np.array([np.max(A[i,:]) for i in range(n)])

    for k in range(n-1):
        
        #Find the pivot element
        pivot_idx = np.argmax(abs(A[p[k:n],k])/s[k:n])
        
        # Swap the pivot element
        p[[k, pivot_idx+k]] = p[[pivot_idx+k, k]]
        
        # Compute scaling factors
        A[p[k+1:n],k] = A[p[k+1:n],k]/A[p[k],k]
        
        # Subtract multiple of row 
        A[p[k+1:n],k+1:n] = A[p[k+1:n],k+1:n] - A[p[k+1:n],k][:,None]*A[p[k],k+1:n]
    
    return A, p

#####################################################################################################

def PLUSolve(A,b):
    
    n = A.shape[0]
    
    A, p = GaussianEliminationPivotVectorized(A)
    
    A = A[p,:]
    b = b[p]
    
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = b[i] - A[i,0:i]@x[0:i]
        
    for i in range(n-1,-1,-1):
        x[i] = (x[i] - A[i,i+1:n]@x[i+1:n])/A[i,i]
        
    A = ((np.tril(A,-1) + np.eye(3))@np.triu(A))
    pT = np.array([np.where(p == i)[0] for i in range(A.shape[0])]).flatten()
    A = A[pT,:]
        
    return x

