from LinAlgFunctions import *

A = np.array([[60, 30, 20],
              [30, 20, 15],
              [20, 15, 12]], dtype = float)

b = np.array([110, 65, 47], dtype = float)

x = PLUSolve(A,b)