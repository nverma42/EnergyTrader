import osqp
import numpy as np
import scipy.sparse as sp

def optimize_action(action):
    P = sp.csc_matrix([[4, 1], [1, 2]])
    q = np.array([1, 1])
    A = sp.csc_matrix([[1.0, 1.0]])
    l = np.array([1.0])
    u = np.array([1.0])

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u)
    res = prob.solve()

    print(res.x)
