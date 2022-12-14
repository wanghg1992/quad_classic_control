from __future__ import print_function

import numpy as np
import casadi as ca

if __name__ == '__main__':
    # SX
    x = ca.MX.sym('x', 5, 3)
    y = ca.SX.sym('y', 5)
    Z = ca.SX.sym('z', 4, 2)
    print(x[1, 2])
    print(y)
    print(Z[0, 1])
    f = x ** 2 + 10
    f = ca.sqrt(f)
    print('f:', f)
    B1 = ca.SX.zeros(4, 5)
    B2 = ca.SX(4, 5)
    B4 = ca.SX.eye(4)
    print('B1:', B1)
    print('B2:', B2)
    print('B4:', B4)

    # MX
    x = ca.MX.sym('x', 2, 2)
    print('x[0, 0]:', x[0, 0])

    # DM
    C = ca.DM(2, 3)
    C_dense = C.full()
    C_sparse = C.sparse()
    print('C:', C)
    print('C_dense:', C_dense)
    print('C_sparse:', C_sparse)

    # QP(Low-level interface)
    H = 2 * ca.DM.eye(2)
    A = ca.DM.ones(1, 2)
    g = ca.DM.zeros(2)
    lba = 10
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    opts = {'print_problem': False, 'print_time': False, 'printLevel': 'none'}
    S = ca.conic('S', 'qpoases', qp, opts)
    r = S(h=H, g=g, a=A, lba=lba)
    x_opt = r['x']
    print('x_opt:', x_opt)

    opti = ca.Opti()
    x = opti.variable()
    y = opti.variable()
    opti.minimize((y - x ** 2) ** 2)
    opti.subject_to(x ** 2 + y ** 2 == 1)
    opti.subject_to(x + y >= 1)
    opti.solver('ipopt')
    opti.solver("ipopt", dict(print_time=False), dict(print_level=False))
    sol = opti.solve()
    print(sol.value(x))
    print(sol.value(y))


    # bug test
    from numpy import *
    from casadi import *

    set_printoptions(linewidth=3000, threshold=sys.maxsize)

    size = 30
    rand = matrix(random.random(size=(29, 30))) / 100
    rand[0, :] = 0
    print(rand)
    H = 2 * DM(rand.T * rand)
    A = DM.ones(1, size)
    g = DM.zeros(size)
    uba = 1000.
    lba = -1000

    qp = {}
    qp['h'] = H.sparsity()
    qp['a'] = A.sparsity()
    opt = {'error_on_fail': False}
    S = conic('S', 'qpoases', qp, opt)
    print(S)

    r = S(h=H, g=g, a=A, lba=lba, uba=uba)
    x_opt = r['x']
    print('x_opt: ', x_opt)
