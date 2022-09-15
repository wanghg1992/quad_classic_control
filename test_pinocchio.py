from __future__ import print_function

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import numpy as np
import pinocchio as pin

import casadi as ca


class Planner:
    def __init__(self):
        self.pb = np.asarray([0] * 12)

    def step(self):
        self.pb[2] = 0.12
        return self.pb


class Controller:
    def __init__(self):
        self.torque = np.asarray([0] * 12)

    def step_high_slope(self, p, o):
        kp = np.asarray([100] * 6)
        ddp = np.asarray([0] * 6)
        ddp[0:3] = np.multiply(kp[0:3], o[0] - p[0:3])

        return self.torque


if __name__ == '__main__':
    # id = "gym_env:Quadruped-v0"
    # env = gym.make(id)
    # env.reset()
    # env.step_torque(range(12))
    # env.render()
    # input("press any key to continue...")
    # env.close()

    # pin
    ## sample model
    model = pin.buildSampleModelManipulator()
    data = model.createData()
    q = pin.neutral(model)
    dq = pin.utils.zero(model.nv)
    ddq = pin.utils.zero(model.nv)
    tau = pin.rnea(model, data, q, dq, ddq)
    # print('tau = ', tau.T)

    ## quadruped model
    root = pin.JointModelFreeFlyer()
    model = pin.buildModelFromUrdf(pybullet_data.getDataPath() + '/urdf/quadruped_robot/quadruped_robot.urdf',
                                   root)
    # model = pin.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf')
    data = model.createData()

    # print("number of joints, bodys, frames:", model.njoints, model.nbodies, model.nframes)
    # print("joint:", [name for name in model.names])
    # print("frame:",[frame.name for frame in model.frames])
    # print("nq, nv:",model.nq,model.nv)
    # print("nqs:",[nq for nq in model.nqs])  # root_joint in linear and quaternion?
    # print("nvs:",[nv for nv in model.nvs])

    qmax = np.matrix(np.full([model.nq, 1], np.pi))
    q = pin.randomConfiguration(model, -qmax, qmax)
    # dq = np.matrix(np.random.rand(model.nv, 1))
    # ddq = np.matrix(np.random.rand(model.nv, 1))
    q = np.matrix([0.5] * model.nq).T
    dq = np.matrix([1.0] * model.nv).T
    ddq = np.matrix([1.0] * model.nv).T
    tau = pin.rnea(model, data, q, dq, ddq)
    # print('tau = ', tau.T)

    ### kinemaitcs and jacbian
    q = np.asarray([0] * model.nq)
    pin.forwardKinematics(model, data, q)
    # for name, oMi in zip(model.names, data.oMi):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #            .format(name, *oMi.translation.T.flat)))
    pin.updateFramePlacements(model, data)
    # for frame, oMf in zip(model.frames, data.oMf):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #            .format(frame.name, *oMf.translation.T.flat)))
    pin.computeJointJacobians(model, data)
    J1 = pin.computeFrameJacobian(model, data, q, model.getFrameId('LFFoot_link'), pin.ReferenceFrame.LOCAL)
    # print('J1 = ', J1)
    J2 = pin.getFrameJacobian(model, data, model.getFrameId('LFFoot_link'), pin.ReferenceFrame.LOCAL)
    # print('J2 = ', J2)

    # M*ddq+nle=tau
    M = pin.crba(model, data, q)
    nle = np.matrix(pin.nle(model, data, q, dq)).T
    # print("M:", M)
    # print("nle:", nle)
    tau = M * ddq + nle
    print('tau = ', tau.T)

    # M*ddq+nle=tau+Jt*f
    Jrf = pin.computeFrameJacobian(model, data, q, model.getFrameId('RFFoot_link'), pin.ReferenceFrame.LOCAL)[0:3, :]
    Jlf = pin.computeFrameJacobian(model, data, q, model.getFrameId('LFFoot_link'), pin.ReferenceFrame.LOCAL)[0:3, :]
    Jrh = pin.computeFrameJacobian(model, data, q, model.getFrameId('RHFoot_link'), pin.ReferenceFrame.LOCAL)[0:3, :]
    Jlh = pin.computeFrameJacobian(model, data, q, model.getFrameId('LHFoot_link'), pin.ReferenceFrame.LOCAL)[0:3, :]
    J = np.append(Jrf, Jlf, axis=0)
    J = np.append(J, Jrh, axis=0)
    J = np.append(J, Jlh, axis=0)
    J = np.asmatrix(J)
    f = np.matrix([0.4, 0.5, 6] * 4).T
    tau = M * ddq + nle - J.T * f
    print('tau = ', tau.T)

    # x = tau-qdd_delta-force_delta
    # H = ca.DM.eye(18)
    # for i in range(6, 18):
    #     H[i, i] = 0
    # # A = ca.DM.eye(18)
    # g = ca.DM.zeros(18)
    # lbx = M * ddq + nle - J.T*f
    # ubx = M * ddq + nle - J.T*f
    # qp = {'h': H.sparsity()}
    # S = ca.conic('S', 'qpoases', qp)
    # r = S(h=H, g=g, lbx=lbx, ubx=ubx)
    # x_opt = r['x']

    # tau + J.T*f = M * ddq + nle
    # x = tau(18)-qdd_delta(6)-force_delta(12)
    H = ca.DM.eye(6 + 12)
    # for i in range(18):
    #     H[i, i] = 0
    g = ca.DM.zeros(6 + 12)
    # A = ca.DM.eye(18)
    A = ca.DM.zeros(6, 6 + 12)
    A[0:6, 0:6] = -M[0:6, 0:6]
    A[0:6, 6:18] = J.T[0:6, :]
    lba = ca.DM.zeros(6)
    lba[0:6] = (M * ddq + nle - J.T * f)[0:6, :]
    uba = lba
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    S = ca.conic('S', 'qpoases', qp)
    r = S(h=H, g=g, a=A, lba=lba, uba=uba)
    x_opt = r['x']
    print('x_opt:', x_opt)
    ddq[0:6, 0] = ddq[0:6, 0] + x_opt[0:6]
    f[0:12, 0] = f[0:12, 0] + x_opt[6:18]
    print('M * ddq + nle - J.T*f:', M * ddq + nle - J.T * f)

    # tau + J.T*f = M * ddq + nle
    # x = tau(18)-qdd_delta(6)-force_delta(12)
    H = ca.DM.eye(18 + 6 + 12)
    for i in range(18):
        H[i, i] = 0
    g = ca.DM.zeros(18 + 6 + 12)
    # A = ca.DM.eye(18)
    A = ca.DM.zeros(24, 36)
    A[0:18, 0:18] = ca.DM.eye(18)
    A[0:18, 18:24] = -M[0:18, 0:6]
    A[0:18, 24:36] = J.T
    for i in range(6):
        A[18 + i, i] = 1
    lba = ca.DM.zeros(24)
    lba[0:18] = M * ddq + nle - J.T * f
    uba = lba
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    S = ca.conic('S', 'qpoases', qp)
    r = S(h=H, g=g, a=A, lba=lba, uba=uba)
    x_opt = r['x']
    print('x_opt:', x_opt)
    ddq[0:6, 0] = ddq[0:6, 0] + x_opt[18:24]
    f[0:12, 0] = f[0:12, 0] + x_opt[24:36]
    print('M * ddq + nle - J.T*f:', M * ddq + nle - J.T * f)

    # a = np.array([0, 1.0])
    # a[0] = 0.1
    # print(a)
