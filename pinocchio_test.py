from __future__ import print_function

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import numpy as np
import pinocchio as pin


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
    q = np.matrix([0.5]*model.nq).T
    dq = np.matrix([1]*model.nv).T
    ddq = np.matrix([1]*model.nv).T
    tau = pin.rnea(model, data, q, dq, ddq)
    # print('tau = ', tau.T)

    ### kinemaitcs and jacbian
    q = np.asarray([0]*model.nq)
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
    f = np.matrix([0.4, 0.5, 6]*4).T
    tau = M * ddq + nle - J.T*f
    print('tau = ', tau.T)
