from __future__ import print_function

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import numpy as np
import pinocchio as pin
import casadi as ca

from scipy.spatial.transform import Rotation as R

class Estimation:
    def __init__(self):
        self.pb = np.asarray([0] * 12)

    def step(self):
        self.pb[2] = 0.12
        return self.pb


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

    def classical_control(self, observation):
        baseLinearVelocityLocal = observation[6:9]
        # print("baseLinearVelocityLocal:", baseLinearVelocityLocal)
        foot_pos_cmd = observation[20:32]
        step_phase = observation[12:16]
        contact = observation[16:20]
        footHold = np.asarray(baseLinearVelocityLocal[0:2]) * 1.0 / self.step_freq * (0.5 + 0.05)
        action = [0] * 12
        kp = [20, -20, 0]
        kd = [2, -2, 0]
        for leg in range(4):
            if contact[leg] == 1:
                action[leg * 3 + 0] = kp[1] * (observation[4]) + kd[1] * (observation[10]) * 1
                action[leg * 3 + 1] = kp[0] * (observation[3]) + kd[0] * (observation[9]) * 1
                # action[leg * 3 + 2] = 2.0 * (-self.stand_height - foot_pos_cmd[leg * 3 + 2])
            elif step_phase[leg] == 0:
                action[leg * 3 + 0] = 10.0 * (footHold[0] - foot_pos_cmd[leg * 3 + 0])
                action[leg * 3 + 1] = 10.0 * (footHold[1] - foot_pos_cmd[leg * 3 + 1])
            # else:
            #     action[leg * 3 + 2] = -0.05*2
        return action

    # def


if __name__ == '__main__':
    id = "gym_env:Quadruped-v0"
    env = gym.make(id)

    ## quadruped model
    root = pin.JointModelFreeFlyer()
    model = pin.buildModelFromUrdf(pybullet_data.getDataPath() + '/urdf/quadruped_robot/quadruped_robot.urdf',
                                   root)
    # model = pin.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf')
    data = model.createData()

    body_pos_des = np.asmatrix([0, 0, 0.14])
    body_pos_fdb = np.asmatrix(env.reset()[0:3])
    body_rot_des = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    body_rot_fdb = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    body_acc_des = np.asmatrix([0] * 6).T

    torque = [.0] * 12
    for i in range(100000):
        # body_force = Mfb*foot_fore

        # Mfb0 = np.append(np.array(np.eye(3)),pin.skew(p[0:3]),axis=0)
        # Mfb1 = np.append(np.array(np.eye(3)),pin.skew(p[3:6]),axis=0)
        # Mfb2 = np.append(np.array(np.eye(3)),pin.skew(p[6:9]),axis=0)
        # Mfb3 = np.append(np.array(np.eye(3)),pin.skew(p[9:12]),axis=0)
        # Mfb = np.append(Mfb0,Mfb1,axis=1)
        # Mfb = np.append(Mfb,Mfb2,axis=1)
        # Mfb = np.append(Mfb,Mfb3,axis=1)
        #
        # foot_force_cmd = np.linalg.pinv(Mfb)*body_force_cmd
        # body_acc_cmd = body_force_cmd/3

        # simulation
        torque = list(torque[0:3])+ list(torque[6:9])+ list(torque[3:6])+ list(torque[9:12])
        [o, pb, vb, js] = env.step_torque(torque)
        # print("pb:", pb)
        # print("vb:", vb)
        # print("js:", js)
        js = list(js[0:3]) + list(js[6:9]) + list(js[3:6]) + list(js[9:12])

        pj = [i[0] for i in js]
        vj = [i[1] for i in js]
        q = np.asmatrix(list(pb[0] + pb[1]) + pj).T
        v = np.asmatrix(list(vb[0] + vb[1]) + vj).T

        body_pos_fdb[0:3] = pb[0]
        body_rot_fdb = R.from_quat(pb[1])

        body_acc_des[0:3, 0] = np.asmatrix(80 * (body_pos_des - body_pos_fdb)).T
        body_acc_des[3:6, 0] = np.asmatrix(80 * (body_rot_des * body_rot_fdb.inv()).as_rotvec()).T

        # body_acc_des = body_acc_des -8*np.asmatrix(vb[0]+vb[1]).T

        p = np.asmatrix([0] * 12).T
        print("torque:", torque)
        print("body_pos_fdb:", body_pos_fdb)

        # q[3] = pb[1][3]
        # q[4] = pb[1][0]
        # q[5] = pb[1][1]
        # q[6] = pb[1][2]
        # q[3] = 1
        # q[4] = 0
        # q[5] = 0
        # q[6] = 0
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        # for frame, oMf in zip(model.frames, data.oMf):
        #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
        #            .format(frame.name, *oMf.translation.T.flat)))
        pin.computeJointJacobians(model, data, q)
        Jrf = np.matrix(pin.getFrameJacobian(model, data, model.getFrameId('RF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jlf = np.matrix(pin.getFrameJacobian(model, data, model.getFrameId('LF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jrh = np.matrix(pin.getFrameJacobian(model, data, model.getFrameId('RH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jlh = np.matrix(pin.getFrameJacobian(model, data, model.getFrameId('LH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jfoot = np.append(Jrf[0:3, :], Jlf[0:3, :], axis=0)
        Jfoot = np.append(Jfoot, Jrh[0:3, :], axis=0)
        Jfoot = np.append(Jfoot, Jlh[0:3, :], axis=0)

        pin.computeJointJacobiansTimeVariation(model, data, q, v)
        Jdrf = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('RF4_joint'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jdlf = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('LF4_joint'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jdrh = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('RH4_joint'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jdlh = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('LH4_joint'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        Jdfoot = np.append(Jdrf[0:3, :], Jdlf[0:3, :], axis=0)
        Jdfoot = np.append(Jdfoot, Jdrh[0:3, :], axis=0)
        Jdfoot = np.append(Jdfoot, Jdlh[0:3, :], axis=0)

        pin.crba(model, data, q)
        M = np.asmatrix(data.M)
        ddx = np.zeros(12).reshape(12, 1)
        JF = Jfoot
        JF_pinv = np.linalg.inv(M)*JF.T*np.linalg.inv(JF*np.linalg.inv(M)*JF.T)
        ddq = JF_pinv * (ddx - Jdfoot * v)
        N1 = np.eye(18) - JF_pinv*JF

        JB = np.matrix(pin.getFrameJacobian(model, data, model.getFrameId('root_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        JdB = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('root_joint'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        JB_pinv = np.linalg.inv(M)*JB.T*np.linalg.inv(JB*np.linalg.inv(M)*JB.T)
        # N1 = np.eye(18) - JB_pinv*JB
        J2 = JB
        Jd2 = JdB
        J2_pre = J2*N1
        J2_pre_dpinv = np.linalg.inv(M)*J2_pre.T*np.linalg.inv(J2_pre*np.linalg.inv(M)*J2_pre.T)
        ddx = body_acc_des
        ddq = ddq + J2_pre_dpinv * (ddx - Jd2 * v - J2 * ddq)

        pin.forwardKinematics(model, data, q, v, ddq)
        a = pin.getFrameAcceleration(model, data,model.getFrameId('LFFoot_link'),pin.LOCAL_WORLD_ALIGNED)
        # print("a LFFoot: ", a)
        a = pin.getFrameAcceleration(model, data,model.getFrameId('body_link'),pin.LOCAL_WORLD_ALIGNED)
        # print("a body: ", a)

        M = pin.crba(model, data, q)
        nle = np.matrix(pin.nle(model, data, q, v)).T
        f = np.matrix([.0, .0, .0] * 4).T
        tor = np.matrix([.0, .0, .0] * 6).T

        # tau + J.T*f = M * ddq + nle
        # x = tau(18)-force(12)
        H = ca.DM.eye(18 + 12)
        # for i in range(18):
        #     H[i, i] = 0
        g = ca.DM.zeros(18 + 12)

        # A = ca.DM.eye(18)
        A = ca.DM.zeros(34, 30)
        A[0:18, 0:18] = ca.DM.eye(18)
        A[0:18, 18:30] = JF.T
        mu_c = 0.3
        # friction cone constrain
        A[18:22, 18:21] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[22:26, 21:24] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[26:30, 24:27] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[30:34, 27:30] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        # A[0:18, 18:30] = JF.T.zeros()
        # A[0:18, 18:30] = ca.DM.zeros(18, 12)
        lba = ca.DM.zeros(34)
        lba[0:18] = M * ddq + nle
        lba[18:34] = -1000

        uba = ca.DM(lba)
        uba[18:34] = 0
        ubx = ca.DM.zeros(30)
        ubx[0:6] = 0.000001
        ubx[6:18] = 20
        ubx[18:30] = 200
        lbx = ca.DM(-ubx)
        lbx[20:30:3] = 0

        qp = {'h': H.sparsity(), 'a': A.sparsity()}
        opts = {'printLevel': 'none'}
        # opts = {'printLevel': 'none', 'error_on_fail': False}
        S = ca.conic('S', 'qpoases', qp, opts)
        r = S(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        x_opt = r['x']
        # print('x_opt:', x_opt)

        tor[0:18, 0] = x_opt[0:18]
        f[0:12, 0] = x_opt[18:30]
        # print('M * ddq + nle - J.T*f:', M * ddq + nle - JF.T * f)

        torque = tor[6:18]

        pin.aba(model, data, q, v, tor)
        a_f = data.ddq
        print('a_f:', a_f)

        # env.render()
    input("press any key to continue...")
    env.close()
