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

import math

class Estimation:
    def __init__(self, model):
        self.q_ = np.asarray([0] * 19)
        self.pb_ = np.asarray([0, 0, 0.16, 0., 0., 0., 1.])
        self.pf_ = np.asarray([0] * 12)
        self.vb_ = np.asarray([0, 0, 0, 0., 0., 0.])
        self.model = model
        self.data = self.model.createData()

        self.JB = np.asmatrix(np.zeros([6, 18]))
        self.Jrf = np.asmatrix(np.zeros([6, 18]))
        self.Jlf = np.asmatrix(np.zeros([6, 18]))
        self.Jrh = np.asmatrix(np.zeros([6, 18]))
        self.Jlh = np.asmatrix(np.zeros([6, 18]))
        self.Jfoot = np.asmatrix(np.zeros([12, 18]))

        self.JdB = np.asmatrix(np.zeros([6, 18]))
        self.Jdrf = np.asmatrix(np.zeros([6, 18]))
        self.Jdlf = np.asmatrix(np.zeros([6, 18]))
        self.Jdrh = np.asmatrix(np.zeros([6, 18]))
        self.Jdlh = np.asmatrix(np.zeros([6, 18]))
        self.Jdfoot = np.asmatrix(np.zeros([12, 18]))

    def step(self):
        model = self.model
        data = self.data

        pin.forwardKinematics(model, data, q_)
        pin.updateFramePlacements(model, data)
        # for frame, oMf in zip(model.frames, data.oMf):
        #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
        #            .format(frame.name, *oMf.translation.T.flat)))
        pf_rf_ = data.oMf[model.getFrameId('RF4_joint')].translation
        pf_lf_ = data.oMf[model.getFrameId('LF4_joint')].translation
        pf_rh_ = data.oMf[model.getFrameId('RH4_joint')].translation
        pf_lh_ = data.oMf[model.getFrameId('LH4_joint')].translation
        self.pf_ = np.append(pf_rf_, pf_lf_, axis=0)
        self.pf_ = np.append(self.pf_, pf_rh_, axis=0)
        self.pf_ = np.append(self.pf_, pf_lh_, axis=0)

        pin.computeJointJacobians(model, data, q_)
        self.JB = np.matrix(
            pin.getFrameJacobian(model, data, model.getFrameId('root_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jrf = np.matrix(
            pin.getFrameJacobian(model, data, model.getFrameId('RF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jlf = np.matrix(
            pin.getFrameJacobian(model, data, model.getFrameId('LF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jrh = np.matrix(
            pin.getFrameJacobian(model, data, model.getFrameId('RH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jlh = np.matrix(
            pin.getFrameJacobian(model, data, model.getFrameId('LH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jfoot = np.append(self.Jrf[0:3, :], self.Jlf[0:3, :], axis=0)
        self.Jfoot = np.append(self.Jfoot, self.Jrh[0:3, :], axis=0)
        self.Jfoot = np.append(self.Jfoot, self.Jlh[0:3, :], axis=0)

        pin.computeJointJacobiansTimeVariation(model, data, q_, dq_)
        self.JdB = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('root_joint'),
                                                          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jdrf = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('RF4_joint'),
                                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jdlf = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('LF4_joint'),
                                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jdrh = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('RH4_joint'),
                                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jdlh = np.matrix(pin.getFrameJacobianTimeVariation(model, data, model.getFrameId('LH4_joint'),
                                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        self.Jdfoot = np.append(self.Jdrf[0:3, :], self.Jdlf[0:3, :], axis=0)
        self.Jdfoot = np.append(self.Jdfoot, self.Jdrh[0:3, :], axis=0)
        self.Jdfoot = np.append(self.Jdfoot, self.Jdlh[0:3, :], axis=0)
        return [self.pb_, self.Jfoot, self.Jdfoot]

class Planner:
    def __init__(self):
        self.pb = np.asarray([0, 0, 0.16, 0., 0., 0., 1.])
        self.vb = np.asarray([0] * 6)

class Controller:
    def __init__(self, model):
        self.torque = np.asarray([0] * 12)
        self.body_pos_des = np.asmatrix([0, 0, 0.16])
        self.body_rot_des = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.body_acc_des = np.asmatrix([0] * 6).T
        self.body_pos_fdb = np.asmatrix(env.reset()[0:3])
        self.body_rot_fdb = R.from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.model = model
        self.data = self.model.createData()

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

    def optimal_control(self, est, plan):
        q_ = est.q_
        dq_ = est.dq_
        JB = est.JB
        Jfoot = est.Jfoot
        JdB = est.JdB
        Jdfoot = est.Jdfoot
        model = self.model
        data = self.data

        self.body_pos_des = plan.pb[0:3]
        self.body_rot_des = R.from_quat(plan.pb[3:7])
        self.body_pos_fdb = est.pb_[0:3]
        self.body_rot_fdb = R.from_quat(est.pb_[3:7])

        self.body_acc_des[0:3, 0] = np.asmatrix(80 * (self.body_pos_des - self.body_pos_fdb)).T
        self.body_acc_des[3:6, 0] = np.asmatrix(300 * (self.body_rot_des * self.body_rot_fdb.inv()).as_rotvec()).T

        self.body_acc_des = self.body_acc_des - 8 * np.asmatrix(est.vb_).T
        self.foot_acc_des = np.asmatrix(200*(plan.pf - est.pf_)).T

        # WBC task 1: contact foot not slip
        pin.crba(model, data, q_)
        M = np.asmatrix(data.M)
        # ddx = np.zeros(12).reshape(12, 1)
        ddx = self.foot_acc_des
        JF = Jfoot
        JF_pinv = np.linalg.inv(M) * JF.T * np.linalg.inv(JF * np.linalg.inv(M) * JF.T)
        ddq = JF_pinv * (ddx - Jdfoot * dq_)
        N1 = np.eye(18) - JF_pinv * JF

        # WBC task 2: body control
        JB_pinv = np.linalg.inv(M) * JB.T * np.linalg.inv(JB * np.linalg.inv(M) * JB.T)
        # N1 = np.eye(18) - JB_pinv*JB
        J2 = JB
        Jd2 = JdB
        J2_pre = J2 * N1
        J2_pre_dpinv = np.linalg.inv(M) * J2_pre.T * np.linalg.inv(J2_pre * np.linalg.inv(M) * J2_pre.T)
        ddx = self.body_acc_des
        ddq = ddq + J2_pre_dpinv * (ddx - Jd2 * dq_ - J2 * ddq)

        # floating base dynamics
        pin.forwardKinematics(model, data, q_, dq_, ddq)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('LFFoot_link'), pin.LOCAL_WORLD_ALIGNED)
        # a = pin.getFrameAcceleration(model, data,model.getFrameId('LFFoot_link'), pin.LOCAL)
        # print("a LFFoot: ", a)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('body_link'), pin.LOCAL_WORLD_ALIGNED)
        # a = pin.getFrameAcceleration(model, data,model.getFrameId('body_link'), pin.LOCAL)
        # print("a body: ", a)

        M = pin.crba(model, data, q_)
        nle = np.matrix(pin.nle(model, data, q_, dq_)).T
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
        # opts = {'printLevel': 'none'}
        opts = {'printLevel': 'none', 'error_on_fail': False}
        S = ca.conic('S', 'qpoases', qp, opts)
        r = S(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        x_opt = r['x']
        # print('x_opt:', x_opt)

        tor[0:18, 0] = x_opt[0:18]
        f[0:12, 0] = x_opt[18:30]
        # print('M * ddq + nle - J.T*f:', M * ddq + nle - JF.T * f)
        # print('ddq:', np.linalg.inv(M) * (tor + JF.T * f - nle))

        torque = tor[6:18]

        pin.aba(model, data, q_, dq_, tor)
        a_f = data.ddq
        # print('a_f:', a_f)

        return torque

    def step(self, est, plan):
        return self.optimal_control(est, plan)



if __name__ == '__main__':
    id = "gym_env:Quadruped-v0"
    env = gym.make(id)

    ## quadruped model
    root = pin.JointModelFreeFlyer()
    model = pin.buildModelFromUrdf(pybullet_data.getDataPath() + '/urdf/quadruped_robot/quadruped_robot.urdf',
                                   root)
    # model = pin.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf')
    data = model.createData()

    torque = [.0] * 12

    est = Estimation(model)
    plan = Planner()
    control = Controller()
    for i in range(100000):

        # simulation
        torque = list(torque[0:3])+ list(torque[6:9])+ list(torque[3:6])+ list(torque[9:12])
        [o_, pb_, vb_, js_] = env.step_torque(torque)
        js_ = list(js_[0:3]) + list(js_[6:9]) + list(js_[3:6]) + list(js_[9:12])

        pj_ = [i[0] for i in js_]
        vj_ = [i[1] for i in js_]
        q_ = np.asmatrix(list(pb_[0] + pb_[1]) + pj_).T
        dq_ = np.asmatrix(list(vb_[0] + vb_[1]) + vj_).T

        est.q_ = q_
        est.dq_ = dq_
        est.pb_[0:3] = pb_[0]
        est.pb_[3:7] = pb_[1]
        est.vb_[0:3] = vb_[0]
        est.vb_[3:6] = vb_[1]
        est.step()

        torque = control.step(est, plan)

        # env.render()
    input("press any key to continue...")
    env.close()
