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

from sensor_msgs import msg as smsg
from geometry_msgs import msg as gmsg
from geometry_msgs.msg import Transform
import tf2_ros
import rospy
import tf.transformations as transformations
from tf.transformations import quaternion_from_euler
from nav_msgs import msg as nmsg
from visualization_msgs import msg as vmsg

class PinInter:
    # leg order: LF-LH-RF-RH
    # def __init__(self):
    def to_pin(self, input, offset):
        return np.concatenate([input[0: offset], input[offset + 3: offset + 6], input[offset + 9: offset + 12],
                input[offset + 0: offset + 3], input[offset + 6: offset + 9]])
    def from_pin(self, input, offset):
        return np.concatenate([input[0: offset], input[offset + 6: offset + 9], input[offset + 0: offset + 3], input[offset + 9: offset + 12],
                input[offset + 3: offset + 6]])

class Estimation:
    def __init__(self, model):
        self.q_ = np.asarray([0.] * 19)
        self.dq_ = np.asarray([0.] * 18)
        self.tor_ = np.asarray([0.] * 18)
        self.pb_ = np.asarray([0, 0, 0.16, 0., 0., 0., 1.])
        self.pf_ = np.asarray([0.] * 12)
        self.vb_ = np.asarray([0.] * 6)
        self.vf_ = np.asarray([0.] * 12)
        self.vb_body_ = np.asarray([0.] * 6)
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

        self.pin_inter = PinInter()

    def step(self):
        model = self.model
        data = self.data
        q_ = self.q_
        dq_ = self.dq_

        pin_q_ = self.pin_inter.to_pin(self.q_, 7)
        pin_dq_ = self.pin_inter.to_pin(self.dq_, 6)

        pin.forwardKinematics(model, data, pin_q_, pin_dq_)
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
        vf_rf_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('RF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_lf_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('LF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_rh_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('RH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_lh_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('LH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        self.vf_ = np.append(vf_rf_[0:3, :], vf_lf_[0:3, :], axis=0)
        self.vf_ = np.append(self.vf_, vf_rh_[0:3, :], axis=0)
        self.vf_ = np.append(self.vf_, vf_lh_[0:3, :], axis=0)

        self.getBaseVelocityLocal()

        pin.computeJointJacobians(model, data, pin_q_)
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

        pin.computeJointJacobiansTimeVariation(model, data, pin_q_, pin_dq_)
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

    def getBaseVelocityLocal(self):
        yaw = transformations.euler_from_quaternion(self.pb_[3:7])[2]
        r = R.from_euler('xyz', [0., 0., yaw])
        r = np.asmatrix(r.as_matrix())

        linearVelocity = np.asarray(self.vb_[0:3]).reshape(3,1)
        angularVelocity = np.asarray(self.vb_[3:6]).reshape(3,1)

        self.vb_body_ = [(r * linearVelocity).reshape(1, 3).tolist()[0],
                (r * angularVelocity).reshape(1, 3).tolist()[0]]

        return self.vb_body_

class Planner:
    def __init__(self, model):
        self.timer = .0
        self.dt = 0.001
        self.iterPerPlan = 10
        self.dtPlan = self.iterPerPlan * self.dt
        self.horizon = 200
        self.gait = 'trot'
        self.segments = np.asarray([0, 0, 0, 0])
        self.duration = np.asarray([0, 0, 0, 0])
        self.offset = np.asarray([0, 0, 0, 0])
        self.contact_state = np.asarray([0, 0, 0, 0])
        self.swing_phase = np.asarray([.0, .0, .0, .0])
        self.contact_phase = np.asarray([.0, .0, .0, .0])
        self.first_swing = np.asarray([True, True, True, True])
        self.first_contact = np.asarray([True, True, True, True])
        self.phase_abnormal = False
        self.pf_init = np.asarray([.0] * 12)
        self.pf_hold = np.asarray([.0] * 12)
        self.pf = np.asarray([.0] * 12)
        self.ph_body = np.asarray([0.0915, -0.08, .0,   0.0915, 0.08, .0,   -0.0915, -0.08, .0,   -0.0915, 0.08, .0])
        self.pb = np.asarray([.0, .0, 0.16, 0., 0., 0., 1.])
        self.vb = np.asarray([.0] * 6)
        self.model = model
        self.data = self.model.createData()

    def get_contact_target(self, time):
        if self.gait == 'trot':
            self.segments = np.asarray([20, 20, 20, 20])
            self.duration = np.asarray([10, 10, 10, 10])
            self.offset = np.asarray([0, 10, 10, 0])
        # iter = time / self.dt
        iterPlan = time / self.dtPlan
        for leg in range(4):
            normIterPlan = (iterPlan + self.segments[leg] - self.offset[leg]) % self.segments[leg]
            self.contact_state[leg] = normIterPlan < self.duration[leg]
        return self.contact_state

    def get_step_phase(self, time):
        if self.gait == 'trot':
            self.segments = np.asarray([20, 20, 20, 20])
            self.duration = np.asarray([10, 10, 10, 10])
            self.offset = np.asarray([0, 10, 10, 0])
        iter = time / self.dt
        # iterPlan = time / self.dtPlan
        for leg in range(4):
            normIter = (iter + self.segments[leg] * self.iterPerPlan  - self.offset[leg] * self.iterPerPlan ) % ( self.segments[leg] * self.iterPerPlan )
            if normIter < self.duration[leg] * self.iterPerPlan:
                self.contact_phase[leg] = normIter / float(self.duration[leg] * self.iterPerPlan)
                self.swing_phase[leg] = 0
            else:
                self.contact_phase[leg] = 0
                self.swing_phase[leg] = (normIter - self.duration[leg] * self.iterPerPlan) / float( (self.segments[leg]- self.duration[leg]) * self.iterPerPlan )
        return [self.contact_phase, self.swing_phase]

    def update_step_phase(self):
        self.get_step_phase(self.timer)

    def update_body_target(self):
        body_pos_des = np.asmatrix([0., 0., 0.16])
        body_rot_des = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        return [body_pos_des, body_rot_des]

    def update_foot_target(self, est_data):
        for leg in range(4):
            if self.contact_phase[leg] > 0.0001:        # contact phase
                self.first_swing[leg] = True
                # if self.first_contact[leg]:
                self.pf[leg*3 + 0] = self.pf_hold[leg*3 + 0]
                self.pf[leg*3 + 1] = self.pf_hold[leg*3 + 1]
                self.pf[leg*3 + 2] = self.pf_hold[leg*3 + 2]

            elif self.swing_phase[leg] > 0.0001:        # swing_phase
                self.first_contact[leg] = True
                if self.first_swing[leg]:
                    self.pf_init = est_data.pf_
                self.pf_hold[leg*3 + 0] = self.ph_body[leg*3 + 0]
                self.pf_hold[leg*3 + 1] = self.ph_body[leg*3 + 1]
                self.pf_hold[leg*3 + 2] = self.ph_body[leg*3 + 2]
                self.pf[leg*3 + 0] = self.pf_init[leg*3 + 0] + (self.pf_hold[leg*3 + 0] - self.pf_init[leg*3 + 0])*self.swing_phase[leg]
                self.pf[leg*3 + 1] = self.pf_init[leg*3 + 0] + (self.pf_hold[leg*3 + 0] - self.pf_init[leg*3 + 0])*self.swing_phase[leg]
                self.pf[leg*3 + 2] = self.pf_init[leg*3 + 0] + (math.cos((self.swing_phase[leg]-0.5)*3.14/2)+1)/2*0.06

    def phase_abnormal_handle(self):
        if 0 == [[e > 1.0 for e in self.contact_phase] + [e > 1.0 for e in self.swing_phase]].count(True):
            self.phase_abnormal = False
        else:
            self.phase_abnormal = True

    def step(self, est_data):
        if not self.phase_abnormal:
            self.timer = self.timer + self.dt
        self.update_step_phase()
        self.phase_abnormal_handle()
        self.update_body_target()
        self.update_foot_target(est_data)
        return self.pb


class Controller:
    def __init__(self, model):
        self.tor = np.asarray([.0] * 18)
        self.body_pos_des = np.asmatrix([.0, .0, 0.16])
        self.body_rot_des = R.from_matrix([[1., .0, .0], [.0, 1., .0], [0., 0., 1.]])
        self.body_acc_des = np.asmatrix([0.] * 6).T
        self.body_pos_fdb = np.asmatrix(env.reset()[0:3])
        self.body_rot_fdb = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.model = model
        self.data = self.model.createData()
        self.pin_inter = PinInter()

    def step_high_slope(self, p, o):
        kp = np.asarray([100.] * 6)
        ddp = np.asarray([0.] * 6)
        ddp[0:3] = np.multiply(kp[0:3], o[0] - p[0:3])

        return self.tor

    def classical_control(self, observation):
        baseLinearVelocityLocal = observation[6:9]
        # print("baseLinearVelocityLocal:", baseLinearVelocityLocal)
        foot_pos_cmd = observation[20:32]
        step_phase = observation[12:16]
        contact = observation[16:20]
        footHold = np.asarray(baseLinearVelocityLocal[0:2]) * 1.0 / self.step_freq * (0.5 + 0.05)
        action = [0.] * 12
        kp = [20., -20., 0.]
        kd = [2., -2., 0.]
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
        pin_q_ = self.pin_inter.to_pin(est.q_, 7)
        pin_dq_ = self.pin_inter.to_pin(est.dq_, 6)
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
        pin.crba(model, data, pin_q_)
        M = np.asmatrix(data.M)
        # ddx = np.zeros(12).reshape(12, 1)
        ddx = self.foot_acc_des
        JF = Jfoot
        JF_pinv = np.linalg.inv(M) * JF.T * np.linalg.inv(JF * np.linalg.inv(M) * JF.T)
        pin_ddq = JF_pinv * (ddx - Jdfoot * pin_dq_)
        N1 = np.eye(18) - JF_pinv * JF

        # WBC task 2: body control
        JB_pinv = np.linalg.inv(M) * JB.T * np.linalg.inv(JB * np.linalg.inv(M) * JB.T)
        # N1 = np.eye(18) - JB_pinv*JB
        J2 = JB
        Jd2 = JdB
        J2_pre = J2 * N1
        J2_pre_dpinv = np.linalg.inv(M) * J2_pre.T * np.linalg.inv(J2_pre * np.linalg.inv(M) * J2_pre.T)
        ddx = self.body_acc_des
        pin_ddq = pin_ddq + J2_pre_dpinv * (ddx - Jd2 * pin_dq_ - J2 * pin_ddq)

        # floating base dynamics
        pin.forwardKinematics(model, data, pin_q_, pin_dq_, pin_ddq)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('LFFoot_link'), pin.LOCAL_WORLD_ALIGNED)
        # a = pin.getFrameAcceleration(model, data,model.getFrameId('LFFoot_link'), pin.LOCAL)
        # print("a LFFoot: ", a)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('body_link'), pin.LOCAL_WORLD_ALIGNED)
        # a = pin.getFrameAcceleration(model, data,model.getFrameId('body_link'), pin.LOCAL)
        # print("a body: ", a)

        M = pin.crba(model, data, pin_q_)
        nle = np.matrix(pin.nle(model, data, pin_q_, pin_dq_)).T
        f = np.matrix([.0, .0, .0] * 4).T
        pin_tor = np.matrix([.0, .0, .0] * 6).T

        # tau + J.T*f = M * ddq + nle
        # x = tau(18)-force(12)-delta_ddx(6)
        H = ca.DM.eye(18 + 12 +6)
        # for i in range(18):
        #     H[i, i] = 0
        g = ca.DM.zeros(18 + 12 +6)

        # A = ca.DM.eye(18)
        A = ca.DM.zeros(34, 36)
        A[0:18, 0:18] = ca.DM.eye(18)
        A[0:18, 18:30] = JF.T
        A[0:18, 30:36] = -M * J2_pre_dpinv
        mu_c = 3.3
        # friction cone constrain
        A[18:22, 18:21] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[22:26, 21:24] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[26:30, 24:27] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        A[30:34, 27:30] = np.asmatrix([[1, 0, -mu_c], [-1, 0, -mu_c], [0, 1, -mu_c], [0, -1, -mu_c]])
        # A[0:18, 18:30] = JF.T.zeros()
        # A[0:18, 18:30] = ca.DM.zeros(18, 12)
        lba = ca.DM.zeros(34)
        lba[0:18] = M * pin_ddq + nle
        lba[18:34] = -1000

        uba = ca.DM(lba)
        uba[18:34] = 0
        ubx = ca.DM.zeros(36)
        ubx[0:6] = 0.000001
        ubx[6:18] = 200.
        for leg in range(4):
            if plan.contact_phase[leg] > 0.00001:
                ubx[18 + leg * 3 + 0: 18 + leg*3 + 3] = 200
            else:
                ubx[18 + leg * 3 + 0: 18 + leg * 3 + 3] = 0.01
        # ubx[18:30] = 200
        # ubx[30:33] = 50.5
        # ubx[33:36] = 5.5
        ubx[30:33] = 50.0
        ubx[33:36] = 50.0
        lbx = ca.DM(-ubx)
        lbx[20:30:3] = 0.

        qp = {'h': H.sparsity(), 'a': A.sparsity()}
        # opts = {'printLevel': 'none'}
        opts = {'printLevel': 'none', 'error_on_fail': False}
        S = ca.conic('S', 'qpoases', qp, opts)
        r = S(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        x_opt = r['x']
        # print('x_opt:', x_opt)

        pin_tor[0:18, 0] = x_opt[0:18]
        f[0:12, 0] = x_opt[18:30]
        delta_ddx = np.matrix([.0]*6).T
        delta_ddx[0: 6, 0] = x_opt[30: 36]
        # delta_ddq[0: 6] = x_opt[30: 36]
        # print('tor == M * ddq + nle - J.T*f:', M * (pin_ddq + J2_pre_dpinv * delta_ddx) + nle - JF.T * f)
        # print('ddq:', np.linalg.inv(M) * (pin_tor + JF.T * f - nle))

        pin.forwardKinematics(model, data, pin_q_, pin_dq_, pin_ddq + J2_pre_dpinv * delta_ddx)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('LFFoot_link'), pin.LOCAL_WORLD_ALIGNED)
        # print("a LFFoot: ", a)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('LHFoot_link'), pin.LOCAL_WORLD_ALIGNED)
        # print("a LHFoot: ", a)
        a = pin.getFrameAcceleration(model, data, model.getFrameId('body_link'), pin.LOCAL_WORLD_ALIGNED)
        # print("a body cmd: ", a)

        self.tor = self.pin_inter.from_pin(pin_tor, 6)

        pin.aba(model, data, pin_q_, pin_dq_, pin_tor + est.Jfoot.T * f)
        pin_ddq_ = self.pin_inter.from_pin(data.ddq, 6)
        # print('pin_ddq_:', pin_ddq_)

        return self.tor

    def step(self, est, plan):
        return self.optimal_control(est, plan)

class RosPublish:
    def __init__(self, model):
        self.model = model

        # msg publish
        # publish path
        self.path_buffer_length = 40
        self.rf_foot_path = nmsg.Path()
        self.lf_foot_path = nmsg.Path()
        self.rh_foot_path = nmsg.Path()
        self.lh_foot_path = nmsg.Path()
        self.foot_path = [self.rf_foot_path, self.lf_foot_path, self.rh_foot_path, self.lh_foot_path]
        self.rf_foot_path_pub = rospy.Publisher('rf_foot_trajectory', nmsg.Path, queue_size=10)
        self.lf_foot_path_pub = rospy.Publisher('lf_foot_trajectory', nmsg.Path, queue_size=10)
        self.rh_foot_path_pub = rospy.Publisher('rh_foot_trajectory', nmsg.Path, queue_size=10)
        self.lh_foot_path_pub = rospy.Publisher('lh_foot_trajectory', nmsg.Path, queue_size=10)
        self.foot_path_pub = [self.rf_foot_path_pub, self.lf_foot_path_pub, self.rh_foot_path_pub, self.lh_foot_path_pub]

        # publish point
        self.foot_point = [gmsg.PointStamped(), gmsg.PointStamped(), gmsg.PointStamped(), gmsg.PointStamped()]
        self.foot_point_pub = [rospy.Publisher('rf_foot_point', gmsg.PointStamped, queue_size=10),
                               rospy.Publisher('lf_foot_point', gmsg.PointStamped, queue_size=10),
                               rospy.Publisher('rh_foot_point', gmsg.PointStamped, queue_size=10),
                               rospy.Publisher('lh_foot_point', gmsg.PointStamped, queue_size=10)]

        # publish pose
        # self.foot_pose = [gmsg.PoseStamped(), gmsg.PoseStamped(), gmsg.PoseStamped(), gmsg.PoseStamped()]
        # self.foot_pose_pub = [rospy.Publisher('rf_foot_pose', gmsg.PoseStamped, queue_size=10),
        #                        rospy.Publisher('lf_foot_pose', gmsg.PoseStamped, queue_size=10),
        #                        rospy.Publisher('rh_foot_pose', gmsg.PoseStamped, queue_size=10),
        #                        rospy.Publisher('lh_foot_pose', gmsg.PoseStamped, queue_size=10)]

        # publish wrench
        # self.foot_force = [gmsg.WrenchStamped(), gmsg.WrenchStamped(), gmsg.WrenchStamped(), gmsg.WrenchStamped()]
        # self.foot_force_pub = [rospy.Publisher('rf_foot_force', gmsg.WrenchStamped, queue_size=10),
        #                        rospy.Publisher('lf_foot_force', gmsg.WrenchStamped, queue_size=10),
        #                        rospy.Publisher('rh_foot_force', gmsg.WrenchStamped, queue_size=10),
        #                        rospy.Publisher('lh_foot_force', gmsg.WrenchStamped, queue_size=10)]

        # publish marker
        self.foot_marker = [vmsg.Marker(), vmsg.Marker(), vmsg.Marker(), vmsg.Marker()]
        self.foot_marker_pub = [rospy.Publisher('rf_foot_marker', vmsg.Marker, queue_size=10),
                               rospy.Publisher('lf_foot_marker', vmsg.Marker, queue_size=10),
                               rospy.Publisher('rh_foot_marker', vmsg.Marker, queue_size=10),
                               rospy.Publisher('lh_foot_marker', vmsg.Marker, queue_size=10)]

        # publish joint state
        self.joint_state = smsg.JointState()
        self.joint_state_pub = rospy.Publisher('joint_state', smsg.JointState, queue_size=10)
        self.joint_name = ['RF1_joint', 'RF2_joint', 'RF3_joint',
                           'LF1_joint', 'LF2_joint', 'LF3_joint',
                           'RH1_joint', 'RH2_joint', 'RH3_joint',
                           'LH1_joint', 'LH2_joint', 'LH3_joint']
        for joint in range(12):
            self.joint_state.name.append(self.joint_name[joint])
            self.joint_state.position.append(.0)
            self.joint_state.velocity.append(.0)
            self.joint_state.effort.append(.0)

        rospy.init_node('talker', anonymous=True)

    def step(self, est, plan, control):
        self.pub_tf(est, plan)
        self.pub_path(est, plan)
        self.pub_point(est, plan)
        # self.pub_pose(est, plan)
        # self.pub_wrench(est, plan)
        self.pub_marker(est, plan)
        self.pub_state(est, plan, control)

    def pub_tf(self, est, plan):

        def pub_tf_body_link():
            tf_body_ = Transform(
                translation=gmsg.Vector3(est.pb_[0], est.pb_[1], est.pb_[2]),
                rotation=gmsg.Quaternion(est.pb_[3], est.pb_[4], est.pb_[5], est.pb_[6])
            )

            st = gmsg.TransformStamped()
            st.header.frame_id = 'world'
            st.child_frame_id = 'body_link'
            st.transform = tf_body_
            st.header.stamp = rospy.Time.now()

            br = tf2_ros.TransformBroadcaster()
            br.sendTransform(st)

        def pub_tf_leg_link():
            def pub_tf_single_link(plink, clink, p_offset, axis, angle):
                translation = gmsg.Vector3(p_offset[0], p_offset[1], p_offset[2])
                if 'x' == axis:
                    quat = quaternion_from_euler(angle, 0, 0)
                elif 'y' == axis:
                    quat = quaternion_from_euler(0, angle, 0)
                rotation = gmsg.Quaternion(quat[0], quat[1], quat[2], quat[3])

                tf_link_ = Transform(translation, rotation)

                st = gmsg.TransformStamped()
                st.header.frame_id = plink
                st.child_frame_id = clink
                st.transform = tf_link_
                st.header.stamp = rospy.Time.now()

                br = tf2_ros.TransformBroadcaster()
                br.sendTransform(st)

            pub_tf_single_link('body_link', 'RF1_link', [0.0915, -0.08, 0], 'x', est.q_[7 + 0])
            pub_tf_single_link('RF1_link', 'RF2_link', [0.0, 0.0, -0.046], 'y', est.q_[7 + 1])
            pub_tf_single_link('RF2_link', 'RF3_link', [0.0, 0.0, -0.066], 'y', est.q_[7 + 2])
            pub_tf_single_link('RF3_link', 'RFFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'LF1_link', [0.0915,  0.08, 0], 'x', est.q_[7 + 3])
            pub_tf_single_link('LF1_link', 'LF2_link', [0.0, 0.0, -0.046], 'y', est.q_[7 + 4])
            pub_tf_single_link('LF2_link', 'LF3_link', [0.0, 0.0, -0.066], 'y', est.q_[7 + 5])
            pub_tf_single_link('LF3_link', 'LFFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'RH1_link', [-0.0915, -0.08, 0], 'x', est.q_[7 + 6])
            pub_tf_single_link('RH1_link', 'RH2_link', [0.0, 0.0, -0.046], 'y', est.q_[7 + 7])
            pub_tf_single_link('RH2_link', 'RH3_link', [0.0, 0.0, -0.066], 'y', est.q_[7 + 8])
            pub_tf_single_link('RH3_link', 'RHFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'LH1_link', [-0.0915, 0.08, 0], 'x', est.q_[7 + 9])
            pub_tf_single_link('LH1_link', 'LH2_link', [0.0, 0.0, -0.046], 'y', est.q_[7 + 10])
            pub_tf_single_link('LH2_link', 'LH3_link', [0.0, 0.0, -0.066], 'y', est.q_[7 + 11])
            pub_tf_single_link('LH3_link', 'LHFoot_link', [0.0, 0.0, -0.065], 'x', 0)


        pub_tf_body_link()
        pub_tf_leg_link()

    def pub_path(self, est, plan):

        def pub_foot_path(leg_id):
            self.foot_path[leg_id].header.frame_id = "world"
            self.foot_path[leg_id].header.stamp = rospy.Time.now()

            pose = gmsg.PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = rospy.Time.now()
            # pose.pose.position.x = est.pf_[leg_id*3 + 0]
            # pose.pose.position.y = est.pf_[leg_id*3 + 1]
            # pose.pose.position.z = est.pf_[leg_id*3 + 2]
            pose.pose.position.x = plan.pf[leg_id*3 + 0]
            pose.pose.position.y = plan.pf[leg_id*3 + 1]
            pose.pose.position.z = plan.pf[leg_id*3 + 2]
            quaternion = quaternion_from_euler(
                0, 0, 0)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            # self.foot_path[leg_id].poses.clear()
            # self.foot_path[leg_id].poses.append(pose)
            # set path buffer
            if len(self.foot_path[leg_id].poses) < self.path_buffer_length:
                self.foot_path[leg_id].poses.append(pose)
            while len(self.foot_path[leg_id].poses) > self.path_buffer_length:
                self.foot_path[leg_id].poses.pop()
            if len(self.foot_path[leg_id].poses) == self.path_buffer_length:
                for i in range(0, self.path_buffer_length-1):
                    self.foot_path[leg_id].poses[i] = self.foot_path[leg_id].poses[i+1]
                self.foot_path[leg_id].poses[self.path_buffer_length-1] = pose

            self.foot_path_pub[leg_id].publish(self.foot_path[leg_id])

        pub_foot_path(0)
        pub_foot_path(1)
        pub_foot_path(2)
        pub_foot_path(3)

    def pub_point(self, est, plan):
        def pub_foot_point(leg_id):
            self.foot_point[leg_id].header.frame_id = 'world'
            self.foot_point[leg_id].header.stamp = rospy.Time.now()
            self.foot_point[leg_id].point.x = est.pf_[leg_id*3 + 0]
            self.foot_point[leg_id].point.y = est.pf_[leg_id*3 + 1]
            self.foot_point[leg_id].point.z = est.pf_[leg_id*3 + 2]
            self.foot_point_pub[leg_id].publish(self.foot_point[leg_id])

        pub_foot_point(0)
        pub_foot_point(1)
        pub_foot_point(2)
        pub_foot_point(3)

    # def pub_pose(self, est, plan):
    #     def pub_foot_pose(leg_id):
    #         self.foot_pose[leg_id].header.frame_id = 'world'
    #         self.foot_pose[leg_id].header.stamp = rospy.Time.now()
    #         self.foot_pose[leg_id].pose.position.x = 0
    #         self.foot_pose[leg_id].pose.position.y = 0
    #         self.foot_pose[leg_id].pose.position.z = 1
    #         self.foot_pose[leg_id].pose.orientation.x = 0
    #         self.foot_pose[leg_id].pose.orientation.y = 0
    #         self.foot_pose[leg_id].pose.orientation.z = 0
    #         self.foot_pose[leg_id].pose.orientation.w = 1
    #         self.foot_pose_pub[leg_id].publish(self.foot_pose[leg_id])
    #
    #     pub_foot_pose(0)
    #     pub_foot_pose(1)
    #     pub_foot_pose(2)
    #     pub_foot_pose(3)

    # def pub_wrench(self, est, plan):
    #     def pub_foot_force(leg_id):
    #         self.foot_force[leg_id].header.frame_id = 'LFFoot_link'
    #         self.foot_force[leg_id].header.stamp = rospy.Time.now()
    #         self.foot_force[leg_id].wrench.force.x = 0
    #         self.foot_force[leg_id].wrench.force.y = 0
    #         self.foot_force[leg_id].wrench.force.z = 1
    #         self.foot_force[leg_id].wrench.torque.x = 0
    #         self.foot_force[leg_id].wrench.torque.y = 0
    #         self.foot_force[leg_id].wrench.torque.z = 0
    #         self.foot_force_pub[leg_id].publish(self.foot_force[leg_id])
    #
    #     pub_foot_force(0)
    #     pub_foot_force(1)
    #     pub_foot_force(2)
    #     pub_foot_force(3)

    def pub_marker(self, est, plan):
        def pub_foot_marker(leg_id):
            self.foot_marker[leg_id].header.frame_id = 'world'
            self.foot_marker[leg_id].header.stamp = rospy.Time.now()
            self.foot_marker[leg_id].ns = 'my_namespace'
            self.foot_marker[leg_id].id = leg_id
            self.foot_marker[leg_id].type = vmsg.Marker.ARROW
            self.foot_marker[leg_id].action = vmsg.Marker.MODIFY
            self.foot_marker[leg_id].pose.position.x = est.pf_[leg_id*3 + 0]
            self.foot_marker[leg_id].pose.position.y = est.pf_[leg_id*3 + 1]
            self.foot_marker[leg_id].pose.position.z = est.pf_[leg_id*3 + 2]
            self.foot_marker[leg_id].pose.orientation.x = 0
            self.foot_marker[leg_id].pose.orientation.y = 0
            self.foot_marker[leg_id].pose.orientation.z = 0
            self.foot_marker[leg_id].pose.orientation.w = 1
            self.foot_marker[leg_id].scale.x = plan.contact_phase[leg_id]*0.08
            self.foot_marker[leg_id].scale.y = 0.01
            self.foot_marker[leg_id].scale.z = 0.01
            self.foot_marker[leg_id].color.a = 0.5
            self.foot_marker[leg_id].color.r = 1.0
            self.foot_marker[leg_id].color.g = 0.0
            self.foot_marker[leg_id].color.b = 0.0

            self.foot_marker_pub[leg_id].publish(self.foot_marker[leg_id])

        pub_foot_marker(0)
        pub_foot_marker(1)
        pub_foot_marker(2)
        pub_foot_marker(3)

    def pub_state(self, est, plan, control):
        def pub_joint_state():
            self.joint_state.header.frame_id = 'world'
            self.joint_state.header.stamp = rospy.Time.now()
            for joint in range(12):
                # self.joint_state.effort[joint] = est.tor_[joint]
                self.joint_state.effort[joint] = control.tor[6 + joint]
            self.joint_state_pub.publish(self.joint_state)

        pub_joint_state()

if __name__ == '__main__':
    id = "gym_env:Quadruped-v0"
    env = gym.make(id)

    ## quadruped model
    root = pin.JointModelFreeFlyer()
    model = pin.buildModelFromUrdf(pybullet_data.getDataPath() + '/urdf/quadruped_robot/quadruped_robot.urdf',
                                   root)
    # model = pin.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf')

    torque = [.0] * 12

    est = Estimation(model)
    plan = Planner(model)
    control = Controller(model)
    rospub = RosPublish(model)
    for i in range(100000):

        # simulation
        # leg order: LF-RF-LH-RH
        torque = list(control.tor[6 + 3:6 + 6]) + list(control.tor[6 + 0:6 + 3]) + list(control.tor[6 + 9:6 + 12]) \
                 + list(control.tor[6 + 6:6 + 9])
        [o_, pb_, vb_, js_] = env.step_torque(torque)
        # js_ = list(js_[0:3]) + list(js_[6:9]) + list(js_[3:6]) + list(js_[9:12])
        js_ = list(js_[3:6]) + list(js_[0:3]) + list(js_[9:12]) + list(js_[6:9])


        pj_ = [i[0] for i in js_]
        vj_ = [i[1] for i in js_]
        tj_ = [i[3] for i in js_]
        q_ = np.asmatrix(list(pb_[0] + pb_[1]) + pj_).T
        dq_ = np.asmatrix(list(vb_[0] + vb_[1]) + vj_).T
        tor_ = np.asmatrix([.0]*6 + tj_).T


        est.q_ = q_
        est.dq_ = dq_
        est.tor_ = tor_
        est.pb_[0:3] = pb_[0]
        est.pb_[3:7] = pb_[1]
        est.vb_[0:3] = vb_[0]
        est.vb_[3:6] = vb_[1]
        est.step()

        plan.step(est)

        torque = control.step(est, plan)

        rospub.step(est, plan, control)

        # env.render()
    input("press any key to continue...")
    env.close()
