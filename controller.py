import numpy as np

from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import casadi as ca

from utility import Utility


class Controller:
    def __init__(self, model):
        self.tor = np.matrix([.0] * 18).T
        self.body_pos_des = np.matrix([.0, .0, 0.16])
        self.body_rot_des = R.from_matrix([[1., .0, .0], [.0, 1., .0], [0., 0., 1.]])
        self.body_acc_des = np.matrix([0.] * 6).T
        self.body_pos_fdb = np.matrix([.0] * 3).T
        self.body_rot_fdb = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.model = model
        self.data = self.model.createData()
        self.ut = Utility()

    def step_high_slope(self, p, o):
        kp = np.array([100.] * 6)
        ddp = np.array([0.] * 6)
        ddp[0:3] = np.multiply(kp[0:3], o[0] - p[0:3])

        return self.tor

    def classical_control(self, observation):
        baseLinearVelocityLocal = observation[6:9]
        # print("baseLinearVelocityLocal:", baseLinearVelocityLocal)
        foot_pos_cmd = observation[20:32]
        step_phase = observation[12:16]
        contact = observation[16:20]
        footHold = np.array(baseLinearVelocityLocal[0:2]) * 1.0 / self.step_freq * (0.5 + 0.05)
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
        pin_q_ = self.ut.to_pin(est.q_, 7)
        pin_dq_ = self.ut.to_pin(est.dq_, 6)
        JB = est.JB
        Jfoot = est.Jfoot
        JdB = est.JdB
        Jdfoot = est.Jdfoot
        model = self.model
        data = self.data

        self.body_pos_des = plan.pb[0:3]
        self.body_rot_des = R.from_quat(self.ut.m2l(plan.pb[3:7]))
        self.body_pos_fdb = est.pb_[0:3]
        self.body_rot_fdb = R.from_quat(self.ut.m2l(est.pb_[3:7]))
        self.body_vel_des = plan.vb
        self.body_vel_fdb = est.vb_

        self.body_acc_des[0:3] = 200. * (self.body_pos_des - self.body_pos_fdb) \
                                 + 20. * (self.body_vel_des[0:3] - self.body_vel_fdb[0:3])
        self.body_acc_des[3:6] = 200. * np.matrix((self.body_rot_des * self.body_rot_fdb.inv()).as_rotvec()).T \
                                 + 20. * (self.body_vel_des[3:6] - self.body_vel_fdb[3:6])

        foot_kp = np.array([.0] * 12)
        foot_kd = np.array([.0] * 12)
        for leg in range(4):
            if plan.swing_phase[leg] > 0.0001:
                foot_kp[leg * 3 + 0: leg * 3 + 3] = np.array([200., 200., 200.])
                foot_kd[leg * 3 + 0: leg * 3 + 3] = np.array([200., 200., 200.])
            else:
                foot_kp[leg * 3 + 0: leg * 3 + 3] = np.array([0., 0., 0.])
                foot_kd[leg * 3 + 0: leg * 3 + 3] = np.array([100., 100., 100.])
        self.foot_acc_des = np.diag(foot_kp) * (plan.pf - est.pf_) + np.diag(foot_kd) * (plan.vf - est.vf_) + \
                            plan.af

        # WBC task 1: contact foot not slip
        pin.crba(model, data, pin_q_)
        M = np.matrix(data.M)
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
        H = ca.DM.eye(18 + 12 + 6)
        # H = ca.DM.zeros(18 + 12 +6)
        for diag in range(0, 30):
            H[diag, diag] = 0.00001
        for diag in range(30, 36):
            H[diag, diag] = 10
        H[31, 31] = 1
        # for i in range(18):
        #     H[i, i] = 0
        g = ca.DM.zeros(18 + 12 + 6)

        # A = ca.DM.eye(18)
        A = ca.DM.zeros(34, 36)
        A[0:18, 0:18] = ca.DM.eye(18)
        A[0:18, 18:30] = JF.T
        A[0:18, 30:36] = -M * J2_pre_dpinv
        mu_c = 0.3
        # friction cone constrain
        for leg in range(4):
            if plan.contact_phase[leg] > 0.00001:
                A[leg * 4 + 18:leg * 4 + 22, leg * 3 + 18:leg * 3 + 21] = np.matrix(
                    [[1., 0., -mu_c], [-1., 0., -mu_c], [0., 1., -mu_c], [0., -1., -mu_c]])
            else:
                A[leg * 4 + 18:leg * 4 + 22, leg * 3 + 18:leg * 3 + 21] = np.matrix(
                    [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        # A[0:18, 18:30] = JF.T.zeros()
        # A[0:18, 18:30] = ca.DM.zeros(18, 12)
        lba = ca.DM.zeros(34)
        lba[0:18] = M * pin_ddq + nle
        # lba[0:18] = M * (pin_ddq + J2_pre_dpinv * (ddx - Jd2 * pin_dq_ - J2 * pin_ddq)) + nle
        lba[18:34] = -1000000.0

        uba = ca.DM(lba)
        uba[18:34] = 0
        ubx = ca.DM.zeros(36)
        ubx[0:6] = 0.0001
        ubx[6:18] = 200.
        for leg in range(4):
            if plan.contact_phase[leg] > 0.00001:
                ubx[18 + leg * 3 + 0: 18 + leg * 3 + 3] = 40.
            else:
                ubx[18 + leg * 3 + 0: 18 + leg * 3 + 3] = 0.001
        # ubx[18:30] = 200
        # ubx[30:33] = 50.5
        # ubx[33:36] = 5.5
        ubx[30:33] = 50.0
        ubx[33:36] = 50.0
        lbx = ca.DM(-ubx)
        lbx[20:30:3] = 0.

        qp = {'h': H.sparsity(), 'a': A.sparsity()}
        # opts = {'printLevel': 'none'}
        opts = {'error_on_fail': False, 'max_schur': 100, 'printLevel': 'none'}
        S = ca.conic('S', 'qpoases', qp, opts)
        r = S(h=H, g=g, a=A, lba=lba, uba=uba, lbx=lbx, ubx=ubx)
        x_opt = r['x']
        # print('x_opt:', x_opt)

        pin_tor[0:18] = x_opt[0:18]
        f[0:12] = x_opt[18:30]
        delta_ddx = np.matrix([.0] * 6).T
        delta_ddx[0: 6] = x_opt[30: 36]
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

        self.body_vel_des[0:3] = 10 * (self.body_pos_des - self.body_pos_fdb) \
                                 + (self.body_vel_des[0:3]) * 0
        self.body_vel_des[3:6] = 10 * np.matrix((self.body_rot_des * self.body_rot_fdb.inv()).as_rotvec()).T \
                                 + (self.body_vel_des[3:6]) * 0
        self.foot_vel_des = 0.005 * (plan.pf - est.pf_) + plan.vf * 0
        # WBC vel task 1: body control
        dx = self.foot_vel_des * 1
        JF = Jfoot
        JF_pinv = JF.T * np.linalg.inv(JF * JF.T)
        pin_dq = JF_pinv * dx
        N1 = np.eye(18) - JF_pinv * JF
        # WBC vel task 2: body control
        JB_pinv = JB.T * np.linalg.inv(JB * JB.T)
        # N1 = np.eye(18) - JB_pinv*JB
        J2 = JB
        Jd2 = JdB
        J2_pre = J2 * N1
        J2_pre_dpinv = J2_pre.T * np.linalg.inv(J2_pre * J2_pre.T)
        dx = self.body_vel_des
        pin_dq = pin_dq + J2_pre_dpinv * (dx - J2 * pin_dq)

        self.tor = self.ut.from_pin(pin_tor * 1 + 0.0 * (pin_dq - pin_dq_) - 0.000 * pin_dq_, 6)

        pin.aba(model, data, pin_q_, pin_dq_, pin_tor + est.Jfoot.T * f)
        ddq_ = self.ut.from_pin(data.ddq, 6)
        # print('ddq_:', ddq_)

        return self.tor

    def step(self, est, plan):
        return self.optimal_control(est, plan)
