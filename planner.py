import numpy as np
import math

from scipy.spatial.transform import Rotation as R
import tf.transformations as transformations

from utility import Utility


class SimpleBezier:
    def __init__(self):
        self.time = .0
        self.pi = np.matrix([.0] * 3).T
        self.pf = np.matrix([.0] * 3).T
        self.height = .0

        self.p = np.matrix([.0] * 3).T
        self.v = np.matrix([.0] * 3).T
        self.a = np.matrix([.0] * 3).T

    def set_line(self, t, pi, pf, h):
        self.time = t
        self.pi = pi
        self.pf = pf
        self.height = h

    def compute_point(self, phase):
        x = phase
        delta_p = self.pf - self.pi
        self.p = self.pi + delta_p * (x * x * x + 3. * (x * x * (1. - x)))
        self.v = delta_p * (6. * x * (1. - x)) / self.time
        self.a = delta_p * (6. - 12. * x) / self.time / self.time

        if phase < 0.5:
            x = phase * 2
            delta_p_z = self.height
            self.p[2] = self.pi[2] + delta_p_z * (x * x * x + 3. * (x * x * (1. - x)))
            self.v[2] = delta_p_z * (6. * x * (1. - x)) * 2 / self.time
            self.a[2] = delta_p_z * (6. - 12. * x) * 4 / self.time / self.time
        else:
            x = (phase - 0.5) * 2
            delta_p_z = self.pf[2] - (self.pi[2] + self.height)
            self.p[2] = self.pi[2] + self.height + delta_p_z * (x * x * x + 3. * (x * x * (1. - x)))
            self.v[2] = delta_p_z * (6. * x * (1. - x)) * 2 / self.time
            self.a[2] = delta_p_z * (6. - 12. * x) * 4 / self.time / self.time

    def get_position(self):
        return self.p

    def get_velocity(self):
        return self.v

    def get_acceleration(self):
        return self.a


class Planner:
    def __init__(self, model, common_para):
        self.timer = .0
        self.dt = common_para.dt
        self.iterPerPlan = 5
        self.dtPlan = self.iterPerPlan * self.dt
        self.horizon = 200
        self.gait = 'trot'
        self.segments = np.array([0, 0, 0, 0])
        self.duration = np.array([0, 0, 0, 0])
        self.offset = np.array([0, 0, 0, 0])
        self.contact_state = np.array([0, 0, 0, 0])
        self.swing_phase = np.array([.0, .0, .0, .0])
        self.contact_phase = np.array([.0, .0, .0, .0])
        self.step_period = .0
        self.first_swing = np.array([True, True, True, True])
        self.first_contact = np.array([True, True, True, True])
        self.phase_abnormal = False
        self.pf_init = np.matrix([.0] * 12).T
        self.pf_hold = np.matrix([.0] * 12).T
        self.pf = np.matrix([.0] * 12).T
        self.vf = np.matrix([.0] * 12).T
        self.af = np.matrix([.0] * 12).T
        self.ph_body = np.matrix([0.0915, -0.08, .0, 0.0915, 0.08, .0, -0.0915, -0.08, .0, -0.0915, 0.08, .0]).T
        self.pb = np.matrix([.0, .0, 0.16, 0., 0., 0., 1.]).T
        self.vb = np.matrix([.0] * 6).T
        self.model = model
        self.data = self.model.createData()

        self.stand_height = 0.16

        self.foot_trajectory = [SimpleBezier(), SimpleBezier(), SimpleBezier(), SimpleBezier()]

        self.ut = Utility()

    def get_contact_target(self, time):
        if self.gait == 'trot':
            self.segments = np.array([10, 10, 10, 10])
            self.duration = np.array([5, 5, 5, 5])
            self.offset = np.array([0, 5, 5, 0])
            self.step_period = self.segments[0] * self.dtPlan
        # iter = time / self.dt
        iterPlan = time / self.dtPlan
        for leg in range(4):
            normIterPlan = (iterPlan + self.segments[leg] - self.offset[leg]) % self.segments[leg]
            self.contact_state[leg] = normIterPlan < self.duration[leg]
        return self.contact_state

    def get_step_phase(self, time):
        if self.gait == 'trot':
            self.segments = np.array([10, 10, 10, 10])
            self.duration = np.array([5, 5, 5, 5])
            self.offset = np.array([0, 5, 5, 0])
            self.step_period = self.segments[0] * self.dtPlan
        iter = round(time / self.dt)
        # iterPlan = time / self.dtPlan
        for leg in range(4):
            normIter = (iter + self.segments[leg] * self.iterPerPlan - self.offset[leg] * self.iterPerPlan) % (
                    self.segments[leg] * self.iterPerPlan) + 1
            if normIter <= self.duration[leg] * self.iterPerPlan:
                self.contact_phase[leg] = normIter / float(self.duration[leg] * self.iterPerPlan)
                self.swing_phase[leg] = 0
            else:
                self.contact_phase[leg] = 0
                self.swing_phase[leg] = (normIter - self.duration[leg] * self.iterPerPlan) / float(
                    (self.segments[leg] - self.duration[leg]) * self.iterPerPlan)
        return [self.contact_phase, self.swing_phase]

    def update_step_phase(self):
        self.get_step_phase(self.timer)

    def update_body_target(self, rece):
        # body_pos_des = np.matrix([0., 0., 0.16])
        # body_rot_des = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.pb[0:2] = self.pb[0:2] + rece.body_vel[0:2] * self.dt
        self.pb[2] = self.stand_height + rece.body_pos_offset[2]
        self.vb[0:2] = rece.body_vel[0:2]
        return self.pb

    def update_foot_target(self, est):
        for leg in range(4):
            if self.contact_phase[leg] > 0.0001:  # contact phase
                self.first_swing[leg] = True
                # if self.first_contact[leg]:
                self.pf[leg * 3 + 0] = est.pf_[leg * 3 + 0]
                self.pf[leg * 3 + 1] = est.pf_[leg * 3 + 1]
                self.pf[leg * 3 + 2] = est.pf_[leg * 3 + 2]
                self.vf[leg * 3 + 0] = 0
                self.vf[leg * 3 + 1] = 0
                self.vf[leg * 3 + 2] = 0
                self.af[leg * 3 + 0] = 0
                self.af[leg * 3 + 1] = 0
                self.af[leg * 3 + 2] = 0

            elif self.swing_phase[leg] > 0.0001:  # swing phase
                self.first_contact[leg] = True
                if self.first_swing[leg]:
                    self.first_swing[leg] = False
                    self.pf_init[leg * 3 + 0: leg * 3 + 3] = est.pf_[leg * 3 + 0: leg * 3 + 3]
                # self.pf_hold[leg*3 + 0] = self.ph_body[leg*3 + 0]
                # self.pf_hold[leg*3 + 1] = self.ph_body[leg*3 + 1]
                # self.pf_hold[leg*3 + 2] = self.ph_body[leg*3 + 2]
                # self.pf_hold[leg*3 + 0] = self.pf_init[leg*3 + 0]
                # self.pf_hold[leg*3 + 1] = self.pf_init[leg*3 + 1]
                # self.pf_hold[leg*3 + 2] = self.pf_init[leg*3 + 2]

                yaw = transformations.euler_from_quaternion(self.ut.m2l(est.pb_[3:7]))[2]
                rot_b2w = np.matrix([[math.cos(-yaw), -math.sin(-yaw)],
                                     [math.sin(-yaw), math.cos(-yaw)]])
                Kp = np.diag([-0.03, -0.01])
                self.pf_hold[leg * 3 + 0: leg * 3 + 2] = \
                    ( \
                                est.pb_[0: 2] + est.vb_[0: 2] * self.step_period / 2.0 \
                                + rot_b2w * Kp * rot_b2w.T * (self.vb[0: 2] - est.vb_[0: 2]) \
                                + rot_b2w * self.ph_body[leg * 3 + 0: leg * 3 + 2] \
                        )
                self.pf_hold[leg * 3 + 2] = self.pf_init[leg * 3 + 2]
                swing_time = (self.segments[leg] - self.duration[leg]) * self.dtPlan

                # self.pf[leg * 3 + 0] = self.pf_init[leg * 3 + 0] + (
                #             self.pf_hold[leg * 3 + 0] - self.pf_init[leg * 3 + 0]) * self.swing_phase[leg]
                # self.pf[leg * 3 + 1] = self.pf_init[leg * 3 + 1] + (
                #             self.pf_hold[leg * 3 + 1] - self.pf_init[leg * 3 + 1]) * self.swing_phase[leg]
                # self.pf[leg * 3 + 2] = self.pf_init[leg * 3 + 2] + (
                #             math.cos((self.swing_phase[leg] - 0.5) * 3.14 * 2) + 1) / 2 * 0.03
                # self.vf[leg * 3 + 0] = 0
                # self.vf[leg * 3 + 1] = 0
                # self.vf[leg * 3 + 2] = -math.sin(
                #     (self.swing_phase[leg] - 0.5) * 3.14 * 2) / 2 * 0.03 * 3.14 * 2 / swing_time
                # self.af[leg * 3 + 0] = 0
                # self.af[leg * 3 + 1] = 0
                # self.af[leg * 3 + 2] = -math.cos(
                #     (self.swing_phase[leg] - 0.5) * 3.14 * 2) / 2 * 0.03 * 3.14 * 2 * 3.14 * 2 / swing_time / swing_time

                self.foot_trajectory[leg].set_line(swing_time, self.pf_init[leg * 3 + 0: leg * 3 + 3],
                                                   self.pf_hold[leg * 3 + 0: leg * 3 + 3], 0.03)
                self.foot_trajectory[leg].compute_point(self.swing_phase[leg])
                self.pf[leg * 3 + 0: leg * 3 + 3] = self.foot_trajectory[leg].get_position()
                self.vf[leg * 3 + 0: leg * 3 + 3] = self.foot_trajectory[leg].get_velocity()
                self.af[leg * 3 + 0: leg * 3 + 3] = self.foot_trajectory[leg].get_acceleration()

    def phase_abnormal_handle(self):
        if 0 == [[e > 1.0 for e in self.contact_phase] + [e > 1.0 for e in self.swing_phase]].count(True):
            self.phase_abnormal = False
        else:
            self.phase_abnormal = True

    def step(self, rece, est):
        if not self.phase_abnormal:
            self.timer = self.timer + self.dt
        self.update_step_phase()
        self.phase_abnormal_handle()
        self.update_body_target(rece)
        self.update_foot_target(est)
        return self.pb
