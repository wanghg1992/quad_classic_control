import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from scipy.spatial.transform import Rotation as R
import tf.transformations as transformations

import casadi as ca

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


class TrajectoryOpti:
    def __init__(self):
        self.dtPlan = 0.05
        self.nLines = 4
        # self.line_para = np.matrix(np.zeros([3, self.nLines, 6])) # 3axis nLines 6para
        self.para_sequence = [np.zeros([6, 3])] * self.nLines  # 3axis nLines 6para
        self.position_sequence = [np.zeros([12, 0])] * self.nLines
        self.contact_sequence = [np.zeros([4, 0])] * self.nLines
        self.polygon_sequence = [] * self.nLines
        self.polygon_radius = 0.04
        self.start_time = 0
        self.end_time = self.nLines * self.dtPlan
        self.start_position = np.matrix([.0, .0, .14]).T
        self.end_position = np.matrix([.0, .0, .14]).T
        self.start_velocity = np.matrix([.0] * 3).T
        self.end_velocity = np.matrix([.0] * 3).T

        self.line_positions = []

        self.opti = ca.Opti()

    def eta(self, t):
        return [t ** 5., t ** 4., t ** 3., t ** 2., t, 1.]

    def d_eta(self, t):
        return [5. * t ** 4., 4. * t ** 3., 3. * t ** 2., 2. * t, 1., 0.]

    def dd_eta(self, t):
        return [20. * t ** 3., 12. * t ** 2., 6. * t, 2., 0., 0.]

    def update_polygon_sequence(self):
        self.polygon_sequence = []
        for i in range(self.nLines):
            # test
            # self.support_polygon.append([np.array([1., 1., 2.]), np.array([-1., -1., -5.])])

            support_point = []
            for leg in range(4):
                if self.contact_sequence[i][leg] == 1:
                    support_point.append(self.position_sequence[i][leg * 3 + 0:leg * 3 + 3])
            sp_sub = []
            if len(support_point) == 2:
                x1 = support_point[0][0]
                y1 = support_point[0][1]
                x2 = support_point[1][0]
                y2 = support_point[1][1]
                sp_sub.append(np.array([y2 - y1, x1 - x2, y1 * x2 - y2 * x1 - self.polygon_radius]))
                sp_sub.append(np.array([-(y2 - y1), -(x1 - x2), -(y1 * x2 - y2 * x1 + self.polygon_radius)]))
            self.polygon_sequence.append(sp_sub)

    def update_opti_struct(self, herizon):
        self.nLines = herizon
        self.para_sequence = [np.zeros([6, 3])] * self.nLines  # 3axis nLines 6para

    def step(self, ploy_seq):
        self.opti = ca.Opti()

        dt = self.dtPlan
        st = self.start_time
        sp = self.start_position
        ep = self.end_position
        sv = self.start_velocity
        ev = self.end_velocity

        # xa = self.opti.variable(6, self.nLines)
        # ya = self.opti.variable(6, self.nLines)
        # za = self.opti.variable(6, self.nLines)
        # a = [self.opti.variable(6, 3)] * self.nLines
        a = [self.opti.variable(6, 3) for i in range(self.nLines)]

        # Qacc = self.opti.parameter(6, 6)
        # self.opti.set_value(Qacc, ca.DM.eye(6))
        # Qacc = ca.DM.zeros(6, 6)
        Qacc = ca.DM([
            [400. / 7. * dt ** 7, 40. * dt ** 6, 24. * dt ** 5, 10. * dt ** 4, 0, 0],
            [40. * dt ** 6, 28.8 * dt ** 5, 18. * dt ** 4, 8. * dt ** 3, 0, 0],
            [24. * dt ** 5, 18. * dt ** 4, 12. * dt ** 3, 6. * dt ** 2, 0, 0],
            [10. * dt ** 4, 8. * dt ** 3, 6. * dt ** 2, 4. * dt, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # cost1 = ca.mtimes(xa[0, :], Qacc)
        # cost1 = ca.mtimes(cost1, ca.reshape(xa, 6, 1))
        # cost1 = xa[:, 0].T @ Qacc @ xa[:, 0] + xa[:, 1].T @ Qacc @ xa[:, 1]

        # constrain1 = eta(0).T @ xa[:, 0] == sp[0]
        # constrain2 = eta(dt).T @ xa[:, 0] - eta(0).T @ xa[:, 1] == 0
        # constrain3 = eta(dt).T @ xa[:, 1] == ep[0]

        cost = 0
        constr_p = []
        constr_v = []
        constr_zmp = []
        for i in range(self.nLines):
            for axis in range(3):
                # cost: acc
                cost = cost + a[i][:, axis].T @ Qacc @ a[i][:, axis]
                # constrain: smooth
                if i == 0:
                    constr_p.append(ca.DM(self.eta(0)).T @ a[i][:, axis] == sp[axis])
                    constr_v.append(ca.DM(self.d_eta(0)).T @ a[i][:, axis] == sv[axis])
                else:
                    constr_p.append(
                        ca.DM(self.eta(dt)).T @ a[i - 1][:, axis] - ca.DM(self.eta(0)).T @ a[i][:, axis] == 0)
                    constr_v.append(
                        ca.DM(self.d_eta(dt)).T @ a[i - 1][:, axis] - ca.DM(self.d_eta(0)).T @ a[i][:, axis] == 0)
                    if i == self.nLines - 1:
                        constr_p.append(ca.DM(self.eta(dt)).T @ a[i][:, axis] == ep[axis])
                        constr_v.append(ca.DM(self.d_eta(dt)).T @ a[i][:, axis] == ev[axis])
            # constrain: zmp
            for sp in ploy_seq[i]:
                g = 9.8
                p0 = [ca.DM(self.eta(0)).T @ a[i][:, axis] for axis in range(3)]
                p1 = [ca.DM(self.eta(dt)).T @ a[i][:, axis] for axis in range(3)]
                ddp0 = [ca.DM(self.dd_eta(0)).T @ a[i][:, axis] for axis in range(3)]
                ddp1 = [ca.DM(self.dd_eta(dt)).T @ a[i][:, axis] for axis in range(3)]
                constr_zmp.append(sp[0] * ddp0[0] * p0[2] + sp[1] * ddp0[1] * p0[2] + sp[2] * (ddp0[2] + g) >= 0)
                constr_zmp.append(sp[0] * ddp1[0] * p1[2] + sp[1] * ddp1[1] * p1[2] + sp[2] * (ddp1[2] + g) >= 0)

        self.opti.minimize(cost)
        # self.opti.subject_to(constrain1)
        # self.opti.subject_to(constrain2)
        # self.opti.subject_to(constrain3)
        self.opti.subject_to(constr_p)
        self.opti.subject_to(constr_v)
        self.opti.subject_to(constr_zmp)

        self.opti.solver('ipopt')

        # self.opti.solver('qpoases')
        p_opts = {"expand": True, "print_out": False, "print_time": False}
        s_opts = {"max_iter": 50, "print_level": 0}
        self.opti.solver('ipopt', p_opts, s_opts)

        sol = self.opti.solve()

        self.para_sequence = [sol.value(a[i]) for i in range(self.nLines)]

        # print(sol.value(a[0]))
        # print(ca.DM(self.eta(0)).T @ sol.value(a[0]))
        # print(ca.DM(self.eta(dt)).T @ sol.value(a[0]))
        # print(d_eta(0).T @ sol.value(xa))
        # print(d_eta(dt).T @ sol.value(xa))

    def get_pva(self, t):
        time = t
        position = []
        for i in range(self.nLines):
            if time < self.dtPlan:
                position = np.matrix(
                    [(np.matrix(self.eta(time)) * np.matrix(self.para_sequence[i][:, axis]).T)[0, 0]
                     for axis in range(3)]
                ).T
                velocity = np.matrix(
                    [(np.matrix(self.d_eta(time)) * np.matrix(self.para_sequence[i][:, axis]).T)[0, 0]
                     for axis in range(3)]
                ).T
                acceleration = np.matrix(
                    [(np.matrix(self.dd_eta(time)) * np.matrix(self.para_sequence[i][:, axis]).T)[0, 0]
                     for axis in range(3)]
                ).T
                break
            else:
                time = time - self.dtPlan
        return [position, velocity, acceleration]

    def update_line_positions(self):
        self.line_positions = []
        for i in range(self.nLines):
            for point in range(100):
                self.line_positions.append(
                    [(np.matrix(self.eta(point / 100. * self.dtPlan)) * np.matrix(self.para_sequence[i][:, axis]).T)[
                         0, 0]
                     for axis in range(3)]
                )

    def plot_lines(self):
        x = [e[0] for e in self.line_positions]
        y = [e[1] for e in self.line_positions]
        z = [e[2] for e in self.line_positions]
        fig = plt.figure(1)
        plt.plot(np.array(x))
        plt.xlabel('t Axes')
        fig = plt.figure(2)
        ax = plt.axes(projection="3d")
        ax.plot3D(np.array(x), np.array(y), np.array(z))
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        plt.show()


class Planner:
    def __init__(self, model, common_para):
        self.timer = .0
        self.dt = common_para.dt
        self.iterPerPlan = 10
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
        self.pb = np.matrix([.0, .0, 0.14, 0., 0., 0., 1.]).T
        self.vb = np.matrix([.0] * 6).T
        self.vb_user = np.matrix([.0] * 6).T
        self.foot_sequence = []
        self.polygon_sequence = []
        self.polygon_radius = 0.02
        self.model = model
        self.data = self.model.createData()

        self.stand_height = 0.14

        self.plan_time_start = 0.
        self.body_traj_para_sequence = []
        self.traj_opti = TrajectoryOpti()
        self.traj_opti.update_opti_struct(self.horizon)
        self.plan_ready = False

        self.foot_trajectory = [SimpleBezier(), SimpleBezier(), SimpleBezier(), SimpleBezier()]

        self.ut = Utility()

    def step(self, rece, est):
        if not self.phase_abnormal:
            self.timer = self.timer + self.dt
        self.update_step_phase()
        self.phase_abnormal_handle()
        self.update_body_target(rece)
        self.update_foot_target(est)
        iter = round(self.timer / self.dt)
        if (iter % self.iterPerPlan) == 0:
            self.plan_time_start = self.timer
            self.update_foot_sequence(est, self.horizon * self.dtPlan)
            self.update_polygon_sequence()
            self.traj_opti.start_position = est.pb_[0:3]
            self.traj_opti.end_position = est.pb_[0:3] + self.vb_user[0:3] * self.horizon * self.dtPlan
            self.traj_opti.step(self.polygon_sequence)
            self.traj_opti.update_line_positions()
            # self.traj_opti.plot_lines()
            self.body_traj_para_sequence = self.traj_opti.para_sequence
            self.plan_ready = True
        return self.pb

    def get_contact_target(self, time):
        contact_state = np.array([0, 0, 0, 0])
        if self.gait == 'trot':
            self.segments = np.array([10, 10, 10, 10])
            self.duration = np.array([5, 5, 5, 5])
            self.offset = np.array([0, 5, 5, 0])
            self.step_period = self.segments[0] * self.dtPlan
        # iter = time / self.dt
        iterPlan = time / self.dtPlan
        for leg in range(4):
            normIterPlan = (iterPlan + self.segments[leg] - self.offset[leg]) % self.segments[leg]
            contact_state[leg] = normIterPlan < self.duration[leg]
        return contact_state

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
        self.vb_user[0:2] = rece.body_vel[0:2]
        if self.plan_ready:
            pva = self.traj_opti.get_pva(self.timer - self.plan_time_start)
            self.pb[0:2] = pva[0][0:2]
            self.vb[0:2] = pva[1][0:2]
            # self.ab[0:3] = pva[2]

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
                Kp = np.diag([-0.03, -0.015])
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

    def update_foot_sequence(self, est, herizon_time):
        iterPlan = 0
        foot_pisition = est.pf_
        foot_position_sequence = [np.zeros([12, 1]) for i in range(self.horizon)]
        foot_contact_sequence = [np.zeros([4, 1]) for i in range(self.horizon)]
        while self.dtPlan * iterPlan < herizon_time:
            contact_target = self.get_contact_target(self.timer + self.dtPlan * iterPlan)
            foot_contact_sequence[iterPlan] = np.matrix(contact_target).T
            for leg in range(4):
                if contact_target[leg] == 0:
                    foot_pisition[leg * 3 + 0] = foot_pisition[leg * 3 + 0] + self.vb[0] * self.dtPlan
                    foot_pisition[leg * 3 + 1] = foot_pisition[leg * 3 + 1] + self.vb[1] * self.dtPlan
            foot_position_sequence[iterPlan] = foot_pisition.copy()
            iterPlan = iterPlan + 1
        self.foot_sequence = [foot_contact_sequence, foot_position_sequence]
        return [foot_contact_sequence, foot_position_sequence]

    def update_polygon_sequence(self):
        self.polygon_sequence = []
        foot_contact_sequence = self.foot_sequence[0]
        foot_position_sequence = self.foot_sequence[1]
        for i in range(self.horizon):
            # test
            # self.polygon_sequence.append([np.array([1., 1., 2.]), np.array([-1., -1., -5.])])

            support_point = []
            for leg in range(4):
                if foot_contact_sequence[i][leg] == 1:
                    support_point.append(foot_position_sequence[i][leg * 3 + 0:leg * 3 + 3])
            sp_sub = []
            if len(support_point) == 2:
                x1 = support_point[0][0]
                y1 = support_point[0][1]
                x2 = support_point[1][0]
                y2 = support_point[1][1]
                sp_sub.append(np.array([y2 - y1, x1 - x2, y1 * x2 - y2 * x1 - self.polygon_radius]))
                sp_sub.append(np.array([-(y2 - y1), -(x1 - x2), -(y1 * x2 - y2 * x1 + self.polygon_radius)]))
            else:
                print("support polygon error!!!!!")
            self.polygon_sequence.append(sp_sub)


if __name__ == '__main__':
    traj_opti = TrajectoryOpti()
    traj_opti.nLines = 2
    traj_opti.para_sequence = [np.zeros([6, 3])] * traj_opti.nLines  # 3axis nLines 6para
    traj_opti.position_sequence = [np.zeros([12, 0])] * traj_opti.nLines
    traj_opti.contact_sequence = [np.zeros([4, 0])] * traj_opti.nLines
    # traj_opti.support_polygon = [] * self.nLines
    traj_opti.position_sequence[0] = np.array([1, -1, 2, 1, 1, 2, -1, -1, 2, -1, 1, 2])
    traj_opti.position_sequence[1] = np.array([1, -1, 2, 1, 1, 2, -1, -1, 2, -1, 1, 2])
    traj_opti.contact_sequence[0] = np.array([1, 0, 0, 1])
    traj_opti.contact_sequence[1] = np.array([1, 0, 0, 1])
    traj_opti.update_polygon_sequence()
    traj_opti.step(traj_opti.nLines, traj_opti.polygon_sequence)
    traj_opti.update_position()
    traj_opti.plot_lines()
