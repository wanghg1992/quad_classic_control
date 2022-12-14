import numpy as np

from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import casadi as ca

from utility import Utility


class Task:
    def __init__(self, A, b, D, f, wb, wf):
        self.A = A
        self.b = b
        self.D = D
        self.f = f
        self.wb = wb
        self.wf = wf

    def __add__(self, other):
        A = np.concatenate((self.A, other.A), axis=0)
        b = np.concatenate((self.b, other.b), axis=0)
        D = np.concatenate((self.D, other.D), axis=0)
        f = np.concatenate((self.f, other.f), axis=0)
        wb = np.concatenate((self.wb, other.wb), axis=0)
        wf = np.concatenate((self.wf, other.wf), axis=0)
        return Task(A, b, D, f, wb, wf)


class NspWbc:
    def __init__(self):
        self.task_number = 0

        self.x_i_pre = None
        self.N_i_pre = None
        self.C_i_pre = None
        self.wb_i_pre = None
        self.D_i_pre = None
        self.f_i_pre = None
        self.wf_i_pre = None

        self.solution = None

    def weighted_pseudo_inverse(self, matrix, weight=None):
        if weight is None:
            weight = np.matrix(np.eye(matrix.shape[1]))
        return np.linalg.inv(weight) * matrix.T * np.linalg.pinv(matrix * np.linalg.inv(weight) * matrix.T)

    def add_task(self, task):
        if self.task_number == 0:
            x_im1_pre = np.matrix(np.zeros([task.A.shape[1], 1]))
            N_im1_pre = np.matrix(np.eye(task.A.shape[1]))
            C_im1_pre = None
            D_im1_pre = np.matrix([[]] * task.A.shape[1]).T
            f_im1_pre = np.matrix([]).T
            wb_im1_pre = np.matrix([]).T
            wf_im1_pre = np.matrix([]).T
        else:
            x_im1_pre = np.matrix(self.x_i_pre)
            N_im1_pre = np.matrix(self.N_i_pre)
            C_im1_pre = np.matrix(self.C_i_pre)
            wb_im1_pre = np.matrix(self.wb_i_pre)
            D_im1_pre = np.matrix(self.D_i_pre)
            f_im1_pre = np.matrix(self.f_i_pre)
            wf_im1_pre = np.matrix(self.wf_i_pre)

        qp_decision_num = task.A.shape[1]

        # remove zero rows
        Abw = np.concatenate((task.A, task.b, task.wb), axis=1)
        Abw = Abw[[np.any(task.A[i] < -1e-6) or np.any(task.A[i] > 1e-6) for i in range(task.A.shape[0])], :]
        task.A = Abw[:, 0:qp_decision_num]
        task.b = Abw[:, qp_decision_num:qp_decision_num + 1]
        task.wb = Abw[:, qp_decision_num + 1:qp_decision_num + 2]
        Dfw = np.concatenate((task.D, task.f, task.wf), axis=1)
        Dfw = Dfw[[np.any(task.D[i] < -1e-6) or np.any(task.D[i] > 1e-6) for i in range(task.D.shape[0])], :]
        task.D = Dfw[:, 0:qp_decision_num]
        task.f = Dfw[:, qp_decision_num:qp_decision_num + 1]
        task.wf = Dfw[:, qp_decision_num + 1:qp_decision_num + 2]

        A_i_pre = task.A * N_im1_pre
        A_i_pre_dpinv = self.weighted_pseudo_inverse(A_i_pre)  # to-do: add dynamic consistent
        self.N_i_im1 = np.matrix(np.eye(qp_decision_num)) - A_i_pre_dpinv * A_i_pre
        self.N_i_pre = N_im1_pre * self.N_i_im1

        self.x_i_pre = x_im1_pre + A_i_pre_dpinv * (task.b - task.A * x_im1_pre)
        if C_im1_pre is None:
            self.C_i_pre = A_i_pre_dpinv
        else:
            self.C_i_pre = np.concatenate((self.N_i_im1 * C_im1_pre, A_i_pre_dpinv), axis=1)

        self.wb_i_pre = np.concatenate((wb_im1_pre, task.wb), axis=0)

        self.D_i_pre = np.concatenate((D_im1_pre, task.D), axis=0)
        self.f_i_pre = np.concatenate((f_im1_pre, task.f), axis=0)
        self.wf_i_pre = np.concatenate((wf_im1_pre, task.wf), axis=0)

        self.task_number = self.task_number + 1

    def clear_tasks(self):
        self.task_number = 0

    def get_solution(self):
        w = np.concatenate((self.wb_i_pre, self.wf_i_pre), axis=0)

        qp_H = ca.DM(np.diag(w.A[:, 0]))
        qp_g = ca.DM(np.zeros((w.shape[0], 1)))
        qp_A = ca.DM(np.zeros((self.D_i_pre.shape[0], w.shape[0])))
        qp_A[:, 0:self.wb_i_pre.shape[0]] = self.D_i_pre * self.C_i_pre
        qp_A[:, self.wb_i_pre.shape[0]:] = -np.eye(self.wf_i_pre.shape[0])

        qp_uba = self.f_i_pre - self.D_i_pre * self.x_i_pre

        qp = {'h': qp_H.sparsity(), 'a': qp_A.sparsity()}
        # opts = {'error_on_fail': False, 'max_schur': 100, 'printLevel': 'none'}
        opts = {'error_on_fail': True}
        S = ca.conic('S', 'osqp', qp, opts)
        r = S(h=qp_H, g=qp_g, a=qp_A, uba=qp_uba)
        x_opt = r['x']
        self.solution = self.x_i_pre + self.C_i_pre * np.matrix(x_opt)[0:self.wb_i_pre.size, 0]
        return self.solution
        # print('x_opt:', x_opt)


class HoWbc:
    def __init__(self):
        self.task_number = 0

        self.x_i_pre = None
        self.N_i_pre = None
        self.D_i_pre = None
        self.f_i_pre = None
        self.v_i_pre = None

        self.solution = None

    def weighted_pseudo_inverse(self, matrix, weight=None):
        if weight is None:
            weight = np.matrix(np.eye(matrix.shape[1]))
        return np.linalg.inv(weight) * matrix.T * np.linalg.pinv(matrix * np.linalg.inv(weight) * matrix.T)

    def add_task(self, task):
        if self.task_number == 0:
            x_im1_pre = np.matrix(np.zeros([task.A.shape[1], 1]))
            N_im1_pre = np.matrix(np.eye(task.A.shape[1]))
            D_im1_pre = np.matrix([[]] * task.A.shape[1]).T
            f_im1_pre = np.matrix([]).T
            v_im1_pre = np.matrix([]).T
        else:
            x_im1_pre = np.matrix(self.x_i_pre)
            N_im1_pre = np.matrix(self.N_i_pre)
            D_im1_pre = np.matrix(self.D_i_pre)
            f_im1_pre = np.matrix(self.f_i_pre)
            v_im1_pre = np.matrix(self.v_i_pre)

        qp_decision_num = task.A.shape[1]

        # remove zero row
        Abw = np.concatenate((task.A, task.b, task.wb), axis=1)
        Abw = Abw[[np.any(task.A[i] < -1e-6) or np.any(task.A[i] > 1e-6) for i in range(task.A.shape[0])], :]
        task.A = Abw[:, 0:qp_decision_num]
        task.b = Abw[:, qp_decision_num:qp_decision_num + 1]
        task.wb = Abw[:, qp_decision_num + 1:qp_decision_num + 2]
        Dfw = np.concatenate((task.D, task.f, task.wf), axis=1)
        Dfw = Dfw[[np.any(task.D[i] < -1e-6) or np.any(task.D[i] > 1e-6) for i in range(task.D.shape[0])], :]
        task.D = Dfw[:, 0:qp_decision_num]
        task.f = Dfw[:, qp_decision_num:qp_decision_num + 1]
        task.wf = Dfw[:, qp_decision_num + 1:qp_decision_num + 2]

        qp_slack_num = task.D.shape[0]
        qp_variable_num = qp_decision_num + qp_slack_num

        AN = task.A * N_im1_pre
        AN = AN[[np.any(AN[i] < -1e-6) or np.any(AN[i] > 1e-6) for i in range(AN.shape[0])], :]

        qp_H = ca.DM(np.eye(qp_variable_num))
        qp_H[0:qp_decision_num, 0:qp_decision_num] = AN.T * AN
        qp_g = ca.DM(np.zeros((qp_variable_num, 1)))
        qp_g[0:qp_decision_num, 0] = AN.T * (task.A * x_im1_pre - task.b)
        qp_A = ca.DM(np.zeros((D_im1_pre.shape[0] + qp_slack_num, qp_variable_num)))
        qp_A[:, 0:qp_decision_num] = np.concatenate((D_im1_pre, task.D), axis=0) * N_im1_pre
        qp_A[D_im1_pre.shape[0]:, qp_decision_num:] = -np.eye(qp_slack_num)
        qp_uba = ca.DM(np.zeros((D_im1_pre.shape[0] + qp_slack_num, 1)))
        qp_uba[0:D_im1_pre.shape[0], 0] = f_im1_pre - D_im1_pre * x_im1_pre + v_im1_pre
        qp_uba[D_im1_pre.shape[0]:, 0] = task.f - task.D * x_im1_pre

        qp = {'h': qp_H.sparsity(), 'a': qp_A.sparsity()}
        opts = {'error_on_fail': True }
        S = ca.conic('S', 'osqp', qp, opts)
        r = S(h=qp_H, g=qp_g, a=qp_A, uba=qp_uba)
        x_opt = r['x']
        self.solution = x_im1_pre + N_im1_pre * np.matrix(x_opt)[0:qp_decision_num]
        # print('x_opt:', x_opt)

        self.x_i_pre = self.solution
        A_i_pre = AN
        A_i_pre_dpinv = self.weighted_pseudo_inverse(A_i_pre)  # to-do: add dynamic consistent
        self.N_i_im1 = np.matrix(np.eye(qp_decision_num)) - A_i_pre_dpinv * A_i_pre
        self.N_i_pre = N_im1_pre * self.N_i_im1
        # self.N_i_pre = N_im1_pre * (np.eye(qp_decision_num) - np.linalg.pinv(task.A * N_im1_pre) * task.A * N_im1_pre)

        self.D_i_pre = np.concatenate((D_im1_pre, task.D), axis=0)
        self.f_i_pre = np.concatenate((f_im1_pre, task.f), axis=0)
        self.v_i_pre = np.concatenate((v_im1_pre, x_opt[qp_decision_num:, 0]), axis=0)

        self.task_number = self.task_number + 1

    def clear_tasks(self):
        self.task_number = 0

    def get_solution(self):
        return self.solution


class Wbc:
    def __init__(self, est):
        self.est = est
        self.ut = Utility()
        self.decision_variable_num = 18 + 12 + 12
        self.tasks = list([])
        self.nsp_wbc = NspWbc()
        self.ho_wbc = HoWbc()
        self.solution = None

    def formulate_foot_acc_task(self, foot_acc_des):
        A = np.matrix(np.zeros([12, self.decision_variable_num]))
        A[0:12, 0:18] = self.est.Jfoot
        b = foot_acc_des - self.est.Jdfoot * self.ut.to_pin(self.est.dq_, 6)
        wb = np.matrix([1000] * 12).T
        D = np.matrix([[]] * self.decision_variable_num).T
        f = np.matrix([]).T
        wf = np.matrix([]).T

        self.foot_acc_task = Task(A, b, D, f, wb, wf)
        return self.foot_acc_task

    def formulate_body_acc_task(self, body_acc_des):
        A = np.matrix(np.zeros([6, self.decision_variable_num]))
        A[0:6, 0:18] = self.est.JB
        b = body_acc_des - self.est.JdB * self.ut.to_pin(self.est.dq_, 6)
        wb = np.matrix([1] * 6).T
        D = np.matrix([[]] * self.decision_variable_num).T
        f = np.matrix([]).T
        wf = np.matrix([]).T
        self.body_acc_task = Task(A, b, D, f, wb, wf)
        return self.body_acc_task

    def formulate_eom_task(self):
        S = np.matrix(np.zeros((18, 12)))
        S[6:18, :] = np.eye(12)
        A = np.matrix(np.zeros([18, self.decision_variable_num]))
        A[0:18, 0:18] = self.est.M
        A[0:18, 18:30] = -S
        A[0:18, 30:42] = -self.est.Jfoot.T
        b = -self.est.nle
        wb = np.matrix([100000] * 18).T

        acc_max = 10000.
        torque_max = 20.
        force_max = 100.
        D = np.matrix(np.concatenate((np.eye(self.decision_variable_num), -np.eye(self.decision_variable_num)), axis=0))
        f = np.matrix(np.ones((self.decision_variable_num * 2, 1)))
        f[0:18, 0] = np.matrix(np.ones((18, 1)) * acc_max)
        f[18:30, 0] = np.matrix(np.ones((12, 1)) * torque_max)
        f[30:42, 0] = np.matrix(np.ones((12, 1)) * force_max)
        f[42:60, 0] = np.matrix(np.ones((18, 1)) * acc_max)
        f[60:72, 0] = np.matrix(np.ones((12, 1)) * torque_max)
        f[72:84, 0] = np.matrix(np.ones((12, 1)) * force_max)
        wf = np.matrix([1] * self.decision_variable_num * 2).T
        self.eom_task = Task(A, b, D, f, wb, wf)
        return self.eom_task

    def formulate_foot_force_task(self, contact_state):
        # swing force is zero
        Sw = np.matrix(np.zeros((12, 12)))
        for foot in range(4):
            if contact_state[foot] == 0:
                Sw[foot * 3 + 0:foot * 3 + 3, foot * 3 + 0:foot * 3 + 3] = np.eye(3)
        A = np.matrix(np.zeros([12, self.decision_variable_num]))
        A[0:12, 30:42] = Sw * np.eye(12)
        b = Sw * np.matrix(np.zeros([12, 1]))
        wb = np.matrix([1] * 12).T

        # no slip for contact foot
        mu = 0.3
        D = np.matrix(np.zeros([16, self.decision_variable_num]))
        for foot in range(4):
            if contact_state[foot] > 0:
                D[foot * 4:(foot + 1) * 4, 30 + foot * 3:30 + (foot + 1) * 3] = np.matrix(
                    [[1, 0, -mu], [-1, 0, -mu], [0, 1, -mu], [0, -1, -mu]])
        f = np.matrix(np.zeros((16, 1)))
        wf = np.matrix([1] * 16).T
        self.foot_force_task = Task(A, b, D, f, wb, wf)
        return self.foot_force_task

    def step(self, control_object, wbc_type):
        self.tasks = list([])
        self.tasks.append(self.formulate_eom_task())
        self.tasks.append(self.formulate_foot_acc_task(control_object.foot_acc_des))
        self.tasks.append(self.formulate_foot_force_task(control_object.contact_state))
        self.tasks.append(self.formulate_body_acc_task(control_object.body_acc_des))

        if wbc_type is 'NspWbc':
            self.nsp_wbc.clear_tasks()
            for task in self.tasks:
                self.nsp_wbc.add_task(task)
            self.solution = self.nsp_wbc.get_solution()
        elif wbc_type is 'HoWbc':
            self.ho_wbc.clear_tasks()
            for task in self.tasks:
                self.ho_wbc.add_task(task)
            self.solution = self.ho_wbc.get_solution()


class ControlObject:
    def __init__(self):
        self.body_acc_des = None
        self.foot_acc_des = None
        self.contact_force_des = None
        self.contact_state = None


class Controller:
    def __init__(self, model, est):
        self.tor = np.matrix([.0] * 18).T
        self.body_pos_des = np.matrix([.0, .0, 0.16])
        self.body_rot_des = R.from_matrix([[1., .0, .0], [.0, 1., .0], [0., 0., 1.]])
        self.body_acc_des = np.matrix([0.] * 6).T
        self.body_pos_fdb = np.matrix([.0] * 3).T
        self.body_rot_fdb = R.from_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.model = model
        self.data = self.model.createData()
        self.ut = Utility()
        self.control_object = ControlObject()
        self.wbc = Wbc(est)

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
        self.body_pos_des = plan.pb[0:3].copy()
        self.body_rot_des = R.from_quat(self.ut.m2l(plan.pb[3:7]))
        self.body_pos_fdb = est.pb_[0:3].copy()
        self.body_rot_fdb = R.from_quat(self.ut.m2l(est.pb_[3:7]))
        self.body_vel_des = plan.vb.copy()
        self.body_vel_fdb = est.vb_.copy()

        body_kp = np.array([10., 10., 50., 50., 50., 50.])
        body_kd = np.array([50., 50., 100., 20., 20., 20.])
        self.body_acc_des[0:3] = np.diag(body_kp[0:3]) * (self.body_pos_des - self.body_pos_fdb) \
                                 + np.diag(body_kd[0:3]) * (self.body_vel_des[0:3] - self.body_vel_fdb[0:3])
        self.body_acc_des[3:6] = np.diag(body_kp[3:6]) * np.matrix(
            (self.body_rot_des * self.body_rot_fdb.inv()).as_rotvec()).T \
                                 + np.diag(body_kd[3:6]) * (self.body_vel_des[3:6] - self.body_vel_fdb[3:6])

        foot_kp = np.array([.0] * 12)
        foot_kd = np.array([.0] * 12)
        for leg in range(4):
            if plan.swing_phase[leg] > 0.0001:
                foot_kp[leg * 3 + 0: leg * 3 + 3] = np.array([200., 200., 200.])
                foot_kd[leg * 3 + 0: leg * 3 + 3] = np.array([200., 200., 1000.])
            else:
                foot_kp[leg * 3 + 0: leg * 3 + 3] = np.array([0., 0., 0.])
                foot_kd[leg * 3 + 0: leg * 3 + 3] = np.array([100., 100., 100.])
        self.foot_acc_des = np.diag(foot_kp) * (plan.pf - est.pf_) + np.diag(foot_kd) * (plan.vf - est.vf_) + \
                            plan.af

        self.control_object.body_acc_des = self.body_acc_des
        self.control_object.foot_acc_des = self.foot_acc_des
        self.control_object.contact_state = plan.contact_phase

        self.wbc.step(self.control_object, 'NspWbc')
        # self.wbc.step(self.control_object, 'HoWbc')
        self.tor = np.matrix([[.0]] * 18)
        self.tor[6:18] = self.ut.from_pin(self.wbc.solution[18:30], 0)

        return self.tor

    def step(self, est, plan):
        return self.optimal_control(est, plan)
