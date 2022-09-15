from __future__ import print_function

import numpy as np

import pinocchio as pin
import casadi as ca

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data

from utility import Utility
from estimator import Estimator
from planner import Planner
from controller import Controller
from publisher import RosPublish


class CommonParameter:
    def __init__(self):
        self.dt = 0.001


if __name__ == '__main__':
    id = "gym_env:Quadruped-v0"
    env = gym.make(id)

    ## quadruped model
    root = pin.JointModelFreeFlyer()
    model = pin.buildModelFromUrdf(pybullet_data.getDataPath() + '/urdf/quadruped_robot/quadruped_robot.urdf',
                                   root)
    # model = pin.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf')

    torque = [.0] * 12

    cp = CommonParameter()
    cp.dt = 1.0 / 240

    est = Estimator(model, cp)
    plan = Planner(model, cp)
    control = Controller(model)
    rospub = RosPublish(model)

    for i in range(100000):
        # simulation
        # leg order: LF-RF-LH-RH
        torque = list(control.tor[6 + 3:6 + 6, 0]) + list(control.tor[6 + 0:6 + 3, 0]) + list(
            control.tor[6 + 9:6 + 12, 0]) \
                 + list(control.tor[6 + 6:6 + 9, 0])
        [o_, pb_, vb_, js_] = env.step_torque(torque)
        # js_ = list(js_[0:3]) + list(js_[6:9]) + list(js_[3:6]) + list(js_[9:12])
        js_ = list(js_[3:6]) + list(js_[0:3]) + list(js_[9:12]) + list(js_[6:9])

        pj_ = [i[0] for i in js_]
        vj_ = [i[1] for i in js_]
        tj_ = [i[3] for i in js_]
        q_ = np.matrix(list(pb_[0] + pb_[1]) + pj_).T
        dq_ = np.matrix(list(vb_[0] + vb_[1]) + vj_).T
        tor_ = np.matrix([.0] * 6 + tj_).T

        est.q_ = q_
        est.dq_ = dq_
        est.tor_ = tor_
        est.pb_[0:3] = np.matrix(pb_[0]).T
        est.pb_[3:7] = np.matrix(pb_[1]).T
        est.vb_[0:3] = np.matrix(vb_[0]).T
        est.vb_[3:6] = np.matrix(vb_[1]).T
        est.step()

        plan.step(est)

        torque = control.step(est, plan)

        rospub.step(est, plan, control)

        # env.render()
    input("press any key to continue...")
    env.close()
