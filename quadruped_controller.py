from __future__ import print_function

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

import numpy as np
import pinocchio

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
        kp = np.asarray([100]*6)
        ddp = np.asarray([0]*6)
        ddp[0:3] = np.multiply(kp[0:3],o[0]-p[0:3])

        return self.torque





if __name__ == '__main__':
    # id = "gym_env:Quadruped-v0"
    # env = gym.make(id)
    # env.reset()
    # env.step_torque(range(12))
    # env.render()
    # input("press any key to continue...")
    # env.close()

    # pinocchio
    model = pinocchio.buildSampleModelManipulator()
    model = pinocchio.buildSampleModelManipulator()
    data = model.createData()

    q = pinocchio.neutral(model)
    v = pinocchio.utils.zero(model.nv)
    a = pinocchio.utils.zero(model.nv)

    tau = pinocchio.rnea(model, data, q, v, a)
    print('tau = ', tau.T)


    root = pinocchio.JointModelFreeFlyer()
    model = pinocchio.buildModelFromUrdf(pybullet_data.getDataPath()+'/urdf/quadruped_robot/quadruped_robot.urdf', root)
    data = model.createData()
    qmax = np.matrix(np.full([model.nq, 1], np.pi))
    q = pinocchio.randomConfiguration(model, -qmax, qmax)
    v = np.matrix(np.random.rand(model.nv, 1))
    a = np.matrix(np.random.rand(model.nv, 1))
    tau = pinocchio.rnea(model, data, q, v, a)
    print('tau = ', tau.T)
