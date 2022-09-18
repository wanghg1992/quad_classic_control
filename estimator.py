import numpy as np

import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import tf.transformations as transformations

from utility import Utility


class Estimator:
    def __init__(self, model, common_para):
        self.dt = common_para.dt

        self.q_ = np.matrix([0.] * 19).T
        self.dq_ = np.matrix([0.] * 18).T
        self.tor_ = np.matrix([0.] * 18).T
        self.pb_ = np.matrix([0, 0, 0.16, 0., 0., 0., 1.]).T
        self.pf_ = np.matrix([0.] * 12).T
        self.vb_ = np.matrix([0.] * 6).T
        self.vf_ = np.matrix([0.] * 12).T
        self.last_vf_ = np.matrix([0.] * 12).T
        self.ab_ = np.matrix([0.] * 6).T
        self.af_ = np.matrix([0.] * 12).T
        self.vb_body_ = np.matrix([0.] * 6).T
        self.model = model
        self.data = self.model.createData()

        self.JB = np.matrix(np.zeros([6, 18]))
        self.Jrf = np.matrix(np.zeros([6, 18]))
        self.Jlf = np.matrix(np.zeros([6, 18]))
        self.Jrh = np.matrix(np.zeros([6, 18]))
        self.Jlh = np.matrix(np.zeros([6, 18]))
        self.Jfoot = np.matrix(np.zeros([12, 18]))

        self.JdB = np.matrix(np.zeros([6, 18]))
        self.Jdrf = np.matrix(np.zeros([6, 18]))
        self.Jdlf = np.matrix(np.zeros([6, 18]))
        self.Jdrh = np.matrix(np.zeros([6, 18]))
        self.Jdlh = np.matrix(np.zeros([6, 18]))
        self.Jdfoot = np.matrix(np.zeros([12, 18]))

        self.foot_force_ = np.array([0.] * 12)
        self.contact_state_ = np.array([0.] * 4)

        self.ut = Utility()

    def step(self):
        model = self.model
        data = self.data

        pin_q_ = self.ut.to_pin(self.q_, 7)
        pin_dq_ = self.ut.to_pin(self.dq_, 6)

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
        self.pf_ = np.matrix(self.pf_).T
        vf_rf_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('RF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_lf_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('LF4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_rh_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('RH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        vf_lh_ = np.matrix(
            pin.getFrameVelocity(model, data, model.getFrameId('LH4_joint'), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).T
        self.last_vf_ = self.vf_
        self.vf_ = np.append(vf_rf_[0:3, :], vf_lf_[0:3, :], axis=0)
        self.vf_ = np.append(self.vf_, vf_rh_[0:3, :], axis=0)
        self.vf_ = np.append(self.vf_, vf_lh_[0:3, :], axis=0)
        self.af_ = (self.vf_ - self.last_vf_) / self.dt

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
        yaw = transformations.euler_from_quaternion(self.ut.m2l(self.pb_[3:7]))[2]
        r = R.from_euler('xyz', [0., 0., yaw])
        r = np.matrix(r.as_matrix())

        linearVelocity = np.array(self.vb_[0:3]).reshape(3, 1)
        angularVelocity = np.array(self.vb_[3:6]).reshape(3, 1)

        self.vb_body_ = [(r * linearVelocity).reshape(1, 3).tolist()[0],
                         (r * angularVelocity).reshape(1, 3).tolist()[0]]

        return self.vb_body_
