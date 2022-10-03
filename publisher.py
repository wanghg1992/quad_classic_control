from sensor_msgs import msg as smsg
from geometry_msgs import msg as gmsg
from geometry_msgs.msg import Transform
import tf2_ros
import rospy
from tf.transformations import quaternion_from_euler
from nav_msgs import msg as nmsg
from visualization_msgs import msg as vmsg

from utility import Utility


class RosPublish:
    def __init__(self, model):
        self.model = model

        # msg publish
        # publish path
        self.path_buffer_length = 80
        self.body_path = nmsg.Path()
        self.foot_path = [nmsg.Path(), nmsg.Path(), nmsg.Path(), nmsg.Path()]
        self.body_path_pub = rospy.Publisher('body_trajectory', nmsg.Path, queue_size=10)
        self.foot_path_pub = [rospy.Publisher('rf_foot_trajectory', nmsg.Path, queue_size=10),
                              rospy.Publisher('lf_foot_trajectory', nmsg.Path, queue_size=10),
                              rospy.Publisher('rh_foot_trajectory', nmsg.Path, queue_size=10),
                              rospy.Publisher('lh_foot_trajectory', nmsg.Path, queue_size=10)]

        # publish point
        self.body_point = gmsg.PointStamped()
        self.zmp_point = gmsg.PointStamped()
        self.foot_point = [gmsg.PointStamped(), gmsg.PointStamped(), gmsg.PointStamped(), gmsg.PointStamped()]
        self.body_point_pub = rospy.Publisher('body_point', gmsg.PointStamped, queue_size=10)
        self.zmp_point_pub = rospy.Publisher('zmp_point', gmsg.PointStamped, queue_size=10)
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
        self.foot_force = [gmsg.WrenchStamped(), gmsg.WrenchStamped(), gmsg.WrenchStamped(), gmsg.WrenchStamped()]
        self.foot_force_pub = [rospy.Publisher('rf_foot_force', gmsg.WrenchStamped, queue_size=10),
                               rospy.Publisher('lf_foot_force', gmsg.WrenchStamped, queue_size=10),
                               rospy.Publisher('rh_foot_force', gmsg.WrenchStamped, queue_size=10),
                               rospy.Publisher('lh_foot_force', gmsg.WrenchStamped, queue_size=10)]
        self.foot_name = ['RFFoot_link', 'LFFoot_link', 'RHFoot_link', 'LHFoot_link']

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

        # publish debug data
        self.debug_data = smsg.JointState()
        self.debug_data_pub = rospy.Publisher('debug_data', smsg.JointState, queue_size=10)
        self.debug_data_name = ['swing_phase_rf', 'contact_phase_rf', 'pf_init_z_rf', 'pf_z_rf', 'pf_z_rf_',
                                'vf_z_rf', 'vf_z_rf_', 'pf_y_rf', 'pf_y_rf_',
                                'vf_y_rf', 'vf_y_rf_']
        for i in range(len(self.debug_data_name)):
            self.debug_data.name.append(self.debug_data_name[i])
            self.debug_data.position.append(.0)
            self.debug_data.velocity.append(.0)
            self.debug_data.effort.append(.0)

        rospy.init_node('talker', anonymous=True)

        self.ut = Utility()

    def step(self, est, plan, control):
        self.pub_tf(est, plan)
        self.pub_path(est, plan)
        self.pub_point(est, plan)
        # self.pub_pose(est, plan)
        # self.pub_wrench(est, plan)
        self.pub_marker(est, plan)
        self.pub_state(est, plan, control)
        self.pub_data(est, plan, control)

    def pub_tf(self, est, plan):

        def pub_tf_body_link():
            pb_ = self.ut.m2l(est.pb_)
            tf_body_ = Transform(
                translation=gmsg.Vector3(pb_[0], pb_[1], pb_[2]),
                rotation=gmsg.Quaternion(pb_[3], pb_[4], pb_[5], pb_[6])
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

            q_ = self.ut.m2l(est.q_)

            pub_tf_single_link('body_link', 'RF1_link', [0.0915, -0.08, 0], 'x', q_[7 + 0])
            pub_tf_single_link('RF1_link', 'RF2_link', [0.0, 0.0, -0.046], 'y', q_[7 + 1])
            pub_tf_single_link('RF2_link', 'RF3_link', [0.0, 0.0, -0.066], 'y', q_[7 + 2])
            pub_tf_single_link('RF3_link', 'RFFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'LF1_link', [0.0915, 0.08, 0], 'x', q_[7 + 3])
            pub_tf_single_link('LF1_link', 'LF2_link', [0.0, 0.0, -0.046], 'y', q_[7 + 4])
            pub_tf_single_link('LF2_link', 'LF3_link', [0.0, 0.0, -0.066], 'y', q_[7 + 5])
            pub_tf_single_link('LF3_link', 'LFFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'RH1_link', [-0.0915, -0.08, 0], 'x', q_[7 + 6])
            pub_tf_single_link('RH1_link', 'RH2_link', [0.0, 0.0, -0.046], 'y', q_[7 + 7])
            pub_tf_single_link('RH2_link', 'RH3_link', [0.0, 0.0, -0.066], 'y', q_[7 + 8])
            pub_tf_single_link('RH3_link', 'RHFoot_link', [0.0, 0.0, -0.065], 'x', 0)

            pub_tf_single_link('body_link', 'LH1_link', [-0.0915, 0.08, 0], 'x', q_[7 + 9])
            pub_tf_single_link('LH1_link', 'LH2_link', [0.0, 0.0, -0.046], 'y', q_[7 + 10])
            pub_tf_single_link('LH2_link', 'LH3_link', [0.0, 0.0, -0.066], 'y', q_[7 + 11])
            pub_tf_single_link('LH3_link', 'LHFoot_link', [0.0, 0.0, -0.065], 'x', 0)

        pub_tf_body_link()
        pub_tf_leg_link()

    def pub_path(self, est, plan):

        pf = self.ut.m2l(plan.pf)

        def pub_body_path(plan):
            self.body_path.header.frame_id = "world"
            self.body_path.header.stamp = rospy.Time.now()

            # for p in plan.
            self.body_path.poses.clear()
            pb_sequence = plan.traj_opti.line_positions
            for pb in pb_sequence:
                pose = gmsg.PoseStamped()
                pose.header.frame_id = 'world'
                pose.header.stamp = rospy.Time.now()
                pose.pose.position.x = pb[0]
                pose.pose.position.y = pb[1]
                pose.pose.position.z = pb[2]
                quaternion = quaternion_from_euler(
                    0, 0, 0)
                pose.pose.orientation.x = quaternion[0]
                pose.pose.orientation.y = quaternion[1]
                pose.pose.orientation.z = quaternion[2]
                pose.pose.orientation.w = quaternion[3]
                self.body_path.poses.append(pose)

            self.body_path_pub.publish(self.body_path)

        def pub_foot_path(leg_id):
            self.foot_path[leg_id].header.frame_id = "world"
            self.foot_path[leg_id].header.stamp = rospy.Time.now()

            pose = gmsg.PoseStamped()
            pose.header.frame_id = 'world'
            pose.header.stamp = rospy.Time.now()
            # pose.pose.position.x = est.pf_[leg_id*3 + 0]
            # pose.pose.position.y = est.pf_[leg_id*3 + 1]
            # pose.pose.position.z = est.pf_[leg_id*3 + 2]
            pose.pose.position.x = pf[leg_id * 3 + 0]
            pose.pose.position.y = pf[leg_id * 3 + 1]
            pose.pose.position.z = pf[leg_id * 3 + 2]
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
                for i in range(0, self.path_buffer_length - 1):
                    self.foot_path[leg_id].poses[i] = self.foot_path[leg_id].poses[i + 1]
                self.foot_path[leg_id].poses[self.path_buffer_length - 1] = pose

            self.foot_path_pub[leg_id].publish(self.foot_path[leg_id])

        pub_body_path(plan)
        pub_foot_path(0)
        pub_foot_path(1)
        pub_foot_path(2)
        pub_foot_path(3)

    def pub_point(self, est, plan):

        def pub_body_point():
            self.body_point.header.frame_id = 'world'
            self.body_point.header.stamp = rospy.Time.now()
            self.body_point.point.x = est.pb_[0]
            self.body_point.point.y = est.pb_[1]
            self.body_point.point.z = est.pb_[2]
            self.body_point_pub.publish(self.body_point)

        def pub_zmp_point():
            self.zmp_point.header.frame_id = 'world'
            self.zmp_point.header.stamp = rospy.Time.now()
            self.zmp_point.point.x = plan.zmp[0]
            self.zmp_point.point.y = plan.zmp[1]
            self.zmp_point.point.z = 0
            self.zmp_point_pub.publish(self.zmp_point)

        def pub_foot_point(leg_id):
            self.foot_point[leg_id].header.frame_id = 'world'
            self.foot_point[leg_id].header.stamp = rospy.Time.now()
            # self.foot_point[leg_id].point.x = est.pf_[leg_id*3 + 0]
            # self.foot_point[leg_id].point.y = est.pf_[leg_id*3 + 1]
            # self.foot_point[leg_id].point.z = est.pf_[leg_id*3 + 2]
            self.foot_point[leg_id].point.x = pf_init[leg_id * 3 + 0]
            self.foot_point[leg_id].point.y = pf_init[leg_id * 3 + 1]
            self.foot_point[leg_id].point.z = pf_init[leg_id * 3 + 2]
            # self.foot_point[leg_id].point.x = plan.pf_hold[leg_id*3 + 0]
            # self.foot_point[leg_id].point.y = plan.pf_hold[leg_id*3 + 1]
            # self.foot_point[leg_id].point.z = plan.pf_hold[leg_id*3 + 2]
            self.foot_point_pub[leg_id].publish(self.foot_point[leg_id])

        pf_init = self.ut.m2l(plan.pf_init)
        pub_body_point()
        pub_zmp_point()
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

    def pub_wrench(self, est, plan):
        def pub_foot_force(leg_id):
            self.foot_force[leg_id].header.frame_id = self.foot_name[leg_id]
            self.foot_force[leg_id].header.stamp = rospy.Time.now()
            self.foot_force[leg_id].wrench.force.x = est.foot_force_[leg_id * 3 + 0] / 10.0
            self.foot_force[leg_id].wrench.force.y = est.foot_force_[leg_id * 3 + 1] / 10.0
            self.foot_force[leg_id].wrench.force.z = est.foot_force_[leg_id * 3 + 2] / 10.0
            self.foot_force[leg_id].wrench.torque.x = 0
            self.foot_force[leg_id].wrench.torque.y = 0
            self.foot_force[leg_id].wrench.torque.z = 0
            self.foot_force_pub[leg_id].publish(self.foot_force[leg_id])

        pub_foot_force(0)
        pub_foot_force(1)
        pub_foot_force(2)
        pub_foot_force(3)

    def pub_marker(self, est, plan):

        pf_ = self.ut.m2l(est.pf_)

        def pub_foot_marker(leg_id):
            self.foot_marker[leg_id].header.frame_id = 'world'
            self.foot_marker[leg_id].header.stamp = rospy.Time.now()
            self.foot_marker[leg_id].ns = 'my_namespace'
            self.foot_marker[leg_id].id = leg_id
            self.foot_marker[leg_id].type = vmsg.Marker.ARROW
            self.foot_marker[leg_id].action = vmsg.Marker.MODIFY
            self.foot_marker[leg_id].pose.position.x = pf_[leg_id * 3 + 0]
            self.foot_marker[leg_id].pose.position.y = pf_[leg_id * 3 + 1]
            self.foot_marker[leg_id].pose.position.z = pf_[leg_id * 3 + 2]
            self.foot_marker[leg_id].pose.orientation.x = 0
            self.foot_marker[leg_id].pose.orientation.y = 0
            self.foot_marker[leg_id].pose.orientation.z = 0
            self.foot_marker[leg_id].pose.orientation.w = 1
            self.foot_marker[leg_id].scale.x = plan.contact_phase[leg_id] * 0.08
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

        tor = self.ut.m2l(control.tor)

        def pub_joint_state():
            self.joint_state.header.frame_id = 'world'
            self.joint_state.header.stamp = rospy.Time.now()
            for joint in range(12):
                # self.joint_state.effort[joint] = est.tor_[joint]
                self.joint_state.effort[joint] = tor[6 + joint]
            self.joint_state_pub.publish(self.joint_state)

        pub_joint_state()

    def pub_data(self, est, plan, control):

        def pub_debug_data():
            self.debug_data.header.frame_id = 'world'
            self.debug_data.header.stamp = rospy.Time.now()
            self.debug_data.position[0] = plan.swing_phase[0]
            self.debug_data.position[1] = plan.contact_phase[0]
            self.debug_data.position[2] = plan.pf_init[2]
            self.debug_data.position[3] = plan.pf[2]
            self.debug_data.position[4] = est.pf_[2]
            self.debug_data.position[5] = plan.vf[2]
            self.debug_data.position[6] = est.vf_[2]
            self.debug_data.position[7] = plan.pf[1]
            self.debug_data.position[8] = est.pf_[1]
            self.debug_data.position[9] = plan.vf[1]
            self.debug_data.position[10] = est.vf_[1]
            self.debug_data_pub.publish(self.debug_data)

        pub_debug_data()
