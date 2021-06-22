#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
import numpy as np 
from umeyama import align_umeyama
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray

class Server:
    def __init__(self, scale_publisher, rot_publisher, trans_publisher): 
        print("init")
        self.true_pos = [] 
        self.est_pos_orb = [] 
        self.scale_publisher = scale_publisher
        self.rot_publisher = rot_publisher
        self.trans_publisher = trans_publisher
    
    def true_pos_callback(self, msg):
        if not self.true_pos:
            self.true_pos.append(msg)
        else:
            if msg:
                if msg.header.stamp.to_sec() - self.true_pos[len(self.true_pos)-1].header.stamp.to_sec() > 0.03:
                    self.true_pos.append(msg)
                    if len(self.true_pos) > 50: 
                        del self.true_pos[0]

    def est_pos_orb_callback(self, msg):
        if msg:
            self.est_pos_orb.append(msg)
            if len(self.est_pos_orb) > 50: 
                del self.est_pos_orb[0]

                self.update_trans_variables()

    def update_trans_variables(self):
        # return, if not enough points are available
        # since no scale can be computed
        if (len(self.est_pos_orb) < 50) or (len(self.true_pos) < 50):
            print("not enough data available, waiting...")
            return

        # return the scale if it already has been calculated    
        else:
            # get minimum and maximum time for each queue to figure out, 
            # how many points can be considered for alignment. This is only done once!
            min_orb = np.min([pose_oi.header.stamp.to_sec() for pose_oi in self.est_pos_orb])
            max_orb = np.max([pose_oi.header.stamp.to_sec() for pose_oi in self.est_pos_orb])

            min_true = np.min([point_oi.header.stamp.to_sec() for point_oi in self.true_pos])
            max_true = np.max([point_oi.header.stamp.to_sec() for point_oi in self.true_pos])

            print("orb")
            print(min_orb)
            print(max_orb)
            print(min_true)
            print(max_true)

            thresh_min = np.max([min_orb, min_true])
            thresh_max = np.min([max_true, max_orb])

            # cut off the queues
            orb_oi = [pose_oi for pose_oi in self.est_pos_orb if pose_oi.header.stamp.to_sec() > thresh_min]
            true_oi = [pose_oi for pose_oi in self.true_pos if pose_oi.header.stamp.to_sec() > thresh_min]


            # for the shorter remaining queue, get the matching point
            if len(orb_oi) <= len(true_oi): 
                orb_oi_final = orb_oi
                true_oi_final = []
                for pose_oi in orb_oi: 
                    diffs_oi = [np.abs(pose_oi.header.stamp.to_sec() - point_oi.header.stamp.to_sec()) for point_oi in true_oi]
                    true_oi_final.append(true_oi[diffs_oi.index(min(diffs_oi))])

            else:
                true_oi_final = true_oi
                orb_oi_final = []
                for pose_oi in true_oi: 
                    diffs_oi = [np.abs(pose_oi.header.stamp.to_sec() - point_oi.header.stamp.to_sec()) for point_oi in orb_oi]
                    orb_oi_final.append(orb_oi[diffs_oi.index(min(diffs_oi))])

            # now do the alignment and compute the scale
            x_orb = [pose_oi.pose.position.x for pose_oi in orb_oi_final]
            y_orb = [pose_oi.pose.position.y for pose_oi in orb_oi_final]
            z_orb = [pose_oi.pose.position.z for pose_oi in orb_oi_final]

            x_true = [pose_oi.pose.position.x for pose_oi in true_oi_final]
            y_true = [pose_oi.pose.position.y for pose_oi in true_oi_final]
            z_true = [pose_oi.pose.position.z for pose_oi in true_oi_final]
            
            orb_points = np.column_stack((x_orb, y_orb, z_orb))
            true_points = np.column_stack((x_true, y_true, z_true))
            s, R, t = align_umeyama(true_points, orb_points)
            print(R)
            R = R.reshape([9,])
            print(R)

            # finally publish the computed scale, matrix and vector
            if s>0:
                self.scale_publisher.publish(Float64(s))
            if sum(np.isnan(R)) == 0:
                rot = Float32MultiArray()
                rot.data = R.tolist()
                self.rot_publisher.publish(rot)
            if sum(np.isnan(t)) == 0:
                t_out = Float32MultiArray()
                t_out.data = t.tolist()
                self.trans_publisher.publish(t_out)


    
if __name__ == '__main__':
    rospy.init_node("scale_update")
    rospy.Rate(1)
    pub_scale = rospy.Publisher("/scale", Float64, queue_size=1)
    pub_rot = rospy.Publisher("/rotation_matrix", Float32MultiArray, queue_size=1)
    pub_trans = rospy.Publisher("/translation", Float32MultiArray, queue_size=1)
    server = Server(pub_scale, pub_rot, pub_trans)
    rospy.Subscriber('/ardrone/true_position', PoseStamped , server.true_pos_callback)
    rospy.Subscriber('/orb/pose', PoseStamped, server.est_pos_orb_callback)
    rospy.spin()



