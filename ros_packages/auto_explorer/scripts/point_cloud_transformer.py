#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
import numpy as np 
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Float32MultiArray

class Server:
    def __init__(self, true_pq_pub, true_pose_pub): 
        print("init")
        self.pq = None
        self.scale = None 
        self.rot_mat = None 
        self.trans = None
        self.orb_pos = None 
        self.true_pq_pub = true_pq_pub
        self.true_position_pub = true_pose_pub
    
    def pq_callback(self, msg):
        if msg:
            self.pq = msg
            self.update_pq_pose()

    def scale_callback(self, msg):
        if msg:
            self.scale = msg.data

    def trans_callback(self, msg):
        print(msg.data)
        if msg:
            if len(msg.data) == 3:
                self.trans = np.array(msg.data)
    
    def rot_mat_callback(self, msg):
        if msg:
            if len(msg.data) == 9:
                self.rot_mat = np.array(msg.data)

    def orb_pos_callback(self, msg):
        if msg:
            self.orb_pos = msg

    def update_pq_pose(self): 
        # wait for all values to be there: 
        print(self.scale)
        print(self.rot_mat)
        print(self.trans)
        if (not self.scale) or (not isinstance(self.rot_mat, (np.ndarray, np.generic) )) or (not isinstance(self.trans, (np.ndarray, np.generic) )):
            print("waiting for all data...")
            return
        else:
            print("found all data for computation!")

        # pointcloud
        pq = PointCloud()
        orb_points = self.pq.points
        print(self.rot_mat)
        R = self.rot_mat.reshape([3,3])
        print(R)

        for p in orb_points: 
            p_coord = np.array([p.x, p.y, p.z])
            p_trans = np.array(self.scale * np.dot(R, p_coord) + self.trans)
            
            p_out = Point32()
            p_out.x = p_trans[0]
            p_out.y = p_trans[1]
            p_out.z = p_trans[2]
            pq.points.append(p_out)


        # bottom and top 
        for x_oi in np.linspace(-30, 30, 100): 
            for y_oi in np.linspace(-30, 30, 100): 
                for z_oi in [0, 15]: 
                    p_out = Point32()
                    p_out.x = x_oi
                    p_out.y = y_oi
                    p_out.z = z_oi
                    pq.points.append(p_out)

        # left and right wall 
        for x_oi in [-30, 30]: 
            for y_oi in np.linspace(-30, 30, 100): 
                for z_oi in np.linspace(0, 15, 50): 
                    p_out = Point32()
                    p_out.x = x_oi
                    p_out.y = y_oi
                    p_out.z = z_oi
                    pq.points.append(p_out)

        # front and back wall
        for y_oi in [-30, 30]: 
            for x_oi in np.linspace(-30, 30, 100): 
                for z_oi in np.linspace(0, 15, 50): 
                    p_out = Point32()
                    p_out.x = x_oi
                    p_out.y = y_oi
                    p_out.z = z_oi
                    pq.points.append(p_out)

        pq.header.frame_id = "world"
        self.true_pq_pub.publish(pq)

        # true position 
        true_pos = PointStamped()
        true_pos.header.stamp = self.orb_pos.header.stamp
        orb_pos = self.orb_pos.pose.position
        orb_pos_coord = np.array([orb_pos.x, orb_pos.y, orb_pos.z])
        orb_pos_trans = np.array(self.scale * np.dot(R, p_coord) + self.trans)
        true_pos.point.x = orb_pos_trans[0]
        true_pos.point.y = orb_pos_trans[1]
        true_pos.point.z = orb_pos_trans[2]
        self.true_position_pub.publish(true_pos) 



    
if __name__ == '__main__':
    rospy.init_node("pq_transformer")
    rospy.Rate(5)
    pub_pq = rospy.Publisher("/point_cloud_transformed", PointCloud, queue_size=1)
    pub_pos = rospy.Publisher("/orb_position_transformed", PointStamped, queue_size=1)
    server = Server(pub_pq, pub_pos)
    rospy.Subscriber('/scale', Float64 , server.scale_callback)
    rospy.Subscriber('/rotation_matrix', Float32MultiArray , server.rot_mat_callback)
    rospy.Subscriber('/translation', Float32MultiArray , server.trans_callback)
    rospy.Subscriber('/orb/pose', PoseStamped, server.orb_pos_callback)
    rospy.Subscriber('/orb/map_points', PointCloud, server.pq_callback)
    rospy.spin()



