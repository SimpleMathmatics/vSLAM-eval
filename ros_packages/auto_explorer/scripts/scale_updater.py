#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
import numpy as np 
from umeyama import align_umeyama

class Server:
    def __init__(self, scale_publisher): 
        print("init")
        self.est_pos_vel = [] 
        self.est_pos_orb = [] 
        self.scale_publisher = scale_publisher
        self.scale = None
    
    def est_pos_vel_callback(self, msg):
        if msg:
            self.est_pos_vel.append(msg)
            if len(self.est_pos_vel) > 25: 
                del self.est_pos_vel[0]

    def est_pos_orb_callback(self, msg):
        if msg:
            print(len(self.est_pos_orb))
            self.est_pos_orb.append(msg)
            if len(self.est_pos_orb) > 25: 
                del self.est_pos_orb[0]

                self.update_scale()

    def update_scale(self):
        # return, if not enough points are available
        # since no scale can be computed
        if (len(self.est_pos_orb) < 25) or (len(self.est_pos_vel) < 25):
            return

        # return the scale if it already has been calculated    
        elif self.scale:
            return
        else:
            # get minimum and maximum time for each queue to figure out, 
            # how many points can be considered for alignment. This is only done once!
            min_orb = np.min([pose_oi.header.stamp.to_sec() for pose_oi in self.est_pos_orb])
            max_orb = np.max([pose_oi.header.stamp.to_sec() for pose_oi in self.est_pos_orb])

            min_vel = np.min([point_oi.header.stamp.to_sec() for point_oi in self.est_pos_vel])
            max_vel = np.max([point_oi.header.stamp.to_sec() for point_oi in self.est_pos_vel])

            thresh_min = np.max([min_orb, min_vel])
            thresh_max = np.min([max_vel, max_orb])

            # cut off the queues
            orb_oi = [pose_oi for pose_oi in self.est_pos_orb if pose_oi.header.stamp.to_sec() > thresh_min]
            vel_oi = [point_oi for point_oi in self.est_pos_vel if point_oi.header.stamp.to_sec() > thresh_min]

            # for the shorter remaining queue, get the matching point
            if len(orb_oi) <= len(vel_oi): 
                orb_oi_final = orb_oi
                vel_oi_final = []
                for pose_oi in orb_oi: 
                    diffs_oi = [np.abs(pose_oi.header.stamp.to_sec() - point_oi.header.stamp.to_sec()) for point_oi in vel_oi]
                    vel_oi_final.append(vel_oi[diffs_oi.index(min(diffs_oi))])

            else:
                vel_oi_final = vel_oi
                orb_oi_final = []
                for pose_oi in vel_oi: 
                    diffs_oi = [np.abs(pose_oi.header.stamp.to_sec() - point_oi.header.stamp.to_sec()) for point_oi in orb_oi]
                    vel_oi_final.append(vel_oi[diffs_oi.index(min(diffs_oi))])


            # now do the alignment and compute the scale
            x_orb = [pose_oi.pose.position.x for pose_oi in orb_oi_final]
            y_orb = [pose_oi.pose.position.y for pose_oi in orb_oi_final]
            z_orb = [pose_oi.pose.position.z for pose_oi in orb_oi_final]

            x_vel = [point_oi.point.x for point_oi in vel_oi_final]
            y_vel = [point_oi.point.y for point_oi in vel_oi_final]
            z_vel = [point_oi.point.z for point_oi in vel_oi_final]
            
            orb_points = np.column_stack((x_orb, y_orb, z_orb))
            vel_points = np.column_stack((x_vel, y_vel, z_vel))

            s, R, t = align_umeyama(vel_points, orb_points)

            # finally publish the computed scale 
            self.scale_publisher.publish(Float64(s))
            self.scale = s



    
if __name__ == '__main__':
    rospy.init_node("scale_update")
    rospy.Rate(5)
    pub = rospy.Publisher("/scale_estimation", Float64, queue_size=1)
    server = Server(pub)
    rospy.Subscriber('/drone_position_init', PointStamped , server.est_pos_vel_callback)
    rospy.Subscriber('/orb/pose', PoseStamped, server.est_pos_orb_callback)
    rospy.spin()



