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

class Server:
    def __init__(self, pub_ts): 
        print("init")
        self.true_pos = [] 
        self.pub_ts = pub_ts 

    def true_pos_callback(self, msg):
        if msg:
            # transform to pose 
            pose_oi = PoseStamped()
            pose_oi.header.stamp = rospy.Time.now()
            pose_oi.pose = msg.pose[len(msg.pose) - 1]
            self.pub_ts.publish(pose_oi)



    
if __name__ == '__main__':
    rospy.init_node("timestamp_adder")
    rospy.Rate(100)
    pub_ts = rospy.Publisher("/ardrone/true_position", PoseStamped, queue_size=1)
    server = Server(pub_ts)
    rospy.Subscriber('/gazebo/model_states', ModelStates , server.true_pos_callback)
    rospy.spin()



