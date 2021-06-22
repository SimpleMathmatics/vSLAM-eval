#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from ardrone_autonomy.msg import Navdata

class Server:
    def __init__(self, position_publisher): 
        print("init")
        self.navdata = None 
        self.position = None 
        self.position_publisher = position_publisher
    
    def navdata_callback(self, msg):
        self.navdata = msg

        # initialize the publishing 
        if self.position == None: 
            init_point = PointStamped()
            init_point.header.stamp = rospy.Time.now()
            init_point.header.frame_id = "init"
            init_point.point.x = 0
            init_point.point.y = 0
            init_point.point.z = 0
            self.position_publisher.publish(init_point)

    def position_callback(self, msg):
        if msg:
            self.position = msg

        self.update_position()

    def update_position(self):
        # get velocity in x, y and z direction
        x_vel = self.navdata.vz
        y_vel = self.navdata.vy
        z_vel = self.navdata.vz

        if x_vel is not None and y_vel is not None and z_vel is not None: 
            # update positions
            curr_time = rospy.Time.now()
            time_diff = (curr_time - self.position.header.stamp).to_sec()
            new_point = PointStamped()
            new_point.header.stamp = curr_time
            new_point.header.frame_id = "init"
            new_point.point.x = self.position.point.x + time_diff * x_vel / 1000
            new_point.point.y = self.position.point.y + time_diff * y_vel / 1000
            new_point.point.z = self.position.point.z + time_diff * z_vel / 1000
            self.position_publisher.publish(new_point)
            print(new_point)
            rospy.sleep(0.1)

    
if __name__ == '__main__':
    rospy.init_node("position_update")
    rospy.Rate(10)
    pub = rospy.Publisher("/drone_position_init", PointStamped, queue_size=1)
    server = Server(pub)
    rospy.Subscriber('/drone_position_init', PointStamped , server.position_callback)
    rospy.Subscriber('/ardrone/navdata', Navdata, server.navdata_callback)
    rospy.spin()



