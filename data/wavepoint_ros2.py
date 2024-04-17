import rclpy
from rclpy.node import Node
#from tf_transformations import euler_from_quaternion

import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from time import gmtime, strftime



#file = open(strftime('~/rcws/logs/wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')
file = open(strftime('./wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')
file.write('%s, %s, %s, %s\n' % ("px", "py", "yaw", "odom_speed"))

file1 = open(strftime('./wp-%Y-%m-%d-%H-%M-%S',gmtime())+'_drive.csv', 'w')
file1.write('%s, %s\n' % ("st_angle", "drive_speed"))

class WavePointCollector(Node):
    def __init__(self):
        super().__init__('wave_point_collector')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'
        odom_topic = '/pf/pose/odom'
        odom_topic = '/ego_racecar/odom'

        # TODO: create subscribers and publishers
        

        self.odom_subscriber = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        
        # self.scan_subscriber = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_subscriber = self.create_subscription(AckermannDriveStamped, drive_topic, self.drive_callback, 10)

        # TODO: set PID gains
        self.kp = 1.5
        self.ki = 0.0
        self.kd = 0.15

        # TODO: store history
        self.integral = 0
        self.prev_error = 0
        self.error = 0

        # TODO: store any necessary values you think you'll need
        self.first = True

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
    def drive_callback(self, data):
	    file1.write('%f, %f\n' % (data.drive.steering_angle, data.drive.speed))
    	
    
    def odom_callback(self, data):
        quaternion = np.array([data.pose.pose.orientation.x, 
                           data.pose.pose.orientation.y, 
                           data.pose.pose.orientation.z, 
                           data.pose.pose.orientation.w])
        print(quaternion)

        euler = self.euler_from_quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        speed = np.linalg.norm(np.array([data.twist.twist.linear.x, 
                                data.twist.twist.linear.y, 
                                data.twist.twist.linear.z]))
        file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                     data.pose.pose.position.y,
                                     euler[2],
                                     speed))
            




def main(args=None):
    rclpy.init(args=args)
    print("Collecting wavepoint Initialized")
    wavepoint_collector_node = WavePointCollector()
    rclpy.spin(wavepoint_collector_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wavepoint_collector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
