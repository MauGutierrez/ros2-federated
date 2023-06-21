import rclpy
import threading
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC_ROBOT_1 = "/camera_robot/Right"
TOPIC_ROBOT_2 = "/camera_robot/Left"   

class CameraSubscriber_1(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.image_msg = Image()
        self.subscription_topic_1 = self.node.create_subscription(Image, TOPIC_ROBOT_1, self.callback_topic_1, 30)
        self.publisher_image_right_ = self.node.create_publisher(Image, '/camera/Right', 1)

    def callback_topic_1(self, image_msg):
        # self.node.get_logger().info('I heard: "%s"' % image_msg.data)
        self.node.get_logger().info('I heard topic: "%s"' % TOPIC_ROBOT_1)
        self.image_msg.height = np.shape(image_msg)[0]
        self.image_msg.width = np.shape(image_msg)[1]
        self.image_msg.encoding = "bgr8"
        self.image_msg.is_bigendian = False
        self.image_msg.step = np.shape(image_msg)[2] * np.shape(image_msg)[1]
        self.image_msg.data = np.array(image_msg).tobytes()
    
    def run(self):
        rate = self.node.create_rate(1)
        while rclpy.ok():
            self.publisher_image_right_.publish(self.image_msg)
            rate.sleep()

class CameraSubscriber_2(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.image_msg = Image()
        self.subscription_topic_2 = self.node.create_subscription(Image, TOPIC_ROBOT_2, self.callback_topic_2, 30)
        self.publisher_image_left_ = self.node.create_publisher(Image, '/camera/Left', 1)

    def callback_topic_2(self, image_msg):
        # self.node.get_logger().info('I heard: "%s"' % image_msg.data)
        self.node.get_logger().info('I heard topic: "%s"' % TOPIC_ROBOT_2)
        self.image_msg.height = np.shape(image_msg)[0]
        self.image_msg.width = np.shape(image_msg)[1]
        self.image_msg.encoding = "bgr8"
        self.image_msg.is_bigendian = False
        self.image_msg.step = np.shape(image_msg)[2] * np.shape(image_msg)[1]
        self.image_msg.data = np.array(image_msg).tobytes()
    
    def run(self):
        rate = self.node.create_rate(1)
        while rclpy.ok():
            self.publisher_image_left_.publish(self.image_msg)
            rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    node = Node('multithreaded_node')

    # Initialize subscriber 1 and 2 for the two unity cameras
    camera_unity_subscriber_1 = CameraSubscriber_1(node)
    camera_unity_subscriber_2 = CameraSubscriber_2(node)

    camera_unity_subscriber_1.start()
    camera_unity_subscriber_2.start()

    rclpy.spin(node)

    camera_unity_subscriber_1.join()
    camera_unity_subscriber_2.join()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()