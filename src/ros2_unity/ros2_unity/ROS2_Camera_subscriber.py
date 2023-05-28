import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = "/camera_robot/front"

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, TOPIC, self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, image_msg):
        self.get_logger().info('I heard: "%s"' % image_msg.data)


def main(args=None):
    rclpy.init(args=args)

    camera_subscriber = CameraSubscriber()

    rclpy.spin(camera_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()