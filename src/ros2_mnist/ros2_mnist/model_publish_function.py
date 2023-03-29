import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from cv_bridge import CvBridge
import os
import cv2
import numpy as np
import tensorflow as tf

class ModelPublisher(Node):
    def __init__(self):
        super().__init__('model_publisher')

        self._cv_bridge = CvBridge()
        # Recreate the exact same model, including its weights and the optimizer
        self.new_model = tf.keras.models.load_model(os.path.join(get_package_share_directory('ros2_mnist'),
                            "my_model.h5"))

        self.subscriber_ = self.create_subscription(Image, '/racecar/camera', self.predict_callback, 1)
        self.publisher_ = self.create_publisher(Int64, 'result', 1)


    def predict_callback(self, image_msg):
        # convert image to cv_bridge
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")[:,:,0] 
        cv_image = cv2.resize(cv_image, (28 , 28))     
        cv_image = np.invert(np.array([cv_image]))
        prediction = self.new_model.predict(cv_image)

        # predict the number 
        answer = np.argmax(prediction, 1)[0]
        self.get_logger().info('%d' % answer)

        # convert from numpy array to std.msg 
        msg = Int64()
        msg.data = answer.item()

        # publish the prediction
        self.publisher_.publish(msg)

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
  
    # Create the node
    model_publisher = ModelPublisher()
    
    # Spin the node so the callback function is called.
    rclpy.spin(model_publisher)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    model_publisher.destroy_node()
    
    # Shutdown the ROS client library for Python
    rclpy.shutdown()
        

if __name__ == '__main__':
    main()