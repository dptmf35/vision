import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

# 이미지 메시지 데이터를 어레이 형태로 변환
bridge = CvBridge()


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_sub')
        qos = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(
            Image,  # 임포트 된 메시지 타입
            '/result',  # 토픽리스트에서 조회한 토픽 주소
            self.image_callback,  # 정의한 콜백함수
            qos)
        self.image = np.empty(shape=[1])

    def image_callback(self, data):
        self.image = bridge.imgmsg_to_cv2(data, '8UC3')
        cv2.imshow('img', self.image)
        cv2.waitKey(33)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by Keyboard')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()