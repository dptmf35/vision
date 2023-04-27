import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import stitching
import json

from pathlib import Path
from stitching.image_handler import ImageHandler
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

img_handler = ImageHandler(medium_megapix=0.5, low_megapix=0.4,
                           final_megapix=0.4)
finder = FeatureDetector(nfeatures=500)
matcher = FeatureMatcher(matcher_type='homography', range_width=-1)
camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()
cropper = Cropper()
seam_finder = SeamFinder()
compensator = ExposureErrorCompensator()
warper = Warper(warper_type='spherical')
blender = Blender(blender_type='multiband', blend_strength=5)
subsetter = Subsetter()

settings = {'try_use_gpu' : True, "confidence_threshold" : 0.5}
stitcher = stitching.Stitcher(**settings)

image_list = ['./data/img1.jpg', './data/img2.jpg', './data/img3.jpg']

img_handler.set_img_names(image_list)
medium_imgs = list(img_handler.resize_to_medium_resolution())
low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
final_imgs = list(img_handler.resize_to_final_resolution())

original_size = img_handler.img_sizes[0]
medium_size = img_handler.get_image_size(medium_imgs[0])
low_size = img_handler.get_image_size(low_imgs[0])
final_size = img_handler.get_image_size(final_imgs[0])

features = [finder.detect_features(img) for img in medium_imgs]
keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])

matches = matcher.match_features(features)

indices = subsetter.get_indices_to_keep(features, matches)

medium_imgs = subsetter.subset_list(medium_imgs, indices)
low_imgs = subsetter.subset_list(low_imgs, indices)
final_imgs = subsetter.subset_list(final_imgs, indices)
features = subsetter.subset_list(features, indices)
matches = subsetter.subset_matches(matches, indices)

img_names = subsetter.subset_list(img_handler.img_names, indices)
img_sizes = subsetter.subset_list(img_handler.img_sizes, indices)

img_handler.img_names, img_handler.img_sizes = img_names, img_sizes


cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)

warper.set_scale(cameras)


low_sizes = img_handler.get_low_img_sizes()
camera_aspect = img_handler.get_medium_to_low_ratio()      # since cameras were obtained on medium imgs

warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

final_sizes = img_handler.get_final_img_sizes()
camera_aspect = img_handler.get_medium_to_final_ratio()    # since cameras were obtained on medium imgs

warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

print(final_corners)
print(final_sizes)

mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
lir = cropper.estimate_largest_interior_rectangle(mask)

low_corners = cropper.get_zero_center_corners(low_corners)
rectangles = cropper.get_rectangles(low_corners, low_sizes)
overlap = cropper.get_overlap(rectangles[1], lir)
intersection = cropper.get_intersection(rectangles[1], overlap)

cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

cropped_low_masks = list(cropper.crop_images(warped_low_masks))
cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

lir_aspect = img_handler.get_low_to_final_ratio()  # since lir was obtained on low imgs
cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)


seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]


compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

compensated_imgs = [compensator.apply(idx, corner, img, mask)
                    for idx, (img, mask, corner)
                    in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]


DIM=(640, 480)
# K= np.array([[341.74229398988433,0.0,325.1640543472415],
# 			[0.0,342.9767502673575,260.5383939654134],
# 			[0.0,0.0,1.0]])
# D = np.array([[0.029951123979630775],
# 			[-0.20820190674818156],
# 			[0.2342040898893896],
# 			[-0.06274915635263466]])
fov_type="fov_175"
with open("camera_calibration_data.json", 'r') as f:
    data = json.load(f)
    DIM = data[fov_type]['DIM']
    K = np.array(data[fov_type]['K'])
    D = np.array(data[fov_type]['D'])
f.close()

# print(K, "type:", type(K))


panor = cv2.imread("panorama.jpg")
height, width = panor.shape[:2]
print(f"height {height} x width {width}")

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('panoramas.mp4', fourcc, 30, (1000, 500))

bridge = CvBridge()

class BboxSubscriber(Node) :
    def __init__(self):
        super().__init__('bbox_sub')
        qos = QoSProfile(depth=10)
        self.img1 = self.create_subscription(
            Image,
            '/image_3',
            self.img1_callback,
            qos
        )
        self.img2 = self.create_subscription(
            Image,
            '/image_4',
            self.img2_callback,
            qos
        )
        self.img3 = self.create_subscription(
            Image,
            '/image_5',
            self.img3_callback,
            qos
        )
        self.img1 = np.empty(shape=[1])
        self.img2 = np.empty(shape=[1])
        self.img3 = np.empty(shape=[1])
        self.publisher = self.create_publisher(Image, '/panorama', 10)
        self.timer = self.create_timer(0.01, self.time_callback)
        self.panorama = None

    def img1_callback(self, data):
        data = bridge.imgmsg_to_cv2(data, '8UC3')
        self.img1 = self.undistort_simple(data, DIM, K, D)
        # cv2.imshow('fov_125', self.image_125)
        # cv2.waitKey(2)
        cv2.imwrite('./data/img1.jpg', self.img1)

    def img2_callback(self, data):
        data = bridge.imgmsg_to_cv2(data, '8UC3')
        self.img2 = self.undistort_simple(data, DIM, K, D)
        # cv2.imshow('fov_125', self.image_125)
        # cv2.waitKey(2)
        cv2.imwrite('./data/img2.jpg', self.img2)

    def img3_callback(self, data):
        data = bridge.imgmsg_to_cv2(data, '8UC3')
        self.img3 = self.undistort_simple(data, DIM, K, D)
        cv2.imwrite('./data/img3.jpg', self.img3)

        # cv2.imshow('fov_125', self.image_125)
        # cv2.waitKey(2)
        # imgs = cv2.hconcat([self.img1, self.img2, self.img3])
        # print(self.img1.shape, self.img2.shape, self.img3.shape)

        img_handler.set_img_names(image_list)
        final_imgs = list(img_handler.resize_to_final_resolution())

        final_imgs = subsetter.subset_list(final_imgs, indices)
        img_names = subsetter.subset_list(img_handler.img_names, indices)
        img_sizes = subsetter.subset_list(img_handler.img_sizes, indices)

        img_handler.img_names, img_handler.img_sizes = img_names, img_sizes

        final_sizes = img_handler.get_final_img_sizes()
        camera_aspect = img_handler.get_medium_to_final_ratio()  # since cameras were obtained on medium imgs

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

        cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

        compensated_imgs = [compensator.apply(idx, corner, img, mask)
                            for idx, (img, mask, corner)
                            in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]

        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        # detecion part with YOLOv8
        results = model(panorama, conf=0.7)
        panorama = results[0].plot()
        cv2.imshow("result", panorama)
        cv2.waitKey(2)
        self.panorama = panorama
        out.write(cv2.resize(panorama, (1000, 500)))


    def time_callback(self):
        if self.panorama is not None :
            pano = bridge.cv2_to_imgmsg(self.panorama)
            self.publisher.publish(pano)
            self.get_logger().info("Publishing panorama result")

    def undistort(self, img, DIM, K, D, dim2=None, dim3=None, balance=0.0):
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def undistort_simple(self, img, DIM=DIM, K=K, D=D):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img



def main(args=None) :
    rclpy.init(args=args)
    node = BboxSubscriber()
    try :
        rclpy.spin(node)
    except KeyboardInterrupt :
        node.get_logger().info('Process Stopped by Keyboard')
    finally :
        node.destroy_node()
        rclpy.shutdown()
        out.release()

if __name__ == '__main__' :
    main()