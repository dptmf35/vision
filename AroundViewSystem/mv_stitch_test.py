import stitching
import cv2
import torch
import json
import glob
import numpy as np
from matplotlib import pyplot as plt
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
import time
import cv2 as cv
import numpy as np

img_handler = ImageHandler(medium_megapix=0.5, low_megapix=0.5,
                           final_megapix=0.4)
finder = FeatureDetector(detector='orb', nfeatures=300)

matcher = FeatureMatcher(matcher_type='homography', range_width=-1)

subsetter = Subsetter()

camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()

warper = Warper(warper_type='spherical')
# timelapser = Timelapser('as_is')
cropper = Cropper()
seam_finder = SeamFinder()
compensator = ExposureErrorCompensator()
blender = Blender(blender_type='multiband', blend_strength=5)



setting = {"try_use_gpu": True, "nfeatures" : 500}
stitcher = stitching.Stitcher(**setting)


def undistort(img, save_name, fov_type, dim2=None, dim3=None, balance=0.0):
    # get intrinsic parameters
    with open("camera_calibration_data.json", 'r') as f:
        data = json.load(f)
        DIM = data[fov_type]['DIM']
        K = np.array(data[fov_type]['K'])
        D = np.array(data[fov_type]['D'])
    f.close()
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # save as *_undistorted.jpg
    cv2.imwrite(f'./samples/{save_name}_undistorted.jpg', undistorted_img)
    return undistorted_img

# temp1 = cv2.imread("./samples/15x10_175_10.jpg")
# temp2 = cv2.imread("./samples/15x10_175_10.jpg")
#
#
#
# undistort(temp1, "temp1","fov_150")
# undistort(temp2, "temp2","fov_150")


cap1 = cv2.VideoCapture("./samples/vid1.mp4")
cap2 = cv2.VideoCapture("./samples/vid2.mp4")
_, fra1 = cap1.read()
_, fra2 = cap2.read()

fra1 = undistort(fra1, "fra1", "fov_150")
fra2 = undistort(fra2, "fra2", "fov_150")

image_list = ["./samples/fra1_undistorted.jpg", "./samples/fra2_undistorted.jpg"]

img_handler.set_img_names(image_list)
medium_imgs = list(img_handler.resize_to_medium_resolution())
low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
final_imgs = list(img_handler.resize_to_final_resolution())

original_size = img_handler.img_sizes[0]
medium_size = img_handler.get_image_size(medium_imgs[0])
low_size = img_handler.get_image_size(low_imgs[0])
final_size = img_handler.get_image_size(final_imgs[0])

start = time.time()

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




blender.prepare(final_corners, final_sizes)
for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
    blender.feed(img, mask, corner)
panorama, _ = blender.blend()

# cv2.imshow("pano", panorama)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

while cap1.isOpened() :
    _, fra1 = cap1.read()
    _, fra2 = cap2.read()

    if fra1 is None or fra2 is None :
        break

    undistort(fra1, "fra1", "fov_150")
    undistort(fra2, "fra2", "fov_150")


    # fra1 = cv2.resize(fra1, (0, 0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)
    # fra2 = cv2.resize(fra2, (0, 0), fx=1/2,fy=1/2, interpolation=cv2.INTER_AREA)

    img_handler.set_img_names(image_list)
    final_imgs = list(img_handler.resize_to_final_resolution())
    final_imgs = subsetter.subset_list(final_imgs, indices)
    img_names = subsetter.subset_list(img_handler.img_names, indices)
    img_handler.img_names, img_handler.img_sizes = img_names, img_sizes


    warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

    # cropped_low_masks = list(cropper.crop_images(warped_low_masks))
    # cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
    # low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

    # lir_aspect = img_handler.get_low_to_final_ratio()  # since lir was obtained on low imgs
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

    cv2.imshow("pano", panorama)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


