import stitching
import cv2
import torch
import json
import glob
import numpy as np

# select camera type = [fov_125, fov_150, fov_175, fov_180]

class FrameStitcher() :
    def __init__(self, img_list, detector='sift', confidence_threshold=0.5, warper_type='spherical',
                 try_use_gpu=True, nfeatures=500):
        self.img_list = img_list
        if not torch.cuda.is_available() :
            try_use_gpu = False
        setting = {"detector": detector, "warper_type" : warper_type,"try_use_gpu": try_use_gpu, "nfeatures" : nfeatures}
        print(setting) # check parameter setting
        self.stitcher = stitching.Stitcher(**setting)

    def stitching_frames(self, fov_type):
        for impath in self.img_list:
            self.undistort(impath, fov_type)
        # stitching only with undistorted images
        self.img_list = [impath[:-4] + "_undistorted.jpg" for impath in self.img_list]
        try :
            stitching_result = self.stitcher.stitch(self.img_list)
            cv2.imshow("stitching_result", stitching_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e :
            print(f"Stitching Error : {e}")

    def undistort(self, img_path, fov_type, dim2=None, dim3=None, balance=0.0):
        # get intrinsic parameters
        with open("camera_calibration_data.json", 'r') as f:
            data = json.load(f)
            DIM = data[fov_type]['DIM']
            K = np.array(data[fov_type]['K'])
            D = np.array(data[fov_type]['D'])
        f.close()
        img = cv2.imread(img_path)
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
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
        cv2.imwrite(f'{img_path[:-4]}_undistorted.jpg', undistorted_img)
        # return undistorted_img

if __name__ == '__main__' :
    # test with 150 fov images
    fov_150_imgs = glob.glob("./samples/*")
    fov_150_imgs = [path for path in fov_150_imgs if "undistorted" not in path]
    stitcher = FrameStitcher(fov_150_imgs)
    stitcher.stitching_frames('fov_150')