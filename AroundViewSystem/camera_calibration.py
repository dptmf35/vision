import os, glob
import cv2
import numpy as np
import json

CHECKERBOARD = (14, 9)

# fisheye camera calibration
class CalibrationCam():
    def __init__(self, calib_type):
        self.images = glob.glob(f'./{calib_type}/*')
        self.DIM = None
        self.K = None
        self.D = None
        self.dim1 = None
        self.dim2 = None
        self.dim3 = None
        self.new_K = None
        self.balance = 0.0
        self.scaled_K = None
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        self.objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self._img_shape = None
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

    def get_params(self):
        for fname in self.images:
            img = cv2.imread(fname)
            if self._img_shape == None:
                self._img_shape = img.shape[:2]
            else:
                assert self._img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.subpix_criteria)
                self.imgpoints.append(corners)
        N_OK = len(self.objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                self.objpoints,
                self.imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                self.calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(self._img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
        self.DIM = self._img_shape[::-1]
        self.K = np.array(K.tolist())
        self.D = np.array(D.tolist())

    def undistort(self, dim2=None, dim3=None):
        img = cv2.imread(self.images[0])
        self.dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert self.dim1[0]/self.dim1[1] == self.DIM[0]/self.DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            self.dim2 = self.dim1
        if not dim3:
            self.dim3 = self.dim1
        self.scaled_K = self.K * self.dim1[0] / self.DIM[0]  # The values of K is to scale with image dimension.
        self.scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.scaled_K, self.D, self.dim2, np.eye(3), self.balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.scaled_K, self.D, np.eye(3), self.new_K, self.dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imshow("undistorted", undistorted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def undistort_simple(self):
        img = cv2.imread(self.images[-1])
        h, w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imshow("undistorted", undistorted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #         return undistorted_img

    def get_camera_dict(self):
        self.get_params()
        self.undistort()
        data = {'DIM': self.DIM,
                'K': np.array(self.K).tolist(),
                'D': np.array(self.D).tolist(),
                #                 'new_K':np.array(self.new_K).tolist(),
                #                 'scaled_K':np.array(self.scaled_K).tolist(),
                #                 'balance':self.balance}
                }
        return data


if __name__ == '__main__' :
    data = dict()

    calib = CalibrationCam('15x10_125')
    fov_dict = calib.get_camera_dict()
    data['fov_125'] = fov_dict

    calib = CalibrationCam('15x10_150')
    fov_dict = calib.get_camera_dict()
    data['fov_150'] = fov_dict

    calib = CalibrationCam('15x10_175')
    fov_dict = calib.get_camera_dict()
    data['fov_175'] = fov_dict

    calib = CalibrationCam('15x10_180')
    fov_dict = calib.get_camera_dict()
    data['fov_180'] = fov_dict


    with open("camera_calibration_data.json", "w") as f:
        json.dump(data, f, indent="\t")