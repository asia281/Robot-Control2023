import pickle
import cv2
import numpy as np

MARKER_SIZE = 0.015

f = open('calibration.pckl', 'rb')
cameraMatrix, distCoeffs, _, _ = pickle.load(f)
img = cv2.imread('frame-002.png')

def undistort(img, camera_matrix, dist_coeffs):
    size = img.shape[1::-1]
    alpha = 0.999 # you need to adjust alpha
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, alpha)[0]
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undistorted_img

def task_1(img_draw):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    corners, ids, _ = detector.detectMarkers(img)

    cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)
    cv2.imwrite('task_1.jpg', img_draw)

    return corners, ids

def task_2(corners, img_draw):
    # pose estimation
    marker_points = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                              [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                              [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                              [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]])
    rvecs = []
    tvecs = []
    for c in corners:
        _, r, t = cv2.solvePnP(marker_points, c, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(r)
        tvecs.append(t)
        cv2.drawFrameAxes(img_draw, cameraMatrix, distCoeffs, r, t, 0.01)
    cv2.imwrite('task_2.jpg', img_draw)

    return rvecs, tvecs


def task_3(ids, rvecs, tvecs):

    index_of_1 = np.where(ids == 1)[0][0]
    print(index_of_1)
    # index_of_1 = None
    # for idx, id in enumerate(ids):
    #     if id == 1:
    #         index_of_1 = idx  # index of a marker with id 1

    print(f'Pose of marker (1): ')
    print(f'rvec: {rvecs[index_of_1]}')
    print(f'tvec: {tvecs[index_of_1]}')

# sth doesnt work well
def task_4():
    # draw missing marker
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    TEST_SIZE = 256
    marker_1 = cv2.aruco.generateImageMarker(dictionary, 1, TEST_SIZE)
    black_indices_y, black_indices_x = np.nonzero(marker_1 == 0)
    black_indices_x = (black_indices_x) / TEST_SIZE
    black_indices_y = (black_indices_y) / TEST_SIZE
    black_indices = np.stack([black_indices_x, black_indices_y, np.zeros(black_indices_x.shape[0])], axis=-1)
    black_indices *= MARKER_SIZE
    # white_indices = (np.nonzero(marker_1 == 255).reshape(-1, 2) - TEST_SIZE // 2) / TEST_SIZE

    img2 = cv2.imread('frame-253.png')
    trn = np.array([[0.01428826],
           [0.02174878],
           [0.37597986]])
    rot = np.array([[1.576368],
           [-1.03584672],
           [0.89579336]])

    imgpts = np.rint(cv2.projectPoints(black_indices, rot, trn, cameraMatrix, distCoeffs)[0]).astype(int).reshape(-1,2)
    # img2[...] = 0
    for imgpt in imgpts:
        img2[imgpt[1], imgpt[0], :] = 0

    imgpts2 = np.rint(cv2.projectPoints(np.array([[0,0,0], [0.01, 0.01, 0.01]]), rot, trn, cameraMatrix, distCoeffs)[0]).astype(int).reshape(-1,2)
    cv2.circle(img2, (imgpts2[0][0], imgpts2[0][1]), 5, (0,255,0), -1)
    print(img2.shape)
    cv2.imshow("Image 2 with Missing Marker", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_draw = img.copy()
    crnrs, ids = task_1(img_draw)
    rvecs, tvecs = task_2(crnrs, img_draw)
    task_3(ids, rvecs, tvecs)

    task_4()