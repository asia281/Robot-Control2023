from email.mime import image
import cv2
import numpy as np
import pickle

# Load camera matrix and distortion coefficients from calibration file
with open("calibration.pckl", "rb") as f:
    camera_matrix, dist_coeffs, _, _ = pickle.load(f)

def undistort(img, camera_matrix, dist_coeffs):
    size = img.shape[1::-1]
    # h, w = img.shape[:2]
    # new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    alpha = 0.999 # you need to adjust alpha
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, alpha)[0]
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undistorted_img

# Load the first image
image1 = cv2.imread("frame-002.png")
undistorted_img = image1.copy() #undistort(image1, camera_matrix, dist_coeffs)

cv2.imshow("Undistorted Image 1", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Define aruco parameters
gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_parameters =  cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
corners, ids, _ = aruco_detector.detectMarkers(gray_img)
print(corners)

marker1_index = np.where(ids == 1)[0]
print(marker1_index)

squareLength = 0.026 
markerLength = 0.015
ml2 = markerLength / 2

# If marker 1 is found, get its corners
#if len(marker1_index) > 0:
for corner in corners:
    #marker1_rvec, marker1_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[marker1_index[0]], 0.015, cameraMatrix, distCoeffs)
    marker_points = np.array([[-ml2, ml2, 0], [ml2, ml2, 0], [ml2, -ml2, 0], [-ml2, -ml2, 0]], dtype=np.float32)
    _, marker1_rvec, marker1_tvec = cv2.solvePnP(marker_points, corner, camera_matrix, dist_coeffs)
    print(marker1_rvec, marker1_tvec)
    # Extract rotation and translation matrices
    rvec, _ = cv2.Rodrigues(marker1_rvec)
    tvec = marker1_tvec[0]
    cv2.drawFrameAxes(undistorted_img, camera_matrix, dist_coeffs, marker1_rvec, marker1_tvec, markerLength)
    # cv2.aruco.drawAxis(image1, cameraMatrix, distCoeffs, rvec, tvec, 0.01)

    # Convert rotation matrix to Euler angles (for readability)
    marker1_rot_euler = cv2.Rodrigues(rvec)[0]

    # Display the matrices
    print("Translation matrix (trn):")
    print(tvec)
    print("\nRotation matrix (rot):")
    print(rvec)


# Draw detected markers and coordinate systems
cv2.aruco.drawDetectedMarkers(undistorted_img, corners, ids)

# Display the first image with detected markers and coordinate systems
cv2.imshow("Undistorted Image 1", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Given pose information for marker one in the second image
trn = np.array([[0.01428826], [0.02174878], [0.37597986]])
rot = np.array([[1.576368], [-1.03584672], [0.89579336]])

# Create a blank image for the second image
image2 = np.image1 = cv2.imread("frame-253.png")

# Draw the missing marker (blue color) on the second image using the given pose
marker_size = 0.015
rot_mat, _ = cv2.Rodrigues(rot)
points_3d = np.array([[0, 0, 0], [0, marker_size, 0], [marker_size, marker_size, 0], [marker_size, 0, 0]])
points_3d = np.dot(rot_mat, points_3d.T).T + trn.T
#points_2d, _ = cv2.projectPoints(points_3d, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
points_2d, _ = cv2.projectPoints(points_3d, rot, trn, camera_matrix, dist_coeffs)
print(points_2d)

# [[[ 1.25155792e+10 -2.58996029e+10]]

#  [[ 1.19660050e+10 -2.45144278e+10]]

#  [[ 1.24427130e+10 -2.51304640e+10]]

#  [[ 1.30090609e+10 -2.65427546e+10]]]

print(points_2d)

points_2d = np.int32(points_2d).reshape(-1, 2)

cv2.polylines(image2, [points_2d], isClosed=True, color=(255, 255, 255), thickness=5)

# Display the second image with the missing marker
cv2.imshow("Image 2 with Missing Marker", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
