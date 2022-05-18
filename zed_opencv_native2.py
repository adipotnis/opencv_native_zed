'''
/*****************************************************************************************
 ** This sample demonstrates how to capture stereo images and calibration parameters    **
 ** from the ZED camera with OpenCV without using the ZED SDK. 
 ** This code is modified for generating disparity and depth maps.                          **
 *****************************************************************************************/
'''

import numpy as np
import os
import configparser
import sys
import cv2
import wget
import yaml

def download_calibration_file(serial_number, file_dir) :
    if os.name == 'nt' :
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else :
        hidden_path = '/usr/local/zed/settings/'
    calibration_file = file_dir  #hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url+str(serial_number), out=file_dir)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file

def init_calibration(calibration_file, image_size) :

    cameraMatrix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])


    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)


    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution :
    """
    2208 : '2K'
    1920 : 'FHD' 
    1280 : 'HD'
    672  : 'VGA'
    HD is default
    """
    width = 1280
    height = 720

def main() :
    
    # Access the required serial number and directory ot the camera
    with open("zed_mini.yaml", "r") as stream:
        try:
            # Camera info
            yaml_dict = yaml.safe_load(stream)
            serial_number = yaml_dict.get('serial_number')
            config_path = yaml_dict.get('config_path')
            
            # Parameter values
            minDisparity = yaml_dict.get('minDisparity')
            numDisparities = yaml_dict.get('numDisparities')
            blockSize = yaml_dict.get('blockSize')
            speckleRange = yaml_dict.get('speckleRange')
            speckleWindowSize = yaml_dict.get('speckleWindowSize')

        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    # Open the ZED camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == 0:
        exit(-1)

    image_size = Resolution()

    # Set the video resolution to set value. As camera captures both left and right frame, hence width x 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)

    calibration_file = download_calibration_file(serial_number, os.path.join(config_path, 'SN'+ str(serial_number) +'.conf'))
    if calibration_file  == "":
        print('calibration file missing.')
        exit(1)
    print()
    print("Calibration file found. Loading...")

    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)

    print(f'Resolution = {Resolution.width}x{Resolution.height}')

    print("Press 'esc' to exit")
    print("Press 's' to save image")

    while True :
        # Get a new frame from camera
        retval, frame = cap.read()
        # Extract left and right images from side-by-side
        left_right_image = np.split(frame, 2, axis=1)   # Splits the right and left images in 2 parts
        
        # Display images
        cv2.imshow("left RAW", left_right_image[0])

        #remaps images to the planar form from the fisheye input using the calibration file 
        left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("left RECT", left_rect)
        #cv2.imshow("right RECT", right_rect)

        stereoMatcher = cv2.StereoBM_create()

        # Setting the parameters to required values
        stereoMatcher.setMinDisparity(minDisparity)
        stereoMatcher.setNumDisparities(numDisparities)    # multiple of 16
        stereoMatcher.setBlockSize(blockSize) # odd numbers
        stereoMatcher.setSpeckleRange(speckleRange)
        stereoMatcher.setSpeckleWindowSize(speckleWindowSize)
        
        # Convert to gray
        grayLeft = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        disparity = stereoMatcher.compute(grayLeft, grayRight)
        
        # DEPTH_VISUALIZATION_SCALE = 2048
        # Calculating disparity using the StereoBM algorithm
		# Note: Code returns a 16bit signed single channel image,
		# CV_16S containing a disparity map scaled by 16. Hence it 
		# is essential to convert it to CV_32F and scale it down 16 times.
        # Converting to float32 
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        cv2.imshow('depth', disparity )
        
        # Press 's' to save image
        if cv2.waitKey(30) == 115 :
            print('Saving left.png and right.png')
            cv2.imwrite('left.png',left_rect)
            cv2.imwrite('right.png',right_rect)
            break
        elif cv2.waitKey(30) >= 0 :
            break
    exit(0)

if __name__ == "__main__":
    main()