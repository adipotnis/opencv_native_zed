
import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt


def main():

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

            # image_files
            left_name = yaml_dict.get('left_name')
            right_name = yaml_dict.get('right_name')

        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    imgL = cv2.imread(left_name,0)
    imgR = cv2.imread(right_name,0)

    # Create stereo object
    stereoMatcher = cv2.StereoSGBM_create()

    # Setting the parameters to required values
    stereoMatcher.setMinDisparity(minDisparity)
    stereoMatcher.setNumDisparities(numDisparities)    # multiple of 16
    stereoMatcher.setBlockSize(blockSize) # odd numbers
    stereoMatcher.setSpeckleRange(speckleRange)
    stereoMatcher.setSpeckleWindowSize(speckleWindowSize)

    disparity = stereoMatcher.compute(imgL,imgR)

    # Calculating disparity using the StereoBM algorithm
    # Note: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
    # Converting to float32 

    disparity = disparity.astype(np.float32)
    disparity = cv2.medianBlur(disparity,5)     # medianBlur of the disparity image
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities

    # Measurements
    # Calculation 
    # 35 cm --> 1/disparity value =  1.0655568 (value found for depth by using k_depth as 1)
    # 35 = k_depth /disparity = k_depth * 1.0655568
    # k_depth = 35/ 1.0655568
    # Current Setup 15 May 2022 (constant of proportionality)
    k_depth = 35/ 1.0655568     

    max_val_dis = disparity.max()   # Threshold normalized values below 0
    ret, thresh = cv2.threshold(disparity,0,max_val_dis,cv2.THRESH_TOZERO)

    depth = np.divide(k_depth,thresh, where=thresh!=0)  # Depth map from disparity map using constan of proportionality, skipping zero values
    max_depth = depth.max()
    depth = np.where(depth<=0, max_depth, depth)        # Replacing values which have zero disparity i.e. inf distance with max distance value

    depth = cv2.medianBlur(depth, 5) 

    center = [int(depth.shape[0]/2), int(depth.shape[1]/2)]
    print('Distance at image centre: ', np.average(depth[center[0],center[1]]), 'cm')

    # Images of plots
    cv2.imshow('depth', depth/512)
    cv2.imwrite('depth.png', depth)

    cv2.imshow('left', imgL)
    cv2.imwrite('left_gray.png', imgL)

    cv2.imshow('disparity', disparity)
    cv2.imwrite('disparity.png', disparity * 1024)

    plt.title('depth')

    plt.imshow(depth, 'rainbow')    # Interactive plot
    plt.show()

if __name__ == "__main__":
    main()
# Measurements
# Calculation 
# 35 cm --> 1/disparity value =  1.0655568 (value found for depth by using k_depth as 1)
# 35 = k_depth /disparity = k_depth * 1.0655568
# k_depth = 35/ 1.0655568   (replace value of k_depth here with 1)
# Current Setup 15 May 2022 (constant of proportionality)

# Real    Measured
# 35 cm -> 34.996 cm
# 40 cm -> 39.734196 cm
# 45 cm -> 44.84666 cm
# 50 cm -> 49.645752 cm
# 55 cm -> 54.824768 cm
# 60 cm ->  59.95543 cm
# 65 cm -> 64.99516 cm
# 70 cm -> 70.2192 cm
# 75 cm -> 75.33034 cm
# 80 cm -> 80.37036 cm
# 85 cm -> 85.80356 cm
# 90 cm -> 89.69332 cm
# 95 cm -> 94.215675 cm
# 100 cm -> 99.511826 cm

