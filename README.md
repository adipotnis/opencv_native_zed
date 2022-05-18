# opencv_native_zed
Modified version of opencv native zed 


## 1) zed_mini.yaml

Contains important parameter values and serial number and config file addresses

```
serial_number: 14242888
config_path: '/Library/code'
minDisparity: 0
numDisparities: 128
blockSize: 35
speckleRange: 16
speckleWindowSize: 25
output_file: output.mp4
```

## 2) zed_opencv_native2.py
-	A Viewer program to view disparity maps generated by OpenCV's stereo_BM file.
-	Allows view of the left cam output and the disparity map output.
-	The disparity map is generated is using stereo_BM which has lower accuracy.
-	Disparity output can be converted to depth map by using baseline and focal length.
-	Press 'esc' key to exit.
-	Press 's' key to save png image file with names left.png and right.png.

Run using 

```
python zed_opencv_native2.py
```

## 3) depth_still.py
### Input 
- Takes zed_mini.yaml as input.
- Requires Camera calibration .conf files for image rectification.

### Output

- Displays images of: 
1. Left camera image
2. Disparity image
3. Depth image
4. Depth plot in matplotlib

- Output files are: 
1. left_gray.png
2. disparity.png
3. depth.png
- Object distance calculated is shown on console
```
Distance at image centre:  89.91579 cm
```
