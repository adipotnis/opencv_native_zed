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
- A Viewer program to view disparity maps generated by OpenCV's stereo_BM file.
- Allows view of the left cam output and the disparity map output.
- Disparity output can be converted to depth map by using baseline and focal length.
- Press 'esc' key to exit. 
- Press 's' key to save png image file with names left.png and right.png.

Run using 

```
python zed_opencv_native2.py
```

