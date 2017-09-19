# **SDC-ND Term 1 Project 1: Finding Lane Lines on the Road**

## Objective
### The objective of this project is to implement a lane detection algorithm in Python and OpenCV. This writeup details the processing pipeline, analyzes the results, and reflects on areas for improvement.

---

### 1. Lane Detection Pipeline

My processing pipeline consists of 5 major processing blocks with sub-steps. A given image/video frame will be its input. The output will be an image/video frame of the original image with the a drawn overlay of the detected left lane and right lane.

Figure 1: Input Image
![Alt Text](https://github.com/itscherylanne/CarND-LaneLines-P1/blob/master/test_images/whiteCarLaneSwitch.jpg)


Figure 2: Output Image
![Alt Text](https://github.com/itscherylanne/CarND-LaneLines-P1/blob/master/test_images_output/whiteCarLaneSwitch.jpg)


The processing pipeline is outlined as follows:
* Yellow and White Lane Isolation
  * Color Space Conversion
  * Thresholding / Image Masking
* Edge Detection
  * Color to Grayscale Conversion
  * Gaussian Smoothing
  * Canny Edge detection
* Region of Interest Filtering
* Hough Line Transform
* Detection of Left and Right Lane
  * Sorting of Left and Right Lane
  * Filtering of detected lines
  * Calculated Average of Left and Right Lane

#### 1.1 Yellow and White Lane Isolation

This processing step was added after trying to accomplish the challenge problem at the end of the the assignment. Shadows and variations in illumination would adversely affect the edge detection of the lanes when feeding in the gray scaled image. The edge would not be detected when specific pixels were dark due to shadows. Thus, the image would need to be isolated using a different colorspace.

The color value of the yellow and white lane were analyzed in various colorspaces. RGB, LAB, HSV, HSL, and YCbCr were considered. From playing with various color spaces and hand-picking thresholds, the yellow lane was isolated in the HSV colorspace and the white lane was isolated in HLS colorspace.

The yellow lane was isolated in the HSV colorspace with the following ranges
* H: 10 to 50
* S: 0 to 255
* V:0 to 100

The yellow lane was isolated by the function isolate_yellow_lane():
```
def isolate_yellow_lane(image):
    cvt_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    isolate_image = np.copy(cvt_img)
    th_img = (image[:,:,0] < 10) \
                | (image[:,:,0] > 50) \
                | (image[:,:,1] < 100)
    isolate_image[th_img] = [0,0,0]    
    return isolate_image
```

The white lane was isolated in the HLS colorspace with the following ranges
* H: 0 to 255
* L: 220 to 255
* S: 0 to 255

The white lane was isolated by the function isolate_white_lane():
```
def isolate_white_lane(image):
    cvt_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)    
    isolate_image = np.copy(cvt_img)
    th_img = (image[:,:,1] < 220)
    isolate_image[th_img] = [0,0,0]    
    return isolate_image
```

The two isolated images were bit-wise OR-ed to create the image that will undergo the line detection.
```
yellow_lane = isolate_yellow_lane(image)
white_lane = isolate_white_lane(image)

#Combine detected Yellow and White Lane Images
color_select = cv2.bitwise_or(yellow_lane, white_lane)
```


#### 1.2 Edge Detection
The pipeline for edge detection did not change much from what was provided in lecture. I have found that a 1:3 ratio for the low and high hysteresis thresholds worked best for the Canny edge detection. The parameters I chose were:
* Low threshold: 50
* High threshold: 150

```
gray_img = grayscale(color_select)
blur_img = gaussian_blur (gray_img, 11)
edge_img = canny(blur_img, 50, 150)
```

#### 1.3 Region of Interest Filtering
The region of interest is defined as the area in the frame that is considered to be the driver's lane. In this project, the region of interest was a trapezoid that was scaled by the image size.

The region of interest was applied to the detected edge image to zero-out edge pixels outside of the of the lane.
```
#gather image information, define region of interest
imshape = image.shape
rows = imshape[0]
cols = imshape[1]
vertices = np.array([[(cols*.55,rows*.6),(cols*.45,rows*.6),(cols*.05,rows-1),(cols*.95,rows-1)]], dtype=np.int32)   

#zero out pixels outside of ROI
roi_img = region_of_interest(edge_img, vertices)
```

#### 1.4 Hough Line Transform
The Hough transform takes the filtered edge image and transforms it to parameter space. The parameter space is in polar coordinates which describe a straight line. The parameters that go into the Hough transform form the resolution of the detection and creates bounds on what is detected. The output of the Hough transform are detected line segments.
```
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 10     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments    

hough_img = hough_lines(roi_img, rho, theta, threshold, min_line_length, max_line_gap)
```


#### 1.5 Detection of Left and Right Lane
The output of hough transform is a list of vertices that represent the detected line segment. The list is analyzed and a final detected line was drawn for both the left lane and right lane. This process was done in the function draw_lines().


When analyzing the list, the slope of the lane was calculated to determine where it is the left lane or the right lane. The detected line was considered to be the left lane if it had a positive slope. The right lane was considered to have a negative slope.

Additional checks on the calculated slope were placed to filter out horizontal lines and completely vertical lines. A slope between 0.4 and 0.8 was determined to be the left lane. A slope between -1.0 and -0.6 was determined to be the right lane. The need for additional filtering was determined during the challenge problem.
```
for line in lines:
    for x1,y1,x2,y2 in line:
        if(x1 != x2):
            m = (y2-y1)/(x2-x1)
            if (m < -0.6) & (m > -1.0):
                right_lane_m.append(m)
                right_lane_b.append(y2 - m*x2)
                right_lane_b.append(y1 - m*x1)
            if (m < 0.8) & (m > 0.4):
                left_lane_m.append(m)
                left_lane_b.append(y2 - m*x2)
                left_lane_b.append(y1 - m*x1)   
```

The average lane was calculated and then drawn on the image:
```
if len(left_lane_m) > 0:
    left_m_avg = sum(left_lane_m)/len(left_lane_m)
    left_b_avg = sum(left_lane_b)/len(left_lane_b)
    left_y1 = rows-1
    left_x1 = int((left_y1 - left_b_avg)/ left_m_avg)
    left_y2 = int(rows*0.6)
    left_x2 = int((left_y2 - left_b_avg)/ left_m_avg)
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)

if len(right_lane_m) > 0:
    right_m_avg = sum(right_lane_m)/len(right_lane_m)
    right_b_avg = sum(right_lane_b)/len(right_lane_b)        
    right_y1 = rows-1
    right_x1 = int((right_y1 - right_b_avg)/ right_m_avg)
    right_y2 = int(rows*0.6)
    right_x2 = int((right_y2 - right_b_avg)/ right_m_avg)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
```

### 2. Analysis
When analyzing the test videos, the detected line would not coincide with the lane in the image in all frames.This is due to how the edge was detected in the image and the fact that the final detected line was based on the line information detected in the given frame.

There would be moments when the detected lane would appear to jiggle from frame to frame and at times intersect with the opposite lane line. This was due to inconsistent line detection for a particular lane. There were instances when the detection of the left side of the lane would not match the detection of the right side of the lane. This would create a bias towards the side of the lane that had more of its edge detected.

Another shortcoming with the algorithm is that it can only handle straight lanes. The Hough line transform is what constrains the problem to straight lines. When dealing with the test sets that have a bend in the road, the algorithm would detect the a line that would diverge tangentially to the bend in the road.

### 3. Reflection
This reflection is limited to improving the detection given only the image/video. It should be noted though that the detection can improve when fusing information from other sensors. Having more information about the environment to fuse with the detected image solution can only improve the detection.

The algorithm can improve by using time-history information on the detected line. Using a windowed average of the detected line over a few frames will reduce the amount of variation of the detected line between frames.

The Hough transform was specifically for straight lines. This transform falls short when there are bends on the road. It may be favorable to determine another parameter space that will accommodate curved lines.
