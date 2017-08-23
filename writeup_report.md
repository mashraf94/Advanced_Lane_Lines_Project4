## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Matrices Calculation 
### 1. Camera Calibration
The code for this step is contained in the first code cell of the IPython Notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

To calibrate the camera, the calibration images found at [calib_imgs](./calib_imgs) folder, representing a 9x6 Chessboard:
*The following procedures acquired the calibration matrix* ``mtx`` *which was used to undistort any image took by the car's camera.*
* The function `calibrate()` is used to execute the following steps:
1. Determining Object Points of (x, y, z) coordinates of the chessboard corners in real world.
    * The chessboard is assumed to be fixed on the (x, y) plane, hence z = 0
    * The same Object Points were assigned for each calibration image, thus `objpoints` is just a replicated array of `objp`
2. Converting each image to grayscale, to prepare it for Corner Detection.    
3. Using OpenCV function `cv2.findChessboardCorners()` to detect corners on the 9x6 chessboard
4. Each time all chessboard corners are successfully detected:
    1. Append the detected corners to `imgpoints`
    2. Append the a replica of `objp` to `objpoints`
5. Use OpenCV function `cv2.calibrateCamera()` to acquire the calibration matrix `mtx` and distortion coefficients `dist`
6. Apply the acquired distortion correction parameters to input images using the `cv2.undistort()` function

The following figure represents the undistortion applied to 2 chessboard images:
* Row1: The Chessboard's corners were detected, drawn onto the image, and is then undistorted.
* Row2: The Chessboard's corners aren't all included in the image, hence it's corners weren't detected, nonetheless was undistorted using `mtx` and `dist`

<p align="center">
<img align="center" src="./writup_imgs/calibration.png" alt="alt text">
</p>

### 2. Transformation Matrix
The code for this step is contained in the first code cell of the IPython Notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`). 

To accurately detect lane line's we need to look at the road with a **Bird's Eye View**, through applying a perspective transform to the image, to view the lanes as parallel lines and have the ability to detect and calculate curvature.

*The following procedures acquired the transformation matrix* `M` *&* `Minv` *which was used to warp any image to a Bird's Eye Perspective, and back to the original perspective later.*
The function `transform_matrix()` is used to execute the following steps:
1. Source Points `src` were chosen from a sample image, outlining the lanes in this image.
2. Destination Points `dst` were determined as an erect rectangle, to transfrom `src` onto `dst` achieving the bird's eye perspective.
* Source and Destination Points are Calculated Below:
```python
## Source Points
## Start: Bottom Left Corner -- ClockWise Rotation
src = np.float32([[244.515,685.472],
                  [575.507,462.495],
                  [706.532,462.456],
                  [1061.62,685.42]])

offset = 200

dst = np.float32([[offset,img.shape[0]],
                  [offset,0],
                  [img.shape[1]-offset,0],
                  [img.shape[1]-offset,img.shape[0]]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 244.5 , 685.5     | 200 , 720        | 
| 575.5 , 462.5     | 200 , 0      |
| 706.5 , 462.5     | 1080 , 0      |
| 1061.6 , 685.4      | 1080 , 720        |

3. Using OpenCV function `cv2.getPerspectiveTransform` we acquire transformation matrix `M` to warp images into bird's eye view.
4. Through flipping the `src` and `dst` points in `cv2.getPerspectiveTransform` we get the tranformation matrix `Minv` to unwrap the wrapped images back to it's original state.
* These images represent the `src` points on the Original Image and `dst` points on Warped Image
<p align="center">
<img align="center" src="./writup_imgs/points_warp.png" alt="alt text">
</p>

## Single Image Processing Pipeline 
*The code for this step is contained in the first code cell of the IPython Notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).*

This pipeline `process_img()` is set with threshholds to detect lanes within an image in the RGB color space. If the image was read with `cv2.imread()`, or is in the BGR color space, it is transformed to RGB using the BGR parameter. `BGR=True`
### 1. Remove Image Distortion `cv2.undistort()`

Using OpenCV function `cv2.undistort()`, along with the calculated calibration matrix `mtx`, and distortion parameters `dist`. Input images into the pipeline are undistorted:
<p align="center">
<img align="center" src="./writup_imgs/distortion_correction.png" alt="alt text">
</p>

### 2. Threshholded Binary Image

#### To detect the lane lines in an image, several threshholds were applied separately on the image:
##### 1. Color Threshholds:
Multiple experiments with all the components within different color spaces were applied to different images under changeable lighting and shadowing, including: HSV, HLS, LUV, LAB, YUV, RGB and GrayScale. These threshholds were implemented using the `color_threshhold()` function. *The following proved to be most efficient*
1. LUV Color Space: The L Component proved to be most efficient in detecting the right lane 
   * @threshhold: 210 <= L < 255
2. LAB Color Space: The B Component proved to be most efficient in detecting the left lane
   * @threshhold: 150 <= B < 255
> Since, both detected separate lanes, and were set at appropriate threshholds to produce minimal noise.
>> Therefore, both color components were executed and combined using the OR `|` operator to achieve a complete layover of both binary images.
<p align="center">
<img align="center" src="./writup_imgs/threshholds_color.png" alt="alt text">
</p>

##### 2. Gradient Threshholds:
The Sobel Operator was executed with different calculations to get the gradient in the X and Y direction; `sobel_abs()`, calculating the magnitute of the X and Y outputs; `sobel_mag()`, and the direction of the gradients; `sobel_dir()`. These gradients were introduced to different images, within several threshholds, and these outcomes were concluded:
1. The Gradient in the X direction was the most efficient in calculating the vertical lines, but it suffered a lot of noise under shadows
   * @threshhold 20 <= sobelX < 255, 3x3 kernel
2. The Gradient in the Y direction detected the lanes too (*Specially the right lane*) but also suffered a lot of noise.
   * @threshhold 30 <= sobelY < 255, 3x3 kernel 
> Since, The noises suffered in the X and Y directions were different yet both detected the lanes fairly.
>> Therefore, both gradients were executed and combined using the AND `&` operator to remove the noise from both and still detect the lanes, assisting the color threshholding techniques towards an accurate detection.
<p align="center">
<img align="center" src="./writup_imgs/threshholds_sobel.png" alt="alt text">
</p>

##### 3. Results
Using the discussed color and gradient thresholds a binary image was created using the OR `|` operator, creating the highest detection of lane lines achieved (thresholding steps at lines # through # in `another_file.py`). 

* The following image, represents, the combined result of the color and gradient threshholding techniques.
* *Note: It's clear in the Sobel Combined binary image, how the AND `&` operator fairly removes the noise*
<p align="center">
<img align="center" src="./writup_imgs/threshholds_summary.png" alt="alt text">
</p>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
