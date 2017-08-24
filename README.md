# Advanced_Lane_Lines_Project4
In this project, computer vision techniques are used to accurately detect lanes in a video. 
The video processing pipeline ``process_vid()``, consists of the following steps:
  1. Receiving input frame as an RGB image
  2. Undistorting the image, using the calibration matrix ``mtx``
  3. Applying Multiple threshholds to the undistorted image, including:
      * Gradient: Sobel in both X and Y directions
      * Color: Using L component of LUV colorspace, and B component of LAB colorspace
  4. Warping the image using the Transform Matrix ``M`` into a bird's eye view
  5. Running ``detect_lanes()`` and ``look_ahead()`` functions to detect the lane lines
  6. Smoothing the lanes detected along with the last ``n`` lane detects
  7. Unwarping the detected lanes onto the image and returning the original image along with:
      * The Lane Covered in Green
      * Left Lane highlighted in Blue
      * Right Lane highlighted in Red
      * Lane's Radius of Curvature in meters
      * Car's absolute offset from the center of the lane in meters
      
##### Check this [video](./project_output.mp4) to preview the pipeline's output
##### Here is a [youtube link](https://youtu.be/3YX-kcZqPTE) for my video to stream it online.

*Please check the [writeup report](./writeup_report.md) for further details*
*Also check my implementation contained in this [IPython Notebook](./Advanced_LaneFinding_Project-Process-Notebook.ipynb)
