# Detection of Vehicles on the Same Lane

Course: Advanced Topics in Digital Image Processing

Academic Year: 2020/21

Semester: 2nd

Grade: 19 out of 20

Technologies Used: Python, OpenCV, OpenCL

Brief Description: Development of an application, that from a video and making use of the GPU for image processing, finds lanes and vehicles, and mark in
red the vehicles that are in our lane and in green the others.

The system architecture is composed by 3 image processing methods:

- Canny Edge Detection for lane enchancement (OpenCL);
- Hough Transform for lane lines selection (OpenCL);
- Viola-Jones Algorithm for vehicle detection (OpenCV).

Algorithm Evolution
![plot](images\algorithm_evolution.png)

Demo
![plot](images\demo.png)
