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
![algorithm_evolution](https://user-images.githubusercontent.com/46992334/192883616-f2c39bc0-7a17-4a91-9588-0e49ecf32f1c.png)

Demo
![demo](https://user-images.githubusercontent.com/46992334/192883218-af0e6089-5dab-4fa7-bd94-5276c680daf7.jpg)
