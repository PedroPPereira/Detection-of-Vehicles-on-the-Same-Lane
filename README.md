# Detection of Vehicles on the Same Lane

**Course:** Advanced Topics in Digital Image Processing

**Specialty Area:** Digital Systems

**Academic Year:** 2020/21

**Semester:** 2nd

**Grade:** 19 out of 20

**Technologies Used:** Python, C, OpenCV, OpenCL

**Brief Description:** Development of an application, that from a video and making use of the GPU for image processing, finds lanes and vehicles, and marks in
red the vehicles that are in our lane and in green the other vehicles outside our lane.

**System Architecture:**
- Canny Edge Detection for lane enchancement (OpenCL)
- Hough Transform to find and select the lane lines (OpenCL)
- Viola-Jones Algorithm and Cascade Classifiers for vehicle detection (OpenCV)

---

### Algorithm Evolution

![algorithm_evolution](https://user-images.githubusercontent.com/46992334/192883616-f2c39bc0-7a17-4a91-9588-0e49ecf32f1c.png)
1) Original image
2) Greyscale operation
3) Sobel operation
4) Canny Edge Detection
5) Hough Transform
6) Viola-Jones Algorithm
---

### Video Snippet Demo [[link](https://drive.google.com/file/d/1JZ5Q1saFxpLOkcW3tXRiY8I3pswUT4Tz/view?usp=drive_link)]

![demo](https://user-images.githubusercontent.com/46992334/192883218-af0e6089-5dab-4fa7-bd94-5276c680daf7.jpg)
