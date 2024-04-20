# Course Attendance System
<div align="center">
    <a><img width="720" src="FD_FR_1.jpg" alt="soft"></a>
</div>
Say goodbye to tedious roll calls! This project introduces a comprehensive system for managing attendance, harnessing facial detection and recognition technologies to identify individual students and register their attendance. Developed with Python, RetinaFace, Face_Recognition, and OpenCV, the system offers an efficient and automated solution for monitoring attendance across diverse settings such as educational institutions and workplaces.

Users can upload their images into the system's database, which is then utilized for facial recognition during attendance checks. Recognized faces are cross-referenced with the database, and attendance records are instantly updated in real time. 

This project serves as a compelling demonstration of how computer vision and machine learning can modernize traditional processes, improving efficiency and accuracy.

## Table of Contents

- [Course Attendance System with Facial Recognition](#course-attendance-system-with-facial-recognition)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
      - [Project Description](#project-description)
      - [Website Screenshots](#website-screenshots)
  - [Datasets](#datasets)
      - [Raw Images Data Collection](#raw-images-data-collection)
      - [Raw Screenshots](#raw-screenshots)
      - [Processed Images Summary](#processed-images-summary)
      - [Processed Screenshots](#processed-screenshots)
  - [Method](#method)
      - [RetinaFace](#retinaface)
      - [Retinaface Output Screenshots](#retinaface-output-screenshots)
      - [Face Recognition](#face-recognition)
      - [Face Recognition Output Screenshots](#face-recognition-output-screenshots)
  - [Results](#results)
  - [Technical Information](#technical-information)
  - [Benefits](#benefits)
  - [Applications](#applications)
  - [Future Improvements](#future-improvements)
  - [Citations](#citations)
   
## Introduction

### Project Description
In conventional attendance systems, the manual process of marking attendance is often time-consuming and error-prone. However, with advancements in machine learning and computer vision, we now can automate this process, enhancing efficiency and accuracy.

Introducing our Face Recognition Attendance System, specifically crafted to capitalize on these technologies, delivering a smooth and automated attendance tracking solution. Utilizing face recognition technology, the system effortlessly identifies individuals and records their attendance, eliminating the need for manual input and mitigating the risk of errors or unauthorized entries.

Developed using Python, RetinaFace, Face_Recognition, and OpenCV, our system integrates these tools seamlessly. Python provides the backend operations, OpenCV (Open Source Computer Vision Library), a popular computer vision library, to capture and preprocess images for face recognition, retina face facilitates face detection and Python library face_recognition which is built on top of Dlib provides a simple API for face recognition tasks.

Whether you're an educational institution seeking to streamline attendance management or a corporation aiming to automate employee check-ins, our Face Recognition Attendance System offers a dependable and effective solution.

### Website Screenshots

<figure align="center"> 
  <img src="docs/images/main_page.png" alt="drawing" height="400"/>
  <figcaption>Home Page of the Interface</figcaption>
</figure>


## Datasets:

### Raw Images Data Collection:
To assess the efficacy of the proposed system, a dataset containing thirty-six students was compiled. Facial images of volunteer students are captured using a laptop web camera and saved in a folder. The dataset utilized in this investigation encompasses 100 images for a sample size of thirty-six (36) students totaling 3,600 images, exhibiting a variety of poses, backgrounds, lighting conditions, and facial expressions. The images were standardized to dimensions of 112 pixels in height and 112 pixels in width. Additionally, 69 class group photographs were captured to evaluate the individual recognition of students' faces within group settings.

### Raw Screenshots:
<figure align="center"> 
  <img src="docs/images/main_page.png" alt="drawing" height="400"/>
  <figcaption>Home Page of the Interface</figcaption>
</figure>

### Processed Images Summary:


### Processed Screenshots:
<figure align="center"> 
  <img src="docs/images/main_page.png" alt="drawing" height="400"/>
  <figcaption>Home Page of the Interface</figcaption>
</figure>

## Method:

### RetinaFace:
Retina Face is the state-of-the-art model for facial detection developed as a part of the InsightFace Project. Author Jiankang Deng et al. published a paper in 2019 titled “RetinaFace: Single-stage Dense Face Localisation in the Wild”. It is a deep learning-based cutting-edge facial detector for Python coming with facial landmarks and its detection performance is amazing.

RetinaFace is the face detection module of [insightface](https://github.com/deepinsight/insightface) project. The original implementation is mainly based on mxnet. Then, its tensorflow-based [re-implementation](https://github.com/StanislasBertrand/RetinaFace-tf2) is published by [Stanislas Bertrand](https://github.com/StanislasBertrand). So, this repo is heavily inspired by the study of Stanislas Bertrand. Its source code is simplified and it is transformed to pip compatible but the main structure of the reference model and its pre-trained weights are the same.

**Installation** [![PyPI](https://img.shields.io/pypi/v/retina-face.svg)](https://pypi.org/project/retina-face/) [![Conda](https://img.shields.io/conda/vn/conda-forge/retina-face.svg)](https://anaconda.org/conda-forge/retina-face)

The easiest way to install Retina Face is to download it from [PyPI](https://pypi.org/project/retina-face/). It's going to install the library itself and its prerequisites as well.

```shell
$ pip install retina-face
```

RetinaFace is also available at [`Conda`](https://anaconda.org/conda-forge/retina-face). You can alternatively install the package via conda.

```shell
$ conda install -c conda-forge retina-face
```

Then, you will be able to import the library and use its functionalities.

```python
from retinaface import RetinaFace
```

**Face Detection** 

RetinaFace offers a face-detection function. It expects an exact path of an image as input.

```python
resp = RetinaFace.detect_faces("img1.jpg")
```

Then, it will return the facial area coordinates and some landmarks (eyes, nose, and mouth) with a confidence score.

```json
{
    "face_1": {
        "score": 0.9993440508842468,
        "facial_area": [155, 81, 434, 443],
        "landmarks": {
          "right_eye": [257.82974, 209.64787],
          "left_eye": [374.93427, 251.78687],
          "nose": [303.4773, 299.91144],
          "mouth_right": [228.37329, 338.73193],
          "mouth_left": [320.21982, 374.58798]
        }
  }
}
```

### Retinaface Output Screenshots:
<figure align="center"> 
  <img src="Results Images/download_10.png" alt="drawing" height="720"/>
  <figcaption>Classroom Face Detection using RetinaFace</figcaption>
</figure>


### Face Recognition:
To encode the face from the image we used the Python library face_recognition which is built on top of [Dlib](http://dlib.net/) state-of-the-art face recognition built with deep learning and provides a simple API for face recognition tasks. This also provides a simple `face_recognition` command line tool that lets
you do face recognition on a folder of images from the command line!

**Python Module**:
- You can import the `face_recognition` module and then easily manipulate faces with just a couple of lines of code. It's super easy!
- API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

**Automatically find all the faces in an image**:

```python
import face_recognition

image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the coordinates of each face!
```

### Face Recognition Output Screenshots:
<figure align="center"> 
  <img src="Results Images/15_predicted.jpg" alt="drawing" height="720"/>
  <figcaption>Classroom Face Recognition using face_recognition</figcaption>
</figure>



## Results:

## Facial Detection and Recognition Results

|    Detection       |     Recognition     |  Image Size | Cropped Embeddings | Detection Accuracy (%) | Recognition Accuracy (%) | Recognition on Group Image |
|:-----------------: |:-------------------:|:-----------:|:------------------:|:----------------------:|:------------------------:|:--------------------------:|
| RetinaFace         |     Face Recognition    |    50x50    |         Yes    |          99.4           |           90.3          |            Yes             |
| RetinaFace         |     Face Recognition    |    50x50    |         No     |          99.4           |           88.3          |            Yes             |
| RetinaFace         |     Face Recognition    |   112x112   |         No     |          99.2           |           85.3          |            Yes             |
| RetinaFace         |     Face Recognition    |   600x600   |         No     |          98.6           |           85.1          |            Yes             |
| Face Recognition   |     Face Recognition    |    50x50    |         No     |          65.5           |           72.3          |            Yes             |
| Face Recognition   |     Face Recognition    |   112x112   |         No     |          65.5           |           70.5          |            Yes             |
| Yolo9              |     Face Recognition    |   112x112   |         No     |          51.7           |           58.4          |            Yes             |
| RetinaFace         | haarcascade_frontalface |    50x50    |         No     |          99.4           |           45.2          |            Yes             |
| Dlib               | Face Recognition        |    112x112  |         No     |          96.0           |           92.0          |            Yes             |
| Dlib               | Face Recognition        |    112x112  |         No     |          91.0           |           86.0          |            Yes             |
| RetinaFace         | OpenCV-LBPH        |    Extracted Images  |     No     |          100            |           91.0          |            Yes             |
| InsightFace        | InsightFace        |    1200x1600  |            No     |          99.0           |           91.0          |            Yes             |


## Technical Information:

- **Programming Language**: Python
- **Face Detection**: RetinaFace
- **Face Recognition**:  World's simplest face recognition library (face_recognition) to recognize and manipulate faces from Python.
- **Computer Vision Library**: OpenCV

## Benefits:

- **Effortless Efficiency**: Automates attendance, saving time and resources.
- **Increased Accuracy**: Reduces human error compared to manual roll calls.
- **Enhanced Convenience**: Provides a faster and more user-friendly approach for everyone.
- **Flexible Scalability**: Adapts to accommodate different group sizes.

## Applications:

- **Educational Institutions**: Streamlines attendance management and ensures accurate student records.
- **Workplaces**: Simplifies employee attendance tracking and supports flexible work arrangements.
- **Access Control Systems**: Offers an additional layer of security by integrating face recognition for entry.


## Future Improvements:

In the foreseeable future, there are numerous avenues for refining and broadening the system:

- **Smoother User Experience**:
    - **Visually Appealing Interface**: A more engaging and user-friendly interface is in the works to make interacting with the system a breeze.
    - **Data First, Capture Later**: We're refining the image capture process to ensure all necessary information is entered before an image is added to the database.
    - **Smarter ID Assignment**: Say goodbye to manual counting! We'll optimize ID allocation to automatically fill in missing gaps (e.g., if the IDs are 8001,8002,8003,8004,[ ],8006,8007,8008, the new image's ID will be 8005).

- **Beyond the Local Machine**:
    - **Deployment on Cloud Platforms**: We're aiming to make the system accessible from anywhere by deploying it on platforms like Google Cloud, Azure, AWS, and Heroku freeing it from the limitations of a local machine.

- **Security Boost**:
    - **Enhanced Security**: We're committed to safeguarding your data! We'll implement more robust methods for data handling and user authentication.

- **Expanded Functionality:**:
    - **Student Login**: Students will soon be able to log in using passwords, offering them greater control over their attendance data.
    - **Dedicated Teacher Database**: A separate database for teachers is on the horizon. Teachers will log in using usernames and passwords for secure access.
    - **Teacher-Centric View**: Logged-in teachers will have the power to view student attendance reports specific to the classes they teach.

- **Advanced Features (Future Explorations)**:
    - **Improved Error Handling and User Feedback**: Refining error handling and user feedback will make the system more user-friendly and robust.
    - **Real-time Attendance Updates**: We're exploring implementing real-time updates, so attendance is reflected instantly upon face recognition.
    - **Integration with Other Systems**: Seamless integration with existing systems like learning management systems or student information systems is a potential future endeavor.


## Citations:
[Face Detection and Face Recognition Image](https://www.google.com/url?sa=i&url=https%3A%2F%2Fbrainalyst.in%2Fface-detection-and-face-recognition%2F&psig=AOvVaw3eSregE6CKGlWYOqkyp3Gm&ust=1713629119953000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqGAoTCNjwq7LUzoUDFQAAAAAdAAAAABCBAQ)




















