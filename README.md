# Image Stitching Project	
Final project for subject IMP301

Algorithm:
- Extract features using SIFT, SURF, ORB
- Matching features using BruteForce algorithm
- Find homography matrix using RANSAC algorithm
- Blending and warping image

GUI: 
- Tkinter library in python

The project is to implement a simple image stitching app, when we input a set of images with overlapped fields, we expect to obtain a wide seamless panorama.

## Note
- We have implemented the SIFT algorithm from scratch using Python, however to optimize the execution time in the app we use the API provided by the library.
- Please feel free to check out our implementation for SIFT by the following link [SIFT Implementation using Python](https://github.com/baophuoc1903/Image_Stitching?fbclid=IwAR0-wnj1MYw_xWGjXSaFMScA58ZWn5lynJu5vPBNcBhCLV8oXITzKUhzBQU)

## Dependency
- Numpy
- Matplotlib
- Pillow

To use all none-free features
- Python 3.7.1
- OpenCV 3.4.2

*downgrade opencv version by:*

```
pip install opencv-contrib-python==3.4.2.17 --force-reinstall
```

To run by newest version (some features doesn't work like SIFT,...)
- Python 3.9
- OpenCV 4.5.3

## Usage
- You can directly test project by downloading following .exe file
[Stitching.exe](https://drive.google.com/file/d/1xMpF6NBg3uo-A7PmVw7AO4m-rIYnkiqi/view?usp=sharing)
- Run main.py or Stitching.exe file and follow the guideline in app by clicking guideline button

<img src="https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/GUI.jpg">

## Sample (Step by step)

## Input images
<img src="https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/data/AI1502/2_new.jpg" >   <img src="https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/data/AI1502/3_new.jpg" > 

## Matching
![matching](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Step%20by%20step/matching.jpg)

## Blending
![left](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Step%20by%20step/blending1.jpg)

![right](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Step%20by%20step/blending2.jpg)

![no blending](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Step%20by%20step/blendingTwo.jpg)

![after blending](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Step%20by%20step/resultTwo.jpg)


## Other examples

Result for 4 pics taken from 5th floor FPT University:
![Dom A](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Result%20multiple%20stitching/result_domA.jpg)

Result for 3 pics taken from classroom at FPT University:
![FPT](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Result%20multiple%20stitching/result_AI1502.jpg)

Result for 6 pics taken from Quy Nhon Beach:
![QuyNhonBeach](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Result%20multiple%20stitching/result_QuyNhonBeach.jpg)

Result for 10 pics taken from Quy Nhon city:
![QuyNhonBeach](https://github.com/AnhDuy26/Image-Stitching-Project/blob/master/Images/Result%20multiple%20stitching/result_QuyNhonCity.jpg)
