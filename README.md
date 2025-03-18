Because of the number of images of the data set, it was not posible to push it to this repository, here is the dataset: https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd

📌 Required Libraries:
OpenCV (cv2) → For video capture and processing
NumPy (numpy) → For handling arrays and numerical processing
TensorFlow (tensorflow) → For loading and running the drowsiness detection model
Time (time) → Built into Python, no installation required
📌 Command to Install Dependencies:
Run this in the terminal or within a virtual environment:
 pip install opencv-python numpy tensorflow
If you're using OpenCV in an environment without GUI support and face issues, try this version:
 pip install opencv-python-headless
This installs OpenCV without GUI support, which is useful for running the code on a server.

📌 Installation Check:
After installing, run this in Python to verify everything is set up correctly:
import cv2
import numpy as np
import tensorflow as tf

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)
If all versions print without errors, you're good to go!
