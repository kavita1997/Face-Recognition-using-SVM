# Face-Recognition-using-SVM
# What is Face Recognition?
A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source.
There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database. 
It is also described as a Biometric Artificial Intelligence based application that can uniquely identify a person by analyzing patterns based on the person's facial textures and shape.

# Installation required:

•	OpenCV

•	Numpy

•	Matplotlib

•	os

•	Sklearn

To create a complete project on Face Recognition, we must work on 3 very distinct phases
1.	Face Detection and Data Gathering
2.	Train the dataset
3.	Face Recognition

# Face Detection

The above code will capture the video stream that will be generate by your webcam. 
The most basic task on Face Recognition is “Face Detection”. The most common way to detect a
Face (or any objects) is using “Haar Cascade Classifier”. In this project we use Haar Cascaded
Frontal Face.xml file.

Step 1: Open the camera or webcam

Step 2: Load the Haar cascaded frontal face.xml file

Step 3: Make the directory to save the image sample for each person

Step 4: Resize the frame

Step 5: Take the sample 

Step 6: Store or save the data in the directory


In detection it takes the 100 samples of particular person and saved in a directory. Through this we simply create a dataset, where we will store for each id, a group of photos in gray with the portion that was used for face detection.

# Train the data

For training we use different models in image dataset. These are

•	SVM 

•	CNN and

•	Neural Network

All the models will train the data and model will save in pikel file which will help in face recognition.

Step 1: Splits the dataset in train and test data 

Step 2: Build the model

Step 3: Extract the feature

Step 3: Fit the train data in the model

Step 4: Predict the test data

Step 5: Calculate the Accuracy 

Step 6: Dump the trained model

# Face Recognition

Now, we reached the final phase of our project. Here we will capture a fresh face on our camera and if this person had his face captured and trains before, our recognizer will make a “prediction” returning its id and index, shown how confident the recognizer is with this match.

Step 1: Load the trained model or pickle file

Step 2: Give the input

Step3: Recognize the person





 




