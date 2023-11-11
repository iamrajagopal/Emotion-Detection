# Emotion-Detection
This project aims to detect the emotion of a person in real-time using Convolutional Neural Networks (CNN) and computer vision techniques. The developed system utilizes the Haar cascade frontal face detection algorithm to detect faces in images and then predicts the emotion associated with each detected face. Website link https://faceexpressiondetection.azurewebsites.net/

![image](https://github.com/Rohith766/Emotion-Detection/assets/92597090/4c9c309b-2793-4ae5-a23b-9a15cbca86a8)
 
# Dataset
For training and evaluating the emotion detection model, the "Face Expression Recognition Dataset" from Kaggle was used. 
The dataset can be found at the following link: Face Expression Recognition Dataset(https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).

The dataset consists of images labeled with seven different facial expressions, including:

Angry,
Disgust,
Fear,
Happy,
Sad,
Surprise,
Neutral.

These images were used to train and validate the CNN model for emotion detection.
# System Overview
The system's workflow for emotion detection can be summarized as follows:

Face Detection: The Haar cascade frontal face detection algorithm is applied to identify and locate faces in an image or video stream.

Preprocessing: Detected faces are preprocessed, which involves resizing, normalization, or other techniques to prepare them for input to the CNN model.

Emotion Prediction: The preprocessed face images are passed through the trained CNN model, which predicts the emotion associated with each face.

Visualization: The system may display the detected faces along with their corresponding predicted emotions in real-time or as output on saved images or videos.
# Accuracy 
The trained emotion detection model achieved an accuracy of approximately 70% during testing and evaluation. It is important to note that accuracy can vary based on factors such as dataset quality, model architecture, and training techniques.
