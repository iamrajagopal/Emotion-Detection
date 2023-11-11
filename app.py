import os
from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model("model.h5")

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image = request.files['image']
    
    # Save the image file in the static folder
    image_path = os.path.join(app.root_path, 'static', "temp.jpg")
    image.save(image_path)
    
    # Load the image and convert it to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each face detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_img = gray[y:y+w, x:x+h]  # Extract the face ROI
            resized_img = cv2.resize(face_img, (48, 48))  # Resize to match model input size
            normalized_img = resized_img / 255.0  # Normalize the image
            reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))  # Reshape for model input
            
            # Predict the emotion
            emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion_probs = model.predict(reshaped_img)
            emotion_index = np.argmax(emotion_probs)
            emotion = emotion_labels[emotion_index]
            
            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the result image
        result_path = os.path.join(app.root_path, 'static', "result.jpg")
        cv2.imwrite(result_path, img)
        
        # Render the result template with the result image and predicted expression
        return render_template('result.html', result_image="result.jpg", predicted_expression=emotion)
    else:
        # Render the result template with the input image and "No Expression Detected" message
        return render_template('result.html', result_image="temp.jpg", predicted_expression="No Expression Detected")

if __name__ == '__main__':
    app.run()
