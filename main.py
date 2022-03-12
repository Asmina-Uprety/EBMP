#opening webcam ko code
from tensorflow.keras.models import model_from_json
import cv2
import sys, scipy, numpy as np; print(scipy.__version__, np.__version__, sys.version_info)
from tkinter import *

# music player function


emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

#pathf = 'C:\\Users\\asmin\\anaconda3\\envs\\gputest\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'
# start the webcam feed
video = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = video.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        x1,y1=x + w, y + h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.line(frame, (x, y), (x + 30, y), (255, 0, 255), 6)  # Top left
        cv2.line(frame, (x, y), (x, y + 30), (255, 0, 255), 6)

        cv2.line(frame, (x1, y), (x1 - 30, y), (255, 0, 255), 6)  # Top right
        cv2.line(frame, (x1, y), (x1, y + 30), (255, 0, 255), 6)

        cv2.line(frame, (x, y1), (x + 30, y1), (255, 0, 255), 6)  # Bottom left
        cv2.line(frame, (x, y1), (x, y1 - 30), (255, 0, 255), 6)

        cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 0, 255), 6)  # Bottom left
        cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 0, 255), 6)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
