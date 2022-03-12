from flask import Flask, render_template,Response,request
import numpy as np
import cv2
import os
import time
from pathlib import Path
from tkinter import *
from tensorflow.keras.models import model_from_json

camera = cv2.VideoCapture(0)
app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(),
    mimetype = 'multipart/x-mixed-replace; boundary=frame')


# music player function
def music_player(emotion_str: object) -> object:
    from musicplayer2 import MusicPlayer
    root = Tk()
    print('\nPlaying ' + emotion_str + ' songs')
    MusicPlayer(root, emotion_str)
    root.mainloop()
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)

# load weights into new model
model1.load_weights("model/emotion_model.h5")

print('\n Welcome to Music Player based on Facial Emotion Recognition \n')
print('\n Press \'q\' to exit the music player \n')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# File to append the emotions

def gen():
    with open(str(Path.cwd()) + "\emotion.txt", "w") as emotion_file:
        # start the webcam feed
        # cap = cv2.VideoCapture(0)
        now = time.time()  ###For calculate seconds of video
        future = now + 10

        while True:
            success,frame=camera.read()
            if not success:
                break
            else:
                facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model1.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255),
                                2, cv2.LINE_AA)
                    text = emotion_dict[maxindex]
                    emotion_file.write(emotion_dict[maxindex] + "\n")
                    emotion_file.flush()

                ret, buffer = cv2.imencode('.jpg', frame)
                frame=buffer.tobytes()
            if time.time() > future:  ##after 10 second music will play
                cv2.destroyAllWindows()
                music_player(text)
                future = time.time() + 10
            yield(b'--frame\r\n'
                b'Content_Type: image/jpeg\r\n\r\n' + frame +
                b'\r\n\r\n') #streaming video with flask
@app.route('/after', methods=['GET' , 'POST'])
def after():
    with open(str(Path.cwd()) + "\emotion.txt", "w") as emotion_file:
        img = request.files['file1']
        img.save('static/file.jpg')

        img1 = cv2.imread('static/file.jpg')
        gray_frame = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in num_faces:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped = img1[y:y + h, x:x + w]

        cv2.imwrite('static/after.jpg', img1)

        try:
            cv2.imwrite('static/cropped.jpg', cropped)

        except:
            pass

        try:
            image = cv2.imread('static/cropped.jpg', 0)
        except:
            image = cv2.imread('static/file.jpg', 0)
        image = cv2.resize(image, (48,48))
        image = image/255.0
        image = np.reshape(image, (1,48,48,1))
        #model = load_model("model/emotion_model.h5")
        json_file = open('model/emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)

        emotion_model.load_weights("model/emotion_model.h5")
        prediction = emotion_model.predict(image)
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
        prediction = np.argmax(prediction)
        final_prediction = emotion_dict[prediction]
        return render_template('after.html',data=final_prediction)
        music_player(prediction)




if __name__ == "__main__":
    app.run(debug=True)