import numpy as np
import cv2
import pickle
import tensorflow as tf
import sklearn as sk
from tf.keras.Model import load_model
from sklearn.metrics.pairwise import cosine_similarity


facenet_model = load_model("facenet_keras.h5")
embeddings = np.load("face_embeddings.npy")
with open("face_labels.pkl", "rb") as f:
    label_encoder = pickle.load(f)

video = cv2.VideoCapture(0)
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(haar_cascade_path)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (160, 160)) / 255.0
        emb = facenet_model.predict(np.expand_dims(resized_img, axis=0))[0]

       
        scores = cosine_similarity([emb], embeddings)
        best_match = np.argmax(scores)
        
        if scores[0][best_match] > 0.7:
            name = label_encoder.inverse_transform([best_match])[0]
        else:
            name = "Unknown"

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
