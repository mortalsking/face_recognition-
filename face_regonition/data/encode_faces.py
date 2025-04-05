import numpy as np
import os
import cv2
import tensorflow as tf
import sklearn as sk
from tensorflow import keras
model = tf.keras.Model()
from tensorflow.keras.models import load_model
from sk.preprocessing import LabelEncoder
import pickle


facenet_model = load_model("facenet_keras.h5")  


faces_dir = "faces"
embeddings = []
labels = []

for filename in os.listdir(faces_dir):
    img = cv2.imread(os.path.join(faces_dir, filename))
    img = cv2.resize(img, (160, 160))  
    img = np.expand_dims(img, axis=0) / 255.0  
    emb = facenet_model.predict(img)[0] 
    embeddings.append(emb)
    labels.append(filename.split("_")[1])


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


np.save("face_embeddings.npy", embeddings)
with open("face_labels.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Faces encoded and saved!")
