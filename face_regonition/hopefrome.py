import cv2
import os

video = cv2.VideoCapture(0)
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(haar_cascade_path)

faces_dir = "faces"
os.makedirs(faces_dir, exist_ok=True)

i = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (100, 100))
        filename = f"{faces_dir}/face_{i}.jpg"
        cv2.imwrite(filename, resized_img)
        i += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or i >= 100:  
        break

video.release()
cv2.destroyAllWindows()
