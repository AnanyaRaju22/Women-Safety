import cv2
import tensorflow as tf
from utils.preprocessing import preprocess_face

# Load model
model = tf.keras.models.load_model("gender_classification/model/gender_model.h5")

# Labels
labels = ['Male', 'Female']

# Open camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = preprocess_face(frame, (x, y, w, h))
        pred = model.predict(face)
        gender = labels[int(pred[0][0] > 0.5)]

        # Draw on screen
        color = (0, 255, 0) if gender == 'Female' else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Gender Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
