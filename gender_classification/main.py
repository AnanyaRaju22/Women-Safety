import cv2
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("gender_classification/model/gender_model.h5")

# Labels
labels = ['Male', 'Female']

# Function to preprocess face correctly
def preprocess_face(img, box):
    x, y, w, h = box
    face = img[y:y+h, x:x+w]

    # Resize to match model input size
    face = cv2.resize(face, (249, 249))

    # Convert BGR to RGB and normalize
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0

    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    return face

# Open webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = preprocess_face(frame, (x, y, w, h))
        pred = model.predict(face)

        # Handle sigmoid or softmax output
        if pred.shape[-1] == 1:  # Sigmoid
            prob = float(pred[0][0])
            gender = labels[int(prob > 0.5)]
        else:  # Softmax
            index = int(np.argmax(pred[0]))
            prob = float(pred[0][index])
            gender = labels[index]

        # Color and label
        color = (0, 255, 0) if gender == 'Female' else (255, 0, 0)
        label_text = f"{gender} ({prob:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Gender Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
