import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model("drowsiness_model.h5")

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Diccionario de clases
class_labels = {0: "Drowsy", 1: "Non Drowsy"}

# Captura de video
cap = cv2.VideoCapture(0)

# Variables para la detección de somnolencia
drowsy_start_time = None  # Momento en que se detectó somnolencia por primera vez
alert_triggered = False  # Estado de la alerta

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))  # Detectar rostros

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Tomar el primer rostro detectado

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extraer solo el rostro y redimensionarlo
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (227, 227))
        face_roi = np.expand_dims(face_roi, axis=0) / 255.0  # Normalizar

        # Hacer la predicción
        prediction = model.predict(face_roi)
        label = class_labels[np.argmax(prediction)]

        # Mostrar el estado del usuario en la pantalla
        cv2.putText(frame, f"Estado: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Si el modelo detecta "Drowsy", iniciamos el temporizador
        if label == "Drowsy":
            if drowsy_start_time is None:
                drowsy_start_time = time.time()  # Guardar el tiempo de inicio
            elif time.time() - drowsy_start_time >= 3 :  # 3 segundos han pasado
                print("⚠ ALERTA: Estado de somnolencia detectado por más de 3 segundos ⚠")
                cv2.putText(frame, "ALERTA: Despierta!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                alert_triggered = True

        else:
            # Si no está somnoliento, reiniciar el contador
            drowsy_start_time = None
            alert_triggered = False

    else:
        # Mostrar mensaje si no se detecta rostro
        cv2.putText(frame, "No se detecta rostro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar la imagen en pantalla
    cv2.imshow("Drowsiness Detection", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

