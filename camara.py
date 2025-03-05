import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("drowsiness_model.h5")

# Diccionario de clases
class_labels = {0: "Drowsy", 1: "Non Drowsy"}

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    x1, y1 = width // 2 - 113, height // 2 - 113  # Coordenadas superiores izquierda del rectángulo
    x2, y2 = width // 2 + 113, height // 2 + 113  # Coordenadas inferiores derecha del rectángulo
    
    # Dibujar un rectángulo guía en la pantalla
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Coloca tu rostro dentro del cuadro", (50, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Recortar el área del rostro
    face_roi = frame[y1:y2, x1:x2]
    
    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
        # Redimensionar la imagen al tamaño del modelo (227x227)
        img = cv2.resize(face_roi, (227, 227))
        img = np.expand_dims(img, axis=0)  # Agregar dimensión de batch
        img = img / 255.0  # Normalizar

        # Hacer predicción
        prediction = model.predict(img)
        label = class_labels[np.argmax(prediction)]

        # Mostrar el resultado en pantalla
        cv2.putText(frame, f"Estado: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Drowsiness Detection", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
