import cv2
import numpy as np
import tensorflow as tf
import time
import pygame
import threading
from tensorflow.keras.models import load_model

# Inicializar pygame mixer para sonido
pygame.mixer.init()
ALERT_SOUND_PATH = "alerta.mp3"

# Función para reproducir alerta sonora en segundo plano
def play_alert_sound():
    def _play():
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(ALERT_SOUND_PATH)
            pygame.mixer.music.play()
    threading.Thread(target=_play, daemon=True).start()

# Cargar los modelos
drowsy_model = load_model("mobilenet_drowsiness_best.h5")
yawn_model = load_model("mobilenet_yawn_best.h5")

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Diccionarios de clases
drowsy_class_labels = {0: "drowsy", 1: "non_drowsy"}
yawn_class_labels = {0: "no_yawn", 1: "yawn"}

# Tamaños de entrada
DROWSY_INPUT_SIZE = (224, 224)
YAWN_INPUT_SIZE = (320, 320)

# Inicializar cámara con OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()
print("✅ Cámara iniciada correctamente.")

# Variables de control
drowsy_start_time = None
alert_triggered = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]

            # Redimensionar entradas
            face_drowsy = cv2.resize(face_roi, DROWSY_INPUT_SIZE)
            face_drowsy = np.expand_dims(face_drowsy, axis=0) / 255.0

            face_yawn = cv2.resize(face_roi, YAWN_INPUT_SIZE)
            face_yawn = np.expand_dims(face_yawn, axis=0) / 255.0

            # Predicciones
            drowsy_pred = drowsy_model.predict(face_drowsy, verbose=0)
            yawn_pred = yawn_model.predict(face_yawn, verbose=0)

            drowsy_label = drowsy_class_labels[np.argmax(drowsy_pred)]
            yawn_label = yawn_class_labels[np.argmax(yawn_pred)]

            # Mostrar estado
            cv2.putText(frame, f"Somnolencia: {drowsy_label}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Bostezo: {yawn_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Alerta por somnolencia
            if drowsy_label == "drowsy":
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time >= 3:
                    print("⚠ ALERTA: Estado de somnolencia detectado por más de 3 segundos ⚠")
                    cv2.putText(frame, "ALERTA: Despierta!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if not alert_triggered:
                        play_alert_sound()
                        alert_triggered = True
            else:
                drowsy_start_time = None
                alert_triggered = False
        else:
            cv2.putText(frame, "No se detecta rostro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            drowsy_start_time = None
            alert_triggered = False

        # Mostrar imagen
        cv2.imshow("Drowsiness & Yawn Detection", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrumpido por el usuario.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Recursos liberados.")


