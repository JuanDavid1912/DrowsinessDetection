import cv2
import numpy as np
import tensorflow as tf
import time
import pygame
import threading
from tensorflow.keras.models import load_model
from picamera2 import Picamera2, Preview

# CONFIGURACIÓN GENERAL
ALERT_SOUND_PATH = "alerta.mp3"
DROWSY_INPUT_SIZE = (224, 224)
YAWN_INPUT_SIZE = (320, 320)
DISPLAY = True  # Cambia a False si estás en modo headless
PREDICTION_INTERVAL = 5  # Número de frames entre predicciones

# Inicializar pygame mixer para sonido
pygame.mixer.init()

# Función para reproducir alerta sonora sin duplicar sonidos
def play_alert_sound():
    def _play():
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(ALERT_SOUND_PATH)
            pygame.mixer.music.play()
    threading.Thread(target=_play, daemon=True).start()

# Cargar modelos
drowsy_model = load_model("mobilenet_drowsiness_best.h5")
yawn_model = load_model("mobilenet_yawn_best.h5")

# Cargar clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Etiquetas
drowsy_class_labels = {0: "drowsy", 1: "non_drowsy"}
yawn_class_labels = {0: "no_yawn", 1: "yawn"}

# Configuración de la cámara
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)  # Resolución reducida para rendimiento
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
print("✅ Cámara iniciada correctamente.")

# Variables de control
drowsy_start_time = None
alert_triggered = False
frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]

            frame_count += 1
            if frame_count % PREDICTION_INTERVAL == 0:
                # Preprocesamiento
                face_drowsy = cv2.resize(face_roi, DROWSY_INPUT_SIZE)
                face_drowsy = np.expand_dims(face_drowsy, axis=0) / 255.0

                face_yawn = cv2.resize(face_roi, YAWN_INPUT_SIZE)
                face_yawn = np.expand_dims(face_yawn, axis=0) / 255.0

                # Predicciones
                drowsy_pred = drowsy_model.predict(face_drowsy, verbose=0)
                yawn_pred = yawn_model.predict(face_yawn, verbose=0)

                drowsy_label = drowsy_class_labels[np.argmax(drowsy_pred)]
                yawn_label = yawn_class_labels[np.argmax(yawn_pred)]
            # Mostrar resultados (aunque se actualicen solo cada N frames)
            cv2.putText(frame, f"Somnolencia: {drowsy_label}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Bostezo: {yawn_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Control de alerta
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
            drowsy_start_time = None
            alert_triggered = False
            cv2.putText(frame, "No se detecta rostro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Mostrar ventana si hay pantalla conectada
        if DISPLAY:
            cv2.imshow("Drowsiness & Yawn Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n🛑 Interrupción por teclado. Cerrando programa...")

finally:
    picam2.stop()
    if DISPLAY:
        cv2.destroyAllWindows()
    print("✅ Recursos liberados correctamente.")


