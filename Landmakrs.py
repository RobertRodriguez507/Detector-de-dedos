import cv2
import mediapipe as mp

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

# Función para verificar si los dedos están extendidos
def detectar_dedos(mano_landmarks):
    dedos = {}

    # Tamaño de la imagen
    altura, ancho = 480, 640  # Puedes obtener dinámico si lo necesitas

    # Obtener los puntos clave como pixeles
    def get_px(id):
        lm = mano_landmarks.landmark[id]
        return int(lm.x * ancho), int(lm.y * altura)

    # Pulgar: Compara X porque el pulgar se mueve horizontal
    x4, _ = get_px(4)
    x3, _ = get_px(3)
    dedos["Pulgar"] = "Extendido" if x4 > x3 else "Doblado"

    # Otros dedos: Compara Y (hacia arriba es menor valor)
    dedos["Indice"]  = "Extendido" if mano_landmarks.landmark[8].y < mano_landmarks.landmark[6].y else "Doblado"
    dedos["Medio"]   = "Extendido" if mano_landmarks.landmark[12].y < mano_landmarks.landmark[10].y else "Doblado"
    dedos["Anular"]  = "Extendido" if mano_landmarks.landmark[16].y < mano_landmarks.landmark[14].y else "Doblado"
    dedos["Menique"] = "Extendido" if mano_landmarks.landmark[20].y < mano_landmarks.landmark[18].y else "Doblado"

    return dedos

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar la mano
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener estado de los dedos
            dedos = detectar_dedos(hand_landmarks)

            # Mostrar texto en pantalla
            y_offset = 30
            for dedo, estado in dedos.items():
                texto = f"{dedo}: {estado}"
                cv2.putText(frame, texto, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

    cv2.imshow("Detector de Dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
