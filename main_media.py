import cv2
import mediapipe as mp

# Initialisation des modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Webcam non detectee")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Effet miroir (plus naturel)
    frame = cv2.flip(frame, 1)

    # Conversion en RGB (obligatoire)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection
    results = hands.process(rgb_frame)

    # Si une main est détectée
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les points de la main
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Texte IA
            cv2.putText(
                frame,
                "MAIN DETECTEE (IA)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Projet IA - Mediapipe Hand Tracking", frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()